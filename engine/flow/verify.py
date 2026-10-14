"""
The DLP side of a work state: run its check, say what the DLP saw, write the cycle folder.

    resolve_recipe()   the recipe a work state verifies with for the active profile: a
                       recipe of the profile (name, file or mapping) or one composed from
                       the work state's inline zone checks + the profile's zones / preset /
                       golden (what that part variant looks like)
    check_config()     everything a configuration needs that the pure schema can't see:
                       every profile's recipes resolve, the check named to observe each
                       expected signal (verify.observe) exists in them, and no work state
                       is impossible to confirm
    verify_step()      one DLP check through the recipe runner (its own run folder inside
                       the cycle folder: result.json, overlay, the fault image the projector
                       paints) -> the DLP answer the cycle classifies
    observe()          per expected signal: what the DLP saw (true / false), or None when it
                       could not tell (an INVALID check: unseen is not absent)
    open_cycle() / write_cycle()   the cycle folder: run.log.jsonl + result.json (schema
                       0.4: station config, profile, steps, permissions, disposition)
    run_cycle()        a whole cycle, synchronously (`scan run <configuration>`, tests);
                       an auto-acknowledged prompt is answered at most retries + 1 times
                       per work state, then the unattended cycle ends PULL

The step scans reuse the recipe runner and its run folders; nothing here opens hardware
itself (the runner goes through the engine verbs, so every engine rule applies).
"""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Callable, Optional

from .. import badparts
from .. import disposition as D
from .. import runs as R
from .cycle import Cycle
from .schema import StationConfigError, StationConfiguration, Step, rel_path

STEPS_DIR = "steps"                     # a cycle folder's step run folders


# ------------------------------------------------------------------ recipes of a profile

def _name(*parts: str) -> str:
    """A recipe / run name from parts ([A-Za-z0-9_.-], max 64): long ones keep a hash."""
    n = ".".join(p for p in parts if p)
    if len(n) <= 64:
        return n
    return n[:55] + "." + hashlib.sha1(n.encode("utf-8")).hexdigest()[:8]


def compose(cfg: StationConfiguration, profile: str, step: Step) -> dict:
    """The recipe mapping of a work state's INLINE zone checks: the active profile says where
    the zones are and how to scan (zones, preset, scan, part frame, golden, feedback, pull),
    the work state says what is checked in them."""
    p = cfg.profile(profile)
    zones: dict = {}
    for c in step.verify.get("checks") or []:
        if c["zone"] not in p.zones:
            raise StationConfigError(f"work state {step.id}: profile {p.name} defines no zone "
                                     f"{c['zone']!r} (profiles.{p.name}.zones)")
        z = zones.setdefault(c["zone"], {**copy.deepcopy(p.zones[c["zone"]]), "id": c["zone"],
                                         "checks": []})
        z["checks"].append({"type": c["type"], "id": c["id"],
                            "params": copy.deepcopy(c.get("params") or {})})
    for zid, z in p.zones.items():          # reference-only zones (the part frame's datum)
        if z.get("datum") is True and zid not in zones:
            zones[zid] = {**copy.deepcopy(z), "id": zid, "checks": []}
    preset = p.preset or cfg.parameters.get("preset")
    if not preset:
        raise StationConfigError(f"work state {step.id}: inline checks need a preset "
                                 f"(profiles.{p.name}.preset or parameters.preset)")
    doc = {"schema_version": 1, "name": _name(cfg.station_id, p.name, step.id),
           "description": f"{cfg.station_id}: {step.title or step.id} ({p.name})",
           "preset": preset, "zones": list(zones.values())}
    for k in ("scan", "part", "golden", "feedback", "pull"):
        if getattr(p, k) is not None:
            doc[k] = copy.deepcopy(getattr(p, k))
    return doc


def resolve_recipe(cfg: StationConfiguration, profile: str, step: Step):
    """-> the Recipe this work state verifies with for `profile` (validated; RecipeError /
    StationConfigError say what is wrong)."""
    from .. import api
    p = cfg.profile(profile)
    v = step.verify
    if v.get("kind") == "checks":
        return api.recipe_load(compose(cfg, profile, step))
    if v.get("kind") != "recipe":
        raise StationConfigError(f"work state {step.id} of {cfg.station_id} is a single job, "
                                 "not a DLP check of a flow: `scan run` runs it")
    key = v.get("recipe")
    if key not in p.recipes:
        raise StationConfigError(f"work state {step.id}: profile {p.name} has no recipe "
                                 f"{key!r} (its recipes: {', '.join(p.recipes) or 'none'})")
    return api.recipe_load(copy.deepcopy(p.recipes[key]))


def _observer(step: Step, signal: str, zones: list) -> tuple:
    """The check that observes `signal` in a list of zones [{id, checks: [{id, type, ...}]}]
    -> (zone id, check dict, invert) or (None, None, why). Only the check the work state
    NAMES for that signal (verify.observe): nothing is guessed - a signal no check was named
    for is observed by none, whatever checks the recipe happens to have."""
    ref = (step.verify.get("observe") or {}).get(signal)
    if ref is None:
        return None, None, (f"no check observes {signal}: name it in verify.observe "
                            "({\"" + signal + "\": \"<zone>/<check>\"})")
    for z in zones:
        if z["id"] == ref["zone"]:
            for ch in z.get("checks") or []:
                if ch.get("id") == ref["check"]:
                    return z["id"], ch, bool(ref.get("invert"))
    return None, None, (f"verify.observe names {ref['zone']}/{ref['check']}, which this "
                        "work state's recipe does not have")


def _saw(ch: dict) -> Optional[bool]:
    """What one check result saw, as a boolean: a presence check -> part present (whatever
    it expected); any other check -> it passed. None when it could not tell: INVALID, or a
    verdict that is itself in doubt (`pull`)."""
    if ch.get("verdict") not in ("PASS", "FAIL") or ch.get("pull"):
        return None
    if ch.get("type") == "presence":
        seen = (ch.get("details") or {}).get("present")
        if isinstance(seen, bool):
            return seen
        want = (ch.get("params") or {}).get("expect", "present") == "present"
        return want if ch["verdict"] == "PASS" else not want
    return ch["verdict"] == "PASS"


def observe(step: Step, zones: list, pull_reasons: Optional[list] = None) -> tuple:
    """Per signal the work state expects: what the DLP saw (True / False), or None when it
    could not tell -> ({signal: bool | None}, {signal: "zone/check" | why not}). A reading
    the run itself holds in doubt (`pull_reasons` naming that zone + check: a value in the
    guard band, a PASS its own numbers do not hold up) is not a reading: None."""
    seen, notes = {}, {}
    for sig in step.expects:
        zid, ch, inv = _observer(step, sig, zones or [])
        if ch is None:
            seen[sig], notes[sig] = None, inv
            continue
        v = _saw(ch)
        notes[sig] = f"{zid}/{ch.get('id')}" + (" (inverted)" if inv else "")
        doubt = next((r for r in pull_reasons or [] if isinstance(r, dict)
                      and r.get("zone") == zid and r.get("check") == ch.get("id")), None)
        if v is None:
            notes[sig] += f": {ch.get('verdict')} - {ch.get('message') or 'could not tell'}"
        elif doubt is not None:
            v = None
            notes[sig] += f": in doubt ({doubt.get('reason')}: {doubt.get('message')})"
        seen[sig] = (not v) if (inv and v is not None) else v
    return seen, notes


def check_step(cfg: StationConfiguration, profile: str, step: Step) -> list:
    """Problems of one work state with one profile, beyond the schema: its recipe must
    resolve, each expected signal needs a check that observes it, and the work state must
    be confirmable at all (the observing check passing must mean what `expects` says)."""
    try:
        r = resolve_recipe(cfg, profile, step)
    except Exception as e:                  # noqa: BLE001 - listed, every profile and step
        return [f"work state {step.id}, profile {profile}: {type(e).__name__}: "
                + " ".join(str(e).split())]
    zones = [{"id": z.id, "checks": [{"id": c.id, "type": c.type, "params": dict(c.check.p)}
                                     for c in z.checks]} for z in r.zones]
    out = []
    for sig, want in step.expects.items():
        zid, ch, inv = _observer(step, sig, zones)
        if ch is None:
            out.append(f"work state {step.id}, profile {profile} (recipe {r.name}): expects "
                       f"{sig} but {inv}")
            continue
        when_pass = True
        if ch["type"] == "presence":
            when_pass = ch["params"].get("expect", "present") == "present"
        if inv:
            when_pass = not when_pass
        if when_pass != want:
            out.append(f"work state {step.id}, profile {profile} (recipe {r.name}): can never "
                       f"be CONFIRMED - it expects {sig} {str(want).lower()}, but check "
                       f"{zid}/{ch['id']} passes only when {sig} is {str(when_pass).lower()}")
    return out


def check_config(cfg: StationConfiguration) -> list:
    """Every problem of a configuration the schema could not see (needs the engine: the
    recipe library). [] = every work state can be verified with every profile."""
    if cfg.legacy:
        return []
    out = []
    for name in cfg.profiles:
        for step in cfg.flow:
            out += check_step(cfg, name, step)
    return out


def preset_for(cfg: StationConfiguration, profile: str, recipe):
    """The preset a work state scans with: the profile's (else the configuration's), else
    the recipe's own pinned one. -> Preset (loaded as the session's active one)."""
    from .. import api, config
    s = api._engine()
    want = cfg.preset_for(profile)
    if not want:
        return api._recipe_preset(recipe, None)
    if s.preset is not None and s.preset.name == want and s.preset.mock == s.mock:
        return s.preset
    p = config.load_preset(want, presets_dir=s.presets_dir, store=s.store, mock=s.mock)
    s.preset = p
    return p


# ------------------------------------------------------------------ one DLP check

def dlp_answer(res: dict, step: Step, recipe_name: Optional[str] = None) -> dict:
    """A recipe run's result document -> the DLP answer of the work state (what the cycle
    classifies and the step records): verdict, disposition, what was observed per expected
    signal, the run it came from, a one-line message, the zones in short."""
    zones = res.get("zones") or []
    seen, notes = observe(step, zones, res.get("pull_reasons"))
    msg = ""
    for want in ("FAIL", "INVALID"):
        for z in zones:
            for ch in z.get("checks") or []:
                if not msg and ch.get("verdict") == want:
                    msg = f"{z['id']}/{ch['id']}: {ch.get('message') or want}"
    fb = res.get("feedback") or {}
    return {"step": step.id,                    # the work state this answer is for
            "result": res.get("result"), "disposition": res.get("disposition"),
            "pull_reasons": copy.deepcopy(res.get("pull_reasons") or []),
            "invalid_reason": res.get("invalid_reason"),
            "observed": seen, "observe": notes,
            "run_id": res.get("run_id"), "run_folder": res.get("run_folder"),
            "recipe": recipe_name or (res.get("recipe") or {}).get("name"),
            "preset": (res.get("capture") or {}).get("preset"),
            "scan_id": (res.get("part") or {}).get("scan_id"),
            "message": " ".join(str(msg or res.get("invalid_reason") or "").split()),
            "zones": [{"id": z.get("id"), "verdict": z.get("verdict"),
                       "checks": [{"id": c.get("id"), "type": c.get("type"),
                                   "verdict": c.get("verdict"),
                                   "message": " ".join(str(c.get("message") or "").split())}
                                  for c in z.get("checks") or []]} for z in zones],
            "feedback_rendered": bool(fb.get("rendered")),
            "feedback_refused": fb.get("refused"),
            "warnings": [str(w) for w in (res.get("warnings") or [])[:8]]}


def verify_step(cfg: StationConfiguration, profile: str, step_id: str, *, runs_dir,
                operator_confirmed: bool = False, mock_scene: Optional[str] = None,
                scan=None, mock_place: Optional[Callable] = None) -> dict:
    """Run the DLP check of one work state for the active profile: scan with the profile's
    preset, run the recipe's checks, write a NEW run folder under runs_dir (nothing is
    projected here: the cycle's project_fault action paints the fault afterwards).
    scan: a stored metric scan to check instead of capturing (offline, tests). mock_scene
    (MOCK only): 'scene:<name>[:<variant>]' or 'none' the simulated human places first.
    -> {"dlp": the DLP answer, "run_id", "run_dir", "result" (the run's result.json),
    "exit_code"}. A check that cannot run at all still answers: INVALID, with the error."""
    from .. import api
    from ..recipes import library as recipe_lib
    from ..recipes import runner
    s = api._engine()
    step = cfg.step(step_id)
    recipe = resolve_recipe(cfg, profile, step)
    preset = preset_for(cfg, profile, recipe)
    place = mock_place
    if mock_scene and place is None:
        if not s.mock:
            raise ValueError("mock_scene is MOCK ONLY: physical steps on the real rig are human")
        place = mock_placer(s, mock_scene)
    tier = recipe_lib.tier_of(s.store, api._recipes_dir(), recipe.name) \
        if recipe.source is not None and recipe.source.stem == recipe.name else None
    out = runner.run_recipe(
        s, recipe, preset, runs_dir=runs_dir, operator_confirmed=bool(operator_confirmed),
        project_feedback=False, scan=scan, run_name=_name(cfg.station_id, step.id),
        mock_place=place, tier=tier or ("file" if recipe.source else "mapping"),
        station=cfg.station_id,
        context={"station_config": cfg.source.name if cfg.source else None, "profile": profile,
                 "job": {"config": cfg.source.name if cfg.source else None,
                         "work_state": step.id},
                 badparts.KEY: bad_part_output(cfg)})
    res = out["result"]
    return {"dlp": {**dlp_answer(res, step, recipe.name), "exit_code": out["exit_code"]},
            "run_id": out["run_id"], "run_dir": out["run_dir"], "result": res,
            "exit_code": out["exit_code"]}


def mock_placer(s, scene: str) -> Callable:
    """MOCK only: what the simulated human places before the scan: a named scene, or
    'none' (the empty wall: no part)."""
    from ..capture import mock_scenes
    if str(scene).strip().lower() in ("none", "scene:none"):
        def empty():
            s.bench.clear()
            s.bench.use_stereo()
        return empty
    mock_scenes.parse(scene)                    # refuse a typo before the rig
    return lambda: mock_scenes.place(s.bench, scene)


# ------------------------------------------------------------------ the cycle folder

def runs_dir_for(cfg: StationConfiguration) -> Path:
    """Where a configuration's cycle folders go: $SCANNER_RUNS_DIR, else parameters.runs_dir
    (relative to the config file), else runs/ next to it."""
    env = os.environ.get("SCANNER_RUNS_DIR")
    if env:
        return Path(env)
    base = cfg.source.parent if cfg.source else Path.cwd()
    return rel_path(base, cfg.parameters.get("runs_dir")) or base / "runs"


def open_cycle(cfg: StationConfiguration, profile: str, *, runs_dir=None, mock: bool,
               operator_confirmed: bool = False, source: str = "") -> dict:
    """A NEW cycle folder + its run.log.jsonl (never an existing one). -> {cycle_id,
    cycle_dir, log (RunLog)}; step run folders go under <cycle_dir>/steps/."""
    cycle_id, cycle_dir = R.new_run(runs_dir or runs_dir_for(cfg), cfg.station_id)
    log = R.RunLog(Path(cycle_dir) / "run.log.jsonl", cycle_id)
    log.event("run_start", station=cfg.station_id, mode="MOCK" if mock else "REAL", job="flow",
              config=cfg.source.name if cfg.source else None, profile=profile,
              work_states=[s.id for s in cfg.flow], operator_confirmed=bool(operator_confirmed),
              source=source)
    return {"cycle_id": cycle_id, "cycle_dir": Path(cycle_dir), "log": log}


def cycle_document(cfg: StationConfiguration, record: dict, *, cycle_id: str, cycle_dir,
                   mock: bool, operator_confirmed: bool = False) -> dict:
    """The cycle's result document (the run schema 0.4, engine/runs.py): station config,
    profile, steps [{id, expects, sensor_inputs, dlp_result, outcome, permission_granted,
    actions, disposition}], permissions, the overall verdict and disposition. CONTINUE only
    when every required work state was CONFIRMED."""
    steps = copy.deepcopy(record.get("steps") or [])
    name = cfg.source.name if cfg.source else None
    doc = R.empty_result(run_id=cycle_id, run_folder=str(cycle_dir), mock=mock,
                         operator_confirmed=operator_confirmed, station=cfg.station_id,
                         job={"type": "flow", "base": None, "zone": None, "config": name,
                              "flow": cfg.flow_source,
                              "work_states": [s.id for s in cfg.flow]},
                         part={"role": "test", "label": record.get("profile") or "part"},
                         station_config=name, profile=record.get("profile"))
    result = record.get("result")
    disp = record.get("disposition")
    reasons = copy.deepcopy(record.get("pull_reasons") or [])
    if result not in R.RESULTS:                 # a cycle that never ended: never a pass
        result, disp = "INVALID", "PULL"
        reasons.append(D.reason("error", "the cycle was recorded before it ended"))
    done = [s for s in steps if (s.get("dlp_result") or {}).get("run_id")]
    last = (done[-1]["dlp_result"] if done else {})
    doc["capture"]["preset"] = last.get("preset")
    doc["part"]["scan_id"] = last.get("scan_id")
    doc["steps"] = steps
    doc["permissions"] = dict(record.get("permissions") or {})
    seen = copy.deepcopy(record.get("sensor_mismatches") or [])
    for s in steps:                             # a record without the history: the rows' own
        for m in s.get("sensor_mismatch") or []:
            if not any(x.get("step") == s["id"] and x.get("signal") == m.get("signal")
                       for x in seen):
                seen.append({"step": s["id"], "attempt": s.get("attempts"), **m})
    left = [{"step": s["id"], "outcome": s.get("outcome"), "required": s.get("required")}
            for s in steps if s.get("outcome") != "CONFIRMED"]
    doc["cycle"] = {"ended_by": record.get("ended_by"), "kickout": record.get("kickout"),
                    # true when a sensor and the DLP disagreed at ANY verify of the cycle,
                    # also one put right since (sensor_mismatch_open: still CORRECTED now)
                    "sensor_mismatch": bool(seen),
                    "sensor_mismatch_open": bool(record.get("sensor_mismatch")),
                    "sensor_mismatches": seen,
                    "unconfirmed": left,
                    "work_state": record.get("step"),
                    "outcomes": {s["id"]: s.get("outcome") for s in steps}}
    doc["result"] = doc["engine_verdict"] = result
    doc["disposition"], doc["pull_reasons"] = disp, reasons
    doc["invalid_reason"] = None
    if result == "INVALID":
        doc["invalid_reason"] = "; ".join(str(r.get("message") or r.get("reason"))
                                          for r in reasons[:3]) or "the cycle did not confirm"
    doc["started"] = record.get("started") or doc["started"]
    doc["warnings"] = [w for s in steps for w in ((s.get("dlp_result") or {}).get("warnings")
                                                 or [])][:20]
    if disp == "CONTINUE" and left:             # work states that are not required
        doc["warnings"].append("CONTINUE with work state(s) left unconfirmed (not required): "
                               + ", ".join(f"{x['step']} ({x['outcome']})" for x in left))
    doc["outputs"]["report"] = None
    doc["outputs"][badparts.KEY] = bad_part_output(cfg)
    if doc["outputs"][badparts.KEY] is None:    # the station says nothing: its recipes'
        doc["outputs"][badparts.KEY], why = recipes_bad_part_output(
            cfg, record.get("profile"), done, record.get("kickout"))
        if why:
            doc["warnings"].append(why)
    return doc


def bad_part_output(cfg: StationConfiguration) -> Optional[str]:
    """The configuration's parameters.bad_part_output (relative to its file; "off"; None: it
    says nothing) - for its cycles, and for a work state's check run on its own, where it
    wins over the recipe's (engine/badparts.py). A value that is no folder was refused when
    the configuration was loaded; here it never stops a cycle (None)."""
    try:
        return badparts.configured(cfg.parameters.get(badparts.KEY),
                                   cfg.source.parent if cfg.source else None)
    except ValueError:
        return None


def recipes_bad_part_output(cfg: StationConfiguration, profile: Optional[str], steps: list,
                            kickout=None) -> tuple:
    """The bad-part output of a cycle whose station configuration names none (engine/
    badparts.py): the outputs.bad_part_output of the recipes of the work states it checked
    (`steps`, in flow order) - the one they name; when they name different ones, the one of
    the work state that speaks for the cycle (where it was kicked out, else its last check
    that names one), with a warning. -> (a folder | "off" | None, the warning | None). Never
    raises: a recipe that cannot be read here says nothing."""
    from ..recipes import runner
    said = {}
    for st in steps:
        try:
            v = runner.recipe_bad_part_output(
                resolve_recipe(cfg, profile or cfg.parameters.get("default_profile"),
                               cfg.step(st["id"])))
        except Exception:                       # noqa: BLE001 - never stops a cycle
            continue
        if v is not None:
            said[st["id"]] = v
    if len(set(said.values())) <= 1:
        return next(iter(said.values()), None), None
    kick = kickout.get("step") if isinstance(kickout, dict) else None
    speaker = kick if kick in said else list(said)[-1]
    return said[speaker], (
        f"{badparts.WARNING}: the recipes of this cycle's work states name different bad-part "
        f"outputs ({'; '.join(f'{k}: {v}' for k, v in said.items())}): this cycle uses the "
        f"one of work state {speaker}; set parameters.bad_part_output in the station "
        "configuration to choose one for every cycle")


def write_cycle(cfg: StationConfiguration, record: dict, opened: dict, *, mock: bool,
                operator_confirmed: bool = False, events: Optional[list] = None) -> dict:
    """result.json of a cycle (written once) + run_end in its log. -> the document."""
    log = opened.get("log")
    for ev in events or []:
        log_event(log, ev)
    doc = cycle_document(cfg, record, cycle_id=opened["cycle_id"],
                         cycle_dir=opened["cycle_dir"], mock=mock,
                         operator_confirmed=operator_confirmed)
    out = R.write_result(opened["cycle_dir"], doc)
    if log is not None and not log.closed:
        log.event("run_end", result=doc["result"], disposition=doc["disposition"],
                  pull_reasons=doc["pull_reasons"], exit_code=exit_code(doc),
                  result_json=out.name, invalid_reason=doc["invalid_reason"],
                  permissions=doc["permissions"])
        log.close()
    return doc


def log_event(log, ev: dict) -> None:
    """One cycle event (engine/flow/cycle.py Cycle.events) on the cycle's log."""
    if log is None or log.closed:
        return
    ev = dict(ev)
    log.event(ev.pop("event", "event"), ev.pop("level", "info"), **ev)


def exit_code(doc: dict) -> int:
    """CONTINUE 0, REJECT 1, PULL 5; an ABORTED cycle keeps the documented 1."""
    if doc.get("result") == "ABORTED":
        return 1
    return D.exit_code(doc)


# ------------------------------------------------------------------ a whole cycle, in line

def run_cycle(cfg: StationConfiguration, profile: Optional[str] = None, *,
              inputs: Optional[dict] = None, runs_dir=None, operator_confirmed: bool = False,
              mock_scene: Optional[str] = None, scans: Optional[dict] = None,
              on_prompt: Optional[Callable] = None, project: bool = True,
              hold_s: Optional[float] = None, say: Optional[Callable] = None) -> dict:
    """One cycle of a station configuration's flow, synchronously: every work state is
    verified in turn (no waiting for STEP: the caller is the line), its actions run, the
    cycle folder is written. For `scan run <configuration>`, scripts and tests; the operator
    page and a PLC drive the same cycle step by step through the command bus.

    inputs: the sensor claims {signal: bool} (a mapping, or a callable(step id) -> mapping
    read when each scan starts). mock_scene (MOCK only): the scene placed before every scan
    (a mapping {step id: scene} places one per work state). scans: {step id: stored metric
    scan} checked instead of capturing. on_prompt(prompt) -> True to acknowledge (the
    operator did it), anything else: nobody acknowledges - the cycle ends PULL (fail
    closed; the default). project: paint the faults (project_fault) with the projector.
    -> {cycle_id, cycle_dir, result (the result.json document), exit_code, record}"""
    from .. import api, rules
    s = api._engine()
    name = cfg.default_profile if profile is None else profile
    cfg.profile(name)                           # refuse an unknown profile before a folder
    if cfg.legacy:
        raise StationConfigError(f"{cfg.station_id} is a single-job station config: `scan "
                                 "run` runs its job")
    if not s.mock and not operator_confirmed:
        raise rules.RuleViolation("the real rig needs a human to confirm the physical setup "
                                  "first (part placed, rig untouched): operator_confirmed")
    opened = open_cycle(cfg, name, runs_dir=runs_dir, mock=s.mock,
                        operator_confirmed=operator_confirmed, source="run_cycle")
    cyc = Cycle(cfg, name, meta={"cycle_id": opened["cycle_id"]})
    say = say or (lambda *a: None)
    acks: dict = {}                             # work state -> prompts acknowledged so far

    def claims(step_id: str) -> dict:
        v = inputs(step_id) if callable(inputs) else inputs
        return dict(v or {})

    def drain():
        for ev in cyc.drain_events():
            log_event(opened["log"], ev)

    effects = cyc.begin(force=True)
    try:
        while True:
            drain()
            if not effects:                     # (a fault to paint may still be queued after
                if cyc.state == "DONE":         # the cycle ended: a kickout's project_fault)
                    break
                effects = [("wait", cyc.waiting)]
            name_, payload = effects.pop(0)
            if name_ == "verify":
                step = cyc.step
                req = cyc.request(claims(step.id))
                say(f"  work state {step.index}/{len(cyc.steps)} {step.id}: verifying "
                    f"(sensors say {req['inputs'] or 'nothing'})")
                scene = mock_scene.get(step.id) if isinstance(mock_scene, dict) else mock_scene
                try:
                    s.check_cancel()
                    out = verify_step(cfg, name, step.id,
                                      runs_dir=Path(opened["cycle_dir"]) / STEPS_DIR,
                                      operator_confirmed=operator_confirmed, mock_scene=scene,
                                      scan=(scans or {}).get(step.id))
                    dlp = out["dlp"]
                except rules.Cancelled as e:
                    effects = cyc.abort(f"aborted: {e}")
                    continue
                except Exception as e:      # noqa: BLE001 - a check that can't run is a doubt
                    dlp = {"error": f"{type(e).__name__}: {e}", "result": "INVALID",
                           "disposition": "PULL"}
                if dlp.get("result") == "ABORTED":
                    effects = cyc.abort("aborted while the work state was verified")
                    continue
                effects = cyc.verified(dlp, claims(step.id))
                row = cyc.rows[step.index - 1]
                say(f"    -> {row['outcome']}: {row['message']}")
            elif name_ == "project":
                # what the projector really did is added to the action the cycle recorded
                act = next((a for a in reversed(cyc.rows[cyc._index_of(payload["step"])]["actions"])
                            if a.get("action") == "project_fault"), {})
                if not project:
                    act["shown"] = False
                    act["why"] = "projection is off for this run (--no-feedback / project_faults)"
                elif payload.get("run_folder"):
                    try:
                        api.recipe_feedback(payload["run_folder"], hold_s=hold_s, cancel=s.cancel)
                        act["shown"] = True
                    except Exception as e:  # noqa: BLE001 - a picture never changes a verdict
                        act["shown"] = False
                        act["why"] = f"{type(e).__name__}: {e}"
                        log_event(opened["log"], {"event": "feedback_error", "level": "warning",
                                                  "step": payload.get("step"),
                                                  "message": f"{type(e).__name__}: {e}"})
            elif name_ == "wait" and cyc.state == "WAIT_STEP":
                effects = cyc.step_now()        # in line: the caller is the line's STEP
            elif name_ == "wait" and cyc.state == "WAIT_ACK":
                p = dict(cyc.prompt or {})
                say(f"    prompt: {p.get('message')} ({p.get('detail')})")
                budget = cyc.step.retries + 1   # an in-line cycle never acknowledges for ever
                if acks.get(p.get("step"), 0) >= budget:
                    effects = cyc.give_up(
                        f"work state {p.get('step')}: acknowledged {budget} time(s) and still "
                        f"not confirmed ({p.get('outcome')}) - an unattended cycle ends here",
                        ended_by="unconfirmed", code="not_confirmed")
                elif on_prompt is not None and on_prompt(p) is True:
                    acks[p.get("step")] = acks.get(p.get("step"), 0) + 1
                    effects = cyc.ack("run_cycle")
                else:
                    effects = cyc.give_up(
                        f"work state {p.get('step')}: nobody acknowledged the prompt "
                        f"({p.get('message')!r}) - an unattended cycle ends here")
    except BaseException as e:                  # crashed: the folder still gets its result
        if cyc.state != "DONE":
            cyc.fail(f"{type(e).__name__}: {e}")
        drain()
        write_cycle(cfg, cyc.record(), opened, mock=s.mock,
                    operator_confirmed=operator_confirmed)
        raise
    drain()
    doc = write_cycle(cfg, cyc.record(), opened, mock=s.mock,
                      operator_confirmed=operator_confirmed)
    return {"cycle_id": opened["cycle_id"], "cycle_dir": opened["cycle_dir"], "result": doc,
            "exit_code": exit_code(doc), "record": cyc.record()}
