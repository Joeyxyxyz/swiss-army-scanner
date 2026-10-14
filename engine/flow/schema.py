"""
Station configuration: the parameters, the profile library set and the work states.

    STATION CONFIGURATION  (stations/*.json | .yaml)      what this station is
      parameters    rig preset, library / run folders, the sensor inputs of the line
                    (io.inputs; io.max_age_s: how old a sensor claim may be when a work
                    state is verified, null = no limit), defaults for the work states
      profiles      the PROFILE LIBRARY SET: every part variant the station can be told to
                    expect (long tube, short tube ...) and what each one looks like: its
                    recipes (zones + checks), preset, named zones, golden
      flow          the WORK STATES: an ordered list of steps, inline or a file under flows/

A station configuration is NOT a work state. The configuration says what the station knows
(parameters + the profiles it can call on); the operator / PLC / HTTP selects the active
PROFILE; the WORK STATES are the steps and permissions that let the line continue to the
next thing.

A work state (one step of the flow):

    id, title
    expects       {signal: true | false}: what the line's own sensors / PLC inputs claim at
                  this step (may be empty)
    verify        the DLP check that confirms or corrects that claim:
                    {"recipe": "<key of the active profile's recipes>"}            or
                    {"checks": [{"zone": "<zone of the active profile>", "type": "presence",
                                 "params": {...}}]}                      (inline zone checks)
                  + "observe": {signal: "zone/check"}: the check whose answer IS that signal.
                  Required for EVERY expected signal: nothing is guessed (a door switch is
                  never "confirmed" by a part-presence check because it was the only one)
    permit        the permission granted when the work state is CONFIRMED (bus output
                  PERMIT_<NAME>), true only while it is confirmed in the current cycle
    on_confirm    next | done | <step id>
    on_mismatch   actions when the DLP disagrees with a sensor (CORRECTED)
    on_not_met    actions when sensor and DLP agree on something the work state does not
                  expect: no part where one is needed, a part where none may be (NOT_MET;
                  default: the on_mismatch list)
    on_defect     actions when the part is there but a check fails (DEFECT)
    on_doubt      actions when the DLP cannot tell (DOUBT; default: kickout PULL; a doubt
                  is never a reject, never grants a permission and is never passed over:
                  no goto, and a list that ends without hold / retry / kickout ends the
                  cycle PULL - also on a work state that is not required)
    retries       re-scans the `retry` action (and the RETRY command) may use (default 1)
    timeout_s     the longest the work state may wait for STEP, or for an ACK (each wait
                  from its start; null = no limit); then its cycle ends PULL
    trigger       step (default: verified on STEP / TRIGGER) | auto (verified as soon as it
                  becomes the current work state)
    required      true (default): the cycle is CONTINUE only when it was CONFIRMED. At
                  least one work state of a flow must be required (a cycle never CONTINUEs
                  with nothing confirmed). false: it may be left unconfirmed on a definite
                  finding (NOT_MET / DEFECT) - never in DOUBT or CORRECTED
    allow_skip    false (default); true needs required false (SKIP never grants, and never
                  skips a DOUBT / CORRECTED work state)
    revoke        permissions of other work states dropped when this one is not confirmed
                  (a list, or "all")

Actions (a short text or a mapping; "when_dlp": {signal: bool} limits one to what the DLP
saw): project_fault | prompt <message> | kickout REJECT|PULL [reason] | hold | retry |
goto <step id>.

Backward compatible: an old single-job station config (station_id, preset, job, part) loads
as a configuration with one implicit profile ("default") and a one-step flow ("run": that
job); `scan run` and the operator page run it exactly as before.

validate(doc) -> StationConfiguration, or StationConfigError listing EVERY problem with its
path ("flow[1] (aligned): permit ..."). Pure: no engine, no hardware (the command bus
imports it); recipes are only named here - engine/flow/verify.py resolves them.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 2
SUPPORTED_VERSIONS = (2,)
LEGACY_PROFILE = "default"              # the one implicit profile of a single-job config
LEGACY_STEP = "run"                     # ... and its one implicit work state
OUTCOMES = ("CONFIRMED", "CORRECTED", "NOT_MET", "DEFECT", "DOUBT")
ACTIONS = ("project_fault", "prompt", "kickout", "hold", "retry", "goto")
TERMINAL_ACTIONS = ("kickout", "hold", "goto")      # nothing after them is ever reached
KICKOUTS = ("REJECT", "PULL")
TRIGGERS = ("step", "auto")
ACTION_LISTS = ("on_mismatch", "on_not_met", "on_defect", "on_doubt")
MAX_STEPS = 64
MAX_RETRIES = 10
MAX_TIMEOUT_S = 86400.0
MAX_PROMPT = 200
EXTS = (".json", ".yaml", ".yml")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_PERMIT = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")

TOP_KEYS = {"schema_version", "station_id", "description", "parameters", "profiles", "flow",
            "mock", "_comment"}
LEGACY_TOP_KEYS = {"preset", "library", "mock_library", "runs_dir", "operator_confirmed",
                   "bad_part_output"}
# bad_part_output: where a part judged REJECT or PULL is copied (engine/badparts.py): a
# folder relative to the configuration file, "off", or null (the recipe's / the computer's)
PARAMETER_DEFAULTS = {"preset": None, "library": None, "mock_library": None, "runs_dir": None,
                      "operator_confirmed": False, "default_profile": None, "io": None,
                      "defaults": None, "bad_part_output": None}
STEP_DEFAULTS = {"retries": 1, "timeout_s": None, "trigger": "step"}
PROFILE_KEYS = {"title", "description", "preset", "recipes", "zones", "golden", "scan", "part",
                "feedback", "pull", "mock_scene", "_comment"}
STEP_KEYS = {"id", "title", "expects", "verify", "permit", "on_confirm", *ACTION_LISTS,
             "retries", "timeout_s", "trigger", "required", "allow_skip", "revoke", "_comment"}
VERIFY_KEYS = {"recipe", "checks", "observe", "_comment"}
INLINE_CHECK_KEYS = {"zone", "type", "id", "params", "enabled", "_comment"}
FLOW_FILE_KEYS = {"schema_version", "name", "description", "steps", "defaults", "_comment"}


class StationConfigError(ValueError):
    """The station configuration is missing, unreadable or invalid (exit 2: refused);
    `.problems` lists every problem."""

    def __init__(self, message: str, problems: Optional[list] = None):
        super().__init__(message)
        self.problems = list(problems or [message])


@dataclass
class Profile:
    """One entry of the profile library set: what a part variant looks like."""
    name: str
    title: str = ""
    description: str = ""
    preset: Optional[str] = None            # None: parameters.preset, else the recipe's own
    recipes: dict = field(default_factory=dict)     # key -> recipe name | file | mapping
    zones: dict = field(default_factory=dict)       # id -> {shape, units, coords, ...}
    golden: Optional[dict] = None
    scan: Optional[dict] = None
    part: Optional[dict] = None
    feedback: Optional[dict] = None
    pull: Optional[dict] = None
    mock_scene: Optional[str] = None

    def public(self) -> dict:
        return {"name": self.name, "title": self.title or self.name,
                "description": self.description, "preset": self.preset,
                "recipes": {k: (v if isinstance(v, str) else "(inline)")
                            for k, v in self.recipes.items()},
                "zones": sorted(self.zones), "mock_scene": self.mock_scene}


@dataclass
class Step:
    """One work state: a claim about the line the DLP confirms or corrects."""
    id: str
    title: str = ""
    index: int = 1                          # 1-based position in the flow
    expects: dict = field(default_factory=dict)
    verify: dict = field(default_factory=dict)      # kind recipe | checks | job (legacy)
    permit: Optional[str] = None
    on_confirm: str = "next"
    on_mismatch: list = field(default_factory=list)
    on_not_met: list = field(default_factory=list)
    on_defect: list = field(default_factory=list)
    on_doubt: list = field(default_factory=list)
    retries: int = 1
    timeout_s: Optional[float] = None
    trigger: str = "step"
    required: bool = True
    allow_skip: bool = False
    revoke: list = field(default_factory=list)

    def actions_for(self, outcome: str) -> list:
        return list({"CORRECTED": self.on_mismatch, "NOT_MET": self.on_not_met,
                     "DEFECT": self.on_defect, "DOUBT": self.on_doubt}.get(outcome, []))

    def verify_text(self) -> str:
        v = self.verify
        if v.get("kind") == "recipe":
            return f"recipe {v['recipe']}"
        if v.get("kind") == "checks":
            return ", ".join(f"{c['zone']}/{c['type']}" for c in v["checks"])
        return v.get("job") or "job"

    def observe_text(self) -> dict:
        """signal -> the check that observes it ("zone/check", "(inverted)" when it is)."""
        return {sig: f"{ref['zone']}/{ref['check']}" + (" (inverted)" if ref.get("invert") else "")
                for sig, ref in (self.verify.get("observe") or {}).items()}

    def public(self) -> dict:
        return {"id": self.id, "title": self.title or self.id, "index": self.index,
                "expects": dict(self.expects), "verify": self.verify_text(),
                "observe": self.observe_text(),
                "permit": self.permit, "on_confirm": self.on_confirm,
                **{k: [action_text(a) for a in getattr(self, k)] for k in ACTION_LISTS},
                "retries": self.retries, "timeout_s": self.timeout_s, "trigger": self.trigger,
                "required": self.required, "allow_skip": self.allow_skip,
                "revoke": list(self.revoke)}


@dataclass
class StationConfiguration:
    station_id: str
    description: str = ""
    parameters: dict = field(default_factory=lambda: copy.deepcopy(PARAMETER_DEFAULTS))
    profiles: dict = field(default_factory=dict)        # name -> Profile, in file order
    default_profile: Optional[str] = None
    flow: list = field(default_factory=list)            # [Step]: the work states
    flow_source: Optional[str] = None                   # the flow file, when it is not inline
    signals: list = field(default_factory=list)         # the declared sensor inputs
    signal_info: dict = field(default_factory=dict)     # signal -> {description, source}
    mock: dict = field(default_factory=dict)            # MOCK only: named cases
    legacy: bool = False                                # an old single-job config
    job_type: Optional[str] = None                      # legacy: its job.type
    schema_version: int = SCHEMA_VERSION
    source: Optional[Path] = None
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def permissions(self) -> list:
        """Every permission the flow can grant, in step order."""
        return [s.permit for s in self.flow if s.permit]

    @property
    def profile_names(self) -> list:
        return list(self.profiles)

    @property
    def input_max_age_s(self) -> Optional[float]:
        """parameters.io.max_age_s: the oldest a sensor claim may be when a work state is
        verified (None: no limit beyond "reported since the last cycle ended")."""
        io = self.parameters.get("io")
        v = io.get("max_age_s") if isinstance(io, dict) else None
        return float(v) if _number(v) and v > 0 else None

    def profile(self, name: Optional[str] = None) -> Profile:
        key = self.default_profile if name is None else name
        if key not in self.profiles:
            raise StationConfigError(
                f"station configuration {self.station_id} has no profile {key!r} (profiles: "
                f"{', '.join(self.profiles) or 'none'})")
        return self.profiles[key]

    def step(self, step_id: str) -> Step:
        for s in self.flow:
            if s.id == step_id:
                return s
        raise StationConfigError(f"station configuration {self.station_id} has no work state "
                                 f"{step_id!r} (work states: {', '.join(s.id for s in self.flow)})")

    def preset_for(self, profile: Optional[str] = None) -> Optional[str]:
        """The rig preset a profile pins: its own, else the configuration's (None: the
        recipe's own / the loaded one)."""
        p = self.profiles.get(self.default_profile if profile is None else profile)
        return (p.preset if p is not None else None) or self.parameters.get("preset")

    def mock_case(self, name: Optional[str], profile: Optional[str] = None) -> dict:
        """MOCK only: a named case -> {"name", "scene", "inputs"} (the scene the simulated
        human places for this profile + what the simulated sensors say). None: the default
        case, else the profile's own good part and no sensor claim."""
        cases = self.mock.get("cases") or {}
        key = name if name is not None else self.mock.get("default_case")
        p = self.profiles.get(self.default_profile if profile is None else profile)
        base = p.mock_scene if p is not None else None
        if key is None:
            return {"name": None, "scene": base, "inputs": {}}
        if key not in cases:
            raise StationConfigError(f"station configuration {self.station_id} has no mock "
                                     f"case {key!r} (cases: {', '.join(cases) or 'none'})")
        c = cases[key]
        scene = c.get("scene")
        if scene is None:
            scene = base
            if base and c.get("variant"):
                scene = f"{base}:{c['variant']}"
        return {"name": key, "scene": scene, "inputs": dict(c.get("inputs") or {}),
                "description": c.get("description", "")}

    def public(self) -> dict:
        """What the operator page and a gateway read (no file paths)."""
        return {"station_id": self.station_id, "description": self.description,
                "legacy": self.legacy, "job_type": self.job_type,
                "preset": self.parameters.get("preset"),
                "profiles": [p.public() for p in self.profiles.values()],
                "default_profile": self.default_profile,
                "flow": [s.public() for s in self.flow], "flow_source": self.flow_source,
                "permissions": self.permissions, "signals": list(self.signals),
                "signal_info": copy.deepcopy(self.signal_info),
                "input_max_age_s": self.input_max_age_s,
                "mock_cases": [{"name": k, "description": (v or {}).get("description", "")}
                               for k, v in (self.mock.get("cases") or {}).items()],
                "mock_default_case": self.mock.get("default_case")}


# ------------------------------------------------------------------ reading

def read_doc(path) -> dict:
    """A station configuration / flow file (.json, or .yaml / .yml with PyYAML) -> the raw
    document (a mapping, or for a flow file also a list)."""
    path = Path(path)
    if not path.is_file():
        raise StationConfigError(f"station config not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # optional: PyYAML
        except ImportError as e:
            raise StationConfigError(f"{path.name}: YAML station configs need PyYAML "
                                     "(pip install pyyaml), or use the .json form") from e
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise StationConfigError(f"{path.name}: not valid YAML ({e})") from e
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise StationConfigError(f"{path.name}: not valid JSON ({e})") from e


def is_legacy(doc) -> bool:
    """An old single-job station config: it names neither profiles nor a flow."""
    return isinstance(doc, dict) and "profiles" not in doc and "flow" not in doc


def load(path) -> StationConfiguration:
    """A station configuration file -> StationConfiguration. An old single-job config is
    lifted (lift_legacy: one implicit profile, a one-step flow); its job keys are validated
    where the job runs (ui/run_job.load_station_config)."""
    path = Path(path)
    doc = read_doc(path)
    if not isinstance(doc, dict):
        raise StationConfigError(f"{path.name}: the top level must be a mapping")
    if is_legacy(doc):
        return lift_legacy(doc, source=path)
    return validate(doc, source=path)


def lift_legacy(doc: dict, source: Optional[Path] = None) -> StationConfiguration:
    """An old single-job station config as a configuration: one implicit profile named
    "default" (what the part looks like = the job: its preset, its recipe) and a one-step
    flow ("run": the job itself, verified at START; REJECT on a definite FAIL, PULL on any
    doubt, exactly the job's own disposition). Structural only: nothing is validated here."""
    if not isinstance(doc, dict):
        raise StationConfigError("a station config is a mapping (JSON object)")
    job = doc.get("job") if isinstance(doc.get("job"), dict) else {}
    part = doc.get("part") if isinstance(doc.get("part"), dict) else {}
    sid = str(doc.get("station_id") or (Path(source).stem if source else "station")).strip()
    jt = str(job.get("type", "inspect"))
    recipes = {}
    if isinstance(job.get("recipe"), str) and job["recipe"].strip():
        recipes = {"job": job["recipe"].strip()}
    label = part.get("label") if isinstance(part.get("label"), str) else None
    prof = Profile(name=LEGACY_PROFILE, title=label or sid,
                   description="the single job of this station config",
                   preset=doc.get("preset") if isinstance(doc.get("preset"), str) else None,
                   recipes=recipes,
                   mock_scene=doc.get("mock_part") if isinstance(doc.get("mock_part"), str)
                   else None)
    step = Step(id=LEGACY_STEP, title=f"{jt} job" + (f": {recipes['job']}" if recipes else ""),
                index=1, verify={"kind": "job", "job": jt}, trigger="auto", retries=0,
                on_defect=[{"action": "kickout", "disposition": "REJECT", "reason": ""}],
                on_doubt=[{"action": "kickout", "disposition": "PULL", "reason": ""}])
    params = copy.deepcopy(PARAMETER_DEFAULTS)
    params.update({k: doc.get(k) for k in ("preset", "library", "mock_library", "runs_dir",
                                           "bad_part_output")})
    params["operator_confirmed"] = doc.get("operator_confirmed") is True
    params["default_profile"] = LEGACY_PROFILE
    return StationConfiguration(station_id=sid, description=str(doc.get("_comment") or ""),
                                parameters=params, profiles={LEGACY_PROFILE: prof},
                                default_profile=LEGACY_PROFILE, flow=[step], legacy=True,
                                job_type=jt, schema_version=1,
                                source=Path(source).resolve() if source else None,
                                raw=copy.deepcopy(doc))


def references(doc) -> dict:
    """What a raw station config document names, whatever its form (an old single-job config
    or one with profiles + a flow): {"presets": [every preset it pins], "recipes": [{"ref":
    a recipe name | file | mapping, "preset": the preset that overrides the recipe's own
    pinned one, else None}]}. For the library's delete guards (a preset / recipe a station
    uses is never deleted unasked): read loosely, nothing is validated."""
    out = {"presets": [], "recipes": []}
    if not isinstance(doc, dict):
        return out

    def name(v):
        return v.strip() if isinstance(v, str) and v.strip() else None

    if is_legacy(doc):
        own = name(doc.get("preset"))
        job = doc.get("job") if isinstance(doc.get("job"), dict) else {}
        if own:
            out["presets"].append(own)
        if name(job.get("recipe")):
            out["recipes"].append({"ref": name(job.get("recipe")), "preset": own})
        return out
    params = doc.get("parameters") if isinstance(doc.get("parameters"), dict) else {}
    base = name(params.get("preset")) or name(doc.get("preset"))
    if base:
        out["presets"].append(base)
    profiles = doc.get("profiles") if isinstance(doc.get("profiles"), dict) else {}
    for p in profiles.values():
        if not isinstance(p, dict):
            continue
        own = name(p.get("preset"))
        if own and own not in out["presets"]:
            out["presets"].append(own)
        recipes = p.get("recipes") if isinstance(p.get("recipes"), dict) else {}
        for ref in recipes.values():
            ref = name(ref) if isinstance(ref, str) else (ref if isinstance(ref, dict) else None)
            if ref is not None:
                out["recipes"].append({"ref": ref, "preset": own or base})
    return out


# ------------------------------------------------------------------ validation

class _Errors:
    def __init__(self):
        self.items: list = []

    def add(self, where: str, msg: str) -> None:
        self.items.append(f"{where}: {msg}" if where else msg)


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _count(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _unknown(d: dict, known: set, where: str, E: _Errors, hint: str = "") -> None:
    extra = sorted(set(d) - known)
    if extra:
        E.add(where, f"unknown keys {extra} (known: {', '.join(sorted(known - {'_comment'}))})"
              + hint)


def rel_path(base_dir: Optional[Path], value) -> Optional[Path]:
    """Config paths are relative to the config file, after env expansion."""
    if value in (None, ""):
        return None
    p = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return p if p.is_absolute() or base_dir is None else (Path(base_dir) / p)


def validate(doc: dict, source: Optional[Path] = None) -> StationConfiguration:
    """A raw mapping with profiles + flow -> StationConfiguration (defaults filled in), or
    StationConfigError listing every problem."""
    if not isinstance(doc, dict):
        raise StationConfigError("a station configuration is a mapping (JSON object)")
    E = _Errors()
    base_dir = Path(source).resolve().parent if source else None
    moved = sorted(set(doc) & LEGACY_TOP_KEYS)
    _unknown(doc, TOP_KEYS | LEGACY_TOP_KEYS, "", E)
    if moved:
        E.add("", f"{moved} belong under \"parameters\" in a station configuration with "
              "profiles and a flow")
    ver = doc.get("schema_version", SCHEMA_VERSION)
    if isinstance(ver, bool) or not isinstance(ver, int) or ver not in SUPPORTED_VERSIONS:
        E.add("schema_version", f"{ver!r} is not supported (this engine reads "
              f"{SUPPORTED_VERSIONS}; an old single-job config names neither profiles nor a flow)")
    sid = doc.get("station_id")
    if not isinstance(sid, str) or not _ID.fullmatch(sid.strip()):
        E.add("station_id", "is required: letters, digits, _ . - (max 64 chars), e.g. "
              "example_tube_line")
        sid = Path(source).stem if source else "station"
    sid = sid.strip()
    desc = doc.get("description", "")
    if not isinstance(desc, str):
        E.add("description", "must be text")
        desc = ""
    params, signals, sig_info, defaults = _parameters(doc.get("parameters"), E)
    profiles = _profiles(doc.get("profiles"), params, base_dir, E)
    dflt = params.get("default_profile")
    if dflt is None:
        dflt = next(iter(profiles), None)
    elif dflt not in profiles:
        E.add("parameters.default_profile", f"{dflt!r} is not one of the profiles "
              f"({', '.join(profiles) or 'none'})")
        dflt = next(iter(profiles), None)
    params["default_profile"] = dflt
    steps, flow_source = _flow(doc.get("flow"), defaults, base_dir, E)
    declared = bool(signals)
    for s in steps:
        for sig in s.expects:
            if declared and sig not in signals:
                E.add(f"flow[{s.index - 1}] ({s.id})", f"expects {sig!r}, which is not one of "
                      f"parameters.io.inputs ({', '.join(signals)})")
            elif not declared and sig not in signals:
                signals.append(sig)
    _cross(steps, profiles, params, E)
    mock = _mock(doc.get("mock"), signals, profiles, E)
    if E.items:
        raise StationConfigError(f"station configuration {sid!r}: {len(E.items)} problem(s):"
                                 "\n  - " + "\n  - ".join(E.items), E.items)
    return StationConfiguration(station_id=sid, description=desc, parameters=params,
                                profiles=profiles, default_profile=dflt, flow=steps,
                                flow_source=flow_source, signals=signals, signal_info=sig_info,
                                mock=mock, legacy=False, job_type="flow", schema_version=ver,
                                source=Path(source).resolve() if source else None,
                                raw=copy.deepcopy(doc))


def _parameters(v, E: _Errors) -> tuple:
    """-> (parameters, declared signals, signal info, work-state defaults)."""
    out = copy.deepcopy(PARAMETER_DEFAULTS)
    defaults = dict(STEP_DEFAULTS)
    if v is None:
        return out, [], {}, defaults
    if not isinstance(v, dict):
        E.add("parameters", "must be a mapping")
        return out, [], {}, defaults
    _unknown(v, set(PARAMETER_DEFAULTS) | {"_comment"}, "parameters", E)
    out.update({k: v[k] for k in v if k in PARAMETER_DEFAULTS})
    for k in ("preset", "default_profile"):
        if out[k] is not None and (not isinstance(out[k], str) or not _ID.fullmatch(out[k])):
            E.add(f"parameters.{k}", "must be a name [A-Za-z0-9_.-] or null")
            out[k] = None
    for k in ("library", "mock_library", "runs_dir"):
        if out[k] is not None and not isinstance(out[k], str):
            E.add(f"parameters.{k}", "must be a folder path or null")
            out[k] = None
    if out["bad_part_output"] is not None and not isinstance(out["bad_part_output"], str):
        E.add("parameters.bad_part_output", "must be a folder path, \"off\" or null (where a "
              "part judged REJECT or PULL is copied)")
        out["bad_part_output"] = None
    if not isinstance(out["operator_confirmed"], bool):
        E.add("parameters.operator_confirmed", "must be true or false (unquoted), got "
              f"{out['operator_confirmed']!r}")
        out["operator_confirmed"] = False
    signals, info = [], {}
    io = out["io"]
    if io is not None:
        if not isinstance(io, dict):
            E.add("parameters.io", "must be a mapping {\"inputs\": {...}}")
        else:
            _unknown(io, {"inputs", "max_age_s", "_comment"}, "parameters.io", E)
            age = io.get("max_age_s")
            if age is not None and (not _number(age) or not 0 < age <= MAX_TIMEOUT_S):
                E.add("parameters.io.max_age_s", "must be a number of seconds 0 < s <= "
                      f"{MAX_TIMEOUT_S:g} or null (no limit), got {age!r}")
            ins = io.get("inputs")
            if isinstance(ins, list) and all(isinstance(k, str) for k in ins):
                ins = {k: None for k in ins}
            elif ins is None:
                ins = {}
            elif not isinstance(ins, dict):
                E.add("parameters.io.inputs", "must map each sensor signal to {description, "
                      "source} (or be a list of signal names)")
                ins = {}
            for name, meta in ins.items():
                if not isinstance(name, str) or not _ID.fullmatch(name):
                    E.add("parameters.io.inputs", f"signal name {name!r}: letters, digits, "
                          "_ . - (max 64)")
                    continue
                if meta is not None and not isinstance(meta, (dict, str)):
                    E.add(f"parameters.io.inputs.{name}", "must be {description, source}, a "
                          "text or null")
                    meta = None
                if isinstance(meta, dict):
                    _unknown(meta, {"description", "source", "_comment"},
                             f"parameters.io.inputs.{name}", E)
                signals.append(name)
                info[name] = ({"description": meta} if isinstance(meta, str) else
                              {k: str(meta[k]) for k in ("description", "source")
                               if isinstance(meta, dict) and meta.get(k) is not None})
    d = out["defaults"]
    if d is not None:
        if not isinstance(d, dict):
            E.add("parameters.defaults", "must be a mapping {retries, timeout_s, trigger}")
        else:
            _unknown(d, set(STEP_DEFAULTS) | {"_comment"}, "parameters.defaults", E)
            defaults.update({k: d[k] for k in d if k in STEP_DEFAULTS})
            _step_numbers(defaults, "parameters.defaults", E)
    out["defaults"] = dict(defaults)
    return out, signals, info, defaults


def _step_numbers(d: dict, where: str, E: _Errors) -> None:
    r = d.get("retries")
    if not _count(r) or not 0 <= r <= MAX_RETRIES:
        E.add(where, f"retries must be a whole number 0..{MAX_RETRIES}, got {r!r}")
        d["retries"] = STEP_DEFAULTS["retries"]
    t = d.get("timeout_s")
    if t is not None and (not _number(t) or not 0 < t <= MAX_TIMEOUT_S):
        E.add(where, f"timeout_s must be a number 0 < s <= {MAX_TIMEOUT_S:g} or null, got {t!r}")
        d["timeout_s"] = None
    elif t is not None:
        d["timeout_s"] = float(t)
    if d.get("trigger") not in TRIGGERS:
        E.add(where, f"trigger {d.get('trigger')!r} must be one of {', '.join(TRIGGERS)}")
        d["trigger"] = STEP_DEFAULTS["trigger"]


def _profiles(v, params: dict, base_dir: Optional[Path], E: _Errors) -> dict:
    if not isinstance(v, dict) or not v:
        E.add("profiles", "at least one profile is required: {\"<name>\": {\"recipes\": "
              "{\"<key>\": \"<recipe name>\"}}} (the profile library set)")
        return {}
    out = {}
    for name, p in v.items():
        where = f"profiles.{name}"
        if not isinstance(name, str) or not _ID.fullmatch(name):
            E.add("profiles", f"profile name {name!r}: letters, digits, _ . - (max 64)")
            continue
        if not isinstance(p, dict):
            E.add(where, "must be a mapping")
            continue
        _unknown(p, PROFILE_KEYS, where, E)
        title, desc = p.get("title", ""), p.get("description", "")
        for k, t in (("title", title), ("description", desc)):
            if not isinstance(t, str):
                E.add(where, f"{k} must be text")
        preset = p.get("preset")
        if preset is not None and (not isinstance(preset, str) or not _ID.fullmatch(preset)):
            E.add(where, "preset must be a preset name or null (null: parameters.preset, "
                  "else each recipe's own)")
            preset = None
        recipes = {}
        r = p.get("recipes", {})
        if r is None:
            r = {}
        if not isinstance(r, dict):
            E.add(where, "recipes must map a key to a recipe name, a recipe file or a recipe "
                  "mapping")
            r = {}
        for key, ref in r.items():
            if not isinstance(key, str) or not _ID.fullmatch(key):
                E.add(where, f"recipes key {key!r}: letters, digits, _ . - (max 64)")
            elif isinstance(ref, dict):
                recipes[key] = copy.deepcopy(ref)
            elif isinstance(ref, str) and ref.strip():
                ref = ref.strip()
                if ref.lower().endswith(EXTS):
                    ref = str(rel_path(base_dir, ref))
                recipes[key] = ref
            else:
                E.add(f"{where}.recipes.{key}", "must be a recipe name (scan recipe-list), a "
                      "recipe file (relative to this config) or a recipe mapping")
        zones = p.get("zones", {})
        if zones is None:
            zones = {}
        if not isinstance(zones, dict) or not all(
                isinstance(k, str) and _ID.fullmatch(k) and isinstance(z, dict)
                and not ({"id", "checks"} & set(z)) for k, z in zones.items()):
            E.add(where, "zones must map a zone id to {shape, units, coords} (no id, no checks: "
                  "the work states name the checks)")
            zones = {}
        sections = {}
        for k in ("golden", "scan", "part", "feedback", "pull"):
            sec = p.get(k)
            if sec is not None and not isinstance(sec, dict):
                E.add(where, f"{k} must be a mapping (as in a recipe) or null")
                sec = None
            sections[k] = copy.deepcopy(sec)
        scene = p.get("mock_scene")
        if scene is not None and not isinstance(scene, str):
            E.add(where, "mock_scene must be text (scene:<name>[:<variant>]) or null")
            scene = None
        out[name] = Profile(name=name, title=title if isinstance(title, str) else "",
                            description=desc if isinstance(desc, str) else "", preset=preset,
                            recipes=recipes, zones=copy.deepcopy(zones), mock_scene=scene,
                            **sections)
    return out


def _flow(v, defaults: dict, base_dir: Optional[Path], E: _Errors) -> tuple:
    """-> ([Step], flow file name or None). `flow`: a list of steps, {"steps": [...]} or a
    file (relative to the config) holding either."""
    source = None
    if isinstance(v, str) and v.strip():
        f = rel_path(base_dir, v.strip())
        try:
            v = read_doc(f)
        except StationConfigError as e:
            E.add("flow", f"{v!r}: {e}")
            return [], None
        source = Path(f).name
    where = f"flow ({source})" if source else "flow"
    if isinstance(v, dict):
        _unknown(v, FLOW_FILE_KEYS, where, E)
        if v.get("defaults") is not None:
            if isinstance(v["defaults"], dict):
                defaults = {**defaults, **{k: v["defaults"][k] for k in v["defaults"]
                                           if k in STEP_DEFAULTS}}
                _unknown(v["defaults"], set(STEP_DEFAULTS) | {"_comment"}, f"{where}.defaults", E)
                _step_numbers(defaults, f"{where}.defaults", E)
            else:
                E.add(f"{where}.defaults", "must be a mapping {retries, timeout_s, trigger}")
        v = v.get("steps")
    if not isinstance(v, list) or not v:
        E.add(where, "at least one work state is required: [{id, title, expects, verify, "
              "permit, ...}] (inline, or a flow file relative to this config)")
        return [], source
    if len(v) > MAX_STEPS:
        E.add(where, f"{len(v)} work states: at most {MAX_STEPS}")
    steps, seen = [], set()
    for i, s in enumerate(v[:MAX_STEPS]):
        w = f"flow[{i}]"
        if not isinstance(s, dict):
            E.add(w, "must be a mapping")
            continue
        sid = s.get("id")
        if isinstance(sid, str) and _ID.fullmatch(sid):
            w = f"flow[{i}] ({sid})"
            if sid in seen:
                E.add(w, f"duplicate work state id {sid!r}")
            seen.add(sid)
        else:
            E.add(w, "id is required: letters, digits, _ . -")
            sid = f"step{i + 1}"
        _unknown(s, STEP_KEYS, w, E)
        title = s.get("title", sid)
        if not isinstance(title, str):
            E.add(w, "title must be text")
            title = sid
        nums = {k: s.get(k, defaults[k]) for k in STEP_DEFAULTS}
        _step_numbers(nums, w, E)
        flags = {}
        for k, d in (("required", True), ("allow_skip", False)):
            flags[k] = s.get(k, d)
            if not isinstance(flags[k], bool):
                E.add(w, f"{k} must be true or false")
                flags[k] = d
        if flags["allow_skip"] and flags["required"]:
            E.add(w, "allow_skip needs required: false (a required work state can't be "
                  "skipped and the cycle still CONTINUE)")
        permit = s.get("permit")
        if permit is not None:
            if not isinstance(permit, str) or not _PERMIT.fullmatch(permit):
                E.add(w, f"permit {permit!r} must be a permission name [A-Za-z][A-Za-z0-9_] "
                      "(max 32; the bus output is PERMIT_<NAME>) or null")
                permit = None
            else:
                permit = permit.upper()
        on_confirm = s.get("on_confirm", "next")
        if not isinstance(on_confirm, str) or not on_confirm.strip():
            E.add(w, "on_confirm must be next, done or a work state id")
            on_confirm = "next"
        step = Step(id=sid, title=title, index=len(steps) + 1,
                    expects=_expects(s.get("expects"), w, E), permit=permit,
                    on_confirm=on_confirm.strip(), **nums, **flags)
        step.verify = _verify(s.get("verify"), step, w, E)
        for key in ACTION_LISTS:
            setattr(step, key, _actions(s.get(key), key, step, w, E))
        if s.get("on_not_met") is None:             # default: what a sensor mismatch does
            step.on_not_met = copy.deepcopy(step.on_mismatch)
        if s.get("on_doubt") is None:               # default: a doubt is pulled
            step.on_doubt = [{"action": "kickout", "disposition": "PULL", "reason": ""}]
        step.revoke = s.get("revoke", [])
        raw = s["verify"].get("observe") if isinstance(s.get("verify"), dict) else None
        named = set(step.verify.get("observe") or {}) | set(raw if isinstance(raw, dict) else ())
        for sig in step.expects:                    # nothing is guessed: name the check
            if isinstance(s.get("verify"), dict) and sig not in named:
                E.add(w, f"expects {sig!r} but no check observes it: name the check whose "
                      "answer IS that signal in verify.observe ({\"" + sig + "\": "
                      "\"<zone>/<check>\"}); a signal is never confirmed by a check that was "
                      "not named for it")
        steps.append(step)
    if steps and not any(s.required for s in steps):
        E.add(where, "no work state is required: at least one must be (required: true is the "
              "default), or a cycle could CONTINUE with nothing confirmed")
    _links(steps, E)
    return steps, source


def _expects(v, where: str, E: _Errors) -> dict:
    if v is None:
        return {}
    if not isinstance(v, dict):
        E.add(where, "expects must map a sensor signal to true or false")
        return {}
    out = {}
    for sig, val in v.items():
        if not isinstance(sig, str) or not _ID.fullmatch(sig):
            E.add(where, f"expects: signal name {sig!r}: letters, digits, _ . - (max 64)")
        elif not isinstance(val, bool):
            E.add(where, f"expects.{sig} must be true or false (unquoted), got {val!r}")
        else:
            out[sig] = val
    return out


def _verify(v, step: Step, where: str, E: _Errors) -> dict:
    if not isinstance(v, dict):
        E.add(where, "verify is required: {\"recipe\": \"<key of the profile's recipes>\"} or "
              "{\"checks\": [{\"zone\": ..., \"type\": \"presence\", \"params\": {...}}]}")
        return {"kind": "recipe", "recipe": None, "observe": {}}
    _unknown(v, VERIFY_KEYS, f"{where}.verify", E)
    rec, checks = v.get("recipe"), v.get("checks")
    out = {"observe": _observe(v.get("observe"), step, where, E)}
    if (rec is None) == (checks is None):
        E.add(f"{where}.verify", "give either recipe (a key of the profile's recipes) or "
              "checks (inline zone checks), not both")
        return {"kind": "recipe", "recipe": None, **out}
    if rec is not None:
        if not isinstance(rec, str) or not _ID.fullmatch(rec):
            E.add(f"{where}.verify", "recipe must be a key of the profile's recipes")
            rec = None
        return {"kind": "recipe", "recipe": rec, **out}
    items = []
    if not isinstance(checks, list) or not checks:
        E.add(f"{where}.verify", "checks must be a list of {zone, type, params}")
        checks = []
    seen = set()
    for j, c in enumerate(checks):
        cw = f"{where}.verify.checks[{j}]"
        if not isinstance(c, dict):
            E.add(cw, "must be a mapping {zone, type, params}")
            continue
        _unknown(c, INLINE_CHECK_KEYS, cw, E)
        zone, ctype = c.get("zone"), c.get("type")
        if not isinstance(zone, str) or not _ID.fullmatch(zone):
            E.add(cw, "zone is required: a zone id of the profile's zones")
            continue
        if not isinstance(ctype, str) or not ctype:
            E.add(cw, "type is required (a check id: see `scan recipe-checks`)")
            continue
        cid = c.get("id", ctype)
        if not isinstance(cid, str) or not _ID.fullmatch(cid):
            E.add(cw, "id must be a name [A-Za-z0-9_.-]")
            cid = ctype
        if (zone, cid) in seen:
            E.add(cw, f"duplicate check id {cid!r} in zone {zone} (give each an id)")
        seen.add((zone, cid))
        params = c.get("params") or {}
        if not isinstance(params, dict):
            E.add(cw, "params must be a mapping")
            params = {}
        if c.get("enabled", True) is False:
            continue
        items.append({"zone": zone, "type": ctype, "id": cid, "params": copy.deepcopy(params)})
    return {"kind": "checks", "checks": items, **out}


def _observe(v, step: Step, where: str, E: _Errors) -> dict:
    """verify.observe: signal -> {"zone", "check", "invert"} ("zone/check" for short)."""
    if v is None:
        return {}
    if not isinstance(v, dict):
        E.add(f"{where}.verify", "observe must map a signal to \"<zone>/<check>\"")
        return {}
    out = {}
    for sig, ref in v.items():
        ow = f"{where}.verify.observe.{sig}"
        if sig not in step.expects:
            E.add(ow, f"{sig!r} is not one of this work state's expects "
                  f"({', '.join(step.expects) or 'none'})")
            continue
        if isinstance(ref, str) and ref.count("/") == 1 and all(ref.split("/")):
            zone, check = ref.split("/")
            ref = {"zone": zone, "check": check}
        if not isinstance(ref, dict) or set(ref) - {"zone", "check", "invert"} or not all(
                isinstance(ref.get(k), str) and ref.get(k) for k in ("zone", "check")):
            E.add(ow, "must be \"<zone>/<check>\" or {\"zone\": ..., \"check\": ..., "
                  "\"invert\": false}")
            continue
        inv = ref.get("invert", False)
        if not isinstance(inv, bool):
            E.add(ow, "invert must be true or false")
            inv = False
        out[sig] = {"zone": ref["zone"], "check": ref["check"], "invert": inv}
    return out


def parse_action(a) -> dict:
    """One action as written -> {"action": ..., ...}; ValueError says what is wrong.
        "project_fault" | "hold" | "retry"
        "prompt <message>"           {"action": "prompt", "message": "..."}
        "kickout REJECT|PULL [why]"  {"action": "kickout", "disposition": "PULL", "reason": ".."}
        "goto <step id>"             {"action": "goto", "step": "<id>"}
    A mapping may carry "when_dlp": {signal: true | false} (only when the DLP saw that)."""
    if isinstance(a, str):
        word, _, rest = a.strip().partition(" ")
        rest = rest.strip()
        if len(rest) >= 2 and rest[0] == rest[-1] and rest[0] in "\"'":
            rest = rest[1:-1].strip()
        d = {"action": word.lower()}
        if d["action"] == "prompt":
            d["message"] = rest
        elif d["action"] == "kickout":
            disp, _, why = rest.partition(" ")
            d.update(disposition=disp.upper(), reason=why.strip())
        elif d["action"] == "goto":
            d["step"] = rest
        elif rest:
            raise ValueError(f"{a!r}: {word} takes no argument")
        a = d
    elif isinstance(a, dict):
        a = dict(a)
        if len(a) >= 1 and "action" not in a:
            for k in ACTIONS:               # {"prompt": "..."} / {"kickout": "REJECT"} for short
                if k in a:
                    val = a.pop(k)
                    short = parse_action(f"{k} {val}" if isinstance(val, str) and val else k)
                    a = {**short, **a}
                    break
    else:
        raise ValueError(f"{a!r}: an action is a text or a mapping")
    kind = a.get("action")
    if kind not in ACTIONS:
        raise ValueError(f"action {kind!r} is not one of {', '.join(ACTIONS)}")
    known = {"action", "when_dlp", "_comment"} | {
        "prompt": {"message"}, "kickout": {"disposition", "reason"}, "goto": {"step"}}.get(
        kind, set())
    extra = sorted(set(a) - known)
    if extra:
        raise ValueError(f"{kind}: unknown keys {extra} (known: {', '.join(sorted(known))})")
    out = {"action": kind}
    if kind == "prompt":
        msg = a.get("message")
        if not isinstance(msg, str) or not msg.strip() or len(msg.strip()) > MAX_PROMPT:
            raise ValueError(f"prompt needs a message (1..{MAX_PROMPT} characters)")
        out["message"] = " ".join(msg.split())
    elif kind == "kickout":
        disp = str(a.get("disposition") or "").upper()
        if disp not in KICKOUTS:
            raise ValueError(f"kickout needs REJECT or PULL, got {a.get('disposition')!r}")
        why = a.get("reason", "")
        if not isinstance(why, str) or len(why) > MAX_PROMPT:
            raise ValueError(f"kickout reason must be text (max {MAX_PROMPT} characters)")
        out.update(disposition=disp, reason=" ".join(why.split()))
    elif kind == "goto":
        if not isinstance(a.get("step"), str) or not _ID.fullmatch(a["step"]):
            raise ValueError("goto needs a work state id")
        out["step"] = a["step"]
    when = a.get("when_dlp")
    if when is not None:
        if not isinstance(when, dict) or not when or not all(
                isinstance(k, str) and isinstance(x, bool) for k, x in when.items()):
            raise ValueError("when_dlp must map a signal to true or false (what the DLP saw)")
        out["when_dlp"] = dict(when)
    return out


def action_text(a: dict) -> str:
    """An action back as its short text (the operator page, the logs)."""
    kind = a.get("action")
    t = {"prompt": f"prompt {a.get('message', '')}",
         "kickout": f"kickout {a.get('disposition', '')} {a.get('reason', '')}".rstrip(),
         "goto": f"goto {a.get('step', '')}"}.get(kind, str(kind))
    if a.get("when_dlp"):
        t += " (when the DLP says " + ", ".join(f"{k} {'true' if v else 'false'}"
                                                for k, v in a["when_dlp"].items()) + ")"
    return t


def _actions(v, key: str, step: Step, where: str, E: _Errors) -> list:
    if v is None:
        return []
    if not isinstance(v, list):
        E.add(where, f"{key} must be a list of actions ({', '.join(ACTIONS)})")
        return []
    out = []
    for j, a in enumerate(v):
        aw = f"{where}.{key}[{j}]"
        try:
            act = parse_action(a)
        except ValueError as e:
            E.add(aw, str(e))
            continue
        for sig in act.get("when_dlp") or {}:
            if sig not in step.expects:
                E.add(aw, f"when_dlp names {sig!r}, which is not one of this work state's "
                      f"expects ({', '.join(step.expects) or 'none'})")
        if key == "on_doubt" and act["action"] == "kickout" and act["disposition"] == "REJECT":
            E.add(aw, "kickout REJECT on a doubt: a doubt is never a reject (use kickout PULL)")
        if key == "on_doubt" and act["action"] == "goto":
            E.add(aw, "goto on a doubt: a doubt is never passed over (verify again with hold "
                  "/ retry, or kickout PULL)")
        if act["action"] == "retry" and step.retries < 1:
            E.add(aw, "retry needs retries >= 1 on this work state")
        if out and out[-1]["action"] in TERMINAL_ACTIONS and "when_dlp" not in out[-1]:
            E.add(aw, f"is never reached: {out[-1]['action']} before it ends the list")
        out.append(act)
    return out


def _links(steps: list, E: _Errors) -> None:
    """on_confirm / goto targets, revoke lists and duplicate permissions."""
    ids = {s.id for s in steps}
    permits: dict = {}
    for s in steps:
        if s.permit:
            permits.setdefault(s.permit, []).append(s.id)
    for name, who in permits.items():
        if len(who) > 1:
            E.add("flow", f"permission {name} is granted by {', '.join(who)}: one work state "
                  "per permission")
    for s in steps:
        w = f"flow[{s.index - 1}] ({s.id})"
        if s.on_confirm not in ("next", "done"):
            if s.on_confirm not in ids:
                E.add(w, f"on_confirm {s.on_confirm!r} is not next, done or a work state id "
                      f"({', '.join(sorted(ids))})")
            elif s.on_confirm == s.id:
                E.add(w, "on_confirm names this work state itself")
        for key in ACTION_LISTS:
            for a in getattr(s, key):
                if a["action"] == "goto" and a["step"] not in ids:
                    E.add(f"{w}.{key}", f"goto {a['step']!r}: no such work state "
                          f"({', '.join(sorted(ids))})")
        rv = s.revoke
        if rv == "all":
            s.revoke = [p for p in permits if p != s.permit]
        elif isinstance(rv, list) and all(isinstance(p, str) for p in rv):
            up = [p.upper() for p in rv]
            bad = [p for p in up if p not in permits or p == s.permit]
            if bad:
                E.add(w, f"revoke {bad}: not a permission of another work state "
                      f"({', '.join(permits) or 'none'})")
            s.revoke = [p for p in dict.fromkeys(up) if p not in bad]
        else:
            E.add(w, "revoke must be a list of permission names, or \"all\"")
            s.revoke = []


def _cross(steps: list, profiles: dict, params: dict, E: _Errors) -> None:
    """Every work state must be verifiable with EVERY profile of the set: the recipe key in
    its recipes, the inline checks' zones among its zones (and a preset to scan with)."""
    for s in steps:
        w = f"flow[{s.index - 1}] ({s.id})"
        v = s.verify
        if v.get("kind") == "recipe" and v.get("recipe"):
            missing = [n for n, p in profiles.items() if v["recipe"] not in p.recipes]
            if missing:
                E.add(w, f"verify.recipe {v['recipe']!r} is not in the recipes of profile(s) "
                      f"{', '.join(missing)}: every profile of the set must say what this "
                      "work state looks like")
        elif v.get("kind") == "checks":
            for n, p in profiles.items():
                gone = sorted({c["zone"] for c in v["checks"]} - set(p.zones))
                if gone:
                    E.add(w, f"verify.checks name zone(s) {gone}, which profile {n} does not "
                          "define (profiles.<name>.zones)")
                if not (p.preset or params.get("preset")):
                    E.add(w, f"inline checks need a preset: set profiles.{n}.preset or "
                          "parameters.preset")
        for sig, ref in (v.get("observe") or {}).items():
            if v.get("kind") == "checks" and not any(
                    c["zone"] == ref["zone"] and c["id"] == ref["check"] for c in v["checks"]):
                E.add(f"{w}.verify.observe.{sig}", f"{ref['zone']}/{ref['check']} is not one "
                      "of this work state's checks")


def _mock(v, signals: list, profiles: dict, E: _Errors) -> dict:
    """MOCK only: named cases for the simulated line: the scene the simulated human places
    (the profile's mock_scene, or its ":<variant>", or an explicit scene / "none") and what
    the simulated sensors say."""
    if v is None:
        return {}
    if not isinstance(v, dict):
        E.add("mock", "must be a mapping {default_case, cases}")
        return {}
    _unknown(v, {"default_case", "cases", "_comment"}, "mock", E)
    cases = v.get("cases") or {}
    if not isinstance(cases, dict):
        E.add("mock.cases", "must map a case name to {variant | scene, inputs, description}")
        cases = {}
    out = {}
    for name, c in cases.items():
        w = f"mock.cases.{name}"
        if not isinstance(name, str) or not _ID.fullmatch(name) or not isinstance(c, dict):
            E.add("mock.cases", f"case {name!r}: a name [A-Za-z0-9_.-] mapped to a mapping")
            continue
        _unknown(c, {"variant", "scene", "inputs", "description", "_comment"}, w, E)
        for k in ("variant", "scene", "description"):
            if c.get(k) is not None and not isinstance(c[k], str):
                E.add(w, f"{k} must be text or null")
        ins = c.get("inputs") or {}
        if not isinstance(ins, dict) or not all(
                isinstance(k, str) and isinstance(x, bool) for k, x in ins.items()):
            E.add(w, "inputs must map a sensor signal to true or false")
            ins = {}
        for sig in ins:
            if sig not in signals:
                E.add(w, f"inputs name {sig!r}, which is not a sensor signal of this "
                      f"configuration ({', '.join(signals) or 'none'})")
        out[name] = {"variant": c.get("variant") if isinstance(c.get("variant"), str) else None,
                     "scene": c.get("scene") if isinstance(c.get("scene"), str) else None,
                     "inputs": dict(ins),
                     "description": c.get("description") if isinstance(c.get("description"),
                                                                       str) else ""}
    dflt = v.get("default_case")
    if dflt is not None and dflt not in out:
        E.add("mock.default_case", f"{dflt!r} is not one of the cases ({', '.join(out)})")
        dflt = None
    return {"default_case": dflt, "cases": out}
