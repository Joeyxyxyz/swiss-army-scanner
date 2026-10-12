"""Station configuration -> profiles -> work states (engine/flow/, the command bus), MOCK only.

A station configuration (stations/*.json) is NOT a work state: it holds the station's
parameters and its profile library set (long tube / short tube). The work states are the
steps of its flow: a claim of the line's own sensors that the DLP confirms or corrects, and
the permission that lets the line continue. Here: the schema and its validator, every
outcome of a verify (CONFIRMED / CORRECTED / NOT_MET / DEFECT / DOUBT), the actions (prompt +
ACK, hold, retry, kickout, goto), the permissions (granted only on CONFIRMED; dropped on
abort / home / fault / a profile or configuration change), the bus commands (START, STEP,
ACK, RETRY, SKIP, SELECT_PROFILE, the deprecated SELECT_WORKSTATE alias), the DLP checks on
synthetic scans of the mock tube scenes, the cycle folder's result.json, the old single-job
configs (still run), and `scan run` / `flow-show` on the command line. No hardware: the
state machines are pure, the scans synthetic; one command-line cycle goes through the full
mock capture path."""

import contextlib
import copy
import io
import itertools
import json
import shutil
import sys
from pathlib import Path

import pytest

from engine import api, rules
from engine import disposition as D
from engine import runs as R
from engine.control import commands as C
from engine.flow import schema as FS
from engine.flow import verify as FV
from engine.flow.cycle import OUTCOME_CODES, Cycle, CycleError, classify, job_outcome
from engine.recipes import runner
from ui import cli
from ui import run_job as RJ

import recipe_synth as RS
from conftest import ROOT

EXAMPLE = ROOT / "stations" / "example_tube_line.json"
FLOW = ROOT / "flows" / "example_tube_line.json"
PRESET = "bench_9in_metric_bright"
LEGACY = ("example_cell_station.json", "example_metric_check.json",
          "example_recipe_box_lid.json")


@pytest.fixture(autouse=True)
def _fresh_engine():
    yield
    api._session = None


# ================================================================== helpers

def _doc(**over) -> dict:
    """A small configuration with two profiles and the example's three work states, inline."""
    d = {"schema_version": 2, "station_id": "line1",
         "parameters": {"preset": PRESET, "io": {"inputs": {"part_present": None}}},
         "profiles": {"long": {"recipes": {"presence": "tube_long_presence",
                                           "alignment": "tube_long_alignment",
                                           "end_face": "tube_long_end_face"}},
                      "short": {"recipes": {"presence": "tube_short_presence",
                                            "alignment": "tube_short_alignment",
                                            "end_face": "tube_short_end_face"}}},
         "flow": copy.deepcopy(json.loads(FLOW.read_text(encoding="utf-8"))["steps"])}
    d.update(over)
    return d


def _cfg(steps=None, **over) -> FS.StationConfiguration:
    d = _doc(**over)
    if steps is not None:
        d["flow"] = steps
    return FS.validate(d)


def dlp(result="PASS", disposition="CONTINUE", run="r1", **observed) -> dict:
    """A DLP answer as engine/flow/verify.dlp_answer gives it."""
    return {"result": result, "disposition": disposition, "observed": dict(observed),
            "run_id": run, "run_folder": f"runs/{run}", "feedback_rendered": True,
            "pull_reasons": ([] if disposition != "PULL" else
                             [D.reason("too_few_points", "only 3 points", "seat", "presence")]),
            "message": "" if result == "PASS" else "seat/check: it failed"}


OK = dlp(part_present=True)
ABSENT = dlp("FAIL", "REJECT", part_present=False)
FAILS = dlp("FAIL", "REJECT", part_present=True)       # the part is there, a check fails
GOOD = dlp(part_present=True)                           # the part is there, every check passes


def _walk(cyc: Cycle, answers, sensors=None, ack=True) -> Cycle:
    """Drive a cycle like `scan run` does: every verify answered from `answers` (in order),
    STEP given at once, prompts acknowledged (or not: the caller sees it waiting)."""
    answers = list(answers)
    effects = cyc.begin()
    while cyc.state != "DONE":
        if cyc.state == "WAIT_STEP":
            effects = cyc.step_now()
        elif cyc.state == "VERIFYING":
            cyc.request({"part_present": True} if sensors is None else sensors)
            if not answers:
                return cyc
            effects = cyc.verified(answers.pop(0), sensors)
        elif cyc.state == "WAIT_ACK":
            if not ack:
                return cyc
            effects = cyc.ack("test")
    assert effects == [("done", None)] or cyc.state == "DONE"
    return cyc


# ================================================================== schema

def test_the_example_is_a_configuration_with_a_profile_set_and_a_flow():
    cfg = FS.load(EXAMPLE)
    assert not cfg.legacy and cfg.station_id == "example_tube_line"
    assert cfg.profile_names == ["long_tube", "short_tube"] and cfg.default_profile == "long_tube"
    assert [s.id for s in cfg.flow] == ["loaded", "aligned", "end_face"]
    assert [s.title for s in cfg.flow] == ["Tube loaded?", "Tube aligned?", "End face OK?"]
    assert cfg.permissions == ["LOAD_DONE", "CLAMP", "TRANSFER"]
    assert cfg.signals == ["part_present"] and cfg.flow_source == "example_tube_line.json"
    loaded, aligned, face = cfg.flow
    assert loaded.expects == {"part_present": True} and loaded.verify["recipe"] == "presence"
    assert loaded.verify["observe"]["part_present"] == {"zone": "seat", "check": "presence",
                                                        "invert": False}
    assert [a["action"] for a in loaded.on_mismatch] == ["project_fault", "prompt", "prompt",
                                                         "hold"]
    assert loaded.on_mismatch[1] == {"action": "prompt", "when_dlp": {"part_present": False},
                                     "message": "no tube found: load or clear the sensor"}
    assert loaded.on_not_met == loaded.on_mismatch          # the default
    # every work state asks "is that tube there or not?" of the same named check, so a tube
    # that went missing at any step is a missing tube (CORRECTED / NOT_MET), never a dent
    for s in cfg.flow:
        assert s.expects == {"part_present": True}
        assert s.verify["observe"] == {"part_present": {"zone": "seat", "check": "presence",
                                                        "invert": False}}
        assert s.on_mismatch == loaded.on_mismatch
        assert s.public()["observe"] == {"part_present": "seat/presence"}
    assert aligned.retries == 2
    assert [a["action"] for a in aligned.on_defect] == ["project_fault", "prompt", "retry"]
    assert face.on_defect[-1] == {"action": "kickout", "disposition": "REJECT",
                                  "reason": "end face not OK"}
    for s in cfg.flow:                                      # a doubt is pulled, by default
        assert s.on_doubt == [{"action": "kickout", "disposition": "PULL", "reason": ""}]
        assert s.trigger == "step" and s.required and not s.allow_skip
    assert cfg.preset_for("short_tube") == PRESET
    assert cfg.profile("short_tube").recipes["end_face"] == "tube_short_end_face"
    # MOCK: the simulated line's cases (scene per profile + what the sensor says)
    assert cfg.mock_case(None)["name"] == "all_good"
    assert cfg.mock_case("dented", "short_tube") == {
        "name": "dented", "scene": "scene:tube_short:dent", "inputs": {"part_present": True},
        "description": cfg.mock["cases"]["dented"]["description"]}
    assert cfg.mock_case("sensor_false_part_present")["inputs"] == {"part_present": False}
    assert cfg.mock_case("sensor_true_part_absent")["scene"] == "scene:tube_long:absent"
    with pytest.raises(FS.StationConfigError, match="no mock case"):
        cfg.mock_case("nope")
    pub = cfg.public()
    assert [p["name"] for p in pub["profiles"]] == ["long_tube", "short_tube"]
    assert pub["flow"][0]["on_mismatch"][1].startswith("prompt no tube found")
    assert json.loads(json.dumps(pub)) == pub               # plain data: the page reads it


@pytest.mark.parametrize("name", LEGACY)
def test_an_old_single_job_config_is_one_implicit_profile_and_a_one_step_flow(name):
    cfg = FS.load(ROOT / "stations" / name)
    assert cfg.legacy and cfg.profile_names == ["default"] and cfg.default_profile == "default"
    (step,) = cfg.flow
    assert step.id == "run" and step.trigger == "auto" and step.verify["kind"] == "job"
    assert step.expects == {} and step.permit is None and cfg.permissions == []
    assert step.on_defect[0]["disposition"] == "REJECT" and step.on_doubt[0]["disposition"] == "PULL"
    old = RJ.load_station_config(ROOT / "stations" / name)   # the job itself: read as before
    assert old.station_id == cfg.station_id and old.job_type == cfg.job_type
    if old.job_type == "recipe":
        assert cfg.profiles["default"].recipes == {"job": old.recipe}
    assert RJ.StationConfigError is FS.StationConfigError


def test_run_job_refuses_to_read_a_flow_as_one_job():
    with pytest.raises(RJ.StationConfigError, match="profiles and a flow"):
        RJ.load_station_config(EXAMPLE)


def test_flow_inline_as_a_mapping_or_a_file(tmp_path):
    steps = _doc()["flow"]
    assert [s.id for s in FS.validate(_doc()).flow] == ["loaded", "aligned", "end_face"]
    cfg = FS.validate(_doc(flow={"steps": steps, "defaults": {"retries": 3, "trigger": "auto"}}))
    assert cfg.flow[0].retries == 3 and cfg.flow[1].retries == 2 and cfg.flow[0].trigger == "auto"
    (tmp_path / "flows").mkdir()
    (tmp_path / "flows" / "f.json").write_text(json.dumps({"schema_version": 2, "steps": steps}))
    f = tmp_path / "st.json"
    f.write_text(json.dumps(_doc(flow="flows/f.json")))
    cfg = FS.load(f)
    assert cfg.flow_source == "f.json" and len(cfg.flow) == 3 and cfg.source == f.resolve()
    f.write_text(json.dumps(_doc(flow="flows/missing.json")))
    with pytest.raises(FS.StationConfigError, match="not found"):
        FS.load(f)
    with pytest.raises(FS.StationConfigError, match="at least one work state"):
        FS.validate(_doc(flow=[]))


def test_a_bad_configuration_lists_every_problem():
    bad = _doc(preset="p", colour="red")                    # a legacy key at the top, a typo
    bad["station_id"] = "bad id"
    bad["parameters"]["default_profile"] = "medium"
    bad["parameters"]["defaults"] = {"retries": 99, "timeout_s": -1, "trigger": "sometimes"}
    bad["profiles"]["short"]["recipes"].pop("alignment")
    bad["flow"] = [
        {"id": "a", "expects": {"part_present": "true", "door_closed": True},
         "verify": {"recipe": "presence", "observe": {"lid_on": "seat/presence"}},
         "permit": "1st", "on_doubt": ["kickout REJECT"], "on_mismatch": ["hold", "retry"],
         "allow_skip": True, "revoke": ["NOPE"], "colour": "red"},
        {"id": "a", "verify": {"recipe": "alignment"}, "permit": "GO", "on_confirm": "nowhere",
         "on_defect": ["goto elsewhere", "dance", {"prompt": ""}], "retries": 0,
         "on_not_met": ["retry"], "timeout_s": "soon"},
        {"id": "c", "verify": {"recipe": "end_face", "checks": []}, "permit": "go"},
        {"id": "d"},
    ]
    bad["mock"] = {"default_case": "none", "cases": {"x": {"inputs": {"lid_on": True}}}}
    with pytest.raises(FS.StationConfigError) as e:
        FS.validate(bad)
    text = "\n".join(e.value.problems)
    for needle in (
            "unknown keys ['colour']", "belong under \"parameters\"", "station_id: is required",
            "default_profile: 'medium' is not one of the profiles",
            "retries must be a whole number 0..10", "timeout_s must be a number",
            "trigger 'sometimes' must be one of step, auto",
            "expects.part_present must be true or false (unquoted), got 'true'",
            "expects 'door_closed', which is not one of parameters.io.inputs",
            "observe.lid_on: 'lid_on' is not one of this work state's expects",
            "permit '1st' must be a permission name",
            "kickout REJECT on a doubt: a doubt is never a reject",
            "is never reached: hold before it ends the list",
            "allow_skip needs required: false",
            "revoke ['NOPE']: not a permission of another work state",
            "flow[0] (a): unknown keys ['colour']",
            "duplicate work state id 'a'",
            "permission GO is granted by a, c: one work state per permission",
            "on_confirm 'nowhere' is not next, done or a work state id",
            "goto 'elsewhere': no such work state",
            "action 'dance' is not one of", "prompt needs a message",
            "retry needs retries >= 1",
            "give either recipe (a key of the profile's recipes) or checks",
            "verify is required",
            "verify.recipe 'alignment' is not in the recipes of profile(s) short",
            "mock.cases.x: inputs name 'lid_on'", "mock.default_case: 'none' is not one of"):
        assert needle in text, needle
    assert len(e.value.problems) >= 25
    with pytest.raises(FS.StationConfigError, match="at least one profile"):
        FS.validate(_doc(profiles={}))
    with pytest.raises(FS.StationConfigError, match="schema_version"):
        FS.validate(_doc(schema_version=7))


def test_actions_as_short_text_or_mappings():
    P = FS.parse_action
    assert P("project_fault") == {"action": "project_fault"}
    assert P('prompt "no tube found: load or clear the sensor"') == {
        "action": "prompt", "message": "no tube found: load or clear the sensor"}
    assert P("kickout reject dented end face") == {"action": "kickout", "disposition": "REJECT",
                                                   "reason": "dented end face"}
    assert P({"kickout": "PULL", "reason": "look at it"})["reason"] == "look at it"
    assert P({"action": "goto", "step": "loaded"}) == {"action": "goto", "step": "loaded"}
    assert P({"prompt": "x", "when_dlp": {"s": False}})["when_dlp"] == {"s": False}
    for bad in ("kickout MAYBE", "hold now", "prompt", "goto", 7, {"action": "prompt"},
                {"action": "hold", "why": "x"}, {"prompt": "x", "when_dlp": {"s": "no"}}):
        with pytest.raises(ValueError):
            P(bad)
    assert FS.action_text(P("kickout PULL")) == "kickout PULL"
    assert "when the DLP says s false" in FS.action_text(P({"hold": True, "when_dlp": {"s": False}}))


def test_delete_guards_see_the_presets_and_recipes_of_a_configuration(tmp_path):
    """A preset / recipe a station configuration's profiles use is found by the library's
    delete guards (they read the old single-job form only before)."""
    from engine.library import presets as preset_lib
    from engine.recipes import library as recipe_lib
    refs = FS.references(json.loads(EXAMPLE.read_text(encoding="utf-8")))
    assert refs["presets"] == [PRESET] and len(refs["recipes"]) == 6
    assert {r["preset"] for r in refs["recipes"]} == {PRESET}
    old = FS.references(json.loads((ROOT / "stations" / LEGACY[2]).read_text(encoding="utf-8")))
    assert old == {"presets": [], "recipes": [{"ref": "box_lid", "preset": None}]}
    stations = ROOT / "stations"
    assert "example_tube_line.json" in recipe_lib.station_refs(stations, "tube_long_presence")
    assert recipe_lib.station_refs(stations, "box_lid") == ["example_recipe_box_lid.json"]
    assert "example_tube_line.json" in preset_lib.station_refs(stations, PRESET)
    # a profile with no preset of its own runs its recipe's pinned preset: found through it
    d = _doc()
    d["parameters"]["preset"] = None
    (tmp_path / "st").mkdir()
    (tmp_path / "st" / "line.json").write_text(json.dumps(d))
    s = api.configure(tmp_path / "lib", mock=True)
    got = preset_lib.recipe_refs(s.store, PRESET, stations_dir=tmp_path / "st")
    assert got["stations"] == ["line.json"] and "tube_long_presence" in got["recipes"]
    assert preset_lib.station_refs(tmp_path / "st", PRESET) == []


# ================================================================== one verify: the outcomes

def _step(**kw) -> FS.Step:
    """One work state; every signal it expects is observed by seat/presence (named: nothing
    is guessed)."""
    verify = {"recipe": "presence"}
    if kw.get("expects"):
        verify["observe"] = {sig: "seat/presence" for sig in kw["expects"]}
    return FS.validate(_doc(flow=[{"id": "s", "verify": verify, **kw}])).flow[0]


def test_every_outcome_of_a_verify():
    s = _step(expects={"part_present": True})
    c = classify(s, {"part_present": True}, OK)
    assert c["outcome"] == "CONFIRMED" and c["sensor_mismatch"] == [] and c["reasons"] == []
    # sensor says present, the DLP says absent
    c = classify(s, {"part_present": True}, ABSENT)
    assert c["outcome"] == "CORRECTED"
    assert c["sensor_mismatch"] == [{"signal": "part_present", "sensor": True, "dlp": False,
                                     "expected": True}]
    assert c["reasons"][0]["reason"] == "sensor_mismatch" and c["reasons"][0]["step"] == "s"
    assert c["message"] == "sensor part_present says true, the DLP says false"
    # the reverse: sensor says absent, the DLP sees the part
    c = classify(s, {"part_present": False}, OK)
    assert c["outcome"] == "CORRECTED" and c["sensor_mismatch"][0]["sensor"] is False
    assert c["sensor_mismatch"][0]["dlp"] is True
    # sensor and DLP agree, on what the work state does not expect
    c = classify(s, {"part_present": False}, ABSENT)
    assert c["outcome"] == "NOT_MET" and c["sensor_mismatch"] == []
    assert "sensor and DLP agree" in c["message"] and c["reasons"][0]["reason"] == "not_confirmed"
    # the part is there, a check fails
    c = classify(s, {"part_present": True}, dlp("FAIL", "REJECT", part_present=True))
    assert c["outcome"] == "DEFECT" and c["message"] == "seat/check: it failed"
    # doubt: INVALID, a guard-band PULL, an unreadable FAIL, a crash, no verdict
    for answer, code in ((dlp("INVALID", "PULL", part_present=None), "invalid"),
                         (dlp("PASS", "PULL", part_present=True), "too_few_points"),
                         (dlp("FAIL", "PULL", part_present=True), "too_few_points"),
                         (dlp("INVALID", "REJECT", part_present=True), "invalid"),
                         ({"error": "RuntimeError: boom"}, "error"),
                         (dlp("ABORTED", "PULL", part_present=True), "aborted"),
                         (None, "invalid"), ({}, "invalid")):
        c = classify(s, {"part_present": True}, answer)
        assert c["outcome"] == "DOUBT" and c["reasons"], answer
        assert code in [r["reason"] for r in c["reasons"]], (answer, c["reasons"])
    # a sensor that never reported, or changed while the DLP scanned
    c = classify(s, {}, OK)
    assert c["outcome"] == "DOUBT" and c["reasons"][0]["reason"] == "sensor_unknown"
    assert "has not reported" in c["message"]
    c = classify(s, {"part_present": "true"}, OK)           # only a real boolean is a claim
    assert c["outcome"] == "DOUBT"
    c = classify(s, {"part_present": True}, OK, {"part_present": False})
    assert c["outcome"] == "DOUBT" and "changed during the scan" in c["message"]
    # a work state that expects nothing: the DLP alone decides
    plain = _step()
    assert classify(plain, {}, GOOD)["outcome"] == "CONFIRMED"
    assert classify(plain, {}, FAILS)["outcome"] == "DEFECT"
    assert classify(plain, {}, dlp("PASS", "PULL"))["outcome"] == "DOUBT"
    assert classify(plain, {}, dlp("INVALID", "PULL"))["outcome"] == "DOUBT"
    # expects false ("the seat must be empty"): a part that is really there is NOT_MET
    empty = _step(expects={"part_present": False})
    assert classify(empty, {"part_present": True}, dlp(part_present=True))["outcome"] == "NOT_MET"
    assert classify(empty, {"part_present": True}, dlp(part_present=False))["outcome"] == "CORRECTED"
    assert classify(empty, {"part_present": False}, dlp(part_present=False))["outcome"] == "CONFIRMED"


def test_nothing_but_a_doubt_free_agreement_confirms():
    """Every combination of verdict, disposition, sensor claim and DLP observation: CONFIRMED
    only for PASS + CONTINUE with sensor == DLP == expected. Nothing else - no INVALID, no
    PULL, no unknown - can grant a permission."""
    n = 0
    steps = {exp: _step(expects={"part_present": exp}) for exp in (True, False)}
    for exp, res, disp, sensor, seen, after in itertools.product(
            (True, False), ("PASS", "FAIL", "INVALID", "ABORTED", None, "bogus"),
            ("CONTINUE", "REJECT", "PULL", None, "bogus"), (True, False, None, "true", 1),
            (True, False, None, "yes"), ("same", True, False, None)):
        s = steps[exp]
        after_ = None if after == "same" else {"part_present": after}
        c = classify(s, {"part_present": sensor}, {"result": res, "disposition": disp,
                                                   "observed": {"part_present": seen}}, after_)
        want = (res == "PASS" and disp == "CONTINUE" and sensor is exp and seen is exp
                and (after == "same" or after is exp))
        assert (c["outcome"] == "CONFIRMED") is want, (exp, res, disp, sensor, seen, after, c)
        n += 1
    assert n == 4800
    assert job_outcome("PASS", "CONTINUE") == "CONFIRMED" and job_outcome("FAIL", "REJECT") == "DEFECT"
    for res, disp in (("PASS", "PULL"), ("INVALID", "REJECT"), ("INVALID", "PULL"), (None, None),
                      ("PASS", None), ("FAIL", "CONTINUE")):
        assert job_outcome(res, disp) == "DOUBT"
    assert job_outcome("ABORTED", "PULL") == "ABORTED"


# ================================================================== the cycle

def test_a_cycle_confirms_every_work_state_grants_in_order_and_continues():
    cyc = Cycle(_cfg(), "long")
    assert cyc.permissions == {"LOAD_DONE": False, "CLAMP": False, "TRANSFER": False}
    assert cyc.begin() == [("wait", "step")] and cyc.state == "WAIT_STEP"
    with pytest.raises(CycleError):
        cyc.ack()
    assert cyc.step_now() == [("verify", {"step": "loaded", "index": 1, "attempt": 1})]
    req = cyc.request({"part_present": True, "other": False})
    assert req["step"] == "loaded" and req["profile"] == "long" and req["inputs"] == {
        "part_present": True}
    assert cyc.verified(OK, {"part_present": True}) == [("wait", "step")]
    assert cyc.permissions == {"LOAD_DONE": True, "CLAMP": False, "TRANSFER": False}
    assert cyc.index == 1 and cyc.rows[0]["permission_granted"] and cyc.rows[0]["outcome"] == "CONFIRMED"
    cyc.step_now()
    cyc.request({"part_present": True})
    cyc.verified(GOOD)
    assert cyc.permissions["CLAMP"] and not cyc.permissions["TRANSFER"]
    cyc.step_now()
    cyc.request({"part_present": True})
    assert cyc.verified(GOOD) == [("done", None)]
    assert cyc.state == "DONE" and (cyc.result, cyc.disposition) == ("PASS", "CONTINUE")
    assert cyc.permissions == {"LOAD_DONE": True, "CLAMP": True, "TRANSFER": True}
    rec = cyc.record()
    assert rec["ended_by"] == "confirmed" and rec["pull_reasons"] == [] and not rec["active"]
    assert [r["outcome"] for r in rec["steps"]] == ["CONFIRMED"] * 3
    assert rec["steps"][0]["sensor_inputs"] == {"part_present": True}
    assert rec["steps"][0]["dlp_result"]["observed"] == {"part_present": True}
    names = [e["event"] for e in cyc.drain_events()]
    assert names[0] == "cycle_start" and names[-1] == "cycle_end" and names.count("permission") == 3
    assert cyc.abort() == [] and cyc.tick() == []            # ended: nothing more happens


def test_corrected_in_both_directions_prompts_holds_and_verifies_again():
    cyc = Cycle(_cfg(), "long")
    _walk(cyc, [ABSENT], {"part_present": True}, ack=False)
    row = cyc.rows[0]
    assert cyc.state == "WAIT_ACK" and row["outcome"] == "CORRECTED" and cyc.sensor_mismatch
    assert row["sensor_mismatch"][0] == {"signal": "part_present", "sensor": True, "dlp": False,
                                         "expected": True}
    assert not any(cyc.permissions.values()) and not row["permission_granted"]
    # the fault (the missing-part zone) is painted, then the prompt for THIS direction
    assert [a["action"] for a in row["actions"]] == ["project_fault", "prompt"]
    assert row["actions"][0]["requested"] is True
    assert cyc.prompt["message"] == "no tube found: load or clear the sensor"
    assert cyc.prompt["detail"] == "sensor part_present says true, the DLP says false"
    with pytest.raises(CycleError):
        cyc.step_now()                                      # a prompt is open: ACK first
    # ACK (the operator loaded a tube): hold verifies the work state again
    assert cyc.ack("operator") == [("verify", {"step": "loaded", "index": 1, "attempt": 2})]
    assert cyc.prompt is None and row["earlier"][0]["outcome"] == "CORRECTED"
    assert row["earlier"][0]["actions"][1]["acknowledged"] is True
    cyc.request({"part_present": True})
    assert cyc.verified(OK) == [("wait", "step")]
    assert row["outcome"] == "CONFIRMED" and cyc.permissions["LOAD_DONE"] and not cyc.sensor_mismatch
    # the reverse: the sensor says absent, the DLP sees the tube - the other prompt, and
    # nothing to paint (every check passed: the fault is the sensor)
    cyc = Cycle(_cfg(), "long")
    _walk(cyc, [OK], {"part_present": False}, ack=False)
    row = cyc.rows[0]
    assert row["outcome"] == "CORRECTED" and row["sensor_mismatch"][0]["dlp"] is True
    assert row["actions"][0] == {"action": "project_fault", "requested": False,
                                 "why": "every check passed: there is no fault on the part to paint"}
    assert cyc.prompt["message"].startswith("a tube is on the seat but the sensor says none")
    assert not any(cyc.permissions.values())
    # still wrong after the ACK: the same again (hold never grants anything by itself)
    cyc.ack()
    cyc.request({"part_present": False})
    cyc.verified(OK)
    assert cyc.state == "WAIT_ACK" and row["attempts"] == 2 and not cyc.permissions["LOAD_DONE"]


def test_not_met_runs_the_mismatch_list_by_default_and_a_defect_its_own():
    cyc = Cycle(_cfg(), "long")
    _walk(cyc, [ABSENT], {"part_present": False}, ack=False)
    row = cyc.rows[0]
    assert row["outcome"] == "NOT_MET" and row["sensor_mismatch"] == [] and not cyc.sensor_mismatch
    assert cyc.prompt["message"] == "no tube found: load or clear the sensor"
    # a tube that is there but not this profile's: DEFECT at "Tube loaded?"
    cyc = Cycle(_cfg(), "long")
    _walk(cyc, [dlp("FAIL", "REJECT", part_present=True)], ack=False)
    assert cyc.rows[0]["outcome"] == "DEFECT" and "not the selected profile" in cyc.prompt["message"]


def test_retry_rescans_up_to_the_budget_then_the_cycle_is_pulled():
    cyc = _walk(Cycle(_cfg(), "long"), [OK, FAILS, FAILS, FAILS])
    row = cyc.rows[1]
    assert row["outcome"] == "DEFECT" and row["attempts"] == 3 and row["retries_used"] == 2
    assert [a["action"] for a in row["actions"]] == ["project_fault", "prompt", "retry"]
    assert row["actions"][-1] == {"action": "retry", "exhausted": True, "of": 2}
    assert (cyc.result, cyc.disposition, cyc.ended_by) == ("FAIL", "PULL", "unconfirmed")
    assert cyc.pull_reasons[0]["reason"] == "not_confirmed" and cyc.pull_reasons[0]["step"] == "aligned"
    assert not any(cyc.permissions.values())                # LOAD_DONE dropped with the cycle
    # the row says what stands NOW (never GRANTED next to "permissions: none"), and why
    assert cyc.rows[0]["permission_granted"] is False
    assert cyc.rows[0]["permission_note"] == "dropped: the cycle ended PULL"
    assert cyc.kickout["by"] == "unconfirmed" and cyc.kickout["disposition"] == "PULL"
    assert cyc.rows[2]["outcome"] == "PENDING"
    # re-seated in time: the second scan confirms and the flow goes on
    cyc = _walk(Cycle(_cfg(), "long"), [OK, FAILS, GOOD, GOOD])
    assert cyc.disposition == "CONTINUE" and cyc.rows[1]["attempts"] == 2
    assert cyc.rows[1]["earlier"][0]["outcome"] == "DEFECT"
    # the RETRY command uses the same budget and needs no ACK
    cyc = _walk(Cycle(_cfg(), "long"), [OK, FAILS], ack=False)
    assert cyc.state == "WAIT_ACK"
    assert cyc.retry("plc")[0][0] == "verify" and cyc.rows[1]["retries_used"] == 1
    cyc.request({"part_present": True})
    cyc.verified(FAILS)
    cyc.retry()
    cyc.request({"part_present": True})
    cyc.verified(FAILS)
    with pytest.raises(CycleError, match="no retry left"):
        cyc.retry()


def test_kickout_reject_only_on_a_definite_finding_and_a_doubt_is_pulled():
    cyc = _walk(Cycle(_cfg(), "long"), [OK, GOOD, FAILS])
    row = cyc.rows[2]
    assert row["outcome"] == "DEFECT" and row["disposition"] == "REJECT"
    assert (cyc.result, cyc.disposition, cyc.ended_by) == ("FAIL", "REJECT", "kickout")
    assert cyc.kickout == {"step": "end_face", "disposition": "REJECT",
                           "reason": "end face not OK", "detail": "seat/check: it failed",
                           "by": "kickout"}
    assert cyc.pull_reasons == [] and not any(cyc.permissions.values())
    # the default on a doubt: kickout PULL, with the DLP's own reasons, never a permission
    cyc = _walk(Cycle(_cfg(), "long"), [OK, dlp("INVALID", "PULL", part_present=True)])
    assert cyc.rows[1]["outcome"] == "DOUBT" and (cyc.result, cyc.disposition) == ("INVALID", "PULL")
    assert cyc.kickout["disposition"] == "PULL" and not any(cyc.permissions.values())
    assert {r["reason"] for r in cyc.pull_reasons} == {"too_few_points"}
    assert all(r["step"] == "aligned" for r in cyc.pull_reasons)
    # a kickout REJECT written for a mismatch is still a PULL when the outcome is a doubt:
    # on_not_met defaults to the on_mismatch list, here ending in kickout REJECT
    steps = [{"id": "s", "expects": {"part_present": True},
              "verify": {"recipe": "presence", "observe": {"part_present": "seat/presence"}},
              "permit": "GO", "on_mismatch": ["kickout REJECT sensor wrong"],
              "on_doubt": ["project_fault", "kickout PULL"]}]
    cyc = _walk(Cycle(_cfg(steps), "long"), [ABSENT])
    assert (cyc.result, cyc.disposition) == ("FAIL", "REJECT")       # CORRECTED: definite
    cyc = _walk(Cycle(_cfg(steps), "long"), [dlp("PASS", "PULL", part_present=True)])
    assert cyc.disposition == "PULL" and not cyc.permissions["GO"]
    cyc = Cycle(_cfg(steps), "long")
    cyc.begin()
    cyc.step_now()
    cyc.request({"part_present": True})
    cyc.rows[0]["outcome"] = "DOUBT"                        # whatever list a doubt runs
    assert cyc._kickout({"action": "kickout", "disposition": "REJECT", "reason": ""}) == [
        ("done", None)]
    assert cyc.disposition == "PULL" and "never a reject" in cyc.rows[0]["actions"][-1]["note"]


def test_a_required_work_state_without_actions_ends_the_cycle_pull():
    steps = [{"id": "a", "verify": {"recipe": "presence"}, "permit": "A"},
             {"id": "b", "verify": {"recipe": "alignment"}, "permit": "B"}]
    cyc = _walk(Cycle(_cfg(steps), "long"), [GOOD, FAILS])
    assert (cyc.result, cyc.disposition, cyc.ended_by) == ("FAIL", "PULL", "unconfirmed")
    assert cyc.permissions == {"A": False, "B": False}
    # no kickout ACTION, but the part still leaves the line: the record says where and why
    assert cyc.kickout["by"] == "unconfirmed" and cyc.kickout["step"] == "b"
    assert cyc.kickout["disposition"] == "PULL" and "b was not confirmed" in cyc.kickout["reason"]
    # a work state that is not required is left unconfirmed and the flow goes on: the cycle
    # can still CONTINUE, its permission stays withheld
    steps[0].update(required=False, allow_skip=True)
    cyc = _walk(Cycle(_cfg(steps), "long"), [FAILS, GOOD])
    assert cyc.disposition == "CONTINUE" and cyc.permissions == {"A": False, "B": True}
    assert cyc.rows[0]["outcome"] == "DEFECT" and cyc.rows[0]["actions"][-1]["action"] == "continue"
    # SKIP: only where the flow allows it, and it never grants
    cyc = Cycle(_cfg(steps), "long")
    cyc.begin()
    assert cyc.skip("operator") == [("wait", "step")] and cyc.rows[0]["outcome"] == "SKIPPED"
    assert not cyc.permissions["A"]
    with pytest.raises(CycleError, match="does not allow SKIP"):
        cyc.skip()


def test_goto_and_on_confirm_jumps_reverify_and_never_skip_a_required_work_state():
    steps = [{"id": "a", "verify": {"recipe": "presence"}, "permit": "A"},
             {"id": "b", "verify": {"recipe": "alignment"}, "permit": "B",
              "on_defect": ["goto a"]},
             {"id": "c", "verify": {"recipe": "end_face"}, "permit": "C"}]
    cyc = _walk(Cycle(_cfg(steps), "long"), [GOOD, FAILS, GOOD, GOOD, GOOD])
    # b failed -> back to a: a is verified again (its permission dropped meanwhile)
    assert cyc.disposition == "CONTINUE" and cyc.rows[0]["attempts"] == 2
    assert cyc.rows[0]["earlier"][0]["outcome"] == "CONFIRMED"
    ev = [e for e in cyc.drain_events() if e["event"] == "permission"]
    assert [(e["permission"], e["granted"]) for e in ev] == [
        ("A", True), ("A", False), ("A", True), ("B", True), ("C", True)]
    # on_confirm done / a jump over a required work state: the cycle is NOT a CONTINUE
    steps[1].pop("on_defect")
    steps[0]["on_confirm"] = "c"
    cyc = _walk(Cycle(_cfg(steps), "long"), [GOOD, GOOD])
    assert (cyc.result, cyc.disposition) == ("INVALID", "PULL") and "b (PENDING)" in \
        cyc.pull_reasons[0]["message"]
    assert not any(cyc.permissions.values())
    steps[0]["on_confirm"] = "done"
    assert _walk(Cycle(_cfg(steps), "long"), [GOOD]).disposition == "PULL"
    # a flow that loops for ever ends PULL instead of scanning for ever
    loop = [{"id": "a", "verify": {"recipe": "presence"}, "trigger": "auto",
             "on_defect": ["goto a"]}]
    cyc = Cycle(_cfg(loop), "long")
    effects = cyc.begin()
    while effects and effects[0][0] == "verify":
        cyc.request({})
        effects = cyc.verified(FAILS)
    assert cyc.state == "DONE" and cyc.ended_by == "loop" and cyc.disposition == "PULL"


def test_revoke_abort_fail_and_timeout_drop_permissions():
    steps = _doc()["flow"]
    steps[1]["revoke"] = "all"
    cyc = _walk(Cycle(_cfg(steps), "long"), [OK, FAILS], ack=False)
    assert cyc.state == "WAIT_ACK" and cyc.permissions["LOAD_DONE"] is False     # revoked
    cyc = _walk(Cycle(_cfg(), "long"), [OK, FAILS], ack=False)
    assert cyc.permissions["LOAD_DONE"] is True             # not configured: it stays for now
    assert cyc.abort("aborted by the operator") == [("done", None)]
    assert (cyc.result, cyc.disposition, cyc.ended_by) == ("ABORTED", "PULL", "abort")
    assert not any(cyc.permissions.values()) and cyc.rows[1]["outcome"] == "ABORTED"
    assert cyc.pull_reasons[0]["reason"] == "aborted"
    cyc = _walk(Cycle(_cfg(), "long"), [OK], ack=False)
    cyc = Cycle(_cfg(), "long")
    cyc.begin()
    cyc.step_now()
    assert cyc.fail("start failed: rig busy") == [("done", None)]
    assert (cyc.result, cyc.disposition) == ("INVALID", "PULL") and cyc.rows[0]["outcome"] == "DOUBT"
    # timeout_s: a work state that waits too long for STEP / ACK
    steps = _doc()["flow"]
    steps[1]["timeout_s"] = 30
    now = [100.0]
    cyc = Cycle(_cfg(steps), "long", clock=lambda: now[0])
    cyc.begin()
    cyc.step_now()
    cyc.request({"part_present": True})
    cyc.verified(OK)
    assert cyc.state == "WAIT_STEP" and not cyc.due() and cyc.tick() == []
    now[0] += 31
    assert cyc.due() and cyc.tick() == [("done", None)]
    assert (cyc.result, cyc.disposition, cyc.ended_by) == ("INVALID", "PULL", "timeout")
    assert cyc.pull_reasons[0]["reason"] == "timeout" and not any(cyc.permissions.values())
    assert cyc.rows[1]["outcome"] == "DOUBT" and "within 30 s" in cyc.rows[1]["message"]


# ================================================================== the command bus

def _flow_bus(cfg=None, confirm=False, clock=None):
    """A bus that knows one configuration with profiles + a flow (line.json) and one old
    single-job config (old.json). The owner records what it is asked."""
    calls = []
    h = C.Handlers(start=lambda c, ok: calls.append(("start", c, ok)),
                   verify=lambda req: calls.append(("verify", req)),
                   project=lambda req: calls.append(("project", req["step"])),
                   cycle_begin=lambda info: (calls.append(("begin", info["config"],
                                                           info["profile"])),
                                             {"cycle_id": f"cyc{len(calls)}"})[1],
                   cycle_end=lambda rec: calls.append(("end", rec["result"], rec["disposition"])),
                   abort=lambda: calls.append(("abort",)),
                   home=lambda: calls.append(("home",)),
                   select=lambda c, preset: calls.append(("select", c, preset)))
    kw = {"clock": clock} if clock else {}
    bus = C.CommandBus(h, require_confirmation=confirm, **kw)
    out = C.SimulatedOutput()
    bus.outputs.append(out)
    flows = {"line.json": cfg or _cfg(),
             "old.json": FS.lift_legacy({"station_id": "old", "job": {"type": "metric"}})}
    bus.update_configs(["line.json", "old.json"], flows=flows)
    return bus, calls, out


def _answer(bus, answer):
    """The DLP check the bus started ended with `answer` (what the server's job reports)."""
    assert bus.state == "RUNNING"
    bus.job_finished("run", answer.get("result"), None, disposition=answer.get("disposition"),
                     step=_for_this(bus, answer))


def _for_this(bus, answer: dict) -> dict:
    """The answer as the owner reports it: it names the verify the bus asked for (work
    state, attempt, cycle), as the server's flow_verify job does."""
    row = bus.cycle.row
    return {**answer, "step": row["id"], "attempt": row["attempts"],
            "cycle_id": bus.cycle.meta.get("cycle_id")}


def test_bus_walks_the_work_states_with_permissions_outputs_and_registers():
    bus, calls, out = _flow_bus()
    snap = bus.snapshot()
    assert snap["station_config"] == "line.json" and snap["profile"] == "long"
    assert [p["name"] for p in snap["profiles"]] == ["long", "short"] and snap["legacy"] is False
    assert [s["id"] for s in snap["flow"]] == ["loaded", "aligned", "end_face"]
    assert snap["permissions"] == {"LOAD_DONE": False, "CLAMP": False, "TRANSFER": False}
    assert snap["cycle"] is None and snap["signal_names"] == ["part_present"]
    assert {"PERMIT_LOAD_DONE", "PERMIT_CLAMP", "PERMIT_TRANSFER", "KICKOUT", "PROMPT",
            "SENSOR_MISMATCH"} <= set(out.current)
    assert out.registers == {"result_code": 0, "state_code": 0, "station_config": 1,
                             "profile": 1, "work_state": 0, "work_state_code": 0}
    assert not bus.command("STEP").accepted and not bus.command("ACK").accepted
    bus.set_inputs({"part_present": True}, source="plc")
    r = bus.command("START", "plc")
    assert r.accepted and bus.state == "WAITING" and "waits for STEP" in r.message
    assert calls[-1] == ("begin", "line.json", "long")
    assert not out.current["READY"] and not out.current["BUSY"]
    assert out.registers["state_code"] == C.STATES.index("WAITING") == 6
    assert out.registers["work_state"] == 1 and out.registers["work_state_code"] == 0
    for cmd in ("START", "HOME", "ARM"):                    # a cycle is open
        assert not bus.command(cmd).accepted, cmd
    for cmd, arg in (("SELECT_PROFILE", "short"), ("SELECT_CONFIG", "old.json")):
        r = bus.command(cmd, arg=arg)
        assert not r.accepted and "WAITING" in r.message, r
    assert not bus.command("STEP", arg="aligned").accepted  # not the current work state
    r = bus.command("STEP", arg="loaded")
    assert r.accepted and bus.state == "RUNNING" and out.current["BUSY"]
    kind, req = calls[-1]
    assert kind == "verify" and req["step"] == "loaded" and req["profile"] == "long"
    assert req["inputs"] == {"part_present": True} and req["cycle_id"].startswith("cyc")
    assert req["confirmed"] is False and req["attempt"] == 1
    assert not bus.command("STEP").accepted and not bus.command("SELECT_PROFILE", arg=2).accepted
    _answer(bus, OK)
    assert bus.state == "WAITING" and out.current["PERMIT_LOAD_DONE"]
    assert not out.current["PERMIT_CLAMP"] and not out.current["PASS"]
    assert out.registers["work_state"] == 2 and bus.permissions()["LOAD_DONE"] is True
    assert bus.command("STEP", arg=2).accepted
    _answer(bus, GOOD)
    assert bus.command("TRIGGER").accepted                  # TRIGGER verifies a waiting work state
    _answer(bus, GOOD)
    assert bus.state == "READY" and (bus.last_result, bus.last_disposition) == ("PASS", "CONTINUE")
    assert out.current["PASS"] and out.current["READY"] and not out.current["KICKOUT"]
    assert all(out.current[f"PERMIT_{p}"] for p in ("LOAD_DONE", "CLAMP", "TRANSFER"))
    assert out.registers == {"result_code": 1, "state_code": 0, "station_config": 1,
                             "profile": 1, "work_state": 3,
                             "work_state_code": OUTCOME_CODES["CONFIRMED"]}
    assert calls[-1] == ("end", "PASS", "CONTINUE")
    assert [c[0] for c in calls].count("verify") == 3 and ("start",) not in [c[:1] for c in calls]
    # the next cycle starts with no permission
    assert bus.command("START").accepted
    assert not any(out.current[f"PERMIT_{p}"] for p in ("LOAD_DONE", "CLAMP", "TRANSFER"))
    assert not out.current["PASS"]


def test_bus_prompt_ack_sensor_mismatch_and_kickout_outputs():
    bus, calls, out = _flow_bus()
    bus.set_inputs({"part_present": True})
    bus.command("START")
    bus.command("STEP")
    _answer(bus, ABSENT)
    assert bus.state == "WAITING" and out.current["PROMPT"] and out.current["SENSOR_MISMATCH"]
    assert not out.current["PERMIT_LOAD_DONE"] and ("project", "loaded") in calls
    assert out.registers["work_state_code"] == OUTCOME_CODES["CORRECTED"]
    snap = bus.snapshot()
    assert snap["prompt"]["message"] == "no tube found: load or clear the sensor"
    assert snap["cycle"]["steps"][0]["sensor_mismatch"][0]["signal"] == "part_present"
    r = bus.command("STEP")
    assert not r.accepted and "waits for ACK" in r.message
    r = bus.command("ACK", "operator")
    assert r.accepted and bus.state == "RUNNING" and not out.current["PROMPT"]
    assert calls[-1][1]["attempt"] == 2
    _answer(bus, OK)
    assert out.current["PERMIT_LOAD_DONE"] and not out.current["SENSOR_MISMATCH"]
    assert not bus.command("ACK").accepted and not bus.command("RETRY").accepted
    bus.command("STEP")
    _answer(bus, FAILS)                                     # misaligned: prompt, then retry
    assert out.current["PROMPT"] and bus.command("RETRY").accepted and bus.state == "RUNNING"
    _answer(bus, GOOD)
    bus.command("STEP")
    _answer(bus, FAILS)                                     # dented: kickout REJECT
    assert bus.state == "READY" and (bus.last_result, bus.last_disposition) == ("FAIL", "REJECT")
    assert out.current["FAIL"] and out.current["KICKOUT"] and not out.current["PASS"]
    assert not any(out.current[f"PERMIT_{p}"] for p in ("LOAD_DONE", "CLAMP", "TRANSFER"))
    assert out.registers["result_code"] == 2
    assert out.registers["work_state_code"] == OUTCOME_CODES["DEFECT"]
    assert calls[-1] == ("end", "FAIL", "REJECT")
    assert not bus.command("SKIP").accepted
    # the KICKOUT stays with the cycle until the next one starts
    bus.command("START")
    assert not out.current["KICKOUT"] and not out.current["FAIL"]


def test_bus_never_grants_a_permission_on_invalid_pull_or_a_lost_job():
    for answer in (dlp("INVALID", "PULL", part_present=None), dlp("PASS", "PULL", part_present=True),
                   dlp("INVALID", "REJECT", part_present=True), {"error": "boom"},
                   {"result": "PASS", "disposition": "CONTINUE"},        # nothing observed
                   dlp("PASS", "CONTINUE", part_present="yes"), {}):
        bus, calls, out = _flow_bus()
        seen = []
        bus.outputs.append(type("Spy", (C.OutputBackend,), {
            "publish": lambda self, sig, seen=seen: seen.append(dict(sig)),
            "publish_registers": lambda self, regs: None})())
        bus.set_inputs({"part_present": True})
        bus.command("START")
        bus.command("STEP")
        _answer(bus, answer)
        assert bus.state == "READY" and bus.last_disposition == "PULL", answer
        assert bus.last_result == "INVALID" and out.current["PULL"] and out.current["ALARM"]
        assert not any(v for s in seen for k, v in s.items()
                       if k.startswith("PERMIT_") or k == "PASS"), answer
        assert bus.cycle.rows[0]["outcome"] == "DOUBT" and not bus.cycle.rows[0]["permission_granted"]
        assert calls[-1] == ("end", "INVALID", "PULL")
    # no sensor claim at all: a doubt too (the DLP has nothing to confirm or correct)
    bus, calls, out = _flow_bus()
    bus.command("START")
    bus.command("STEP")
    _answer(bus, OK)
    assert bus.last_disposition == "PULL" and not out.current["PERMIT_LOAD_DONE"]
    assert bus.cycle.pull_reasons[0]["reason"] == "sensor_unknown"
    # a sensor that changes while the DLP scans
    bus, calls, out = _flow_bus()
    bus.set_inputs({"part_present": True})
    bus.command("START")
    bus.command("STEP")
    bus.set_inputs({"part_present": False})
    _answer(bus, OK)
    assert bus.last_disposition == "PULL" and "changed during the scan" in bus.cycle.rows[0]["message"]
    # the job crashed / was lost / its start was refused
    bus, calls, out = _flow_bus()
    bus.set_inputs({"part_present": True})
    bus.command("START")
    bus.command("STEP")
    bus.job_finished("run", None, error="RuntimeError: camera gone")
    assert (bus.last_result, bus.last_disposition) == ("INVALID", "PULL") and bus.state == "READY"
    bus.command("START")
    bus.command("STEP")
    bus.job_lost("gone")
    assert bus.last_disposition == "PULL" and not any(bus.permissions().values())

    def boom(req):
        raise RuntimeError("rig busy")
    bus.h.verify = boom
    bus.command("START")
    r = bus.command("STEP")
    assert not r.accepted and bus.state == "FAULT" and "rig busy" in bus.fault
    assert bus.last_disposition == "PULL" and bus.cycle.state == "DONE"
    assert calls[-1] == ("end", "INVALID", "PULL") and not any(bus.signals()[k] for k in
                                                                  bus.signals() if k.startswith("PERMIT_"))


def test_permissions_drop_on_abort_home_fault_and_a_profile_or_config_change():
    def confirmed_cycle():
        bus, calls, out = _flow_bus()
        bus.set_inputs({"part_present": True})
        bus.command("START")
        for answer in (OK, GOOD, GOOD):
            bus.command("STEP")
            _answer(bus, answer)
        assert all(bus.permissions().values()) and out.current["PERMIT_TRANSFER"]
        return bus, calls, out

    bus, calls, out = confirmed_cycle()
    assert bus.command("ABORT").accepted and not any(bus.permissions().values())
    assert not out.current["PERMIT_TRANSFER"] and bus.history[-2][2] == "PERMISSIONS"
    bus, calls, out = confirmed_cycle()
    assert bus.command("HOME").accepted and not any(out.current[k] for k in out.current
                                                    if k.startswith("PERMIT_"))
    assert not out.current["PASS"]                          # the rig check ends the PASS too
    bus.job_finished("home", "OK")
    assert bus.state == "READY" and not any(bus.permissions().values())
    bus, calls, out = confirmed_cycle()
    bus.command("HOME")
    bus.job_finished("home", "MOVED")
    assert bus.state == "FAULT" and not any(bus.permissions().values())
    bus.command("RESET")
    assert bus.state == "READY" and not any(bus.permissions().values())     # never handed back
    bus, calls, out = confirmed_cycle()
    r = bus.command("SELECT_PROFILE", arg="short")
    assert r.accepted and bus.profile == "short" and not any(bus.permissions().values())
    assert bus.cycle is None and calls[-1] == ("select", "line.json", PRESET)
    assert out.registers["profile"] == 2 and out.registers["work_state"] == 0
    bus, calls, out = confirmed_cycle()
    assert bus.command("SELECT_PROFILE", arg="long").accepted and all(bus.permissions().values())
    r = bus.command("SELECT_CONFIG", arg="old.json")        # the same profile: nothing changed
    assert r.accepted and bus.config == "old.json" and bus.profile == "default"
    # the other configuration has no permission of its own; the deselected one's are
    # published OFF (a backend that writes one coil per signal never keeps one latched)
    assert bus.permissions() == {}
    assert [out.current[f"PERMIT_{p}"] for p in ("LOAD_DONE", "CLAMP", "TRANSFER")] == [False] * 3
    assert not out.current["PASS"] and out.registers["result_code"] == 0
    # mid-cycle: ABORT while a work state waits, and while one is verified
    bus, calls, out = _flow_bus()
    bus.set_inputs({"part_present": True})
    bus.command("START")
    bus.command("STEP")
    _answer(bus, OK)
    assert out.current["PERMIT_LOAD_DONE"]
    r = bus.command("ABORT")
    assert r.accepted and bus.state == "READY" and "cycle aborted" in r.message
    assert (bus.last_result, bus.last_disposition) == ("ABORTED", "PULL") and out.current["PULL"]
    assert not out.current["PERMIT_LOAD_DONE"] and calls[-2] == ("end", "ABORTED", "PULL")
    assert bus.inputs == {}                                 # ABORT forgets the sensor claims
    bus.set_inputs({"part_present": True})                  # the line reports for the next part
    bus.command("START")
    bus.command("STEP")
    _answer(bus, OK)
    bus.command("STEP")
    assert bus.command("ABORT").accepted and bus.state == "ABORTING"
    assert not out.current["PERMIT_LOAD_DONE"]              # at once, not when the job ends
    _answer_aborting(bus, GOOD)
    assert bus.state == "READY" and bus.last_result == "ABORTED" and not any(bus.permissions().values())
    assert bus.cycle.rows[1]["outcome"] == "ABORTED"


def _answer_aborting(bus, answer):
    assert bus.state == "ABORTING"
    bus.job_finished("run", answer["result"], None, disposition=answer["disposition"],
                     step=_for_this(bus, answer))


def test_select_profile_by_name_or_number_disarms_and_names_the_expected_one():
    bus, calls, out = _flow_bus(confirm=True)
    assert bus.command("SELECT_PROFILE", arg=2).accepted and bus.profile == "short"
    assert bus.command("SELECT_PROFILE", arg="1").accepted and bus.profile == "long"
    for bad in ("medium", 9, None, True):
        r = bus.command("SELECT_PROFILE", arg=bad)
        assert not r.accepted and "has no profile" in r.message, bad
    assert bus.command("ARM", confirmed=True).accepted
    assert bus.command("SELECT_PROFILE", arg="long").accepted and bus.state == "ARMED"
    r = bus.command("SELECT_PROFILE", arg="short")
    assert r.accepted and "disarmed" in r.message and bus.state == "READY" and not bus.armed_confirmed
    # START / ARM naming the configuration + profile the sender expects
    r = bus.command("START", arg={"config": "line.json", "profile": "long"}, confirmed=True)
    assert not r.accepted and "the selected profile is short" in r.message
    assert not bus.command("START", arg={"config": 2, "profile": "short"}, confirmed=True).accepted
    assert not bus.command("START", arg={"config": "line.json", "profile": "short"}).accepted
    assert bus.command("START", arg={"config": 1, "profile": 2}, confirmed=True).accepted
    assert calls[-1] == ("begin", "line.json", "short")
    # ARMED + TRIGGER: the cycle starts AND its first work state is verified at once
    bus, calls, out = _flow_bus()
    bus.set_inputs({"part_present": True})
    bus.command("ARM")
    assert bus.command("TRIGGER").accepted and bus.state == "RUNNING"
    _answer(bus, OK)
    for answer in (GOOD, GOOD):
        assert bus.command("TRIGGER").accepted
        _answer(bus, answer)
    assert bus.state == "ARMED" and out.current["PASS"]     # ready for the next part
    # a profile that left the configuration: refused, never replaced by another
    cfg = FS.validate({**_doc(), "profiles": {"long": _doc()["profiles"]["long"]}})
    bus, calls, out = _flow_bus()
    bus.command("SELECT_PROFILE", arg="short")
    bus.update_configs(["line.json", "old.json"], flows={"line.json": cfg})
    r = bus.command("START")
    assert not r.accepted and "profile 'short' is not one of its profiles" in r.message
    assert bus.profile == "short" and bus.last_disposition == "PULL"


def test_bus_timeout_ends_a_waiting_work_state_pull():
    steps = _doc()["flow"]
    steps[0]["timeout_s"] = 10
    now = [0.0]
    bus, calls, out = _flow_bus(_cfg(steps), clock=lambda: now[0])
    bus.command("START")
    bus.tick()
    assert bus.state == "WAITING"
    now[0] = 11.0
    bus.tick()
    assert bus.state == "READY" and (bus.last_result, bus.last_disposition) == ("INVALID", "PULL")
    assert out.current["PULL"] and out.current["ALARM"] and bus.history[-1][2] == "TIMEOUT"
    assert bus.cycle.ended_by == "timeout" and calls[-1] == ("end", "INVALID", "PULL")
    # a prompt nobody acknowledges
    bus.set_inputs({"part_present": True})
    bus.command("START")
    bus.command("STEP")
    _answer(bus, ABSENT)
    assert out.current["PROMPT"]
    now[0] += 60
    r = bus.command("ACK")                                  # every command looks at the clock
    assert not r.accepted and bus.state == "READY" and not out.current["PROMPT"]
    assert bus.cycle.rows[0]["outcome"] == "CORRECTED" and bus.last_disposition == "PULL"


def test_sensor_inputs_are_strict_booleans():
    d = _doc()
    d["parameters"]["io"]["inputs"]["door.closed"] = "the cell door switch"
    bus, calls, out = _flow_bus(FS.validate(d))
    assert bus.set_inputs({"part_present": True, "door.closed": False}, "plc") == {
        "part_present": True, "door.closed": False}
    assert bus.set_inputs({"door.closed": None}) == {"part_present": True}      # forgotten
    assert bus.snapshot()["inputs"] == {"part_present": True}
    for bad in ({"part_present": "true"}, {"part_present": 1}, {"part present": True},
                {"": True}, ["part_present"], {"x": [True]}, {3: True}):
        with pytest.raises(ValueError):
            bus.set_inputs(bad)
    assert bus.inputs == {"part_present": True}
    assert any(h[2] == "INPUTS" and h[1] == "plc" for h in bus.history)
    with pytest.raises(ValueError, match="more than"):
        bus.set_inputs({f"s{i}": True for i in range(C.MAX_INPUTS + 1)})


def test_select_workstate_is_a_deprecated_alias_of_select_config():
    """SELECT_WORKSTATE selected a station configuration, never a work state: kept only as
    an alias, and the reply says so."""
    bus, calls, out = _flow_bus()
    assert "SELECT_WORKSTATE" not in C.COMMANDS
    assert C.DEPRECATED_COMMANDS == {"SELECT_WORKSTATE": "SELECT_CONFIG"}
    r = bus.command("SELECT_WORKSTATE", arg=2)
    assert r.accepted and r.command == "SELECT_CONFIG" and bus.config == "old.json"
    assert "SELECT_WORKSTATE is deprecated: use SELECT_CONFIG" in r.message
    assert bus.history[-1][2] == "SELECT_WORKSTATE"
    r = bus.command("SELECT_WORKSTATE", arg="nope.json")
    assert not r.accepted and "deprecated" in r.message
    snap = bus.snapshot()
    assert "workstate" not in snap and "workstates" not in snap
    assert "workstate" not in snap["registers"] and "station_config" in snap["registers"]
    assert not hasattr(bus, "workstate") and not hasattr(bus, "update_workstates")


def test_an_old_single_job_config_runs_on_the_bus_as_before():
    bus, calls, out = _flow_bus()
    bus.command("SELECT_CONFIG", arg="old.json")
    snap = bus.snapshot()
    assert snap["legacy"] is True and [p["name"] for p in snap["profiles"]] == ["default"]
    assert [s["id"] for s in snap["flow"]] == ["run"] and snap["permissions"] == {}
    r = bus.command("START")
    assert r.accepted and bus.state == "RUNNING" and r.message == "started old.json"
    assert calls[-1] == ("start", "old.json", False)        # its job, not a DLP check of a flow
    assert not bus.command("STEP").accepted
    bus.job_finished("run", "FAIL", disposition="REJECT")
    assert bus.state == "READY" and out.current["FAIL"] and out.current["KICKOUT"]
    assert bus.cycle.legacy and bus.cycle.rows[0]["outcome"] == "DEFECT"
    assert out.registers["work_state"] == 1 and out.registers["work_state_code"] == OUTCOME_CODES["DEFECT"]
    bus.command("START")
    bus.job_finished("run", "PASS", disposition="PULL")
    assert out.current["PULL"] and bus.cycle.rows[0]["outcome"] == "DOUBT"
    assert not any(c[0] in ("begin", "end", "verify") for c in calls)


def test_io_config_maps_sensor_signals_and_permit_outputs():
    doc = json.loads((ROOT / "stations" / "io_example.json").read_text(encoding="utf-8"))
    ins, outs = C.load_io_doc(doc)
    (sim,), (sim_out,) = ins, outs
    bus, calls, _ = _flow_bus()
    sim.start(bus)
    assert sim.set_level("part_present", True) == {"part_present": True}
    assert sim.inject("start").accepted and sim.inject("step").accepted and bus.state == "RUNNING"
    _answer(bus, ABSENT)
    assert sim.inject("ack").accepted
    with pytest.raises(KeyError):
        sim.set_level("Line9", True)
    every = copy.deepcopy(doc)
    for spec in every["inputs"] + every["outputs"]:
        spec["enabled"] = True
    ins2, outs2 = C.load_io_doc(every)
    dio = next(i for i in ins2 if i.name == "digital_io")
    assert dio.signals == {"DI4": "part_present"} and dio.mapping["DI5"] == "STEP"
    mb = next(o for o in outs2 if o.name == "modbus_tcp")
    assert mb.mapping["PERMIT_TRANSFER"] == "coil:27" and mb.options["work_state_register"] == 106
    # the old `workstate_*` options (they named the station config) are gone
    assert not any(k.startswith("workstate") for spec in doc["inputs"] + doc["outputs"]
                   for k in spec.get("options", {}))
    assert C.is_signal("PERMIT_LOAD_DONE") and C.is_signal("KICKOUT")
    for bad in ("PERMIT_", "PERMIT_load", "permit_X", "BOGUS", None):
        assert not C.is_signal(bad), bad
    with pytest.raises(ValueError, match="unknown signals"):
        C.load_io_doc({"outputs": [{"kind": "simulated", "map": {"PERMIT_": "x"}}]})
    with pytest.raises(ValueError, match="signals must map"):
        C.load_io_doc({"inputs": [{"kind": "simulated", "map": {}, "signals": {"DI4": 7}}]})
    C.load_io_doc({"inputs": [{"kind": "simulated", "map": {"x": "SELECT_WORKSTATE"}}]})


# ================================================================== the DLP check (synthetic scans)

@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    """A mock library with the bench's TRUE stereo calibration, a golden per profile and
    synthetic scans of the tube scenes (recipe_synth: the same picture as a mock capture)."""
    root = tmp_path_factory.mktemp("flow") / "library"
    s = api.configure(root, mock=True)
    try:
        p = api.load_preset(PRESET)
        RS.truth_calibration(s, p)
        scans = {}
        for i, (prof, variants) in enumerate((("long", ("ok", "absent", "misaligned", "dent")),
                                              ("short", ("ok", "dent")))):
            g = RS.oracle_scan(s, p, f"scene:tube_{prof}", f"tube_{prof}_alignment_golden",
                               seed=9 + i)
            runner.record_golden(s, api.recipe_load(f"tube_{prof}_alignment"), p, g)
            for j, v in enumerate(variants):
                scans[f"{prof}:{v}"] = RS.oracle_scan(s, p, f"scene:tube_{prof}:{v}",
                                                      f"t_{prof}_{v}", seed=3 + j).id
    finally:
        api._session = None
    return root, scans


@pytest.fixture
def eng(lib):
    s = api.configure(lib[0], mock=True)
    yield s, lib[1]
    api._session = None


def _all(scan_id: str) -> dict:
    return {k: scan_id for k in ("loaded", "aligned", "end_face")}


def test_check_config_finds_what_the_schema_cannot(eng):
    assert FV.check_config(FS.load(EXAMPLE)) == []
    assert api.flow_check(EXAMPLE)["valid"] and api.flow_check(EXAMPLE)["problems"] == []
    assert FV.check_config(FS.load(ROOT / "stations" / LEGACY[0])) == []
    # a recipe that does not exist, in one profile
    d = _doc()
    d["profiles"]["short"]["recipes"]["presence"] = "no_such_recipe"
    probs = FV.check_config(FS.validate(d))
    assert len(probs) == 1 and "work state loaded, profile short" in probs[0]
    assert "no_such_recipe" in probs[0]
    # a signal no check observes: refused by the schema - nothing is guessed, not even when
    # the recipe has exactly one presence check
    d = _doc()
    del d["flow"][1]["verify"]["observe"]
    with pytest.raises(FS.StationConfigError, match="expects 'part_present' but no check observes"):
        FS.validate(d)
    d["flow"][1]["verify"]["observe"] = {"part_present": "seat/position"}
    assert FV.check_config(FS.validate(d)) == []            # named: the check passing = true
    d["flow"][1]["verify"]["observe"] = {"part_present": "seat/nope"}
    assert all("does not have" in p for p in FV.check_config(FS.validate(d)))
    # a work state that can never be confirmed: it expects ABSENT, its check passes on PRESENT
    d = _doc()
    d["flow"][0]["expects"] = {"part_present": False}
    probs = FV.check_config(FS.validate(d))
    assert len(probs) == 2 and all("can never be CONFIRMED" in p for p in probs)
    d["flow"][0]["verify"]["observe"] = {"part_present": {"zone": "seat", "check": "presence",
                                                          "invert": True}}
    assert FV.check_config(FS.validate(d)) == []
    bad = api.flow_check({"not": "a path"}) if False else api.flow_check(ROOT / "nope.json")
    assert not bad["valid"] and "not found" in bad["problems"][0] and bad["config"] is None


def test_the_dlp_says_what_it_saw_per_expected_signal(eng):
    s, scans = eng
    cfg = FS.load(EXAMPLE)
    runs = Path(s.store.root).parent / "verify_runs"
    out = api.flow_verify(cfg, "long_tube", "loaded", runs_dir=runs, scan=scans["long:ok"])
    d = out["dlp"]
    assert (d["result"], d["disposition"]) == ("PASS", "CONTINUE") and d["observed"] == {
        "part_present": True}
    assert d["observe"] == {"part_present": "seat/presence"} and d["recipe"] == "tube_long_presence"
    assert d["preset"] == PRESET and d["feedback_rendered"] and out["exit_code"] == 0
    res = out["result"]
    assert res["station"] == "example_tube_line" and res["station_config"] == EXAMPLE.name
    assert res["profile"] == "long_tube" and res["job"]["work_state"] == "loaded"
    assert Path(out["run_dir"]).parent == runs and (Path(out["run_dir"]) / "result.json").is_file()
    assert not (Path(out["run_dir"]) / "feedback_photo.png").exists()   # nothing projected here
    # an empty seat: the presence check FAILs, and what it SAW is "absent"
    d = api.flow_verify(cfg, None, "loaded", runs_dir=runs, scan=scans["long:absent"])["dlp"]
    assert (d["result"], d["disposition"]) == ("FAIL", "REJECT") and d["observed"] == {
        "part_present": False}
    assert d["message"].startswith("seat/presence: part absent")
    # the other profile's tube: there (the sensor is right), but not this profile's
    d = api.flow_verify(cfg, "long_tube", "loaded", runs_dir=runs, scan=scans["short:ok"])["dlp"]
    assert d["observed"] == {"part_present": True} and d["result"] == "FAIL"
    assert "profile_height" in d["message"]
    # the same scan under ITS profile is fine: the profile says what the part looks like
    d = api.flow_verify(cfg, "short_tube", "loaded", runs_dir=runs, scan=scans["short:ok"])["dlp"]
    assert (d["result"], d["observed"]) == ("PASS", {"part_present": True})
    assert d["recipe"] == "tube_short_presence"
    d = api.flow_verify(cfg, "long_tube", "aligned", runs_dir=runs,
                        scan=scans["long:misaligned"])["dlp"]
    assert d["result"] == "FAIL" and "offset" in d["message"]
    assert d["observed"] == {"part_present": True} and d["step"] == "aligned"
    d = api.flow_verify(cfg, "long_tube", "end_face", runs_dir=runs, scan=scans["long:dent"])["dlp"]
    assert d["result"] == "FAIL" and "defect" in d["message"]
    # an empty seat is never a clean end face, and an unseen part is never "absent"
    d = api.flow_verify(cfg, "long_tube", "end_face", runs_dir=runs, scan=scans["long:absent"])["dlp"]
    assert d["result"] == "FAIL"
    # ... and at every work state the DLP says whether the tube is there: a tube that went
    # missing at "Tube aligned?" / "End face OK?" is seen as missing, not as a bad part
    assert d["observed"] == {"part_present": False} and d["observe"] == {
        "part_present": "seat/presence"}
    d = api.flow_verify(cfg, "long_tube", "aligned", runs_dir=runs, scan=scans["long:absent"])["dlp"]
    assert d["result"] == "FAIL" and d["observed"] == {"part_present": False}
    assert d["message"].startswith("seat/presence: part absent")
    # the other profile's tube at "Tube aligned?": there, but its position can't be measured
    d = api.flow_verify(cfg, "long_tube", "aligned", runs_dir=runs, scan=scans["short:ok"])["dlp"]
    assert (d["result"], d["disposition"]) == ("INVALID", "PULL")
    assert d["observed"] == {"part_present": True}
    with pytest.raises(FS.StationConfigError, match="no work state"):
        api.flow_verify(cfg, "long_tube", "nope", runs_dir=runs, scan=scans["long:ok"])
    with pytest.raises(FS.StationConfigError, match="no profile"):
        api.flow_verify(cfg, "medium_tube", "loaded", runs_dir=runs, scan=scans["long:ok"])
    assert FV._saw({"type": "presence", "verdict": "INVALID"}) is None
    assert FV._saw({"type": "presence", "verdict": "FAIL", "params": {"expect": "absent"}}) is True
    assert FV._saw({"type": "code", "verdict": "FAIL", "pull": "code_unreadable"}) is None


def test_inline_zone_checks_use_the_profiles_zones_and_preset(eng):
    s, scans = eng
    zone = {"shape": "circle", "units": "part_mm", "coords": {"center": [0, 0], "radius": 14.0}}
    d = {"schema_version": 2, "station_id": "inline_line",
         "parameters": {"io": {"inputs": ["part_present"]}},
         "profiles": {"long": {"preset": PRESET, "zones": {"seat": zone},
                               "feedback": {"enabled": True, "hold_s": 1.0}},
                      "short": {"preset": PRESET, "zones": {"seat": zone}}},
         "flow": [{"id": "loaded", "expects": {"part_present": True}, "permit": "LOAD_DONE",
                   "verify": {"checks": [{"zone": "seat", "type": "presence",
                                          "params": {"min_height_mm": 10, "min_area_pct": 35}},
                                         {"zone": "seat", "type": "height", "id": "h",
                                          "params": {"nominal_mm": 62, "tol_mm": 2,
                                                     "min_height_mm": 10}}],
                              "observe": {"part_present": "seat/presence"}}}]}
    cfg = FS.validate(d)
    assert cfg.flow[0].verify_text() == "seat/presence, seat/height" and FV.check_config(cfg) == []
    rec = FV.compose(cfg, "long", cfg.flow[0])
    assert rec["name"] == "inline_line.long.loaded" and rec["preset"] == PRESET
    assert [c["id"] for c in rec["zones"][0]["checks"]] == ["presence", "h"]
    assert rec["feedback"]["hold_s"] == 1.0 and "feedback" not in FV.compose(cfg, "short",
                                                                             cfg.flow[0])
    runs = Path(s.store.root).parent / "inline_runs"
    out = api.flow_run(cfg, "long", inputs={"part_present": True}, runs_dir=runs,
                       scans={"loaded": scans["long:ok"]}, project_faults=False)
    assert out["result"]["disposition"] == "CONTINUE" and out["result"]["permissions"] == {
        "LOAD_DONE": True}
    assert out["result"]["steps"][0]["dlp_result"]["recipe"] == "inline_line.long.loaded"
    out = api.flow_run(cfg, "long", inputs={"part_present": True}, runs_dir=runs,
                       scans={"loaded": scans["long:absent"]}, project_faults=False)
    assert out["result"]["steps"][0]["outcome"] == "CORRECTED" and out["exit_code"] == 5
    # an inline zone a profile does not define, no preset: refused by the schema
    d2 = copy.deepcopy(d)
    d2["profiles"]["short"] = {"zones": {}}
    with pytest.raises(FS.StationConfigError) as e:
        FS.validate(d2)
    assert "name zone(s) ['seat'], which profile short does not define" in str(e.value)
    assert "inline checks need a preset" in str(e.value)
    d3 = copy.deepcopy(d)
    d3["flow"][0]["verify"]["checks"][0]["params"]["min_area_pct"] = 500
    assert any("min_area_pct" in p for p in FV.check_config(FS.validate(d3)))


def test_a_sensor_says_a_part_is_there_when_it_should_not_be(eng):
    """The owner's case: a work state that needs the seat EMPTY. The sensor says a part is
    there. Is it really there, or not? The DLP confirms the sensor (NOT_MET: the unexpected
    part is painted, the operator is prompted to take it off) or corrects it (CORRECTED: the
    sensor is wrong). Only an empty seat that the sensor also calls empty grants LOAD_OK."""
    s, scans = eng
    zone = {"shape": "circle", "units": "part_mm", "coords": {"center": [0, 0], "radius": 14.0}}
    cfg = FS.validate({
        "schema_version": 2, "station_id": "clear_line",
        "parameters": {"preset": PRESET, "io": {"inputs": ["part_present"]}},
        "profiles": {"tube": {"zones": {"seat": zone}}},
        "flow": [{"id": "clear", "title": "Seat clear?", "expects": {"part_present": False},
                  "verify": {"checks": [{"zone": "seat", "type": "presence",
                                         "params": {"expect": "absent", "min_height_mm": 10}}],
                             "observe": {"part_present": "seat/presence"}},
                  "permit": "LOAD_OK",
                  "on_mismatch": [
                      {"prompt": "the sensor says a part is there, the seat is empty: check "
                                 "the sensor", "when_dlp": {"part_present": False}},
                      {"prompt": "a part is on the seat and the sensor missed it: take it off "
                                 "and check the sensor", "when_dlp": {"part_present": True}},
                      "hold"],
                  "on_not_met": ["project_fault", "prompt a part is on the seat: take it off",
                                 "kickout PULL an unexpected part on the seat"]}]})
    assert FV.check_config(cfg) == []
    runs = Path(s.store.root).parent / "clear_runs"

    def cycle(sensor, scan, ack=False):
        out = api.flow_run(cfg, inputs={"part_present": sensor}, runs_dir=runs,
                           scans={"clear": scans[scan]}, acknowledge=ack, project_faults=False)
        return out["result"], out["result"]["steps"][0]

    res, st = cycle(False, "long:absent")                   # empty, and the sensor says so
    assert st["outcome"] == "CONFIRMED" and res["disposition"] == "CONTINUE"
    assert res["permissions"] == {"LOAD_OK": True} and st["dlp_result"]["observed"] == {
        "part_present": False}
    res, st = cycle(True, "long:absent")                    # the sensor is wrong: corrected
    assert st["outcome"] == "CORRECTED" and res["permissions"] == {"LOAD_OK": False}
    assert st["sensor_mismatch"] == [{"signal": "part_present", "sensor": True, "dlp": False,
                                      "expected": False}]
    assert st["actions"][0]["message"].startswith("the sensor says a part is there")
    res, st = cycle(True, "long:ok", ack=True)              # it really is there: take it off
    assert st["outcome"] == "NOT_MET" and st["sensor_mismatch"] == []
    assert st["dlp_result"]["observed"] == {"part_present": True}
    assert st["dlp_result"]["result"] == "FAIL"             # the check found the unexpected part
    assert [a["action"] for a in st["actions"]] == ["project_fault", "prompt", "kickout"]
    assert st["actions"][0]["requested"] is True and st["actions"][1]["acknowledged"] is True
    assert res["disposition"] == "PULL" and res["cycle"]["kickout"]["reason"] == \
        "an unexpected part on the seat"
    assert res["permissions"] == {"LOAD_OK": False}
    res, st = cycle(False, "long:ok")                       # the sensor missed the part
    assert st["outcome"] == "CORRECTED" and st["sensor_mismatch"][0]["dlp"] is True
    assert res["permissions"] == {"LOAD_OK": False}
    # ... and the prompt is the one for THIS direction, not the "seat is empty" one
    assert st["actions"][0]["message"].startswith("a part is on the seat and the sensor missed")


CASES = {
    #  name                   profile       scan               sensor  ack    -> result, disposition, exit, ended_by, outcomes
    "all_good": ("long_tube", "long:ok", True, False,
                 ("PASS", "CONTINUE", 0, "confirmed", ["CONFIRMED", "CONFIRMED", "CONFIRMED"])),
    "short_tube": ("short_tube", "short:ok", True, False,
                   ("PASS", "CONTINUE", 0, "confirmed", ["CONFIRMED", "CONFIRMED", "CONFIRMED"])),
    "sensor_true_part_absent": ("long_tube", "long:absent", True, False,
                                ("INVALID", "PULL", 5, "unacknowledged",
                                 ["CORRECTED", "PENDING", "PENDING"])),
    "sensor_false_part_present": ("long_tube", "long:ok", False, False,
                                  ("INVALID", "PULL", 5, "unacknowledged",
                                   ["CORRECTED", "PENDING", "PENDING"])),
    "no_tube_no_claim": ("long_tube", "long:absent", False, False,
                         ("INVALID", "PULL", 5, "unacknowledged",
                          ["NOT_MET", "PENDING", "PENDING"])),
    "misaligned": ("long_tube", "long:misaligned", True, True,
                   ("FAIL", "PULL", 5, "unconfirmed", ["CONFIRMED", "DEFECT", "PENDING"])),
    "dented": ("short_tube", "short:dent", True, False,
               ("FAIL", "REJECT", 1, "kickout", ["CONFIRMED", "CONFIRMED", "DEFECT"])),
    "wrong_profile": ("long_tube", "short:ok", True, False,
                      ("INVALID", "PULL", 5, "unacknowledged", ["DEFECT", "PENDING", "PENDING"])),
    "sensor_silent": ("long_tube", "long:ok", None, False,
                      ("INVALID", "PULL", 5, "kickout", ["DOUBT", "PENDING", "PENDING"])),
}
STEP_KEYS = {"id", "title", "index", "required", "expects", "sensor_inputs", "dlp_result",
             "outcome", "sensor_mismatch", "permit", "permission_granted", "actions",
             "disposition"}
RESULT_KEYS = {"schema_version", "run_id", "run_folder", "mode", "station", "operator_confirmed",
               "job", "calibration", "capture", "part", "deviation_mm", "deviation_units",
               "threshold_mm", "result", "invalid_reason", "engine_verdict", "disposition",
               "pull_reasons", "warnings", "outputs", "recipe", "zones", "feedback", "software",
               "started", "finished", "station_config", "profile", "steps", "permissions",
               "cycle"}


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_whole_cycle_writes_its_folder_and_result(eng, tmp_path, name):
    s, scans = eng
    profile, scan, sensor, ack, want = CASES[name]
    inputs = {} if sensor is None else {"part_present": sensor}
    out = api.flow_run(EXAMPLE, profile, inputs=inputs, runs_dir=tmp_path / "runs",
                       scans=_all(scans[scan]), acknowledge=ack, project_faults=False)
    cycle_dir = Path(out["cycle_dir"])
    res = json.loads((cycle_dir / "result.json").read_text(encoding="utf-8"))
    assert res == json.loads(json.dumps(out["result"], default=str)) and set(res) == RESULT_KEYS
    result, disp, code, ended, outcomes = want
    assert (res["result"], res["disposition"], out["exit_code"]) == (result, disp, code), res
    assert res["schema_version"] == R.RESULT_SCHEMA_VERSION == "0.4"
    assert res["station"] == "example_tube_line" and res["station_config"] == EXAMPLE.name
    assert res["profile"] == profile and res["job"]["type"] == "flow"
    assert res["job"]["work_states"] == ["loaded", "aligned", "end_face"]
    assert [st["outcome"] for st in res["steps"]] == outcomes
    assert res["cycle"]["ended_by"] == ended and res["cycle"]["outcomes"] == dict(
        zip(("loaded", "aligned", "end_face"), outcomes))
    for st in res["steps"]:
        assert STEP_KEYS <= set(st), sorted(STEP_KEYS - set(st))
        # a permission stands only for a CONFIRMED work state of a cycle that CONTINUEs: the
        # rows never say GRANTED next to `permissions` all false
        assert st["permission_granted"] is (st["outcome"] == "CONFIRMED" and disp == "CONTINUE"), \
            st["id"]
        if st["outcome"] == "CONFIRMED" and disp != "CONTINUE":
            assert st["permission_note"] == f"dropped: the cycle ended {disp}"
        assert res["permissions"][st["permit"]] is st["permission_granted"]
    # CONTINUE only when every required work state was CONFIRMED; else nothing is released
    assert (disp == "CONTINUE") is all(o == "CONFIRMED" for o in outcomes)
    assert all(res["permissions"].values()) is (disp == "CONTINUE")
    assert set(res["permissions"]) == {"LOAD_DONE", "CLAMP", "TRANSFER"}
    assert (res["pull_reasons"] == []) is (disp != "PULL")
    first = res["steps"][0]
    assert first["expects"] == {"part_present": True}
    assert first["sensor_inputs"] == {"part_present": sensor}
    if name.startswith("sensor_") and sensor is not None:
        assert first["sensor_mismatch"] == [{"signal": "part_present", "sensor": sensor,
                                             "dlp": not sensor, "expected": True}]
        assert res["cycle"]["sensor_mismatch"] is True
        assert {"timeout", "sensor_mismatch"} <= {p["reason"] for p in res["pull_reasons"]}
        assert all(p["step"] == "loaded" for p in res["pull_reasons"])
        assert first["actions"][0]["action"] == "project_fault"
        assert first["actions"][0]["requested"] is (sensor is True)     # a missing part: painted
        assert first["actions"][0].get("shown") is (False if sensor else None)  # projection off
    # every cycle that does not CONTINUE says where and why its part leaves the line
    assert (res["cycle"]["kickout"] is None) is (disp == "CONTINUE")
    assert res["cycle"]["unconfirmed"] == [{"step": st["id"], "outcome": st["outcome"],
                                            "required": True}
                                           for st in res["steps"] if st["outcome"] != "CONFIRMED"]
    if name == "dented":
        k = res["cycle"]["kickout"]
        assert (k["step"], k["disposition"], k["reason"], k["by"]) == (
            "end_face", "REJECT", "end face not OK", "kickout")
        assert k["detail"].startswith("end_face/defect")    # what the DLP found, not a guess
        assert res["steps"][2]["actions"][-1]["action"] == "kickout"
        # the fault of a kicked-out part is still handed to the projector (off in this run)
        paint = res["steps"][2]["actions"][0]
        assert paint["action"] == "project_fault" and paint["requested"] is True
        assert paint["shown"] is False and "projection is off" in paint["why"]
    if name == "misaligned":
        assert res["steps"][1]["attempts"] == 3 and res["steps"][1]["actions"][1]["acknowledged"]
    if name == "sensor_silent":
        assert res["pull_reasons"][0]["reason"] == "sensor_unknown"
    # the folder: the cycle's own result + log, one run folder per DLP check under steps/
    verified = sum(st["attempts"] for st in res["steps"])
    step_dirs = sorted(p.name for p in (cycle_dir / "steps").iterdir())
    assert len(step_dirs) == verified and all("example_tube_line." in n for n in step_dirs)
    for st in res["steps"]:
        d = st["dlp_result"]
        if d is not None:
            sub = json.loads((Path(d["run_folder"]) / "result.json").read_text(encoding="utf-8"))
            assert sub["run_id"] == d["run_id"] and sub["job"]["work_state"] == st["id"]
            assert sub["profile"] == profile and sub["steps"][0]["id"] == "run"
    ev = [json.loads(x) for x in (cycle_dir / "run.log.jsonl").read_text().splitlines()]
    names = [e["event"] for e in ev]
    assert names[0] == "run_start" and names[-1] == "run_end" and "cycle_end" in names
    assert all(e["run_id"] == res["run_id"] for e in ev)
    assert names.count("work_state") == verified
    assert ev[-1]["disposition"] == disp and ev[-1]["exit_code"] == code
    assert names.count("permission") >= outcomes.count("CONFIRMED")


def test_a_cycle_that_crashes_or_is_refused_still_fails_closed(eng, tmp_path, monkeypatch):
    s, scans = eng
    with pytest.raises(FS.StationConfigError, match="no profile"):
        api.flow_run(EXAMPLE, "medium", runs_dir=tmp_path / "runs")
    with pytest.raises(FS.StationConfigError, match="single-job"):
        api.flow_run(ROOT / "stations" / LEGACY[0], runs_dir=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()                 # refused before any folder

    def boom(*a, **k):
        raise RuntimeError("camera gone")
    monkeypatch.setattr(FV, "verify_step", boom)
    out = api.flow_run(EXAMPLE, inputs={"part_present": True}, runs_dir=tmp_path / "runs")
    res = out["result"]
    assert (res["result"], res["disposition"], out["exit_code"]) == ("INVALID", "PULL", 5)
    assert res["steps"][0]["outcome"] == "DOUBT" and "camera gone" in res["steps"][0]["message"]
    assert not any(res["permissions"].values())
    assert {p["reason"] for p in res["pull_reasons"]} == {"error"}
    monkeypatch.setattr(FV, "verify_step", lambda *a, **k: (_ for _ in ()).throw(
        rules.Cancelled("aborted by the operator")))
    out = api.flow_run(EXAMPLE, inputs={"part_present": True}, runs_dir=tmp_path / "runs")
    assert (out["result"]["result"], out["result"]["disposition"], out["exit_code"]) == (
        "ABORTED", "PULL", 1)
    assert out["result"]["steps"][0]["outcome"] == "ABORTED"


def test_real_rig_cycle_refused_without_confirmation(tmp_path):
    """A REAL-mode engine refuses an unconfirmed cycle before anything touches the rig."""
    api.configure(tmp_path / "lib", mock=False)
    with pytest.raises(rules.RuleViolation, match="confirm"):
        api.flow_run(EXAMPLE, inputs={"part_present": True}, runs_dir=tmp_path / "runs")
    with pytest.raises(rules.RuleViolation, match="confirm"):
        api.flow_verify(EXAMPLE, None, "loaded", runs_dir=tmp_path / "runs")
    with pytest.raises(ValueError, match="MOCK ONLY"):
        FV.verify_step(FS.load(EXAMPLE), "long_tube", "loaded", runs_dir=tmp_path / "runs",
                       operator_confirmed=True, mock_scene="scene:tube_long")
    assert not (tmp_path / "runs").exists()
    assert "PySpin" not in sys.modules and "hid" not in sys.modules


def test_a_single_job_result_carries_one_implicit_work_state(tmp_path):
    for result, disp, outcome in (("PASS", "CONTINUE", "CONFIRMED"), ("FAIL", "REJECT", "DEFECT"),
                                  ("PASS", "PULL", "DOUBT"), ("INVALID", "PULL", "DOUBT"),
                                  ("ABORTED", "PULL", "ABORTED")):
        run_id, run_dir = R.new_run(tmp_path, "st")
        res = R.empty_result(run_id=run_id, run_folder=str(run_dir), mock=True,
                             operator_confirmed=False, station="st", job={"type": "metric"},
                             part={}, station_config="st.json", profile="default")
        assert res["steps"] == [] and res["permissions"] == {} and res["cycle"] is None
        res.update(result=result, disposition=disp, invalid_reason=None,
                   pull_reasons=[] if disp != "PULL" else [D.reason("invalid", "x")])
        R.write_result(run_dir, res)
        doc = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        (step,) = doc["steps"]
        assert step["id"] == "run" and step["outcome"] == outcome and step["disposition"] == disp
        assert step["permission_granted"] is False and step["permit"] is None
        assert STEP_KEYS <= set(step) and doc["station_config"] == "st.json"
    # a document from before 0.4 written now still gets the keys (never a KeyError downstream)
    run_id, run_dir = R.new_run(tmp_path, "old")
    old = {k: v for k, v in res.items() if k not in ("steps", "permissions", "cycle",
                                                     "station_config", "profile")}
    R.write_result(run_dir, old)
    assert old["steps"][0]["outcome"] == "ABORTED" and old["permissions"] == {}


# ================================================================== command line

def _cli(argv) -> tuple:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = cli.main(argv)
    api._session = None
    return code, buf.getvalue()


def _station(tmp_path, **params) -> Path:
    """The example configuration + its flow in a temp folder (runs under tmp_path)."""
    (tmp_path / "stations").mkdir(exist_ok=True)
    (tmp_path / "flows").mkdir(exist_ok=True)
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    doc["parameters"].update(runs_dir=str(tmp_path / "runs"), mock_library=None, **params)
    f = tmp_path / "stations" / "example_tube_line.json"
    f.write_text(json.dumps(doc), encoding="utf-8")
    shutil.copy(FLOW, tmp_path / "flows" / FLOW.name)
    return f


def test_cli_flow_show_prints_the_configuration_and_lists_problems(tmp_path, lib):
    L = ["--mock", "--library", str(lib[0])]
    code, out = _cli([*L, "flow-show", str(EXAMPLE)])
    assert code == 0, out
    for needle in ("Station configuration example_tube_line", "Profiles (the profile library set)",
                   "long_tube *", "short_tube", "Work states (the flow)", "Tube loaded?",
                   "part_present=true", "PERMIT_LOAD_DONE, PERMIT_CLAMP, PERMIT_TRANSFER",
                   "part_present <- seat/presence",
                   "end_face.on_defect: project_fault -> kickout REJECT end face not OK"):
        assert needle in out, needle
    code, out = _cli([*L, "--json", "flow-show", str(EXAMPLE)])
    doc = json.loads(out)
    assert code == 0 and doc["valid"] and doc["config"]["permissions"] == ["LOAD_DONE", "CLAMP",
                                                                            "TRANSFER"]
    code, out = _cli([*L, "flow-show", str(ROOT / "stations" / LEGACY[1])])
    assert code == 0 and "single-job config: one implicit profile" in out and "default *" in out
    bad = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    bad["profiles"]["short_tube"]["recipes"]["end_face"] = "nope"
    bad["flow"] = str(FLOW)
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    code, out = _cli([*L, "flow-show", str(tmp_path / "bad.json")])
    assert code == cli.EXIT_REFUSED and "work state end_face, profile short_tube" in out
    code, out = _cli([*L, "--json", "flow-show", str(tmp_path / "bad.json")])
    assert code == cli.EXIT_REFUSED and json.loads(out)["valid"] is False


def test_cli_run_refuses_what_does_not_fit_before_any_scan(tmp_path, lib):
    L = ["--mock", "--library", str(lib[0])]
    cfg = str(_station(tmp_path))
    for extra, needle in ((["--profile", "medium"], "has profiles long_tube, short_tube"),
                          (["--input", "part_present=maybe"], "write SIGNAL=true or SIGNAL=false"),
                          (["--input", "door=true"], "not a sensor signal of this configuration"),
                          (["--mock-case", "nope"], "no mock case"),
                          (["--mock-part", "50,40,30"], "use scene:<name>[:<variant>] or none")):
        code, out = _cli([*L, "run", cfg, *extra])
        assert code == cli.EXIT_REFUSED and needle in out, (extra, out)
    assert not (tmp_path / "runs").exists()
    # the flow options make no sense on a single-job config
    code, out = _cli([*L, "run", str(ROOT / "stations" / LEGACY[1]), "--profile", "x"])
    assert code == cli.EXIT_REFUSED and "single-job station config" in out
    # the real rig: a human confirms first (nothing is opened)
    code, out = _cli(["--library", str(tmp_path / "real"), "run", cfg])
    assert code == cli.EXIT_REFUSED and "--operator-confirmed" in out
    code, out = _cli(["--library", str(tmp_path / "real"), "run", cfg, "--mock-case", "dented",
                      "--operator-confirmed"])
    assert code == cli.EXIT_REFUSED and "MOCK ONLY" in out
    assert "PySpin" not in sys.modules


def test_cli_run_walks_a_cycle_through_the_mock_capture_path(tmp_path, lib):
    """`scan --mock run <configuration> --mock-case sensor_true_part_absent`: the simulated
    sensor says a tube is there, the simulated seat is empty. One real mock capture: the DLP
    CORRECTS the sensor, paints the missing-part zone, prompts - and with nobody to
    acknowledge it the unattended cycle ends PULL (exit 5), no permission granted."""
    cfg = _station(tmp_path)
    code, out = _cli(["--mock", "--library", str(lib[0]), "--json", "run", str(cfg),
                      "--mock-case", "sensor_true_part_absent"])
    doc = json.loads(out)
    res = doc["result"]
    assert code == 5 and doc["exit_code"] == 5 and doc["disposition"] == "PULL", out[-600:]
    assert res["mode"] == "MOCK" and res["profile"] == "long_tube"
    first = res["steps"][0]
    assert first["outcome"] == "CORRECTED" and first["sensor_inputs"] == {"part_present": True}
    assert first["dlp_result"]["observed"] == {"part_present": False}
    assert first["dlp_result"]["scan_id"] and first["dlp_result"]["result"] == "FAIL"
    assert [a["action"] for a in first["actions"]] == ["project_fault", "prompt", "unacknowledged"]
    assert first["actions"][0]["requested"] is True and first["actions"][0]["shown"] is True
    assert not any(res["permissions"].values()) and res["cycle"]["sensor_mismatch"]
    (cycle_dir,) = [d for d in (tmp_path / "runs").iterdir() if d.is_dir()]
    assert json.loads((cycle_dir / "result.json").read_text(encoding="utf-8")) == res
    (step_dir,) = list((cycle_dir / "steps").iterdir())
    assert (step_dir / "feedback_dmd.png").is_file() and (step_dir / "overlay.png").is_file()
    events = [json.loads(x)["event"] for x in (cycle_dir / "run.log.jsonl").read_text().splitlines()]
    assert "prompt" in events and events[-1] == "run_end"


# ================================================================== docs

def test_docs_keep_station_configuration_profile_and_work_state_apart():
    """The terminology the whole feature rests on: a station configuration (parameters + the
    profile library set) is not a work state; the work states are the steps and permissions."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    porting = (ROOT / "docs" / "PORTING.md").read_text(encoding="utf-8")
    stations = (ROOT / "stations" / "README.md").read_text(encoding="utf-8")
    flows = (ROOT / "flows" / "README.md").read_text(encoding="utf-8")
    recipes = (ROOT / "recipes" / "README.md").read_text(encoding="utf-8")
    assert "## Terminology: station configuration, profiles, work states" in readme
    assert "## Terminology" in porting and stations.startswith("# Station configurations")
    for doc, name in ((readme, "README"), (porting, "PORTING"), (stations, "stations/README")):
        low = " ".join(doc.split()).lower()
        for needle in ("profile library set", "work state", "not** a work state",
                       "confirms or corrects", "permit_<name>", "sensor_mismatch",
                       "/api/io/inputs", "select_profile"):
            assert needle in low, (name, needle)
        for outcome in ("CONFIRMED", "CORRECTED", "NOT_MET", "DEFECT", "DOUBT"):
            assert outcome in doc, (name, outcome)
    # how a PLC wires its sensors in and reads the permissions
    assert "How a PLC wires its sensors in and reads the permissions" in porting
    for needle in ("`station_config`", "`work_state_code`", "Nothing grants one on a doubt"):
        assert needle in porting, needle
    # the deprecated name is marked as such wherever it still appears
    for doc in (readme, porting):
        for line_ in doc.splitlines():
            if "SELECT_WORKSTATE" in line_ or "/api/cmd/workstate" in line_:
                assert "deprecated" in line_.lower(), line_[:120]
    for doc in (readme, porting, stations, flows, recipes):
        assert "work state IS" not in doc and "Work state (station config)" not in doc
        assert "`workstate` (1-based)" not in doc
    assert "example_tube_line.json" in readme and "flow-show" in stations
    assert "A recipe is not a work state and not a profile" in recipes
    assert "not the station configuration" in flows
    # nothing here claims the flow ran on the rig
    for doc in (readme, stations):
        assert "not yet run on the real rig" in doc or "not yet run on the rig" in doc
    # the example files validate and say what they are
    assert "NOT a work state" in json.loads(EXAMPLE.read_text(encoding="utf-8"))["_comment"]
    assert FS.load(EXAMPLE).flow_source == FLOW.name


def test_a_flow_edited_on_disk_drops_the_last_cycles_permissions_and_a_stray_answer_confirms_nothing():
    bus, calls, out = _flow_bus()
    bus.set_inputs({"part_present": True})
    bus.command("START")
    for answer in (OK, GOOD, GOOD):
        bus.command("STEP")
        _answer(bus, answer)
    assert all(bus.permissions().values())
    # the same files re-read (every page load does it): nothing changes for the line
    bus.update_configs(["line.json", "old.json"], flows={"line.json": _cfg()})
    assert all(bus.permissions().values()) and out.current["PERMIT_TRANSFER"]
    # the flow changed on disk: what the old version granted no longer holds
    steps = _doc()["flow"]
    steps[2]["title"] = "End face and rim OK?"
    bus.update_configs(["line.json", "old.json"], flows={"line.json": _cfg(steps)})
    assert not any(bus.permissions().values()) and not out.current["PERMIT_TRANSFER"]
    assert bus.snapshot()["flow"][2]["title"] == "End face and rim OK?"
    assert not out.current["PASS"] and bus.cycle is None    # ... nor does its PASS
    # a configuration that broke: dropped too, and START is refused
    bus.set_inputs({"part_present": True})
    bus.command("START")
    for answer in (OK, GOOD, GOOD):
        bus.command("STEP")
        _answer(bus, answer)
    assert all(out.current[f"PERMIT_{p}"] for p in ("LOAD_DONE", "CLAMP", "TRANSFER"))
    bus.update_configs(["line.json", "old.json"], unusable={"line.json": "bad JSON"})
    # published as OFF, by name: not merely missing from what the backend is handed
    assert [out.current[f"PERMIT_{p}"] for p in ("LOAD_DONE", "CLAMP", "TRANSFER")] == [False] * 3
    assert not bus.command("START").accepted
    # an answer the cycle is not waiting for never confirms anything (and never raises: the
    # owner's worker thread reports job ends through job_finished)
    bus, calls, out = _flow_bus()
    bus.set_inputs({"part_present": True})
    bus.command("START")
    bus.command("STEP")
    bus.cycle.state = "WAIT_STEP"                           # a cycle out of step with its job
    bus.job_finished("run", "PASS", None, disposition="CONTINUE", step=OK)
    assert bus.state == "READY" and bus.last_disposition == "PULL"
    assert not any(bus.permissions().values()) and "unexpected DLP answer" in bus.cycle.rows[0]["message"]
    # CONTINUE with recorded doubts is not a doubt-free PASS
    s = _step(expects={"part_present": True})
    c = classify(s, {"part_present": True}, {**OK, "pull_reasons": [D.reason("guard_band", "x")]})
    assert c["outcome"] == "DOUBT"


def test_a_reading_in_doubt_is_no_reading_and_each_wait_has_its_own_timeout():
    step = _step(expects={"part_present": True})
    zones = [{"id": "seat", "checks": [{"id": "presence", "type": "presence", "verdict": "PASS",
                                        "details": {"present": True}, "message": "part present"}]}]
    assert FV.observe(step, zones) == ({"part_present": True}, {"part_present": "seat/presence"})
    # the same PASS, but the run holds it in doubt (36 % against a 35 % limit: the guard band)
    doubt = [D.reason("guard_band", "part_pct 36 is within 10 % of its limit", "seat", "presence")]
    seen, notes = FV.observe(step, zones, doubt)
    assert seen == {"part_present": None} and "in doubt (guard_band" in notes["part_present"]
    answer = FV.dlp_answer({"result": "PASS", "disposition": "PULL", "zones": zones,
                            "pull_reasons": doubt}, step)
    assert answer["observed"] == {"part_present": None}
    # ... so a sensor that disagrees with a marginal reading is a DOUBT, never a "corrected"
    assert classify(step, {"part_present": False}, answer)["outcome"] == "DOUBT"
    assert FV.observe(step, zones, [D.reason("guard_band", "x", "other", "presence")])[0] == {
        "part_present": True}
    # timeout_s counts each wait from its start: the scan time is not held against the prompt
    steps = _doc()["flow"]
    steps[0]["timeout_s"] = 30
    now = [0.0]
    cyc = Cycle(_cfg(steps), "long", clock=lambda: now[0])
    cyc.begin()
    now[0] = 20.0
    cyc.step_now()
    cyc.request({"part_present": True})
    now[0] = 45.0                                           # a 25 s scan
    assert not cyc.due() and cyc.tick() == []               # verifying: no wait is running
    cyc.verified(ABSENT)
    assert cyc.state == "WAIT_ACK" and not cyc.due()
    now[0] = 74.0
    assert cyc.tick() == []
    now[0] = 76.0
    assert cyc.tick() == [("done", None)] and cyc.ended_by == "timeout"
    assert cyc.rows[0]["outcome"] == "CORRECTED" and "waiting for ACK" in cyc.rows[0]["actions"][-1]["message"]
    assert {r["reason"] for r in cyc.pull_reasons} == {"timeout", "sensor_mismatch"}


# ================================================================== fail closed: the open questions

def _confirmed_bus(cfg=None, clock=None):
    """A bus whose cycle just ended CONTINUE: PASS on, every permission granted."""
    bus, calls, out = _flow_bus(cfg, clock=clock)
    bus.set_inputs({"part_present": True})
    bus.command("START")
    for answer in (OK, GOOD, GOOD):
        bus.command("STEP")
        _answer(bus, answer)
    assert out.current["PASS"] and all(bus.permissions().values())
    assert out.registers["work_state"] == 3 and out.registers["work_state_code"] == 1
    return bus, calls, out


def _nothing_released(bus, out) -> None:
    """Nothing of the last part is left for the next one to be mistaken for."""
    assert not any(v for k, v in out.current.items() if k.startswith("PERMIT_")), out.current
    assert not out.current["PASS"] and not any(bus.permissions().values())
    assert (out.registers["result_code"], out.registers["work_state"],
            out.registers["work_state_code"]) == (0, 0, 0)
    assert bus.cycle is None and bus.snapshot()["cycle"] is None


def test_a_start_or_trigger_that_starts_nothing_never_leaves_the_last_parts_permits_on():
    """The next part is announced and NOT checked: the last cycle's PASS, its PERMIT_* bits
    and its work-state registers never stand in for it (a PLC reading them after a refused
    trigger would release a part nobody scanned)."""
    # TRIGGER while not armed (ignored)
    bus, calls, out = _confirmed_bus()
    r = bus.command("TRIGGER", "plc")
    assert not r.accepted and "ignored: not armed" in r.message
    _nothing_released(bus, out)
    assert [h[2] for h in bus.history[-3:]] == ["PERMISSIONS", "RELEASED", "TRIGGER"]
    # START without the operator's confirmation (the real rig)
    bus, calls, out = _confirmed_bus()
    bus.require_confirmation = True
    assert not bus.command("START").accepted
    _nothing_released(bus, out)
    # START / TRIGGER naming another configuration or profile than the selected one
    for arg in ({"config": "old.json"}, {"profile": "short"}, {"config": 1, "profile": 2}):
        bus, calls, out = _confirmed_bus()
        r = bus.command("START", arg=arg)
        assert not r.accepted and "the selected" in r.message, arg
        _nothing_released(bus, out)
    # ARM (waiting for the next part), DISARM, and DISARM + TRIGGER on an armed line
    bus, calls, out = _confirmed_bus()
    assert bus.command("ARM").accepted and bus.state == "ARMED"
    _nothing_released(bus, out)
    bus, calls, out = _flow_bus()
    bus.set_inputs({"part_present": True})
    bus.command("ARM")
    bus.command("TRIGGER")
    _answer(bus, OK)
    for answer in (GOOD, GOOD):
        bus.command("TRIGGER")
        _answer(bus, answer)
    assert bus.state == "ARMED" and out.current["PASS"] and all(bus.permissions().values())
    assert bus.command("DISARM").accepted
    _nothing_released(bus, out)
    assert not bus.command("TRIGGER").accepted
    _nothing_released(bus, out)
    # a FAIL / PULL releases nothing and stays (the part still goes to its lane) ...
    bus, calls, out = _flow_bus()
    bus.set_inputs({"part_present": True})
    bus.command("START")
    bus.command("STEP")
    _answer(bus, dlp("INVALID", "PULL", part_present=None))
    assert out.current["PULL"] and out.current["KICKOUT"]
    assert not bus.command("TRIGGER").accepted
    assert out.current["PULL"] and out.current["KICKOUT"] and not out.current["PASS"]
    assert not any(v for k, v in out.current.items() if k.startswith("PERMIT_"))
    # ... and a command refused INSIDE an open cycle leaves that cycle alone
    bus, calls, out = _flow_bus()
    bus.set_inputs({"part_present": True})
    bus.command("START")
    bus.command("STEP")
    _answer(bus, OK)
    assert out.current["PERMIT_LOAD_DONE"]
    for cmd in ("START", "ARM", "HOME"):
        assert not bus.command(cmd).accepted
    assert out.current["PERMIT_LOAD_DONE"] and bus.cycle.active
    bus.command("STEP")
    assert not bus.command("TRIGGER").accepted and not bus.command("START").accepted   # RUNNING
    _answer(bus, GOOD)
    assert out.current["PERMIT_LOAD_DONE"] and out.current["PERMIT_CLAMP"]


def test_a_profile_or_configuration_change_forgets_the_last_cycle_altogether():
    for cmd, arg in (("SELECT_PROFILE", "short"), ("SELECT_CONFIG", "old.json")):
        bus, calls, out = _confirmed_bus()
        assert bus.command(cmd, arg=arg).accepted
        assert not out.current["PASS"] and bus.last_result is None and bus.cycle is None
        assert out.registers["result_code"] == 0 and out.registers["work_state"] == 0
        assert [out.current[f"PERMIT_{p}"] for p in ("LOAD_DONE", "CLAMP", "TRANSFER")] == [False] * 3
        assert bus.inputs == {}
    # a FAIL / PULL is forgotten the same way: nothing of the other profile's cycle is shown
    bus, calls, out = _flow_bus()
    bus.set_inputs({"part_present": True})
    bus.command("START")
    for answer in (OK, GOOD, FAILS):
        bus.command("STEP")
        _answer(bus, answer)
    assert out.current["FAIL"] and out.current["KICKOUT"] and out.registers["result_code"] == 2
    assert bus.command("SELECT_PROFILE", arg="short").accepted
    assert not any(out.current[k] for k in ("PASS", "FAIL", "PULL", "KICKOUT", "ALARM",
                                            "SENSOR_MISMATCH"))
    assert out.registers["result_code"] == 0 and bus.last_disposition is None
    # the same profile again changes nothing
    bus, calls, out = _confirmed_bus()
    assert bus.command("SELECT_PROFILE", arg="long").accepted and out.current["PASS"]


def test_a_renamed_or_removed_permission_is_published_false_not_left_out():
    """A backend that writes one coil per signal it is handed: a permission that is no
    longer in the selected flow must be handed over as False, not dropped from the dict."""
    coils = {}

    class Coils(C.OutputBackend):
        def publish(self, signals):
            coils.update(signals)

    bus, calls, out = _confirmed_bus()
    bus.outputs.append(Coils())
    bus.command("ACK")                                      # (any publish reaches the coils)
    bus._publish()
    assert coils["PERMIT_TRANSFER"] is True
    steps = _doc()["flow"]
    steps[2]["permit"] = "RELEASE"                          # renamed on disk
    bus.update_configs(["line.json", "old.json"], flows={"line.json": _cfg(steps)})
    assert coils["PERMIT_TRANSFER"] is False and coils["PERMIT_RELEASE"] is False
    assert bus.permissions() == {"LOAD_DONE": False, "CLAMP": False, "RELEASE": False}
    assert bus.snapshot()["signals"]["PERMIT_TRANSFER"] is False
    # the old name never comes back on, whatever the new flow confirms
    bus.set_inputs({"part_present": True})
    bus.command("START")
    for answer in (OK, GOOD, GOOD):
        bus.command("STEP")
        _answer(bus, answer)
    assert coils["PERMIT_RELEASE"] is True and coils["PERMIT_TRANSFER"] is False


def test_a_flow_needs_a_required_work_state_and_never_passes_over_an_open_question():
    # the schema: no required work state, a goto out of a doubt
    steps = _doc()["flow"]
    for s in steps:
        s["required"] = False
    with pytest.raises(FS.StationConfigError, match="no work state is required"):
        _cfg(steps)
    steps = _doc()["flow"]
    steps[0]["on_doubt"] = ["goto aligned"]
    with pytest.raises(FS.StationConfigError, match="goto on a doubt"):
        _cfg(steps)
    # an OPTIONAL work state in DOUBT: on_doubt [] or a prompt alone used to let the flow go
    # on to PASS / CONTINUE. A doubt is verified again or the cycle ends PULL.
    for on_doubt in ([], ["prompt look at the tube"]):
        steps = _doc()["flow"]
        steps[1].update(required=False, on_doubt=on_doubt)
        cyc = _walk(Cycle(_cfg(steps), "long"), [OK, dlp("INVALID", "PULL", part_present=None),
                                                 GOOD])
        assert cyc.rows[1]["outcome"] == "DOUBT" and cyc.rows[2]["outcome"] == "PENDING"
        assert (cyc.result, cyc.disposition, cyc.ended_by) == ("INVALID", "PULL", "unconfirmed")
        assert not any(cyc.permissions.values()) and cyc.pull_reasons
        assert all(r["step"] == "aligned" for r in cyc.pull_reasons)
    # ... and so is an optional work state the DLP CORRECTED (sensor and DLP still disagree)
    steps = _doc()["flow"]
    steps[0].update(required=False, on_mismatch=[])
    cyc = _walk(Cycle(_cfg(steps), "long"), [ABSENT, GOOD, GOOD])
    assert cyc.rows[0]["outcome"] == "CORRECTED" and cyc.disposition == "PULL"
    assert cyc.pull_reasons[0]["reason"] == "sensor_mismatch" and cyc.rows[1]["outcome"] == "PENDING"
    # a definite finding the sensor agrees with, on an optional work state, lets the flow go
    # on: the cycle may CONTINUE, its own permission stays withheld, and the record says so
    steps = _doc()["flow"]
    steps[1].update(required=False, on_defect=[])
    cyc = _walk(Cycle(_cfg(steps), "long"), [OK, FAILS, GOOD])
    assert cyc.disposition == "CONTINUE" and cyc.permissions == {
        "LOAD_DONE": True, "CLAMP": False, "TRANSFER": True}
    assert cyc.unconfirmed == [{"step": "aligned", "outcome": "DEFECT", "required": False}]
    assert cyc.record()["unconfirmed"] == cyc.unconfirmed and cyc.kickout is None
    # SKIP never skips an open question
    steps = _doc()["flow"]
    steps[1].update(required=False, allow_skip=True, on_doubt=["prompt look", "hold"])
    cyc = _walk(Cycle(_cfg(steps), "long"), [OK, dlp("INVALID", "PULL", part_present=None)],
                ack=False)
    assert cyc.state == "WAIT_ACK" and cyc.rows[1]["outcome"] == "DOUBT"
    with pytest.raises(CycleError, match="never skipped"):
        cyc.skip("operator")
    # a goto out of a doubt (a flow built without the validator) is refused by the cycle too
    cfg = _cfg()
    cfg.flow[0].on_doubt = [{"action": "goto", "step": "aligned"}]
    cyc = _walk(Cycle(cfg, "long"), [dlp("INVALID", "PULL", part_present=None), GOOD, GOOD])
    assert cyc.disposition == "PULL" and cyc.rows[1]["outcome"] == "PENDING"
    assert cyc.rows[0]["actions"][-1]["refused"] is True and not any(cyc.permissions.values())
    # a cycle with nothing CONFIRMED is never a CONTINUE (flows built without the validator)
    cfg = _cfg()
    for s in cfg.flow:
        s.required, s.on_defect = False, []
    cyc = _walk(Cycle(cfg, "long"), [FAILS, FAILS, FAILS])
    assert (cyc.result, cyc.disposition) == ("INVALID", "PULL")
    assert "no work state CONFIRMED" in cyc.pull_reasons[0]["message"]


def test_no_permission_while_a_required_work_state_before_it_is_unconfirmed():
    """A flow that jumps ahead (goto / on_confirm) grants nothing on the way: PERMIT_CLAMP is
    never on while the required 'loaded' is not CONFIRMED."""
    steps = _doc()["flow"]
    steps[0]["on_mismatch"] = ["goto aligned"]
    cyc = Cycle(_cfg(steps), "long")
    cyc.begin()
    cyc.step_now()
    cyc.request({"part_present": True})
    assert cyc.verified(ABSENT) == [("wait", "step")] and cyc.index == 1    # jumped ahead
    cyc.step_now()
    cyc.request({"part_present": True})
    cyc.verified(GOOD)
    row = cyc.rows[1]
    assert row["outcome"] == "CONFIRMED" and row["permission_granted"] is False
    assert "loaded before it are not confirmed" in row["permission_note"]
    assert not any(cyc.permissions.values())
    cyc.step_now()
    cyc.request({"part_present": True})
    assert cyc.verified(GOOD) == [("done", None)]
    assert cyc.disposition == "PULL" and not any(cyc.permissions.values())
    # the cycle points at the work state that stopped it, and names it in the reasons
    assert cyc.row["id"] == "loaded" and cyc.kickout["step"] == "loaded"
    assert cyc.pull_reasons[0]["reason"] == "sensor_mismatch"
    assert cyc.pull_reasons[0]["step"] == "loaded"
    ev = [e for e in cyc.drain_events() if e["event"] == "permission"]
    assert ev and not any(e["granted"] for e in ev)
    # on the bus: the register never reads CONFIRMED for a cycle that ended PULL this way
    bus, calls, out = _flow_bus(_cfg(steps))
    seen = []
    bus.outputs.append(type("Spy", (C.OutputBackend,), {
        "publish": lambda self, sig: seen.append(dict(sig)),
        "publish_registers": lambda self, regs: None})())
    bus.set_inputs({"part_present": True})
    bus.command("START")
    for answer in (ABSENT, GOOD, GOOD):
        bus.command("STEP")
        _answer(bus, answer)
    assert bus.last_disposition == "PULL" and out.registers["work_state"] == 1
    assert out.registers["work_state_code"] == OUTCOME_CODES["CORRECTED"]
    assert not any(v for s in seen for k, v in s.items() if k.startswith("PERMIT_") or k == "PASS")


def test_sensor_claims_belong_to_one_cycle_and_a_silent_gateway_is_a_doubt():
    """One POST is not reused for ever: every claim is forgotten when its cycle ends (and on
    ABORT, HOME, FAULT, a profile change), so a dead gateway or sensor is a DOUBT at the
    next part - never "sensor and DLP agree"."""
    bus, calls, out = _confirmed_bus()
    assert bus.inputs == {} and bus.history[-1][2] == "INPUTS" and "forgotten" in bus.history[-1][4]
    bus.command("START")                                    # the gateway says nothing more
    bus.command("STEP")
    _answer(bus, OK)
    assert bus.last_disposition == "PULL" and bus.cycle.rows[0]["outcome"] == "DOUBT"
    assert bus.cycle.pull_reasons[0]["reason"] == "sensor_unknown"
    assert "has not reported for this cycle" in bus.cycle.rows[0]["message"]
    assert not any(v for k, v in out.current.items() if k.startswith("PERMIT_"))
    # reported again for the next part (also BEFORE the START / TRIGGER): confirmed
    bus.set_inputs({"part_present": True}, "plc")
    bus.command("ARM")
    bus.command("TRIGGER")
    _answer(bus, OK)
    assert out.current["PERMIT_LOAD_DONE"] and bus.cycle.rows[0]["sensor_inputs"] == {
        "part_present": True}
    # forgotten on ABORT, HOME, FAULT + RESET and a profile change
    for how in ("abort", "home", "fault", "profile"):
        bus, calls, out = _flow_bus()
        bus.set_inputs({"part_present": True})
        if how == "abort":
            bus.command("ABORT")
        elif how == "home":
            bus.command("HOME")
            bus.job_finished("home", "OK")
        elif how == "fault":
            bus.command("HOME")
            bus.job_finished("home", "MOVED")
            assert bus.command("RESET").accepted
        else:
            bus.command("SELECT_PROFILE", arg="short")
        assert bus.inputs == {} and bus.snapshot()["inputs"] == {}, how
        bus.command("START")
        bus.command("STEP")
        _answer(bus, OK)
        assert bus.last_disposition == "PULL", how
        assert bus.cycle.pull_reasons[0]["reason"] == "sensor_unknown", how
    # a claim that goes true -> false -> true while the DLP scans is a change, not agreement
    bus, calls, out = _flow_bus()
    bus.set_inputs({"part_present": True})
    bus.command("START")
    bus.command("STEP")
    bus.set_inputs({"part_present": False})
    bus.set_inputs({"part_present": True})
    _answer(bus, OK)
    assert bus.last_disposition == "PULL"
    assert "changed during the scan (true -> ... -> true)" in bus.cycle.rows[0]["message"]
    # ... while the same value reported again is not a change
    bus, calls, out = _flow_bus()
    bus.set_inputs({"part_present": True})
    bus.command("START")
    bus.command("STEP")
    bus.set_inputs({"part_present": True}, "plc")
    _answer(bus, OK)
    assert out.current["PERMIT_LOAD_DONE"]
    # parameters.io.max_age_s: a claim older than that when the scan starts is no claim
    d = _doc()
    d["parameters"]["io"]["max_age_s"] = 5
    now = [100.0]
    cfg = FS.validate(d)
    assert cfg.input_max_age_s == 5.0 and cfg.public()["input_max_age_s"] == 5.0
    bus, calls, out = _flow_bus(cfg, clock=lambda: now[0])
    bus.set_inputs({"part_present": True})
    bus.command("START")
    now[0] += 4
    bus.command("STEP")
    assert calls[-1][1]["inputs"] == {"part_present": True}
    _answer(bus, OK)
    assert out.current["PERMIT_LOAD_DONE"] and bus.snapshot()["input_age_s"] == {"part_present": 4.0}
    now[0] += 2                                             # 6 s old at the next work state
    bus.command("STEP")
    assert calls[-1][1]["inputs"] == {"part_present": None}
    _answer(bus, GOOD)
    assert bus.last_disposition == "PULL" and bus.cycle.rows[1]["outcome"] == "DOUBT"
    for bad in (0, -1, "5", True, 1e9):
        d["parameters"]["io"]["max_age_s"] = bad
        with pytest.raises(FS.StationConfigError, match="max_age_s"):
            FS.validate(d)
    # a name no station configuration expects is refused, and nothing of the call is applied
    bus, calls, out = _flow_bus()
    with pytest.raises(ValueError, match="part_presnt"):
        bus.set_inputs({"part_present": True, "part_presnt": True})
    assert bus.inputs == {}
    bare = C.CommandBus()                                   # no configuration declares any
    assert bare.set_inputs({"anything": True}) == {"anything": True}


def test_only_the_answer_to_the_verify_the_bus_asked_for_confirms():
    def started(cfg=None):
        bus, calls, out = _flow_bus(cfg)
        bus.set_inputs({"part_present": True})
        bus.command("START")
        bus.command("STEP")
        return bus, calls, out

    plain = [{"id": "a", "verify": {"recipe": "presence"}, "permit": "A"},
             {"id": "b", "verify": {"recipe": "alignment"}, "permit": "B"}]
    # a bare verdict with no DLP answer confirms nothing, not even a work state with no expects
    bus, calls, out = started(_cfg(plain))
    bus.job_finished("run", "PASS", None, disposition="CONTINUE")
    assert bus.last_disposition == "PULL" and not out.current["PERMIT_A"]
    assert "reported no answer" in bus.cycle.rows[0]["message"]
    # the answer of work state 1 replayed for work state 2
    bus, calls, out = started(_cfg(plain))
    first = _for_this(bus, GOOD)
    bus.job_finished("run", "PASS", None, disposition="CONTINUE", step=first)
    assert out.current["PERMIT_A"]
    bus.command("STEP")
    bus.job_finished("run", "PASS", None, disposition="CONTINUE", step=first)
    assert bus.last_disposition == "PULL" and not out.current["PERMIT_B"]
    assert "is for step 'a'" in bus.cycle.rows[1]["message"]
    # an earlier attempt's answer, another cycle's answer, an answer that names nothing
    for wrong, needle in (({"attempt": 7}, "is for attempt 7"),
                          ({"cycle_id": "cyc_other"}, "is for cycle_id 'cyc_other'"),
                          ({"step": None}, "is for step None")):
        bus, calls, out = started()
        bus.job_finished("run", "PASS", None, disposition="CONTINUE",
                         step={**_for_this(bus, OK), **wrong})
        assert bus.last_disposition == "PULL" and needle in bus.cycle.rows[0]["message"], wrong
        assert not any(v for k, v in out.current.items() if k.startswith("PERMIT_"))
    bus, calls, out = started()
    bus.job_finished("run", "PASS", None, disposition="CONTINUE", step=dict(OK))
    assert bus.last_disposition == "PULL"
    assert "does not say which step" in bus.cycle.rows[0]["message"]
    # pull reasons of any shape are doubts: none is dropped for not being a mapping
    s = _step(expects={"part_present": True})
    for reasons in (["guard band"], "guard band", [None, 7], {"reason": "guard_band"}):
        c = classify(s, {"part_present": True}, {**OK, "pull_reasons": reasons})
        assert c["outcome"] == "DOUBT" and c["reasons"], reasons
    assert classify(s, {"part_present": True}, {**OK, "observed": "yes"})["outcome"] == "DOUBT"


def test_the_record_keeps_every_sensor_mismatch_and_what_an_abort_interrupted():
    # CORRECTED -> put right -> CONFIRMED: the cycle still says a sensor was corrected
    cyc = _walk(Cycle(_cfg(), "long"), [ABSENT, OK, GOOD, GOOD])
    rec = cyc.record()
    assert rec["disposition"] == "CONTINUE" and rec["sensor_mismatch"] is False     # none open
    assert rec["sensor_mismatches"] == [{"step": "loaded", "attempt": 1, "signal": "part_present",
                                         "sensor": True, "dlp": False, "expected": True}]
    doc = FV.cycle_document(_cfg(), rec, cycle_id="c1", cycle_dir="runs/c1", mock=True)
    assert doc["cycle"]["sensor_mismatch"] is True and doc["cycle"]["sensor_mismatch_open"] is False
    assert doc["cycle"]["sensor_mismatches"] == rec["sensor_mismatches"]
    assert doc["cycle"]["unconfirmed"] == [] and doc["cycle"]["kickout"] is None
    # ABORT while the mismatch prompt is open: the finding is kept, not overwritten
    cyc = _walk(Cycle(_cfg(), "long"), [ABSENT], ack=False)
    assert cyc.state == "WAIT_ACK" and cyc.rows[0]["outcome"] == "CORRECTED"
    cyc.abort("aborted: ABORT while the cycle waited")
    row = cyc.rows[0]
    assert row["outcome"] == "ABORTED" and row["sensor_mismatch"] == []
    assert row["actions"] == [{"action": "abort", "message": "aborted: ABORT while the cycle waited"}]
    assert row["earlier"][0]["outcome"] == "CORRECTED"
    assert row["earlier"][0]["sensor_mismatch"][0]["signal"] == "part_present"
    assert [r["reason"] for r in cyc.pull_reasons] == ["aborted", "sensor_mismatch"]
    doc = FV.cycle_document(_cfg(), cyc.record(), cycle_id="c2", cycle_dir="runs/c2", mock=True)
    assert doc["cycle"]["sensor_mismatch"] is True and doc["cycle"]["sensor_mismatch_open"] is False
    assert doc["cycle"]["kickout"]["by"] == "abort" and doc["result"] == "ABORTED"
    assert doc["steps"][0]["permission_granted"] is False
    # KICKOUT on the bus for every cycle that did not CONTINUE, kickout action or not
    for answers, by in (([OK, FAILS, FAILS, FAILS], "unconfirmed"), ([OK, GOOD, FAILS], "kickout")):
        bus, calls, out = _flow_bus()
        bus.set_inputs({"part_present": True})
        bus.command("START")
        bus.command("STEP")
        for a in answers:
            if bus.state == "WAITING" and bus.cycle.waiting == "ack":
                bus.command("ACK")
            elif bus.state == "WAITING":
                bus.command("STEP")
            _answer(bus, a)
        if bus.state == "WAITING":
            bus.command("ACK")
        assert bus.state == "READY" and bus.cycle.kickout["by"] == by, (by, bus.cycle.kickout)
        assert out.current["KICKOUT"] and (out.current["PULL"] or out.current["FAIL"])
    bus, calls, out = _flow_bus()
    bus.command("START")
    bus.command("ABORT")
    assert out.current["KICKOUT"] and out.current["PULL"] and bus.cycle.kickout["by"] == "abort"
    bus.command("START")                                    # the next cycle clears it
    assert not out.current["KICKOUT"]


def test_profiles_named_with_digits_are_selected_by_name():
    d = _doc()
    d["profiles"] = {"60": d["profiles"]["long"], "38": d["profiles"]["short"]}
    bus, calls, out = _flow_bus(FS.validate(d))
    assert bus.profile == "60"
    r = bus.command("SELECT_PROFILE", arg="38")
    assert r.accepted and bus.profile == "38", r
    assert bus.command("START", arg={"config": "line.json", "profile": "38"}).accepted
    bus.command("ABORT")
    assert bus.command("SELECT_PROFILE", arg=1).accepted and bus.profile == "60"    # a number
    assert bus.command("SELECT_PROFILE", arg="2").accepted and bus.profile == "38"  # no such name
    assert not bus.command("SELECT_PROFILE", arg="61").accepted
    d["profiles"] = {"2": d["profiles"]["60"], "1": d["profiles"]["38"]}
    bus, calls, out = _flow_bus(FS.validate(d))
    assert bus.command("SELECT_PROFILE", arg="1").accepted and bus.profile == "1"   # the NAME
    assert bus.command("SELECT_PROFILE", arg=1).accepted and bus.profile == "2"     # position 1
    r = bus.command("START", arg={"profile": "1"})
    assert not r.accepted and "the selected profile is 2, not '1'" in r.message


# ================================================================== fail closed: through the DLP checks

def test_a_tube_that_went_missing_is_a_missing_tube_never_a_dent(eng, tmp_path):
    """Every work state of the example asks "is that tube there or not?" of a named presence
    check. A tube taken away after "Tube aligned?" is a sensor / DLP disagreement at "End
    face OK?" (the sensor still says present): prompt, hold - not a DEFECT, not a kickout
    REJECT "dented end face" on an empty seat."""
    s, scans = eng
    ok, absent = scans["long:ok"], scans["long:absent"]

    def cycle(per_step, inputs, **kw):
        out = api.flow_run(EXAMPLE, "long_tube", inputs=inputs, runs_dir=tmp_path / "runs",
                           scans=per_step, project_faults=False, **kw)
        return out, out["result"]

    out, res = cycle({"loaded": ok, "aligned": ok, "end_face": absent}, {"part_present": True})
    assert [st["outcome"] for st in res["steps"]] == ["CONFIRMED", "CONFIRMED", "CORRECTED"]
    face = res["steps"][2]
    assert face["sensor_mismatch"] == [{"signal": "part_present", "sensor": True, "dlp": False,
                                        "expected": True}]
    assert face["dlp_result"]["observe"] == {"part_present": "seat/presence"}
    assert [a["action"] for a in face["actions"]] == ["project_fault", "prompt", "unacknowledged"]
    assert face["actions"][1]["message"] == "no tube found: load or clear the sensor"
    assert (res["result"], res["disposition"], out["exit_code"]) == ("INVALID", "PULL", 5)
    assert res["cycle"]["kickout"]["by"] == "unacknowledged" and res["cycle"]["sensor_mismatch"]
    assert not any(res["permissions"].values())
    assert not any("dent" in str(r.get("message")) for r in res["pull_reasons"])
    # the same at "Tube aligned?"
    out, res = cycle({"loaded": ok, "aligned": absent, "end_face": ok}, {"part_present": True})
    assert [st["outcome"] for st in res["steps"]] == ["CONFIRMED", "CORRECTED", "PENDING"]
    # sensor and DLP AGREE the tube is gone: NOT_MET (the same prompt by default), never a dent
    out, res = cycle({"loaded": ok, "aligned": ok, "end_face": absent},
                     lambda step: {"part_present": step != "end_face"})
    assert res["steps"][2]["outcome"] == "NOT_MET" and res["disposition"] == "PULL"
    # the sensor goes quiet about a tube that is still there: corrected the other way - a
    # work state no longer confirms while the sensor says the seat is empty
    out, res = cycle({"loaded": ok, "aligned": ok, "end_face": ok},
                     lambda step: {"part_present": step == "loaded"})
    assert [st["outcome"] for st in res["steps"]] == ["CONFIRMED", "CORRECTED", "PENDING"]
    assert res["steps"][1]["actions"][1]["message"].startswith("a tube is on the seat but the "
                                                               "sensor says none")
    assert res["permissions"] == {"LOAD_DONE": False, "CLAMP": False, "TRANSFER": False}
    # a real dent, the tube there: DEFECT, kicked out, and the kickout says what the DLP found
    out, res = cycle({"loaded": ok, "aligned": ok, "end_face": scans["long:dent"]},
                     {"part_present": True})
    k = res["cycle"]["kickout"]
    assert res["steps"][2]["outcome"] == "DEFECT" and res["disposition"] == "REJECT"
    assert k["reason"] == "end face not OK" and k["detail"].startswith("end_face/defect")
    # the other profile's tube at "End face OK?": there, its end face is not - REJECTed for
    # that, in the DLP's own words (not for a dent nobody measured)
    out, res = cycle({"loaded": ok, "aligned": ok, "end_face": scans["short:ok"]},
                     {"part_present": True})
    k = res["cycle"]["kickout"]
    assert res["steps"][2]["outcome"] == "DEFECT" and k["detail"].startswith("end_face/presence")
    assert "dent" not in k["reason"] and "dent" not in k["detail"]


def test_an_unattended_cycle_never_acknowledges_a_hold_for_ever(eng, tmp_path):
    """--acknowledge / acknowledge=true with a mismatch nobody puts right: prompt, ACK, hold,
    verify again ... used to run 200 verifies. A work state's prompts are answered at most
    retries + 1 times; then the cycle ends PULL and says why."""
    s, scans = eng
    out = api.flow_run(EXAMPLE, "long_tube", inputs={"part_present": True},
                       runs_dir=tmp_path / "runs", scans=_all(scans["long:absent"]),
                       acknowledge=True, project_faults=False)
    res = out["result"]
    first = res["steps"][0]
    assert first["outcome"] == "CORRECTED" and first["attempts"] == 3       # retries 1: 1 + 2
    assert (res["result"], res["disposition"], out["exit_code"]) == ("INVALID", "PULL", 5)
    assert res["cycle"]["ended_by"] == "unconfirmed"
    assert first["actions"][-1]["action"] == "unconfirmed"
    assert "acknowledged 2 time(s) and still not confirmed" in first["actions"][-1]["message"]
    assert [r["reason"] for r in res["pull_reasons"]] == ["not_confirmed", "sensor_mismatch"]
    assert "goto" not in res["pull_reasons"][0]["message"]
    assert len(res["cycle"]["sensor_mismatches"]) == 3
    assert len(list((Path(out["cycle_dir"]) / "steps").iterdir())) == 3


def test_the_cycle_table_never_says_granted_above_none_granted():
    """`scan run`'s table: a permission that was granted and dropped when the cycle ended
    REJECT / PULL reads "dropped", not GRANTED above "permissions: none granted"."""
    cyc = _walk(Cycle(_cfg(), "long"), [OK, GOOD, FAILS])           # kicked out at end_face
    assert cyc.disposition == "REJECT"
    printed = []

    class Ctx:
        def table(self, head, rows):
            printed.extend(" ".join(str(c) for c in r) for r in rows)

        def say(self, *lines):
            printed.extend(str(x) for x in lines)

    cli._flow_table(Ctx(), {"steps": cyc.record()["steps"]})
    text = "\n".join(printed)
    assert "LOAD_DONE dropped" in text and "CLAMP dropped" in text
    assert "TRANSFER withheld" in text and "GRANTED" not in text
    cyc = _walk(Cycle(_cfg(), "long"), [OK, GOOD, GOOD])
    printed.clear()
    cli._flow_table(Ctx(), {"steps": cyc.record()["steps"]})
    assert "\n".join(printed).count("GRANTED") == 3

