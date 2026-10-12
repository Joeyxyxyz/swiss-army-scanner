"""Station command bus (engine/control/commands.py): the READY / ARMED / RUNNING / ABORTING /
HOMING / FAULT state machine, its stack-light outputs and the placeholder I/O backends."""

import json

import pytest

from engine.control import commands as C
from conftest import ROOT


def _bus(confirm=False):
    calls = []
    h = C.Handlers(start=lambda ws, ok: calls.append(("start", ws, ok)),
                   abort=lambda: calls.append(("abort",)),
                   home=lambda: calls.append(("home",)))
    bus = C.CommandBus(h, ["a.json", "b.json"], require_confirmation=confirm)
    out = C.SimulatedOutput()
    bus.outputs.append(out)
    return bus, calls, out


def test_start_run_pass_back_to_ready_with_outputs():
    bus, calls, out = _bus()
    assert bus.signals()["READY"]
    r = bus.command("START", "button")
    assert r.accepted and bus.state == "RUNNING" and calls == [("start", "a.json", False)]
    assert out.current["BUSY"] and not out.current["READY"]
    assert not bus.command("START").accepted                     # one cycle at a time
    bus.job_finished("run", "PASS")
    assert bus.state == "READY" and out.current["PASS"] and out.current["READY"]
    bus.command("START")
    bus.job_finished("run", "FAIL")
    assert out.current["FAIL"] and not out.current["PASS"]
    bus.command("START")
    bus.job_finished("run", None, error="crashed")
    assert bus.last_result == "INVALID" and out.current["ALARM"]


def test_abort_wins_and_home_only_when_idle_or_after_abort():
    bus, calls, _ = _bus()
    bus.command("START")
    assert not bus.command("HOME").accepted and bus.state == "RUNNING"
    assert not bus.command("SELECT_CONFIG", arg="b.json").accepted
    assert not bus.command("TRIGGER").accepted
    r = bus.command("ABORT", "key")
    assert r.accepted and bus.state == "ABORTING" and calls[-1] == ("abort",)
    assert not bus.command("START").accepted                     # abort is not preempted
    assert bus.command("ABORT").accepted                         # never refused
    assert bus.command("HOME").accepted and bus.state == "ABORTING"   # queued behind it
    bus.job_finished("run", "ABORTED")
    assert bus.state == "HOMING" and calls[-1] == ("home",) and bus.last_result == "ABORTED"
    bus.job_finished("home", "OK")
    assert bus.state == "READY"


def test_home_moved_is_a_fault_until_a_good_home():
    bus, calls, out = _bus()
    assert bus.command("HOME").accepted and bus.state == "HOMING"
    assert out.current["BUSY"]
    assert bus.command("ABORT").accepted and bus.state == "ABORTING"
    bus.job_finished("home", None, error="Cancelled")
    assert bus.state == "FAULT"
    bus2, _, out2 = _bus()
    bus2.command("HOME")
    bus2.job_finished("home", "MOVED")
    assert bus2.state == "FAULT" and out2.current["ALARM"] and not out2.current["READY"]
    assert not bus2.command("START").accepted and not bus2.command("ARM").accepted
    assert bus2.command("HOME").accepted
    bus2.job_finished("home", "OK")
    assert bus2.state == "READY" and bus2.fault is None
    bus2.command("HOME")
    bus2.job_finished("home", "UNKNOWN")
    assert bus2.command("RESET").accepted and bus2.state == "READY"


def test_trigger_only_when_armed_and_stays_armed():
    bus, calls, _ = _bus()
    assert not bus.command("TRIGGER").accepted and calls == []
    assert bus.command("ARM").accepted and bus.state == "ARMED"
    assert bus.command("TRIGGER", "plc").accepted and bus.state == "RUNNING"
    assert not bus.command("TRIGGER").accepted                   # ignored while RUNNING
    bus.job_finished("run", "PASS")
    assert bus.state == "ARMED"                                  # ready for the next part
    assert bus.command("DISARM").accepted and bus.state == "READY"
    bus.command("ARM")
    bus.command("TRIGGER")
    bus.command("ABORT")
    bus.job_finished("run", "ABORTED")
    assert bus.state == "READY"                                  # an abort disarms


def test_config_by_name_or_index_and_refusals():
    bus, calls, _ = _bus()
    assert bus.command("SELECT_CONFIG", arg=2).accepted and bus.config == "b.json"
    assert bus.command("SELECT_CONFIG", arg="1").accepted and bus.config == "a.json"
    assert not bus.command("SELECT_CONFIG", arg=9).accepted
    assert not bus.command("SELECT_CONFIG", arg="nope.json").accepted
    assert not bus.command("TELEPORT").accepted
    empty = C.CommandBus(configs=[])
    assert not empty.command("START").accepted and empty.state == "READY"


def test_real_rig_needs_confirmation_to_start_or_arm():
    bus, calls, _ = _bus(confirm=True)
    assert not bus.command("START").accepted and calls == []
    assert not bus.command("ARM").accepted
    assert bus.command("ARM", confirmed=True).accepted
    assert bus.command("TRIGGER").accepted                       # the arm carried it
    assert calls[-1] == ("start", "a.json", True)


def test_a_refused_start_is_a_fault():
    def boom(ws, ok):
        raise RuntimeError("rig busy")
    bus = C.CommandBus(C.Handlers(start=boom), ["a.json"])
    r = bus.command("START")
    assert not r.accepted and bus.state == "FAULT" and "rig busy" in bus.fault


def test_io_config_and_placeholder_backends():
    ins, outs = C.load_io_config(ROOT / "stations" / "io_example.json")
    assert [type(i) for i in ins] == [C.SimulatedInput]           # hardware ones disabled
    assert [type(o) for o in outs] == [C.SimulatedOutput]
    bus, calls, _ = _bus()
    sim = ins[0]
    sim.start(bus)
    bus.command("ARM")
    assert sim.inject("trigger").accepted and bus.state == "RUNNING"
    assert sim.inject("abort").accepted and bus.state == "ABORTING"
    with pytest.raises(KeyError):
        sim.inject("Line7")
    for cls in (C.CameraGpioInput, C.SerialIOInput, C.DigitalIOInput, C.ModbusTcpInput,
                C.OpcUaInput, C.EtherNetIpInput):
        with pytest.raises(NotImplementedError, match="placeholder"):
            cls({"x": "START"}).start(bus)
    for cls in (C.CameraGpioOutput, C.DigitalIOOutput, C.ModbusTcpOutput, C.OpcUaOutput,
                C.EtherNetIpOutput):
        with pytest.raises(NotImplementedError, match="placeholder"):
            cls().publish(bus.signals())
    doc = json.loads((ROOT / "stations" / "io_example.json").read_text())
    for spec in doc["inputs"] + doc["outputs"]:                   # every mapping is valid
        spec["enabled"] = True
    ins2, outs2 = C.load_io_doc(doc)
    # simulated + camera GPIO / serial / digital I/O + Modbus TCP / OPC UA / EtherNet/IP
    assert len(ins2) == 7 and len(outs2) == 6


def test_bad_io_config_refused(tmp_path):
    f = tmp_path / "io.json"
    f.write_text(json.dumps({"inputs": [{"kind": "simulated", "map": {"x": "FLY"}}]}))
    with pytest.raises(ValueError, match="unknown commands"):
        C.load_io_config(f)
    f.write_text(json.dumps({"outputs": [{"kind": "laser", "map": {}}]}))
    with pytest.raises(ValueError, match="output kind"):
        C.load_io_config(f)


# ------------------------------------------------------------ review regressions

def _owner(confirm=False, configs=("a.json", "b.json", "c.json")):
    """A bus whose owner tracks its jobs: pending() says whether one will report back."""
    calls, jobs = [], []
    h = C.Handlers(start=lambda ws, ok: (calls.append(("start", ws, ok)), jobs.append("run")),
                   abort=lambda: calls.append(("abort",)) or ["j1"],
                   home=lambda: (calls.append(("home", dict(bus.home_args))), jobs.append("home")),
                   pending=lambda: bool(jobs))
    bus = C.CommandBus(h, list(configs), require_confirmation=confirm)
    return bus, calls, jobs


def test_abort_with_no_job_left_settles_instead_of_wedging():
    bus, calls, jobs = _owner()
    bus.command("START")
    jobs.clear()                                       # the run's report was lost
    r = bus.command("ABORT")
    assert r.accepted and bus.state == "READY" and bus.last_result == "ABORTED"
    bus.command("HOME")
    jobs.clear()
    bus.command("ABORT")
    assert bus.state == "FAULT" and "no job left" in bus.fault     # the rig check never ended
    assert bus.command("RESET").accepted
    bus.command("START")
    jobs.clear()
    bus.job_lost("gone")                               # what the owner's reconcile calls
    assert bus.state == "READY" and bus.last_result == "INVALID"
    bus2, calls2, _ = _owner()                         # an owner that doesn't track: no guess
    bus2.h.pending = lambda: None
    bus2.command("START")
    bus2.command("ABORT")
    assert bus2.state == "ABORTING"


def test_abort_reaches_the_owner_in_every_state():
    bus, calls, jobs = _owner()
    r = bus.command("ABORT")                           # READY: a verdict / live view / direct job
    assert r.accepted and calls == [("abort",)] and "j1" in r.message
    bus.command("ARM")
    r = bus.command("ABORT")
    assert bus.state == "READY" and "disarmed" in r.message and calls[-1] == ("abort",)
    bus.command("START")
    bus.command("ABORT")
    assert bus.state == "ABORTING"
    n = len(calls)
    assert bus.command("ABORT").accepted and len(calls) == n + 1    # again: re-issued
    raising = C.CommandBus(C.Handlers(abort=lambda: 1 / 0), ["a.json"])
    r = raising.command("ABORT")
    assert r.accepted and "ZeroDivisionError" in r.message          # never refused


def test_home_pressed_while_aborting_a_rig_check_runs_once_and_never_later():
    bus, calls, jobs = _owner()
    bus.command("HOME", arg={"tol_px": 0.4})
    assert calls[-1] == ("home", {"tol_px": 0.4}) and bus.home_args == {}
    bus.command("ABORT")
    r = bus.command("HOME", arg={"set_reference": True})
    assert r.accepted and bus.home_after_abort
    jobs.clear()
    bus.job_finished("home", None, error="Cancelled: aborted")
    assert bus.state == "HOMING" and calls[-1] == ("home", {"set_reference": True})
    assert not bus.home_after_abort and bus.home_args == {}
    bus.job_finished("home", "OK")
    assert bus.state == "READY"
    homes = sum(1 for c in calls if c[0] == "home")
    bus.command("START")
    bus.command("ABORT")
    bus.job_finished("run", "ABORTED")
    assert bus.state == "READY" and sum(1 for c in calls if c[0] == "home") == homes
    # queued behind an abort, then a further ABORT: cancelled with its options
    bus.command("START")
    bus.command("ABORT")
    bus.command("HOME", arg={"set_reference": True})
    r = bus.command("ABORT")
    assert "HOME is cancelled" in r.message and not bus.home_after_abort and bus.home_args == {}
    bus.job_finished("run", "ABORTED")
    assert bus.state == "READY" and sum(1 for c in calls if c[0] == "home") == homes
    # a refused HOME leaves nothing behind
    bus.command("START")
    assert not bus.command("HOME", arg={"set_reference": True}).accepted
    assert bus.home_args == {} and not bus.home_after_abort


def test_only_a_real_true_confirms():
    bus, calls, _ = _owner(confirm=True)
    for bad in ("false", "true", "0", 1, [1]):
        assert not bus.command("ARM", confirmed=bad).accepted, bad
        assert not bus.command("START", confirmed=bad).accepted, bad
    assert calls == [] and not bus.armed_confirmed
    assert bus.command("ARM", confirmed=True).accepted and bus.armed_confirmed


def test_a_station_configuration_change_while_armed_disarms():
    bus, calls, _ = _owner(confirm=True)
    assert bus.command("ARM", confirmed=True).accepted
    r = bus.command("SELECT_CONFIG", arg="a.json")             # the same one: stays armed
    assert r.accepted and bus.state == "ARMED" and bus.armed_confirmed
    r = bus.command("SELECT_CONFIG", arg="b.json")
    assert r.accepted and "disarmed" in r.message
    assert bus.state == "READY" and not bus.armed_confirmed
    assert not bus.command("TRIGGER").accepted and calls == []
    # START / TRIGGER / ARM naming the expected configuration: refused on a mismatch
    assert not bus.command("START", arg="a.json", confirmed=True).accepted
    assert not bus.command("ARM", arg=1, confirmed=True).accepted and calls == []
    assert bus.command("START", arg=2, confirmed=True).accepted
    assert calls == [("start", "b.json", True)]


def test_a_cancelled_triggered_cycle_does_not_rearm():
    bus, calls, _ = _owner()
    bus.command("ARM")
    bus.command("TRIGGER")
    bus.job_finished("run", "ABORTED")                 # cancelled from the job list
    assert bus.state == "READY" and bus.last_result == "ABORTED"


def test_config_updates_never_switch_the_selection_or_renumber_mid_cycle():
    bus, calls, jobs = _owner(confirm=True)
    out = C.SimulatedOutput()
    bus.outputs.append(out)
    bus.command("SELECT_CONFIG", arg=2)
    bus.command("ARM", confirmed=True)
    bus.update_configs(["a.json", "b.json", "c.json"], unusable={"b.json": "bad JSON"})
    assert bus.state == "READY" and bus.config == "b.json" and not bus.armed_confirmed
    assert bus.history[-1][2] == "DISARM" and bus.registers()["station_config"] == 2
    r = bus.command("START", confirmed=True)
    assert not r.accepted and "bad JSON" in r.message and calls == []
    assert bus.last_result == "INVALID" and out.current["ALARM"]
    assert not bus.command("SELECT_CONFIG", arg=2).accepted
    assert bus.command("SELECT_CONFIG", arg=3).accepted and bus.config == "c.json"
    bus.update_configs(["a.json", "b.json", "c.json"])
    bus.command("START", confirmed=True)
    bus.update_configs(["b.json", "c.json", "d.json"])           # a.json removed mid-run
    assert bus.configs == ["a.json", "b.json", "c.json"] and bus.config == "c.json"
    assert bus.unusable == {"a.json": "its station config is gone"}
    bus.job_finished("run", "PASS")
    bus.update_configs(["b.json", "d.json"])                     # c.json gone while READY
    assert bus.config == "c.json" and bus.configs == ["b.json", "d.json"]
    r = bus.command("START", confirmed=True)
    assert not r.accepted and "gone" in r.message
    fresh = C.CommandBus(configs=[])                             # first list: first usable
    fresh.update_configs(["x.json", "y.json"], unusable={"x.json": "bad"})
    assert fresh.config == "y.json"


# ------------------------------------------------------------ rig findings: ALARM and RESET

def _spy(bus):
    """Every published signal set, in order (what a PLC output module would see)."""
    seen = []
    bus.outputs.append(type("Spy", (C.OutputBackend,), {
        "publish": lambda self, sig: seen.append(dict(sig)),
        "publish_registers": lambda self, regs: None})())
    return seen


def test_reset_then_a_good_home_clears_alarm_but_not_the_pull():
    """Seen on the rig: an INVALID cycle (PULL + ALARM), ABORT during HOME (FAULT), RESET, a
    good HOME -> READY, and ALARM stayed on for ever (it followed the stale INVALID result).
    RESET clears ALARM / FAULT; the stale PULL lamp and its code stay until the next cycle;
    a PASS never comes back."""
    bus, calls, out = _bus()
    seen = _spy(bus)
    bus.command("START")
    bus.job_finished("run", "INVALID", disposition="PULL")
    assert out.current["ALARM"] and out.current["PULL"] and out.current["READY"]
    bus.command("HOME")
    bus.command("ABORT")
    bus.job_finished("home", None, error="Cancelled: aborted")
    assert bus.state == "FAULT" and out.current["ALARM"] and not out.current["READY"]
    r = bus.command("RESET")
    assert r.accepted and bus.state == "READY" and bus.fault is None
    assert not out.current["ALARM"] and out.current["READY"]        # cleared by RESET itself
    assert out.current["PULL"] and "PULL output stays" in r.message  # documented: stale PULL
    assert out.registers["result_code"] == C.RESULT_CODES["INVALID"]
    bus.command("HOME")
    bus.job_finished("home", "OK")
    assert bus.state == "READY" and not out.current["ALARM"] and out.current["PULL"]
    assert not any(s["PASS"] for s in seen)                         # never a PASS in all this
    # the next cycle starts clean; a new INVALID raises ALARM again
    bus.command("START")
    assert not any(out.current[k] for k in ("ALARM", "PULL", "PASS", "FAIL"))
    bus.job_finished("run", "INVALID", disposition="PULL")
    assert out.current["ALARM"] and not bus.alarm_acknowledged


def test_reset_acknowledges_an_idle_alarm_and_a_good_home_does_too():
    bus, calls, out = _bus()
    assert not bus.command("RESET").accepted                        # nothing to acknowledge
    bus.command("START")
    bus.job_finished("run", None, error="crashed")
    assert bus.state == "READY" and out.current["ALARM"]
    r = bus.command("RESET")                                        # READY, ALARM on: accepted
    assert r.accepted and "acknowledged" in r.message and not out.current["ALARM"]
    assert out.current["PULL"] and bus.last_result == "INVALID"     # the result is not rewritten
    assert bus.snapshot()["alarm_acknowledged"] is True
    assert not bus.command("RESET").accepted                        # already acknowledged
    # a good HOME alone acknowledges too (the operator page has no RESET button)
    bus.command("START")
    bus.job_finished("run", "INVALID")
    assert out.current["ALARM"]
    bus.command("HOME")
    bus.job_finished("home", "REFERENCE_SET")
    assert bus.state == "READY" and not out.current["ALARM"] and out.current["PULL"]
    # a rig check that fails is a FAULT: ALARM on again, whatever was acknowledged before
    bus.command("HOME")
    bus.job_finished("home", "MOVED")
    assert bus.state == "FAULT" and out.current["ALARM"]
    assert bus.command("RESET").accepted and not out.current["ALARM"]
    # the ALARM of a START refused on an unusable configuration: RESET acknowledges it too
    bus.update_configs(["a.json", "b.json"], unusable={"a.json": "config invalid"})
    assert not bus.command("START").accepted and out.current["ALARM"]
    assert bus.command("RESET").accepted and not out.current["ALARM"] and out.current["PULL"]


def test_reset_never_brings_a_pass_back_and_a_fail_lamp_stays():
    bus, calls, out = _bus()
    seen = _spy(bus)
    bus.command("START")
    bus.job_finished("run", "PASS", disposition="CONTINUE")
    assert out.current["PASS"]
    bus.command("HOME")
    bus.job_finished("home", "MOVED")
    assert bus.state == "FAULT" and out.current["ALARM"] and not out.current["PASS"]
    assert bus.command("RESET").accepted
    assert not out.current["ALARM"] and not out.current["PASS"] and out.current["READY"]
    assert out.registers["result_code"] == 0 and [s["PASS"] for s in seen].count(True) == 1
    bus.command("START")
    bus.job_finished("run", "INVALID", disposition="REJECT")       # a doubt the line rejects
    assert out.current["FAIL"] and out.current["ALARM"]
    r = bus.command("RESET")
    assert r.accepted and "FAIL output stays" in r.message
    assert out.current["FAIL"] and not out.current["ALARM"]
