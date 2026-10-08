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
    assert not bus.command("SELECT_WORKSTATE", arg="b.json").accepted
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


def test_workstate_by_name_or_index_and_refusals():
    bus, calls, _ = _bus()
    assert bus.command("SELECT_WORKSTATE", arg=2).accepted and bus.workstate == "b.json"
    assert bus.command("SELECT_WORKSTATE", arg="1").accepted and bus.workstate == "a.json"
    assert not bus.command("SELECT_WORKSTATE", arg=9).accepted
    assert not bus.command("SELECT_WORKSTATE", arg="nope.json").accepted
    assert not bus.command("TELEPORT").accepted
    empty = C.CommandBus(workstates=[])
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
