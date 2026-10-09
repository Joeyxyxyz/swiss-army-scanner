"""
Station command bus: one state machine for every command source.

Sources (all call CommandBus.command):
    web UI buttons        /quick START / ABORT / HOME / ARM / TRIGGER
    keyboard              /quick: Space / Enter START, Esc ABORT, H HOME, T TRIGGER,
                          A ARM / DISARM, 1-9 work state
    HTTP                  POST /api/cmd/<start|abort|home|trigger|arm|disarm|workstate>
                          (a PLC / HMI gateway or an automation flow)
    hardware (PLACEHOLDER) InputBackend subclasses below: camera GPIO lines, serial /
                          USB-IO, a 24 V digital I/O module, and the industrial networks
                          (Modbus TCP, OPC UA, EtherNet/IP). Config-driven mapping
                          (stations/io_example.json). Not implemented: they refuse to start.

States and rules:
    READY     idle, work state selectable          START -> RUNNING, ARM -> ARMED
    ARMED     waiting for TRIGGER (part present)   TRIGGER -> RUNNING, DISARM -> READY
    RUNNING   a station run                        ABORT -> ABORTING; done -> READY / ARMED
    ABORTING  abort requested, run winding down    done -> READY (or HOMING if HOME pressed)
    HOMING    rig check                            OK -> READY, MOVED / error -> FAULT
    FAULT     rig check failed / a job crashed     HOME (re-check) or RESET -> READY
ABORT always wins (it is never refused, and in every state it also asks the owner to stop
whatever it runs outside the state machine: a projected verdict, a live view, jobs started
directly); HOME only when idle or after an abort (a HOME pressed while aborting runs once the
aborted job ends - a further ABORT cancels it - and is never kept for later); TRIGGER is
ignored unless ARMED; the work state can't change while a job runs, and changing it while
ARMED disarms (the operator's confirmation was for the old one). START / TRIGGER / ARM may
name the work state the sender expects: refused when the selected one differs. A work state
the owner marked unusable (its config is invalid or gone) is never started: refused, result
INVALID. A cancelled or aborted cycle never re-arms.

Outputs (stack-light style): READY / BUSY / PASS / FAIL / ALARM booleans, published to
every OutputBackend on each change (the UI lamp reads the same signals), plus registers for
a PLC: result_code (0 none, PASS 1, FAIL 2, INVALID 3, ABORTED 4 - a recipe may renumber
them in outputs.result_codes), state_code (index in STATES), workstate (1-based, 0 none).

Work states: the station configs; a recipe work state (job.type recipe) pins its recipe's
preset, and its result codes come from the recipe.

Pure: no engine, no hardware. The owner (the web server) passes handlers that start /
abort the actual jobs and calls job_finished() when they end (also for a job dropped before
it ran); job_lost() settles a busy state whose job can no longer report back.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

STATES = ("READY", "ARMED", "RUNNING", "ABORTING", "HOMING", "FAULT")
COMMANDS = ("START", "ABORT", "HOME", "TRIGGER", "SELECT_WORKSTATE", "ARM", "DISARM", "RESET")
SIGNALS = ("READY", "BUSY", "PASS", "FAIL", "ALARM")
IDLE = ("READY", "ARMED")
BUSY = ("RUNNING", "ABORTING", "HOMING")
RESULT_CODES = {"PASS": 1, "FAIL": 2, "INVALID": 3, "ABORTED": 4}       # 0 = no result yet
REGISTERS = ("result_code", "state_code", "workstate")


@dataclass
class Reply:
    accepted: bool
    command: str
    state: str
    message: str = ""

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class Handlers:
    """What the bus asks its owner to do. start(workstate, confirmed) -> job id;
    abort() and home() -> anything (ids, for the log). home() reads the HOME request's
    options from bus.home_args while it runs."""
    start: Callable = lambda ws, confirmed: None
    abort: Callable = lambda: None
    home: Callable = lambda: None
    # select(workstate, preset) after SELECT_WORKSTATE is accepted: the owner makes that
    # work state's preset (part recipe) the loaded one; None = the station names none
    select: Callable = lambda ws, preset: None
    # pending() -> True while a job the bus started will still report back (job_finished),
    # False when none will (it was dropped / lost), None = the owner doesn't track jobs
    pending: Callable = lambda: None


class CommandBus:
    def __init__(self, handlers: Optional[Handlers] = None, workstates: Optional[list] = None,
                 require_confirmation: bool = False):
        self.h = handlers or Handlers()
        self.workstates = list(workstates or [])
        self.require_confirmation = bool(require_confirmation)
        self.state = "READY"
        self.workstate: Optional[str] = self.workstates[0] if self.workstates else None
        self.workstate_presets: dict = {}            # work state -> the preset it pins
        self.workstate_info: dict = {}               # work state -> {recipe, result_codes}
        self.unusable: dict = {}                     # work state -> why it can't run (owner)
        self.armed_confirmed = False
        self.last_result: Optional[str] = None      # PASS | FAIL | INVALID | ABORTED
        self.fault: Optional[str] = None
        self.home_after_abort = False                # only ever True while ABORTING
        self.home_args: dict = {}                    # the HOME request's options (owner reads)
        self._busy_kind: Optional[str] = None        # 'run' | 'home': what the busy state awaits
        self._return_armed = False
        self.outputs: list = []
        self.history: list = []                      # (time, source, command, accepted, msg)
        self._lock = threading.RLock()

    # ------------------------------------------------------------ outputs
    def signals(self) -> dict:
        s = self.state
        return {"READY": s in IDLE and self.fault is None,
                "BUSY": s in BUSY,
                "PASS": s in IDLE and self.last_result == "PASS",
                "FAIL": s in IDLE and self.last_result == "FAIL",
                "ALARM": s == "FAULT" or (s in IDLE and self.last_result == "INVALID")}

    def registers(self) -> dict:
        """Numbers for a PLC: result_code (the work state's recipe codes, else RESULT_CODES;
        0 = none yet), state_code (index in STATES), workstate (1-based, 0 = none)."""
        codes = (self.workstate_info.get(self.workstate) or {}).get("result_codes") or RESULT_CODES
        ws = self.workstates.index(self.workstate) + 1 if self.workstate in self.workstates else 0
        return {"result_code": int(codes.get(self.last_result, 0)) if self.last_result else 0,
                "state_code": STATES.index(self.state), "workstate": ws}

    def snapshot(self) -> dict:
        with self._lock:
            return {"state": self.state, "workstate": self.workstate,
                    "workstate_preset": self.workstate_presets.get(self.workstate),
                    "workstate_recipe": (self.workstate_info.get(self.workstate) or {}).get(
                        "recipe"),
                    "registers": self.registers(),
                    "workstates": list(self.workstates), "unusable": dict(self.unusable),
                    "last_result": self.last_result,
                    "fault": self.fault, "signals": self.signals(),
                    "require_confirmation": self.require_confirmation,
                    "history": self.history[-20:]}

    def _publish(self) -> None:
        sig = self.signals()
        regs = self.registers()
        for o in self.outputs:
            try:
                o.publish(sig)
                o.publish_registers(regs)
            except Exception as e:          # noqa: BLE001 - an output must not stop the bus
                self._log("bus", "OUTPUT", False, f"{type(o).__name__}: {e}")

    def _log(self, source: str, name: str, accepted: bool, msg: str) -> None:
        self.history.append((_now(), source, name, accepted, msg))
        del self.history[:-200]

    def _go(self, state: str) -> None:
        if state != "ABORTING" and self.home_after_abort:
            # a HOME pressed while aborting lives only as long as that abort: it runs when
            # the aborted job ends (job_finished), or it is forgotten - never fired later
            self.home_after_abort = False
            self.home_args = {}
        self.state = state
        self._publish()

    # ------------------------------------------------------------ commands
    def command(self, name: str, source: str = "api", arg=None, confirmed: bool = False) -> Reply:
        """confirmed: only a real True counts (a "false" string from a gateway is not a yes)."""
        name = str(name).upper()
        with self._lock:
            if name not in COMMANDS:
                r = Reply(False, name, self.state, f"unknown command (one of {', '.join(COMMANDS)})")
            else:
                r = getattr(self, "_c_" + name.lower())(arg, confirmed is True)
            self._log(source, name, r.accepted, r.message)
            return r

    def _no(self, cmd, msg) -> Reply:
        return Reply(False, cmd, self.state, msg)

    def _ok(self, cmd, msg="") -> Reply:
        return Reply(True, cmd, self.state, msg)

    def _unusable(self, ws: Optional[str]) -> Optional[str]:
        """Why work state `ws` can't run (None = it can)."""
        if self.unusable.get(ws):
            return self.unusable[ws]
        if ws not in self.workstates:
            return "it is not one of the station configs any more"
        return None

    def _expect(self, cmd: str, arg) -> Optional[Reply]:
        """START / TRIGGER / ARM naming the work state the sender expects (the work state the
        operator page shows): refused when the selected one differs."""
        if arg is None or arg == "":
            return None
        if self.resolve_workstate(arg) != self.workstate:
            return self._no(cmd, f"the selected work state is {self.workstate}, not {arg!r}: "
                                 "check the setup for the selected one first")
        return None

    def _begin(self, cmd: str, confirmed: bool, arg=None) -> Reply:
        if self.workstate is None:
            return self._no(cmd, "no work state selected")
        refused = self._expect(cmd, arg)
        if refused is not None:
            return refused
        why = self._unusable(self.workstate)
        if why:                                 # never started: the cycle's result is INVALID
            self.last_result = "INVALID"
            if self.state == "ARMED":
                self.armed_confirmed = False
                self._go("READY")
            else:
                self._publish()
            return self._no(cmd, f"work state {self.workstate} is not usable: {why}")
        if self.require_confirmation and not confirmed:
            return self._no(cmd, "the real rig needs the operator's confirmation of the setup")
        self._return_armed = self.state == "ARMED"
        self.last_result = None
        self._busy_kind = "run"
        self._go("RUNNING")
        try:
            self.h.start(self.workstate, confirmed)
        except Exception as e:              # noqa: BLE001 - the start itself was refused
            self.fault = f"start failed: {type(e).__name__}: {e}"
            self._go("FAULT")
            return self._no(cmd, self.fault)
        return self._ok(cmd, f"started {self.workstate}")

    def _c_start(self, arg, confirmed):
        if self.state not in IDLE:
            return self._no("START", f"not idle ({self.state})")
        return self._begin("START", confirmed, arg)

    def _c_trigger(self, arg, confirmed):
        if self.state != "ARMED":
            return self._no("TRIGGER", f"ignored: not armed ({self.state})")
        return self._begin("TRIGGER", confirmed or self.armed_confirmed, arg)

    def _owner_abort(self) -> str:
        """h.abort() (stops what the owner runs: the bus's job, and also a projected verdict,
        a live view, runs started directly); never raises: ABORT is never refused."""
        try:
            touched = self.h.abort()
        except Exception as e:              # noqa: BLE001
            return f"abort handler failed: {type(e).__name__}: {e}"
        if isinstance(touched, (list, tuple)) and touched:
            return "stopped " + ", ".join(str(t) for t in touched)
        return ""

    def _c_abort(self, arg, confirmed):
        """Never refused: it preempts whatever runs, in every state."""
        if self.state in ("RUNNING", "HOMING"):
            self._go("ABORTING")
            note = self._owner_abort()
            msg = "abort requested: stops before the next capture"
        elif self.state == "ABORTING":
            msg = "abort already requested: waiting for the job to stop"
            if self.home_after_abort:           # ABORT wins over a HOME queued behind it
                self.home_after_abort = False
                self.home_args = {}
                msg += "; the queued HOME is cancelled"
            note = self._owner_abort()          # again: whatever was queued since
        elif self.state == "ARMED":
            self.armed_confirmed = False
            self._go("READY")
            note = self._owner_abort()
            msg = "disarmed"
        else:
            note = self._owner_abort()
            msg = "" if note else "nothing running"
        if self.state == "ABORTING" and self.h.pending() is False:
            # nothing left that will report back (a job dropped without a word): settle now
            self.job_lost("aborted: no job left that could report back")
            msg += f"; nothing left running ({self.state})"
        return self._ok("ABORT", "; ".join(m for m in (msg, note) if m))

    def _start_home(self) -> bool:
        """HOMING + h.home() (which reads home_args); a refused home is a FAULT."""
        self._busy_kind = "home"
        self._go("HOMING")
        try:
            self.h.home()
        except Exception as e:              # noqa: BLE001
            self.fault = f"home failed: {type(e).__name__}: {e}"
            self._go("FAULT")
            return False
        finally:
            self.home_args = {}
        return True

    def _c_home(self, arg, confirmed):
        opts = dict(arg) if isinstance(arg, dict) else {}
        if self.state == "ABORTING":
            self.home_after_abort = True
            self.home_args = opts
            return self._ok("HOME", "homing after the abort finishes")
        if self.state in ("RUNNING", "HOMING"):
            return self._no("HOME", f"refused while {self.state}: ABORT first")
        self.armed_confirmed = False
        self.home_args = opts
        if not self._start_home():
            return self._no("HOME", self.fault)
        return self._ok("HOME", "rig check running")

    def _c_select_workstate(self, arg, confirmed):
        if self.state not in IDLE + ("FAULT",):
            return self._no("SELECT_WORKSTATE", f"refused while {self.state}")
        ws = self.resolve_workstate(arg)
        if ws is None:
            return self._no("SELECT_WORKSTATE", f"no work state {arg!r}")
        why = self.unusable.get(ws)
        if why:
            return self._no("SELECT_WORKSTATE", f"{ws} is not usable: {why}")
        note = ""
        if ws != self.workstate and self.state == "ARMED":
            # the operator's confirmation (and the arm) was for the old work state
            self.armed_confirmed = False
            self.state = "READY"
            note = " - disarmed: ARM again for this work state"
        self.workstate = ws
        self._publish()
        preset = self.workstate_presets.get(ws)
        try:
            self.h.select(ws, preset)
        except Exception as e:              # noqa: BLE001 - selected, but its preset didn't load
            return self._ok("SELECT_WORKSTATE",
                            f"{ws} (preset {preset}: {type(e).__name__}: {e}){note}")
        return self._ok("SELECT_WORKSTATE", (f"{ws} (preset {preset})" if preset else ws) + note)

    def _c_arm(self, arg, confirmed):
        if self.state != "READY":
            return self._no("ARM", f"not READY ({self.state})")
        if self.workstate is None:
            return self._no("ARM", "no work state selected")
        refused = self._expect("ARM", arg)
        if refused is not None:
            return refused
        why = self._unusable(self.workstate)
        if why:
            return self._no("ARM", f"work state {self.workstate} is not usable: {why}")
        if self.require_confirmation and not confirmed:
            return self._no("ARM", "the real rig needs the operator's confirmation to arm")
        self.armed_confirmed = bool(confirmed)
        self._go("ARMED")
        return self._ok("ARM", "armed: TRIGGER starts a cycle")

    def _c_disarm(self, arg, confirmed):
        if self.state != "ARMED":
            return self._no("DISARM", f"not armed ({self.state})")
        self.armed_confirmed = False
        self._go("READY")
        return self._ok("DISARM")

    def _c_reset(self, arg, confirmed):
        if self.state != "FAULT":
            return self._no("RESET", f"no fault ({self.state})")
        self.fault = None
        self._go("READY")
        return self._ok("RESET")

    def resolve_workstate(self, arg) -> Optional[str]:
        """A work state name, or a 1-based index (keys 1-9, a PLC number)."""
        if arg is None or isinstance(arg, bool):
            return None
        if isinstance(arg, int) or (isinstance(arg, str) and arg.isdigit()):
            i = int(arg) - 1
            return self.workstates[i] if 0 <= i < len(self.workstates) else None
        return arg if isinstance(arg, str) and arg in self.workstates else None

    def update_workstates(self, names: list, unusable: Optional[dict] = None,
                          presets: Optional[dict] = None, info: Optional[dict] = None) -> None:
        """The owner re-read its work states: `names` = every station config in a stable
        order (usable or not, so a broken config never shifts the others' numbers),
        `unusable` = name -> why it can't run. Never switches the selected work state (one
        that broke or vanished stays selected and is refused). While ARMED or busy the list
        and its numbers stay as they are and only the usability marks change; an ARMED bus
        whose work state became unusable disarms."""
        names = list(names)
        bad = dict(unusable or {})
        with self._lock:
            before = (list(self.workstates), self.workstate)
            if self.state in ("READY", "FAULT"):
                self.workstates = names
                self.workstate_presets = dict(presets or {})
                self.workstate_info = dict(info or {})
                if self.workstate is None:       # nothing selected yet: the first usable one
                    self.workstate = next((n for n in names if n not in bad), None)
            marks = {n: bad[n] for n in self.workstates if n in bad}
            for n in [*self.workstates, self.workstate]:
                if n is not None and n not in names:
                    marks[n] = "its station config is gone"
            self.unusable = marks
            ws = self.workstate
            why = self._unusable(ws) if ws is not None else None
            if self.state == "ARMED" and why:
                self.armed_confirmed = False
                self._go("READY")
                self._log("bus", "DISARM", True, f"work state {ws} is not usable: {why}")
            elif (list(self.workstates), self.workstate) != before:
                self._publish()                  # the workstate register may have moved

    # ------------------------------------------------------------ job events
    def job_finished(self, kind: str, result: Optional[str] = None,
                     error: Optional[str] = None) -> None:
        """kind 'run' (result PASS / FAIL / INVALID / ABORTED) or 'home' (result OK /
        MOVED / UNKNOWN / REFERENCE_SET); error = the job crashed."""
        with self._lock:
            if self.state == "ABORTING" and self._busy_kind not in (None, kind):
                return                          # not the job this abort waits for
            if kind == "run" and self.state in ("RUNNING", "ABORTING"):
                aborted = self.state == "ABORTING"
                self.last_result = "ABORTED" if aborted else (result or "INVALID")
                if error and not aborted:
                    self.last_result = "INVALID"
                if aborted and self.home_after_abort:
                    self.home_after_abort = False   # consumed: the HOME pressed while aborting
                    self._start_home()
                    return
                # an aborted / cancelled cycle disarms: the next part needs a new ARM
                keep = self._return_armed and self.last_result != "ABORTED"
                if not keep:
                    self.armed_confirmed = False
                self._go("ARMED" if keep else "READY")
            elif kind == "home" and self.state in ("HOMING", "ABORTING"):
                if self.state == "ABORTING" and self.home_after_abort:
                    # HOME pressed again while this rig check was being aborted: run it now
                    self.home_after_abort = False
                    self._start_home()
                    return
                if error or result in ("MOVED", "UNKNOWN"):
                    self.fault = error or f"rig check {result}: re-home / recalibrate"
                    self._go("FAULT")
                else:
                    self.fault = None
                    self._go("READY")

    def job_lost(self, why: str = "the job ended without reporting back") -> None:
        """The owner has no job left that will call job_finished (it was dropped or lost):
        settle the busy state instead of waiting for ever. A run ends INVALID (ABORTED when it
        was being aborted) and never re-arms; a rig check ends in FAULT; a HOME pressed while
        aborting still runs."""
        with self._lock:
            if self.state not in BUSY:
                return
            self._return_armed = False
            kind = self._busy_kind or ("home" if self.state == "HOMING" else "run")
            self._log("bus", "JOB_LOST", False, why)
            self.job_finished(kind, None, error=why)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ================================================================ I/O backends

class InputBackend:
    """A hardware command source. start(bus) begins delivering commands (bus.command(
    name, source=self.name)); stop() releases the hardware. `mapping` maps the backend's
    inputs (lines, bits, serial tokens) to commands, from the I/O config."""
    name = "input"

    def __init__(self, mapping: Optional[dict] = None, **options):
        self.mapping = dict(mapping or {})
        self.options = options

    def start(self, bus: CommandBus) -> None:
        raise NotImplementedError(f"{type(self).__name__} is a placeholder: not implemented yet")

    def stop(self) -> None:
        pass


class SimulatedInput(InputBackend):
    """Tests / mock: inject(input_name) acts like that line going active."""
    name = "simulated"

    def start(self, bus: CommandBus) -> None:
        self.bus = bus

    def inject(self, line: str, arg=None) -> Reply:
        cmd = self.mapping.get(line)
        if cmd is None:
            raise KeyError(f"input {line!r} is not mapped (mapping: {self.mapping})")
        return self.bus.command(cmd, source=f"{self.name}:{line}", arg=arg)


class CameraGpioInput(InputBackend):
    """PLACEHOLDER: the machine-vision camera's GPIO input lines as inputs (e.g. Line0 =
    trigger / part present, Line3 = abort; never Line1 / Line2, outputs). Intended GenICam
    nodes (via the camera SDK, never opened here): LineSelector = LineN, LineMode = Input,
    LineInverter, LineStatus (polled) or an EventLine*RisingEdge event, LineDebouncerTime.
    Must share the camera handle with the capture session (one owner of the camera)."""
    name = "camera_gpio"


class SerialIOInput(InputBackend):
    """PLACEHOLDER: a serial / USB-IO box sending text tokens ("START\\n", "TRIG\\n") or
    bit masks; options: port, baud. Mapping: token -> command."""
    name = "serial_io"


class DigitalIOInput(InputBackend):
    """PLACEHOLDER: a 24 V digital I/O module (fieldbus / Modbus-TCP / vendor DLL); options:
    host, unit, poll_ms. Mapping: input bit "DI0".."DI15" -> command (rising edge)."""
    name = "digital_io"


class OutputBackend:
    """Stack-light style outputs: publish({"READY": bool, "BUSY": .., "PASS": .., "FAIL": ..,
    "ALARM": ..}); `mapping` maps signals to the backend's outputs."""
    name = "output"

    def __init__(self, mapping: Optional[dict] = None, **options):
        self.mapping = dict(mapping or {})
        self.options = options

    def publish(self, signals: dict) -> None:
        raise NotImplementedError(f"{type(self).__name__} is a placeholder: not implemented yet")

    def publish_registers(self, registers: dict) -> None:
        """Numeric outputs (result_code, state_code, workstate): a no-op for backends that
        only drive lines (a stack light); fieldbus backends write them to registers / nodes."""


class SimulatedOutput(OutputBackend):
    name = "simulated"

    def __init__(self, mapping: Optional[dict] = None, **options):
        super().__init__(mapping, **options)
        self.log: list = []
        self.current: dict = {}
        self.registers: dict = {}

    def publish(self, signals: dict) -> None:
        self.current = {self.mapping.get(k, k): bool(v) for k, v in signals.items()}
        self.log.append(dict(self.current))

    def publish_registers(self, registers: dict) -> None:
        self.registers = dict(registers)


class CameraGpioOutput(OutputBackend):
    """PLACEHOLDER: camera GPIO output lines (LineSelector = Line1, LineMode = Output,
    LineSource = UserOutputN, UserOutputSelector / UserOutputValue)."""
    name = "camera_gpio"


class DigitalIOOutput(OutputBackend):
    """PLACEHOLDER: 24 V digital outputs DO0..DO15 driving a stack light / PLC inputs."""
    name = "digital_io"


# ---------------------------------------------------- industrial networks (PLACEHOLDERS)
# Clean interfaces, no protocol stack bundled (no new dependencies): start() / publish() /
# publish_registers() raise NotImplementedError naming the intended mapping. A site adds its
# own client by subclassing (or replacing) these and registering the class in INPUTS /
# OUTPUTS under the same kind. The mapping keys are the protocol's addresses as text.
#   inputs  (PLC -> scanner): command bits START / ABORT / HOME / TRIGGER / ARM / DISARM /
#           RESET (rising edge = one command), the work state number (options.workstate_*)
#   outputs (scanner -> PLC): READY / BUSY / PASS / FAIL / ALARM bits, the result_code /
#           state_code / workstate registers (options.*_register / *_node / *_tag)

_PLACEHOLDER = ("{cls} is a placeholder: the {proto} adapter is not implemented (no {proto} "
                "client is bundled). Implement {method} in a subclass with your site's {proto} "
                "stack. Intended mapping: {mapping}")


class ModbusTcpInput(InputBackend):
    """PLACEHOLDER: Modbus TCP client polling the PLC (options: host, port 502, unit, poll_ms,
    workstate_register). Mapping 'coil:N' / 'di:N' (a coil or discrete input address) ->
    command, acted on the rising edge; the holding register options.workstate_register
    (1-based work state number, 0 = keep) selects the work state when it changes."""
    name = "modbus_tcp"
    PROTOCOL = "Modbus TCP"

    def start(self, bus: CommandBus) -> None:
        raise NotImplementedError(_PLACEHOLDER.format(
            cls=type(self).__name__, proto=self.PROTOCOL, method="start(bus)",
            mapping="coils / discrete inputs 'coil:N' -> START / ABORT / HOME / TRIGGER / ARM, "
                    "holding register workstate_register -> SELECT_WORKSTATE"))


class ModbusTcpOutput(OutputBackend):
    """PLACEHOLDER: Modbus TCP writes (options: host, port 502, unit, result_register,
    state_register, workstate_register). Mapping signal -> 'coil:N' (READY / BUSY / PASS /
    FAIL / ALARM); publish_registers writes result_code / state_code / workstate to the
    holding registers named in options."""
    name = "modbus_tcp"
    PROTOCOL = "Modbus TCP"

    def publish(self, signals: dict) -> None:
        raise NotImplementedError(_PLACEHOLDER.format(
            cls=type(self).__name__, proto=self.PROTOCOL, method="publish(signals)",
            mapping="READY / BUSY / PASS / FAIL / ALARM -> coils 'coil:N'"))

    def publish_registers(self, registers: dict) -> None:
        raise NotImplementedError(_PLACEHOLDER.format(
            cls=type(self).__name__, proto=self.PROTOCOL, method="publish_registers(registers)",
            mapping="result_code / state_code / workstate -> holding registers "
                    "(options result_register, state_register, workstate_register)"))


class OpcUaInput(InputBackend):
    """PLACEHOLDER: OPC UA client subscribed to the PLC's nodes (options: endpoint
    'opc.tcp://host:4840', security, workstate_node). Mapping node id ('ns=2;s=Cell.Start')
    -> command on a false -> true change; options.workstate_node (Int) selects the work
    state."""
    name = "opc_ua"
    PROTOCOL = "OPC UA"

    def start(self, bus: CommandBus) -> None:
        raise NotImplementedError(_PLACEHOLDER.format(
            cls=type(self).__name__, proto=self.PROTOCOL, method="start(bus)",
            mapping="Boolean nodes 'ns=..;s=..' -> START / ABORT / HOME / TRIGGER / ARM "
                    "(data-change subscription), workstate_node -> SELECT_WORKSTATE"))


class OpcUaOutput(OutputBackend):
    """PLACEHOLDER: OPC UA writes (options: endpoint, result_node, state_node,
    workstate_node). Mapping signal -> Boolean node id; registers -> Int nodes."""
    name = "opc_ua"
    PROTOCOL = "OPC UA"

    def publish(self, signals: dict) -> None:
        raise NotImplementedError(_PLACEHOLDER.format(
            cls=type(self).__name__, proto=self.PROTOCOL, method="publish(signals)",
            mapping="READY / BUSY / PASS / FAIL / ALARM -> Boolean nodes"))

    def publish_registers(self, registers: dict) -> None:
        raise NotImplementedError(_PLACEHOLDER.format(
            cls=type(self).__name__, proto=self.PROTOCOL, method="publish_registers(registers)",
            mapping="result_code / state_code / workstate -> Int nodes (options result_node, "
                    "state_node, workstate_node)"))


class EtherNetIpInput(InputBackend):
    """PLACEHOLDER: EtherNet/IP (CIP) - the scanner as an explicit-messaging client reading
    PLC tags, or an adapter with an implicit I/O assembly (options: host, slot, poll_ms,
    workstate_tag). Mapping tag or assembly bit ('Cell_Start', 'in:0.3') -> command on the
    rising edge; options.workstate_tag (DINT) selects the work state."""
    name = "ethernet_ip"
    PROTOCOL = "EtherNet/IP"

    def start(self, bus: CommandBus) -> None:
        raise NotImplementedError(_PLACEHOLDER.format(
            cls=type(self).__name__, proto=self.PROTOCOL, method="start(bus)",
            mapping="BOOL tags / input-assembly bits -> START / ABORT / HOME / TRIGGER / "
                    "ARM, workstate_tag (DINT) -> SELECT_WORKSTATE"))


class EtherNetIpOutput(OutputBackend):
    """PLACEHOLDER: EtherNet/IP writes (options: host, slot, result_tag, state_tag,
    workstate_tag). Mapping signal -> BOOL tag or output-assembly bit; registers -> DINT
    tags."""
    name = "ethernet_ip"
    PROTOCOL = "EtherNet/IP"

    def publish(self, signals: dict) -> None:
        raise NotImplementedError(_PLACEHOLDER.format(
            cls=type(self).__name__, proto=self.PROTOCOL, method="publish(signals)",
            mapping="READY / BUSY / PASS / FAIL / ALARM -> BOOL tags / output-assembly bits"))

    def publish_registers(self, registers: dict) -> None:
        raise NotImplementedError(_PLACEHOLDER.format(
            cls=type(self).__name__, proto=self.PROTOCOL, method="publish_registers(registers)",
            mapping="result_code / state_code / workstate -> DINT tags (options result_tag, "
                    "state_tag, workstate_tag)"))


INPUTS = {c.name: c for c in (SimulatedInput, CameraGpioInput, SerialIOInput, DigitalIOInput,
                              ModbusTcpInput, OpcUaInput, EtherNetIpInput)}
OUTPUTS = {c.name: c for c in (SimulatedOutput, CameraGpioOutput, DigitalIOOutput,
                               ModbusTcpOutput, OpcUaOutput, EtherNetIpOutput)}


def load_io_config(path) -> tuple:
    """stations/io_example.json -> ([InputBackend], [OutputBackend]); entries with
    "enabled": false are skipped. Unknown kinds / commands / signals are refused."""
    return load_io_doc(json.loads(Path(path).read_text(encoding="utf-8")))


def load_io_doc(doc: dict) -> tuple:
    """load_io_config on an already-parsed mapping."""
    ins, outs = [], []
    for spec in doc.get("inputs", []):
        if not spec.get("enabled", True):
            continue
        kind = spec.get("kind")
        if kind not in INPUTS:
            raise ValueError(f"input kind {kind!r} must be one of {sorted(INPUTS)}")
        bad = [c for c in spec.get("map", {}).values() if c not in COMMANDS]
        if bad:
            raise ValueError(f"input {kind}: unknown commands {bad}")
        ins.append(INPUTS[kind](spec.get("map"), **spec.get("options", {})))
    for spec in doc.get("outputs", []):
        if not spec.get("enabled", True):
            continue
        kind = spec.get("kind")
        if kind not in OUTPUTS:
            raise ValueError(f"output kind {kind!r} must be one of {sorted(OUTPUTS)}")
        bad = [s for s in spec.get("map", {}) if s not in SIGNALS]
        if bad:
            raise ValueError(f"output {kind}: unknown signals {bad}")
        outs.append(OUTPUTS[kind](spec.get("map"), **spec.get("options", {})))
    return ins, outs
