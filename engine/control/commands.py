"""
Station command bus: one state machine for every command source.

Words (engine/flow/, stations/README.md):
    station configuration   a stations/*.json file: the station's parameters and its PROFILE
                            LIBRARY SET (what each part variant looks like). NOT a work state.
    profile                 the active part variant of the selected configuration (long tube /
                            short tube): SELECT_PROFILE, from the operator, a PLC or HTTP
    work state              one step of the configuration's flow: a claim of the line's own
                            sensors that the DLP confirms or corrects, and the permission it
                            grants. A cycle walks the work states for one part.

Sources (all call CommandBus.command):
    web UI buttons        /quick START / ABORT / HOME / ARM / TRIGGER / STEP / ACK
    keyboard              /quick: Space / Enter START, Esc ABORT, H HOME, T TRIGGER,
                          A ARM / DISARM, N STEP (next work state), K ACK, P next profile,
                          1-9 station configuration
    HTTP                  POST /api/cmd/<start|abort|home|trigger|step|ack|retry|skip|arm|
                          disarm|reset|config|profile|mode> (a PLC / HMI gateway or an
                          automation flow); sensor claims: POST /api/io/inputs {signal: true |
                          false}
    hardware (PLACEHOLDER) InputBackend subclasses below: camera GPIO lines, serial /
                          USB-IO, a 24 V digital I/O module, and the industrial networks
                          (Modbus TCP, OPC UA, EtherNet/IP). Config-driven mapping
                          (stations/io_example.json). Not implemented: they refuse to start.

States and rules:
    READY     idle, configuration / profile selectable   START -> a cycle, ARM -> ARMED
    ARMED     waiting for TRIGGER (part present)   TRIGGER -> a cycle, DISARM -> READY
    RUNNING   the DLP verifies a work state        ABORT -> ABORTING; done -> the next one
    WAITING   a cycle is open and nothing runs: the current work state waits for STEP /
              TRIGGER, or a prompt waits for ACK   ABORT -> READY (the cycle is ABORTED)
    ABORTING  abort requested, run winding down    done -> READY (or HOMING if HOME pressed)
    HOMING    rig check                            OK -> READY, MOVED / error -> FAULT
    FAULT     rig check failed / a job crashed     HOME (re-check) or RESET -> READY
ABORT always wins (it is never refused, and in every state it also asks the owner to stop
whatever it runs outside the state machine: a projected verdict, a live view, jobs started
directly); HOME only when idle or after an abort (a HOME pressed while aborting runs once the
aborted job ends - a further ABORT cancels it - and is never kept for later); TRIGGER is
ignored unless ARMED or a work state waits for it; the station configuration and the profile
can't change while a cycle runs, and changing either while ARMED disarms (the operator's
confirmation was for the old one). START / TRIGGER / ARM may name the configuration (and
profile) the sender expects: refused when the selected one differs. A configuration the
owner marked unusable (its file is invalid or gone, the profile is not in it) is never
started: refused, result INVALID. A cancelled or aborted cycle never re-arms.

Operating mode (engine/modes/; SELECT_MODE, POST /api/cmd/mode {"mode": ...}): `station`
(the default: everything in this module), `quick_check` or `measure`. The bus runs cycles
only in Station mode. In the other two the owner runs its measurement / quick-check jobs
itself and the bus stands aside: START / TRIGGER / ARM are refused, READY and PASS are off
and no permission is granted, so a measurement (which has no verdict at all) or a quick
check never reaches the line's outputs. ABORT and HOME (the rig check) work in every mode.
The mode can't change while a cycle runs; a change forgets the last cycle (as another
configuration does) and, while ARMED, disarms.

The last cycle speaks only for the part it checked. A START / TRIGGER that starts no cycle
while none is open (refused, or ignored because the bus is not armed) announces a part that
was NOT checked, so the last cycle is released: every permission drops, and a cycle that
ended CONTINUE is forgotten with its PASS output, its PASS code and its work-state registers
(a FAIL / PULL stays until the next cycle: they release nothing). ARM, DISARM, HOME and a
flow edited on disk release it the same way; SELECT_CONFIG / SELECT_PROFILE to another one
forgets the last cycle altogether (no PASS, no FAIL, no PULL, result_code 0). A PLC must
therefore act only on what follows a command the bus ACCEPTED (the reply, or the BUSY edge).

The cycle (engine/flow/cycle.py): START begins at the first work state; STEP / TRIGGER
verifies the current one (a work state with trigger `auto` is verified as soon as it is
current): the sensor inputs are read, the DLP scans, and the work state comes out
    CONFIRMED  sensor and DLP agree and every check passes: its permission is granted, the
               next work state becomes current (after the last one: CONTINUE)
    CORRECTED  the DLP disagrees with a sensor: SENSOR_MISMATCH, permission withheld
    NOT_MET    sensor and DLP agree on something the work state does not expect
    DEFECT     the part is there and a check fails
    DOUBT      the DLP cannot tell: PULL, never a permission
and its actions run: project_fault, prompt (PROMPT on until ACK), hold (ACK, then verify
again), retry, kickout REJECT | PULL, goto. ACK acknowledges a prompt / confirms the removal;
RETRY scans the current work state again (its retry budget); SKIP leaves it unconfirmed, only
where the flow allows it (never a DOUBT / CORRECTED one). A single-job station config is a
one-step flow: START runs its job, exactly as before.

The DLP answer the owner reports (job_finished(..., step=<answer>)) must say which verify it
answers: `step` and `attempt` as the bus asked (and `cycle_id` when the cycle has one). An
answer for another work state, attempt or cycle - or none at all - confirms nothing: DOUBT.

Sensor claims (set_inputs; HTTP POST /api/io/inputs; an input backend's `signals` map):
{signal: true | false}, null = unknown. A claim belongs to ONE cycle: every claim is
forgotten when a cycle ends, on ABORT, HOME, FAULT and a configuration / profile change, so
the line reports its sensors for every part (before START / TRIGGER, or before the STEP that
needs them); a configuration's parameters.io.max_age_s also limits how old a claim may be
when a work state is verified. A work state that expects a signal nobody reported (for this
cycle, recently enough) is a DOUBT: a dead gateway is never "sensor and DLP agree". So is a
signal that changed while the DLP scanned, even when it ended where it started. A name no
loaded station configuration declares is refused (a typo never becomes a silent sensor).

Outputs (stack-light style): READY / BUSY / PASS / FAIL / ALARM / PULL / KICKOUT / PROMPT /
SENSOR_MISMATCH booleans and one PERMIT_<NAME> per permission - of the selected configuration
and of every other one this bus has known, which stay published as false (a permission of a
configuration that was deselected, renamed or broke is switched OFF on the line, never just
left out). KICKOUT is on with FAIL or PULL: the last cycle ended REJECT or PULL, by a
kickout action or because nothing confirmed it, and its part leaves the line. All are
published to every OutputBackend on each change (the UI lamp reads the same signals), plus
registers for a PLC: result_code (0 none, PASS 1, FAIL 2, INVALID 3, ABORTED 4, PULL 5 - a
recipe may renumber them in outputs.result_codes), state_code (index in STATES),
station_config (1-based, 0 none), profile (1-based within the configuration, 0 none),
work_state (1-based current / last work state of the cycle, 0 = no cycle yet) and
work_state_code (its outcome: 0 not verified yet, CONFIRMED 1, CORRECTED 2, NOT_MET 3,
DEFECT 4, DOUBT 5, SKIPPED 6, ABORTED 7).

Permissions: PERMIT_<NAME> is true only while its work state is CONFIRMED in the current
cycle (and every required work state before it is). They all drop when a new cycle starts,
when the cycle ends anything but CONTINUE, on ABORT, FAULT, HOME, ARM, DISARM, a START /
TRIGGER that starts no cycle, a configuration / profile change, a flow edited on disk, and
(a flow's `revoke`) when a later work state is not confirmed. Nothing grants one on a doubt.

Dispositions (engine/disposition.py) on top of the verdict: PASS is set only for CONTINUE (a
PASS with no doubt; a cycle: every required work state CONFIRMED), FAIL for REJECT (a
definite FAIL; a cycle: a kickout REJECT), PULL for every doubt - INVALID, ABORTED, a crashed
or never-started job, a PASS in the guard band, an unreadable code, a work state that was
not confirmed: the part goes to the quality-inspection lane. PULL never sets PASS.
result_code shows the PASS code only for CONTINUE, the FAIL code for REJECT, INVALID /
ABORTED as such, and the PULL code for a PULL whose verdict was PASS or FAIL
(engine/disposition.register_key). ALARM is on in FAULT, and while idle after an INVALID
result until it is acknowledged: RESET (also accepted while idle with the ALARM on) or a
good HOME (rig check OK) switches ALARM off; the next INVALID result or FAULT raises it
again. An acknowledged ALARM changes nothing else: the last cycle's PULL / FAIL output and
result code stay until the next cycle starts (they release nothing; a line reads PULL as
"this part goes to quality inspection", not as "the station is faulted"). A PASS belongs to
the cycle that just ended: a rig check (HOME) ends it, so the PASS output and the PASS code
never come back for that cycle (after HOME OK, or a FAULT and RESET). An ABORT while a run
is busy makes that cycle ABORTED (PULL) whatever the job reports after; a START whose
handler fails leaves INVALID (PULL) and a FAULT.

A recipe configuration (a single-job config with job.type recipe) pins its recipe's preset,
and its result codes come from the recipe; a profile pins its own preset.

Deprecated: SELECT_WORKSTATE (and POST /api/cmd/workstate, the `workstate` body key) is the
old name of SELECT_CONFIG - it selected a station configuration, which was never a work
state. Still accepted as an alias; its reply says so.

Pure: no engine, no hardware. The owner (the web server) passes handlers that start /
abort the actual jobs and calls job_finished() when they end (also for a job dropped before
it ran); job_lost() settles a busy state whose job can no longer report back; tick() ends a
work state that waited longer than its timeout.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .. import disposition as D
from ..flow import schema as FS
from ..flow.cycle import OUTCOME_CODES, Cycle, CycleError
from ..modes import DEFAULT_MODE, MODES, TITLES, check_mode

STATES = ("READY", "ARMED", "RUNNING", "ABORTING", "HOMING", "FAULT", "WAITING")
COMMANDS = ("START", "ABORT", "HOME", "TRIGGER", "STEP", "ACK", "RETRY", "SKIP",
            "SELECT_CONFIG", "SELECT_PROFILE", "SELECT_MODE", "ARM", "DISARM", "RESET")
# old names still accepted (the reply says so): SELECT_WORKSTATE selected a station
# configuration, never a work state
DEPRECATED_COMMANDS = {"SELECT_WORKSTATE": "SELECT_CONFIG"}
SIGNALS = ("READY", "BUSY", "PASS", "FAIL", "ALARM", "PULL", "KICKOUT", "PROMPT",
           "SENSOR_MISMATCH")
PERMIT_PREFIX = "PERMIT_"                   # + a permission name of the selected flow
IDLE = ("READY", "ARMED")
BUSY = ("RUNNING", "ABORTING", "HOMING")    # a job runs and will report back
IN_CYCLE = ("RUNNING", "ABORTING", "WAITING")
RESULT_CODES = {"PASS": 1, "FAIL": 2, "INVALID": 3, "ABORTED": 4, "PULL": 5}  # 0 = none yet
REGISTERS = ("result_code", "state_code", "station_config", "profile", "work_state",
             "work_state_code")
MAX_INPUTS = 256
_SIGNAL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_PERMIT_SIGNAL = re.compile(r"^PERMIT_[A-Z][A-Z0-9_]{0,31}$")


def is_signal(name) -> bool:
    """An output signal name: one of SIGNALS, or PERMIT_<NAME>."""
    return isinstance(name, str) and (name in SIGNALS or bool(_PERMIT_SIGNAL.fullmatch(name)))


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
    """What the bus asks its owner to do. start(config, confirmed) -> job id (the job of a
    single-job configuration); verify(request) -> job id (the DLP check of one work state;
    the owner reports it with job_finished("run", ..., step=<the DLP answer>)); abort() and
    home() -> anything (ids, for the log). home() reads the HOME request's options from
    bus.home_args while it runs."""
    start: Callable = lambda config, confirmed: None
    abort: Callable = lambda: None
    home: Callable = lambda: None
    # select(config, preset) after SELECT_CONFIG / SELECT_PROFILE is accepted: the owner
    # makes the preset the active profile pins the loaded one; None = it names none
    select: Callable = lambda config, preset: None
    # pending() -> True while a job the bus started will still report back (job_finished),
    # False when none will (it was dropped / lost), None = the owner doesn't track jobs
    pending: Callable = lambda: None
    # verify(request): run the DLP check of the current work state. request = {config,
    # station, profile, step, index, attempt, inputs (the sensor claims read now),
    # confirmed, + whatever cycle_begin returned}. The answer reported back must carry the
    # request's step and attempt (and cycle_id when there is one): no other answer confirms
    verify: Callable = lambda request: None
    # project(request): paint that work state's fault on the part ({step, run_id, ...})
    project: Callable = lambda request: None
    # cycle_begin(info) -> dict kept with the cycle (its id, its folder); cycle_end(record)
    # when it ended (the owner writes the cycle's result); log(meta, event) per cycle event
    cycle_begin: Callable = lambda info: None
    cycle_end: Callable = lambda record: None
    log: Callable = lambda meta, event: None


class CommandBus:
    def __init__(self, handlers: Optional[Handlers] = None, configs: Optional[list] = None,
                 require_confirmation: bool = False, clock: Callable = time.monotonic):
        self.h = handlers or Handlers()
        self.configs = list(configs or [])          # the station configurations (file names)
        self.require_confirmation = bool(require_confirmation)
        self.clock = clock
        self.state = "READY"
        self.mode: str = DEFAULT_MODE                # station | quick_check | measure
        self.config: Optional[str] = self.configs[0] if self.configs else None
        self.flows: dict = {}                        # config -> StationConfiguration (owner)
        self.profile: Optional[str] = None           # the active profile of self.config
        self.config_presets: dict = {}               # config -> the preset it pins
        self.config_info: dict = {}                  # config -> {recipe, result_codes}
        self.unusable: dict = {}                     # config -> why it can't run (owner)
        self.cycle: Optional[Cycle] = None           # the open cycle, else the last one
        self.inputs: dict = {}                       # sensor claims: signal -> True | False
        self._input_at: dict = {}                    # signal -> clock() of its last report
        self._input_seq: dict = {}                   # signal -> how often its value changed
        self._seq_at: dict = {}                      # ... when the running verify started
        self._asked: Optional[dict] = None           # the verify the bus waits for an answer to
        self._permits: dict = {}                     # every permission name ever known (ordered)
        self.armed_confirmed = False
        self.last_result: Optional[str] = None      # PASS | FAIL | INVALID | ABORTED
        self.last_disposition: Optional[str] = None  # CONTINUE | REJECT | PULL
        self.alarm_acknowledged = False              # RESET / a good HOME since the last result
        self.fault: Optional[str] = None
        self.home_after_abort = False                # only ever True while ABORTING
        self.home_args: dict = {}                    # the HOME request's options (owner reads)
        self._busy_kind: Optional[str] = None        # 'run' | 'home': what the busy state awaits
        self._return_armed = False
        self._confirmed = False                      # the open cycle's operator confirmation
        self._source = "api"
        self._implicit: dict = {}                    # config -> its implicit one-step flow
        self.outputs: list = []
        self.history: list = []                      # (time, source, command, accepted, msg)
        self._lock = threading.RLock()
        self.profile = self._default_profile()

    # ------------------------------------------------------------ configurations
    def flow_of(self, config: Optional[str]) -> Optional[FS.StationConfiguration]:
        """The station configuration behind a config name: the owner's parsed one, else (the
        owner gave none: tests, a bare bus) a single-job configuration with one implicit
        profile and a one-step flow."""
        if config is None:
            return None
        cfg = self.flows.get(config)
        if cfg is None:
            cfg = self._implicit.get(config)
            if cfg is None:
                cfg = self._implicit[config] = FS.lift_legacy(
                    {"station_id": Path(str(config)).stem or "station"})
        return cfg

    def _default_profile(self) -> Optional[str]:
        cfg = self.flow_of(self.config)
        return cfg.default_profile if cfg is not None else None

    def profiles(self) -> list:
        cfg = self.flow_of(self.config)
        return cfg.profile_names if cfg is not None else []

    def permission_names(self) -> list:
        cfg = self.flow_of(self.config)
        return cfg.permissions if cfg is not None else []

    def run_preset(self) -> Optional[str]:
        """The preset the selected configuration + profile pin (None: none named)."""
        cfg = self.flows.get(self.config)
        own = cfg.preset_for(self.profile) if cfg is not None and not cfg.legacy else None
        return own or self.config_presets.get(self.config)

    # ------------------------------------------------------------ outputs
    def signals(self) -> dict:
        s, d, cyc = self.state, self.last_disposition, self.cycle
        sig = {"READY": s in IDLE and self.fault is None and self.mode == DEFAULT_MODE,
               "BUSY": s in BUSY,
               "PASS": s in IDLE and self.last_result == "PASS" and d == "CONTINUE",
               "FAIL": s in IDLE and d == "REJECT",
               "ALARM": s == "FAULT" or (s in IDLE and self.last_result == "INVALID"
                                         and not self.alarm_acknowledged),
               "PULL": s in IDLE and d == "PULL",
               # the part of the last cycle leaves the line (REJECT or PULL), whether a
               # kickout action said so or nothing confirmed it
               "KICKOUT": s in IDLE and d in ("REJECT", "PULL"),
               "PROMPT": s == "WAITING" and cyc is not None and cyc.prompt is not None,
               "SENSOR_MISMATCH": cyc is not None and cyc.sensor_mismatch}
        live = cyc.permissions if (cyc is not None and s != "FAULT") else {}
        mine = set(self.permission_names())
        for name in self.permit_names():    # every name ever known: the others are OFF
            sig[PERMIT_PREFIX + name] = name in mine and live.get(name) is True
        return sig

    def permit_names(self) -> list:
        """Every permission name the outputs carry: the selected configuration's, every
        loaded configuration's, and every one published before (a deselected, renamed or
        broken configuration's permissions are published false, never left out)."""
        for cfg in self.flows.values():
            for n in cfg.permissions:
                self._permits.setdefault(n, None)
        for n in self.permission_names():
            self._permits.setdefault(n, None)
        return list(self._permits)

    def permissions(self) -> dict:
        """permission name -> granted now (what the PERMIT_<NAME> outputs say)."""
        sig = self.signals()
        return {n: sig[PERMIT_PREFIX + n] for n in self.permission_names()}

    def registers(self) -> dict:
        """Numbers for a PLC: result_code (the configuration's recipe codes, else
        RESULT_CODES; 0 = none yet; the code of engine/disposition.register_key: PASS only
        for CONTINUE), state_code (index in STATES), station_config and profile (1-based, 0 =
        none), work_state (1-based current / last work state of the cycle, 0 = no cycle yet)
        and work_state_code (its outcome, engine/flow/cycle.OUTCOME_CODES)."""
        codes = (self.config_info.get(self.config) or {}).get("result_codes") or RESULT_CODES
        n = self.configs.index(self.config) + 1 if self.config in self.configs else 0
        profs = self.profiles()
        key = D.register_key(self.last_result, self.last_disposition)
        code = 0
        if key is not None:
            code = codes.get(key)
            if code is None:                    # older codes without PULL: never the PASS code
                code = codes.get("INVALID", RESULT_CODES["PULL"]) if key == "PULL" else \
                    RESULT_CODES.get(key, 0)
        row = self.cycle.row if self.cycle is not None else None
        return {"result_code": int(code), "state_code": STATES.index(self.state),
                "station_config": n,
                "profile": profs.index(self.profile) + 1 if self.profile in profs else 0,
                "work_state": int(row["index"]) if row is not None else 0,
                "work_state_code": OUTCOME_CODES.get(row["outcome"], 0) if row is not None
                else 0}

    def snapshot(self) -> dict:
        with self._lock:
            cfg = self.flow_of(self.config)
            cyc = self.cycle
            return {"state": self.state,
                    "operating_mode": self.mode, "operating_modes": list(MODES),
                    "station_config": self.config, "station_configs": list(self.configs),
                    "unusable": dict(self.unusable),
                    "profile": self.profile,
                    "profiles": [p.public() for p in cfg.profiles.values()] if cfg else [],
                    "config_preset": self.run_preset(),
                    "config_recipe": (self.config_info.get(self.config) or {}).get("recipe"),
                    "legacy": bool(cfg.legacy) if cfg else None,
                    "flow": [s.public() for s in cfg.flow] if cfg else [],
                    "cycle": cyc.record() if cyc is not None else None,
                    "prompt": dict(cyc.prompt) if cyc is not None and cyc.prompt else None,
                    "permissions": self.permissions(),
                    "inputs": dict(self.inputs),
                    "input_age_s": {k: round(max(0.0, self.clock() - self._input_at[k]), 1)
                                    for k in self.inputs if k in self._input_at},
                    "input_max_age_s": cfg.input_max_age_s if cfg else None,
                    "signal_names": list(cfg.signals) if cfg else [],
                    "registers": self.registers(),
                    "last_result": self.last_result,
                    "last_disposition": self.last_disposition,
                    "alarm_acknowledged": self.alarm_acknowledged,
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

    def _result(self, result: Optional[str], disposition: Optional[str]) -> None:
        """The cycle's result (None, None: a new cycle, nothing yet). A new result is a new
        ALARM to acknowledge: an earlier RESET / good HOME does not cover it."""
        self.last_result, self.last_disposition = result, disposition
        self.alarm_acknowledged = False

    def _go(self, state: str) -> None:
        if state != "ABORTING" and self.home_after_abort:
            # a HOME pressed while aborting lives only as long as that abort: it runs when
            # the aborted job ends (job_finished), or it is forgotten - never fired later
            self.home_after_abort = False
            self.home_args = {}
        if state == "FAULT":
            self._forget_inputs("FAULT")
            self._drop_permissions("FAULT")
        self.state = state
        self._publish()

    def _drop_permissions(self, why: str) -> bool:
        """Every permission off (ABORT, FAULT, HOME, a configuration / profile change ...).
        -> True when one was on."""
        cyc = self.cycle
        if cyc is None:
            return False
        dropped = cyc.drop_permissions(why)
        if cyc.state == "DONE":
            cyc.drain_events()                  # its log is closed: the history says it
        if dropped:
            self._log("bus", "PERMISSIONS", True, f"dropped {', '.join(dropped)}: {why}")
        return bool(dropped)

    def _release(self, why: str) -> bool:
        """The last cycle no longer speaks for the part at the station (a START / TRIGGER
        that started no cycle: a part was announced and not checked; ARM / DISARM; HOME; the
        flow changed on disk): every permission drops, and a cycle that ended CONTINUE is
        forgotten with its PASS output, its PASS code and its work-state registers. A FAIL /
        PULL stays until the next cycle starts (they release nothing). Never called while a
        cycle is open. -> True when an output may have changed."""
        changed = self._drop_permissions(why)
        if self.last_disposition == "CONTINUE":
            self._result(None, None)
            self.cycle = None
            self._log("bus", "RELEASED", True, f"the last cycle's PASS ended: {why}")
            changed = True
        return changed

    def _forget_inputs(self, why: str) -> None:
        """A sensor claim belongs to one cycle: every claim is forgotten when it ends (and on
        ABORT, HOME, FAULT, a configuration / profile change), so the next work state that
        expects a signal needs a fresh report - a dead gateway is a DOUBT, never an old
        claim read as agreement."""
        if not self.inputs:
            return
        for k in self.inputs:
            self._input_seq[k] = self._input_seq.get(k, 0) + 1
        self.inputs, self._input_at = {}, {}
        self._log("bus", "INPUTS", True, f"forgotten: {why}")

    def _claims(self, cfg) -> dict:
        """The sensor claims a verify may read now: those reported for this cycle and (the
        configuration's parameters.io.max_age_s) recently enough."""
        limit = cfg.input_max_age_s if cfg is not None else None
        now = self.clock()
        return {k: v for k, v in self.inputs.items()
                if limit is None or now - self._input_at.get(k, now) <= limit}

    def known_signals(self) -> list:
        """Every sensor signal a loaded station configuration declares or expects."""
        out: dict = {}
        for cfg in self.flows.values():
            for s in cfg.signals:
                out.setdefault(s, None)
        return list(out)

    # ------------------------------------------------------------ sensor claims
    def set_inputs(self, values: dict, source: str = "api") -> dict:
        """Sensor claims from the line: {signal: true | false}; null forgets one (unknown).
        Only real booleans count, and only signals a loaded station configuration declares
        (when any does). -> every claim now held. ValueError on anything else: nothing is
        applied."""
        if not isinstance(values, dict):
            raise ValueError("sensor inputs must be a mapping {signal: true | false}")
        if len(values) > MAX_INPUTS:
            raise ValueError(f"more than {MAX_INPUTS} sensor signals")
        clean = {}
        for k, v in values.items():
            if not isinstance(k, str) or not _SIGNAL.fullmatch(k):
                raise ValueError(f"sensor signal name {k!r}: letters, digits, _ . - (max 64)")
            if v is not None and not isinstance(v, bool):
                raise ValueError(f"sensor input {k} must be a JSON boolean (true / false) or "
                                 f"null (unknown), got {type(v).__name__} {v!r}")
            clean[k] = v
        with self._lock:
            known = self.known_signals()
            stray = sorted(k for k in clean if known and k not in known)
            if stray:
                raise ValueError(f"sensor signal(s) {', '.join(stray)}: not one a station "
                                 f"configuration expects ({', '.join(known)}); check the name "
                                 "(parameters.io.inputs)")
            new = {**self.inputs, **clean}
            new = {k: v for k, v in new.items() if v is not None}
            if len(new) > MAX_INPUTS:
                raise ValueError(f"more than {MAX_INPUTS} sensor signals")
            now = self.clock()
            for k, v in clean.items():
                if self.inputs.get(k) is not v:     # a change (also to / from unknown)
                    self._input_seq[k] = self._input_seq.get(k, 0) + 1
                if v is None:
                    self._input_at.pop(k, None)
                else:
                    self._input_at[k] = now         # a repeated report keeps it fresh
            self.inputs = new
            self._log(source, "INPUTS", True, ", ".join(
                f"{k}={'unknown' if v is None else str(v).lower()}" for k, v in clean.items()))
            return dict(self.inputs)

    # ------------------------------------------------------------ commands
    def command(self, name: str, source: str = "api", arg=None, confirmed: bool = False) -> Reply:
        """confirmed: only a real True counts (a "false" string from a gateway is not a yes)."""
        name = str(name).upper()
        with self._lock:
            self._tick()
            self._source = str(source)
            canon = DEPRECATED_COMMANDS.get(name, name)
            if canon not in COMMANDS:
                r = Reply(False, name, self.state, f"unknown command (one of {', '.join(COMMANDS)})")
            else:
                r = getattr(self, "_c_" + canon.lower())(arg, confirmed is True)
                if canon != name:
                    r.message = (r.message + " " if r.message else "") + \
                        f"[{name} is deprecated: use {canon}]"
            self._log(source, name, r.accepted, r.message)
            return r

    def _no(self, cmd, msg) -> Reply:
        return Reply(False, cmd, self.state, msg)

    def _ok(self, cmd, msg="") -> Reply:
        return Reply(True, cmd, self.state, msg)

    def _unusable(self, config: Optional[str]) -> Optional[str]:
        """Why station configuration `config` can't run (None = it can)."""
        if self.unusable.get(config):
            return self.unusable[config]
        if config not in self.configs:
            return "it is not one of the station configs any more"
        if config == self.config and self.profile not in self.profiles():
            return (f"profile {self.profile!r} is not one of its profiles "
                    f"({', '.join(self.profiles()) or 'none'}): select a profile")
        return None

    def _expect(self, cmd: str, arg) -> Optional[Reply]:
        """START / TRIGGER / ARM naming the station configuration (and profile) the sender
        expects (what the operator page shows): refused when the selected one differs.
        arg: a configuration name / 1-based number, or {"config": ..., "profile": ...}."""
        want_cfg, want_prof = arg, None
        if isinstance(arg, dict):
            want_cfg, want_prof = arg.get("config"), arg.get("profile")
        if want_cfg is not None and want_cfg != "" and self.resolve_config(want_cfg) != self.config:
            return self._no(cmd, f"the selected station configuration is {self.config}, not "
                                 f"{want_cfg!r}: check the setup for the selected one first")
        if want_prof is not None and want_prof != "" and \
                self.resolve_profile(want_prof) != self.profile:
            return self._no(cmd, f"the selected profile is {self.profile}, not {want_prof!r}: "
                                 "check the setup for the selected one first")
        return None

    def _not_started(self, cmd: str, msg: str) -> Reply:
        """A START / TRIGGER that starts no cycle. While none is open, a part was announced
        and is not checked: the last cycle no longer speaks for what is at the station
        (_release) - its permissions and its PASS never stand in for the new part."""
        if self.state not in IN_CYCLE and self._release(f"{cmd} started no cycle"):
            self._publish()
        return self._no(cmd, msg)

    def _held(self) -> str:
        """Why no cycle starts outside Station mode."""
        return (f"the station is in {TITLES[self.mode]} mode (no cycle, no verdict on the line, "
                "no permission): select Station mode to run cycles (SELECT_MODE)")

    def _begin(self, cmd: str, confirmed: bool, arg=None) -> Reply:
        if self.mode != DEFAULT_MODE:
            return self._not_started(cmd, self._held())
        if self.config is None:
            return self._not_started(cmd, "no station configuration selected")
        refused = self._expect(cmd, arg)
        if refused is not None:
            return self._not_started(cmd, refused.message)
        why = self._unusable(self.config)
        if why:                                 # never started: the cycle's result is INVALID
            self._drop_permissions(f"{cmd} refused")
            self.cycle = None                   # the last cycle is not this part's
            self._result("INVALID", "PULL")
            if self.state == "ARMED":
                self.armed_confirmed = False
                self._go("READY")
            else:
                self._publish()
            return self._no(cmd, f"station configuration {self.config} is not usable: {why}")
        if self.require_confirmation and not confirmed:
            return self._not_started(cmd, "the real rig needs the operator's confirmation of "
                                          "the setup")
        cfg = self.flow_of(self.config)
        self._return_armed = self.state == "ARMED"
        self._result(None, None)
        self._confirmed = confirmed
        cyc = self.cycle = Cycle(cfg, self.profile, clock=self.clock)   # a new cycle: no
        if not cyc.legacy:                                              # permission yet
            try:
                meta = self.h.cycle_begin({"config": self.config, "station": cfg.station_id,
                                           "profile": self.profile, "confirmed": confirmed,
                                           "source": self._source})
            except Exception as e:          # noqa: BLE001 - the cycle could not be opened
                return self._start_failed(cmd, e)
            cyc.meta.update(meta if isinstance(meta, dict) else {})
        self._apply(cyc.begin(force=(cmd == "TRIGGER")))
        if self.state == "FAULT":
            return self._no(cmd, self.fault)
        if cyc.legacy:
            return self._ok(cmd, f"started {self.config}")
        return self._ok(cmd, f"cycle started: {self.config}, profile {self.profile}; "
                        + self._where())

    def _where(self) -> str:
        """What the open cycle is doing now (a reply's message)."""
        cyc = self.cycle
        if cyc is None or cyc.row is None:
            return "no cycle"
        row = cyc.row
        at = f"work state {row['index']}/{len(cyc.rows)} {row['id']}"
        if cyc.state == "VERIFYING":
            return f"verifying {at}"
        if cyc.state == "WAIT_STEP":
            return f"{at} waits for STEP"
        if cyc.state == "WAIT_ACK":
            return f"{at} is {row['outcome']}: {(cyc.prompt or {}).get('message', '')} (ACK)"
        return f"cycle ended {cyc.disposition}" + (
            f" ({cyc.kickout['reason']})" if cyc.kickout else "")

    def _start_failed(self, cmd: str, e: BaseException) -> Reply:
        """The job / cycle could not start: never started, so the cycle's result is INVALID,
        its part PULLed, and the station is in FAULT."""
        self.fault = f"start failed: {type(e).__name__}: {e}"
        cyc = self.cycle
        if cyc is not None and cyc.state != "DONE":
            if cyc.legacy:
                cyc.legacy_done("INVALID", "PULL")
            else:
                cyc.fail(self.fault)
                self._cycle_closed()
        self._result("INVALID", "PULL")
        self._go("FAULT")
        return self._no(cmd, self.fault)

    def _cycle_events(self) -> None:
        cyc = self.cycle
        if cyc is None:
            return
        for ev in cyc.drain_events():
            if cyc.legacy:
                continue
            try:
                self.h.log(dict(cyc.meta), ev)
            except Exception as e:          # noqa: BLE001 - a log must not stop the bus
                self._log("bus", "CYCLE_LOG", False, f"{type(e).__name__}: {e}")

    def _cycle_closed(self) -> None:
        """The cycle ended: its events and its record go to the owner (who writes its
        result); never raises."""
        cyc = self.cycle
        self._cycle_events()
        if cyc is None or cyc.legacy:
            return
        try:
            self.h.cycle_end(cyc.record())
        except Exception as e:              # noqa: BLE001 - the outputs already say it
            self._log("bus", "CYCLE_END", False, f"{type(e).__name__}: {e}")

    def _apply(self, effects: list) -> None:
        """Act on what the cycle asked for (engine/flow/cycle.py effects)."""
        cyc = self.cycle
        for name, payload in effects:
            if name == "verify":
                self._busy_kind = "run"
                self._go("RUNNING")
                try:
                    if cyc.legacy:
                        self.h.start(self.config, self._confirmed)
                    else:
                        self._cycle_events()
                        req = cyc.request(self._claims(cyc.cfg))
                        # the answer must name this verify (job_finished -> _step_finished)
                        self._asked = {"step": req["step"], "attempt": req["attempt"],
                                       "cycle_id": cyc.meta.get("cycle_id")}
                        self._seq_at = dict(self._input_seq)
                        self.h.verify({**req, "confirmed": self._confirmed})
                except Exception as e:      # noqa: BLE001 - the start itself was refused
                    self._start_failed("START", e)
                    return
            elif name == "project":
                try:
                    self.h.project({**payload, **cyc.meta})
                except Exception as e:      # noqa: BLE001 - a picture never stops the flow
                    self._log("bus", "PROJECT", False, f"{type(e).__name__}: {e}")
            elif name == "wait":
                self._go("WAITING")
            elif name == "done":
                self._result(cyc.result, cyc.disposition)
                self._cycle_closed()
                self._forget_inputs("the cycle ended")
                keep = self._return_armed and cyc.result != "ABORTED"
                if not keep:
                    self.armed_confirmed = False
                self._go("ARMED" if keep else "READY")
        self._cycle_events()

    def _c_start(self, arg, confirmed):
        if self.state not in IDLE:
            return self._not_started("START", f"not idle ({self.state})" + (
                ": a cycle is open - STEP / ACK it, or ABORT" if self.state == "WAITING" else ""))
        return self._begin("START", confirmed, arg)

    def _c_trigger(self, arg, confirmed):
        if self.state == "WAITING" and self.cycle is not None and self.cycle.waiting == "step":
            return self._step("TRIGGER", arg)
        if self.state != "ARMED":
            return self._not_started("TRIGGER", f"ignored: not armed ({self.state})")
        return self._begin("TRIGGER", confirmed or self.armed_confirmed, arg)

    def _waiting(self, cmd: str, what: str) -> Optional[Reply]:
        """STEP needs a work state waiting for it, ACK / RETRY a prompt: else refused."""
        cyc = self.cycle
        if self.state != "WAITING" or cyc is None or not cyc.active:
            return self._no(cmd, f"no cycle is waiting ({self.state})")
        if cyc.waiting != what:
            return self._no(cmd, "the cycle waits for " + (
                "ACK: a prompt is open" if cyc.waiting == "ack" else "STEP, not for an "
                "acknowledgement") + f" ({self._where()})")
        return None

    def _step(self, cmd: str, arg) -> Reply:
        refused = self._waiting(cmd, "step")
        if refused is not None:
            return refused
        cyc = self.cycle
        want = arg.get("step") if isinstance(arg, dict) else arg
        if want not in (None, "") and str(want) not in (cyc.row["id"], str(cyc.row["index"])):
            return self._no(cmd, f"the current work state is {cyc.row['index']} "
                                 f"({cyc.row['id']}), not {want!r}")
        self._apply(cyc.step_now())
        if self.state == "FAULT":
            return self._no(cmd, self.fault)
        return self._ok(cmd, self._where())

    def _c_step(self, arg, confirmed):
        return self._step("STEP", arg)

    def _c_ack(self, arg, confirmed):
        refused = self._waiting("ACK", "ack")
        if refused is not None:
            return refused
        self._apply(self.cycle.ack(self._source))
        if self.state == "FAULT":
            return self._no("ACK", self.fault)
        return self._ok("ACK", "acknowledged; " + self._where())

    def _c_retry(self, arg, confirmed):
        refused = self._waiting("RETRY", "ack")
        if refused is not None:
            return refused
        try:
            effects = self.cycle.retry(self._source)
        except CycleError as e:
            return self._no("RETRY", str(e))
        self._apply(effects)
        if self.state == "FAULT":
            return self._no("RETRY", self.fault)
        return self._ok("RETRY", self._where())

    def _c_skip(self, arg, confirmed):
        cyc = self.cycle
        if self.state != "WAITING" or cyc is None or not cyc.active:
            return self._no("SKIP", f"no cycle is waiting ({self.state})")
        try:
            effects = cyc.skip(self._source)
        except CycleError as e:
            return self._no("SKIP", str(e))
        self._apply(effects)
        if self.state == "FAULT":
            return self._no("SKIP", self.fault)
        return self._ok("SKIP", "skipped (its permission is not granted); " + self._where())

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
        """Never refused: it preempts whatever runs, in every state. Every permission drops
        and every sensor claim is forgotten."""
        self._forget_inputs("ABORT")
        if self.state in ("RUNNING", "HOMING"):
            self._drop_permissions("ABORT")
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
        elif self.state == "WAITING":           # an open cycle with nothing running: it ends
            cyc = self.cycle
            if cyc is not None and cyc.active:
                cyc.abort("aborted: ABORT while the cycle waited")
                self._cycle_closed()
            self._result("ABORTED", "PULL")
            self.armed_confirmed = False
            self._go("READY")
            note = self._owner_abort()
            msg = "cycle aborted"
        elif self.state == "ARMED":
            self.armed_confirmed = False
            self._release("ABORT")
            self._go("READY")
            note = self._owner_abort()
            msg = "disarmed"
        else:                                   # idle: nothing of the bus's runs. The last
            if self._drop_permissions("ABORT"):  # cycle's result stays (it is that part's
                self._publish()                 # own); its permissions do not
            note = self._owner_abort()
            msg = "" if note else "nothing running"
        if self.state == "ABORTING" and self.h.pending() is False:
            # nothing left that will report back (a job dropped without a word): settle now
            self.job_lost("aborted: no job left that could report back")
            msg += f"; nothing left running ({self.state})"
        return self._ok("ABORT", "; ".join(m for m in (msg, note) if m))

    def _start_home(self) -> bool:
        """HOMING + h.home() (which reads home_args); a refused home is a FAULT. A PASS
        (CONTINUE) ends here: the rig check never hands the old cycle's PASS output or PASS
        code back (FAIL / PULL stay: they release nothing), every permission drops and every
        sensor claim is forgotten."""
        self._forget_inputs("HOME")
        self._release("HOME")
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
        if self.state in ("RUNNING", "HOMING", "WAITING"):
            return self._no("HOME", f"refused while {self.state}: ABORT first")
        self.armed_confirmed = False
        self.home_args = opts
        if not self._start_home():
            return self._no("HOME", self.fault)
        return self._ok("HOME", "rig check running")

    def _select(self, cmd: str, note: str) -> Reply:
        """After the configuration / profile changed: the owner loads the preset it pins."""
        self._publish()
        preset = self.run_preset()
        what = self.config if cmd == "SELECT_CONFIG" else f"profile {self.profile}"
        try:
            self.h.select(self.config, preset)
        except Exception as e:              # noqa: BLE001 - selected, but its preset didn't load
            return self._ok(cmd, f"{what} (preset {preset}: {type(e).__name__}: {e}){note}")
        return self._ok(cmd, (f"{what} (preset {preset})" if preset else what) + note)

    def _c_select_config(self, arg, confirmed):
        """Select the station configuration (its parameters + profile library set). Refused
        while a cycle runs. A change selects the configuration's default profile, drops every
        permission and forgets the last cycle: its PASS / FAIL / PULL outputs, its result
        code, its work states and the sensor claims made for it."""
        if self.state not in IDLE + ("FAULT",):
            return self._no("SELECT_CONFIG", f"refused while {self.state}")
        name = self.resolve_config(arg)
        if name is None:
            return self._no("SELECT_CONFIG", f"no station configuration {arg!r}")
        why = self.unusable.get(name)
        if why:
            return self._no("SELECT_CONFIG", f"{name} is not usable: {why}")
        note = ""
        if name != self.config:
            if self.state == "ARMED":
                # the operator's confirmation (and the arm) was for the old configuration
                self.armed_confirmed = False
                self.state = "READY"
                note = " - disarmed: ARM again for this station configuration"
            self._drop_permissions("another station configuration was selected")
            self._forget_inputs("another station configuration was selected")
            self.config = name
            self.profile = self._default_profile()
            self.cycle = None
            self._result(None, None)            # the last cycle belonged to the other one
        return self._select("SELECT_CONFIG", note)

    def _c_select_profile(self, arg, confirmed):
        """Select the active profile of the loaded station configuration (which part variant
        the work states check: long tube / short tube). Refused while a cycle runs. A change
        drops every permission and forgets the last cycle: its PASS / FAIL / PULL outputs,
        its result code, its work states and the sensor claims made for it."""
        if self.state not in IDLE + ("FAULT",):
            return self._no("SELECT_PROFILE", f"refused while {self.state}: a cycle is "
                                              "running" if self.state in IN_CYCLE else
                            f"refused while {self.state}")
        if self.config is None:
            return self._no("SELECT_PROFILE", "no station configuration selected")
        name = self.resolve_profile(arg)
        if name is None:
            return self._no("SELECT_PROFILE", f"station configuration {self.config} has no "
                                              f"profile {arg!r} (profiles: "
                                              f"{', '.join(self.profiles()) or 'none'})")
        note = ""
        if name != self.profile:
            if self.state == "ARMED":
                self.armed_confirmed = False
                self.state = "READY"
                note = " - disarmed: ARM again for this profile"
            self._drop_permissions("another profile was selected")
            self._forget_inputs("another profile was selected")
            self.profile = name
            self.cycle = None
            self._result(None, None)            # the last cycle checked the other profile
        return self._select("SELECT_PROFILE", note)

    def _c_select_mode(self, arg, confirmed):
        """Select the operating mode (engine/modes/): station, quick_check or measure.
        Refused while a cycle runs or a job is busy. A change drops every permission and
        forgets the last cycle - its PASS / FAIL / PULL outputs, its result code, its work
        states and the sensor claims made for it - and, while ARMED, disarms. Outside Station
        mode no cycle starts and READY, PASS and every permission stay off."""
        if self.state not in IDLE + ("FAULT",):
            return self._no("SELECT_MODE", f"refused while {self.state}" + (
                ": a cycle is running" if self.state in IN_CYCLE else ""))
        want = arg.get("mode") if isinstance(arg, dict) else arg
        try:
            mode = check_mode(want)
        except ValueError as e:
            return self._no("SELECT_MODE", str(e))
        note = ""
        if mode != self.mode:
            if self.state == "ARMED":           # armed for cycles of the mode being left
                self.armed_confirmed = False
                self.state = "READY"
                note = " - disarmed"
            self._drop_permissions("another operating mode was selected")
            self._forget_inputs("another operating mode was selected")
            self.mode = mode
            self.cycle = None
            self._result(None, None)            # the last cycle belonged to the other mode
        self._publish()
        return self._ok("SELECT_MODE", f"{TITLES[mode]} mode" + note)

    def _c_arm(self, arg, confirmed):
        if self.mode != DEFAULT_MODE:
            return self._no("ARM", self._held())
        if self.state != "READY":
            return self._no("ARM", f"not READY ({self.state})")
        if self.config is None:
            return self._no("ARM", "no station configuration selected")
        refused = self._expect("ARM", arg)
        if refused is not None:
            return refused
        why = self._unusable(self.config)
        if why:
            return self._no("ARM", f"station configuration {self.config} is not usable: {why}")
        if self.require_confirmation and not confirmed:
            return self._no("ARM", "the real rig needs the operator's confirmation to arm")
        self.armed_confirmed = bool(confirmed)
        self._release("ARM")                    # armed for the NEXT part: the last one is over
        self._go("ARMED")
        return self._ok("ARM", "armed: TRIGGER starts a cycle")

    def _c_disarm(self, arg, confirmed):
        if self.state != "ARMED":
            return self._no("DISARM", f"not armed ({self.state})")
        self.armed_confirmed = False
        self._release("DISARM")
        self._go("READY")
        return self._ok("DISARM")

    def _c_reset(self, arg, confirmed):
        """Acknowledge: FAULT -> READY, and the ALARM output off (also the ALARM an INVALID
        cycle left while idle). The cycle's result stays as it is: PULL / FAIL and the
        result code are not cleared (they release nothing), a PASS is never handed back, and
        no permission comes back."""
        if self.state == "FAULT":
            self.fault = None
            self.alarm_acknowledged = True
            self._go("READY")
            return self._ok("RESET", "fault cleared" + self._stale_note())
        if self.state in IDLE and self.signals()["ALARM"]:
            self.alarm_acknowledged = True
            self._publish()
            return self._ok("RESET", "alarm acknowledged" + self._stale_note())
        return self._no("RESET", f"no fault and no alarm ({self.state})")

    def _stale_note(self) -> str:
        """What RESET leaves on: the last cycle's PULL / FAIL (until the next cycle)."""
        d = self.last_disposition
        if d not in ("PULL", "REJECT"):
            return ""
        return (f"; the last cycle's {'PULL' if d == 'PULL' else 'FAIL'} output stays until the "
                "next cycle")

    @staticmethod
    def _pick(arg, names: list) -> Optional[str]:
        """A name, or a 1-based position. A text that IS a name is that name, also when it
        is all digits (profiles "60" / "38"); a number, or a text of digits that names
        nothing, is the position."""
        if arg is None or isinstance(arg, bool):
            return None
        if isinstance(arg, str) and arg in names:
            return arg
        if isinstance(arg, int) or (isinstance(arg, str) and arg.isdigit()):
            i = int(arg) - 1
            return names[i] if 0 <= i < len(names) else None
        return None

    def resolve_config(self, arg) -> Optional[str]:
        """A station configuration name, or a 1-based index (keys 1-9, a PLC number)."""
        return self._pick(arg, self.configs)

    def resolve_profile(self, arg) -> Optional[str]:
        """A profile name of the selected configuration, or its 1-based index (a JSON number;
        a name always wins for a text)."""
        return self._pick(arg, self.profiles())

    def update_configs(self, names: list, unusable: Optional[dict] = None,
                       presets: Optional[dict] = None, info: Optional[dict] = None,
                       flows: Optional[dict] = None) -> None:
        """The owner re-read its station configurations: `names` = every config file in a
        stable order (usable or not, so a broken one never shifts the others' numbers),
        `unusable` = name -> why it can't run, `flows` = name -> StationConfiguration
        (profiles + work states; a name without one is a single-job configuration). Never
        switches the selected configuration or profile (one that broke or vanished stays
        selected and is refused). While ARMED or in a cycle the list and its numbers stay as
        they are and only the usability marks change; an ARMED bus whose configuration became
        unusable disarms."""
        names = list(names)
        bad = dict(unusable or {})
        with self._lock:
            before = (list(self.configs), self.config, self.profile, self.signals(),
                      self.last_result)
            if self.state in ("READY", "FAULT"):
                self.configs = names
                self.config_presets = dict(presets or {})
                self.config_info = dict(info or {})
                self.flows = dict(flows or {})
                self._implicit = {}
                cyc, now = self.cycle, self.flows.get(self.config)
                if cyc is not None and not cyc.legacy and (
                        now is None or now.public() != cyc.cfg.public()):
                    # the last cycle's permissions were granted by another version of the flow
                    self._release("the station configuration changed on disk")
                if self.config is None:          # nothing selected yet: the first usable one
                    self.config = next((n for n in names if n not in bad), None)
                    self.profile = None
                if self.profile is None:
                    self.profile = self._default_profile()
            marks = {n: bad[n] for n in self.configs if n in bad}
            for n in [*self.configs, self.config]:
                if n is not None and n not in names:
                    marks[n] = "its station config is gone"
            self.unusable = marks
            cfg = self.config
            why = self._unusable(cfg) if cfg is not None else None
            if self.state == "ARMED" and why:
                self.armed_confirmed = False
                self._go("READY")
                self._log("bus", "DISARM", True,
                          f"station configuration {cfg} is not usable: {why}")
            elif (list(self.configs), self.config, self.profile, self.signals(),
                  self.last_result) != before:
                self._publish()                  # the registers / PERMIT outputs may have moved

    # ------------------------------------------------------------ job events
    def job_finished(self, kind: str, result: Optional[str] = None,
                     error: Optional[str] = None, disposition: Optional[str] = None,
                     step: Optional[dict] = None) -> None:
        """kind 'run' (result PASS / FAIL / INVALID / ABORTED, disposition CONTINUE / REJECT
        / PULL: the run's own, else PASS -> CONTINUE, FAIL -> REJECT, anything else PULL; one
        that does not fit the result is PULL; `step`: the DLP answer of a work state's check,
        engine/flow/verify.py, for a cycle of a flow) or 'home' (result OK / MOVED / UNKNOWN
        / REFERENCE_SET); error = the job crashed (PULL)."""
        with self._lock:
            if self.state == "ABORTING" and self._busy_kind not in (None, kind):
                return                          # not the job this abort waits for
            if kind == "run" and self.state in ("RUNNING", "ABORTING"):
                aborted = self.state == "ABORTING"
                cyc = self.cycle
                if cyc is not None and not cyc.legacy and cyc.state != "DONE":
                    self._step_finished(cyc, aborted, result, error, disposition, step)
                    return
                res = "ABORTED" if aborted else (result or "INVALID")
                if (error and not aborted) or res not in ("PASS", "FAIL", "INVALID", "ABORTED"):
                    res = "INVALID"             # a crash, or not a verdict at all (a
                                                # measurement's status): never a pass
                self._result(res, D.of_result(
                    res, None if (aborted or error) else disposition,
                    error=bool(error) or aborted))
                if cyc is not None and cyc.legacy and cyc.state != "DONE":
                    cyc.legacy_done(self.last_result, self.last_disposition)
                    cyc.drain_events()
                self._forget_inputs("the cycle ended")
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
                else:                           # a good rig check also acknowledges the ALARM
                    self.fault = None
                    self.alarm_acknowledged = True
                    self._go("READY")

    def _step_finished(self, cyc: Cycle, aborted: bool, result, error, disposition,
                       step: Optional[dict]) -> None:
        """The DLP check of the open cycle's current work state ended."""
        asked, self._asked = self._asked, None
        if aborted or result == "ABORTED":      # ABORT (or its job was cancelled): the cycle
            cyc.abort("aborted: ABORT while the work state was verified")   # is ABORTED
            self._result("ABORTED", "PULL")
            self._cycle_closed()
            self._forget_inputs("the cycle ended")
            if aborted and self.home_after_abort:
                self.home_after_abort = False
                self._start_home()
                return
            self.armed_confirmed = False
            self._go("READY")
            return
        if error:
            dlp = {"error": str(error), "result": "INVALID", "disposition": "PULL"}
        else:                                   # only the answer to THIS verify counts
            stray = self._stray(step, asked)
            dlp = step if stray is None else {"error": stray, "result": "INVALID",
                                              "disposition": "PULL"}
        row = cyc.row
        changed = [s for s in (row["expects"] if row is not None else ())
                   if self._input_seq.get(s, 0) != self._seq_at.get(s, 0)]
        try:
            effects = cyc.verified(dlp, self._claims(cyc.cfg), changed)
        except CycleError as e:                 # not the answer the cycle waited for: nothing
            effects = cyc.fail(f"unexpected DLP answer: {e}")     # is confirmed by it
        self._apply(effects)

    @staticmethod
    def _stray(answer, asked: Optional[dict]) -> Optional[str]:
        """Why a reported DLP answer is not the answer to the verify the bus asked for (None:
        it is). A bare verdict without an answer, an answer that does not say which work
        state / attempt it is for, or one for another work state, attempt or cycle confirms
        nothing."""
        if not isinstance(answer, dict):
            return ("the DLP check reported no answer for this work state (a verdict alone "
                    "confirms nothing)")
        if asked is None:
            return "the DLP answer was not asked for"
        for key in ("step", "attempt", "cycle_id"):
            if key == "cycle_id" and asked.get(key) is None:
                continue
            if key not in answer:
                return f"the DLP answer does not say which {key} it is for"
            if answer[key] != asked[key]:
                return (f"the DLP answer is for {key} {answer[key]!r}, not the one being "
                        f"verified ({asked[key]!r})")
        return None

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

    # ------------------------------------------------------------ time
    def _tick(self) -> None:
        cyc = self.cycle
        if cyc is None or self.state != "WAITING" or not cyc.due():
            return
        effects = cyc.tick()
        if effects:
            self._log("bus", "TIMEOUT", True, cyc.row["message"] if cyc.row else "")
            self._apply(effects)

    def tick(self) -> None:
        """A work state that waited for STEP / ACK longer than its timeout_s ends the cycle
        PULL. The owner calls this now and then (every command does too)."""
        with self._lock:
            self._tick()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ================================================================ I/O backends

class InputBackend:
    """A hardware command source. start(bus) begins delivering commands (bus.command(
    name, source=self.name)) and sensor claims (bus.set_inputs({signal: level},
    source=self.name)); stop() releases the hardware. `mapping` maps the backend's inputs
    (lines, bits, serial tokens) to commands, `signals` maps inputs whose LEVEL is a sensor
    claim (a part-present switch) to the signal name the work states expect; both from the
    I/O config."""
    name = "input"

    def __init__(self, mapping: Optional[dict] = None, signals: Optional[dict] = None,
                 **options):
        self.mapping = dict(mapping or {})
        self.signals = dict(signals or {})
        self.options = options

    def start(self, bus: CommandBus) -> None:
        raise NotImplementedError(f"{type(self).__name__} is a placeholder: not implemented yet")

    def stop(self) -> None:
        pass


class SimulatedInput(InputBackend):
    """Tests / mock: inject(input_name) acts like that line going active (a command);
    set_level(input_name, level) like a sensor line changing level (a sensor claim)."""
    name = "simulated"

    def start(self, bus: CommandBus) -> None:
        self.bus = bus

    def inject(self, line: str, arg=None) -> Reply:
        cmd = self.mapping.get(line)
        if cmd is None:
            raise KeyError(f"input {line!r} is not mapped (mapping: {self.mapping})")
        return self.bus.command(cmd, source=f"{self.name}:{line}", arg=arg)

    def set_level(self, line: str, level: Optional[bool]) -> dict:
        sig = self.signals.get(line)
        if sig is None:
            raise KeyError(f"input {line!r} is not a sensor signal (signals: {self.signals})")
        return self.bus.set_inputs({sig: level}, source=f"{self.name}:{line}")


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
    host, unit, poll_ms. Mapping: input bit "DI0".."DI15" -> command (rising edge); signals:
    input bit -> sensor claim (its level, e.g. DI4 -> part_present)."""
    name = "digital_io"


class OutputBackend:
    """Stack-light style outputs: publish({"READY": bool, "BUSY": .., "PASS": .., "FAIL": ..,
    "ALARM": .., "PULL": .., "KICKOUT": .., "PROMPT": .., "SENSOR_MISMATCH": ..,
    "PERMIT_<NAME>": ..}); `mapping` maps signals to the backend's outputs. PULL = route the
    part to the quality-inspection lane (never set together with PASS); PERMIT_<NAME> = the
    work state that grants it is CONFIRMED in this cycle (the line may do the next thing)."""
    name = "output"

    def __init__(self, mapping: Optional[dict] = None, **options):
        self.mapping = dict(mapping or {})
        self.options = options

    def publish(self, signals: dict) -> None:
        raise NotImplementedError(f"{type(self).__name__} is a placeholder: not implemented yet")

    def publish_registers(self, registers: dict) -> None:
        """Numeric outputs (result_code, state_code, station_config, profile, work_state,
        work_state_code): a no-op for backends that only drive lines (a stack light);
        fieldbus backends write them to registers / nodes."""


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
#   inputs  (PLC -> scanner): command bits START / ABORT / HOME / TRIGGER / STEP / ACK / ARM
#           / DISARM / RESET (rising edge = one command), the station configuration and
#           profile numbers (options.station_config_* / profile_*), sensor claims (`signals`:
#           a bit's level -> the signal a work state expects)
#   outputs (scanner -> PLC): READY / BUSY / PASS / FAIL / ALARM / PULL / KICKOUT / PROMPT /
#           SENSOR_MISMATCH / PERMIT_<NAME> bits, the result_code / state_code /
#           station_config / profile / work_state / work_state_code registers
#           (options.*_register / *_node / *_tag); PULL routes the part to the
#           quality-inspection lane, a PERMIT bit lets the line do the next thing

_PLACEHOLDER = ("{cls} is a placeholder: the {proto} adapter is not implemented (no {proto} "
                "client is bundled). Implement {method} in a subclass with your site's {proto} "
                "stack. Intended mapping: {mapping}")


class ModbusTcpInput(InputBackend):
    """PLACEHOLDER: Modbus TCP client polling the PLC (options: host, port 502, unit, poll_ms,
    station_config_register, profile_register). Mapping 'coil:N' / 'di:N' (a coil or discrete
    input address) -> command, acted on the rising edge; signals 'di:N' -> sensor claim (its
    level); the holding registers options.station_config_register / profile_register
    (1-based numbers, 0 = keep) select the station configuration / the profile when they
    change."""
    name = "modbus_tcp"
    PROTOCOL = "Modbus TCP"

    def start(self, bus: CommandBus) -> None:
        raise NotImplementedError(_PLACEHOLDER.format(
            cls=type(self).__name__, proto=self.PROTOCOL, method="start(bus)",
            mapping="coils / discrete inputs 'coil:N' -> START / ABORT / HOME / TRIGGER / STEP "
                    "/ ACK / ARM, discrete inputs -> sensor claims (signals), holding registers "
                    "station_config_register -> SELECT_CONFIG, profile_register -> "
                    "SELECT_PROFILE"))


class ModbusTcpOutput(OutputBackend):
    """PLACEHOLDER: Modbus TCP writes (options: host, port 502, unit, result_register,
    state_register, station_config_register, profile_register, work_state_register,
    work_state_code_register). Mapping signal -> 'coil:N' (READY / BUSY / PASS / FAIL / ALARM
    / PULL / KICKOUT / PROMPT / SENSOR_MISMATCH / PERMIT_<NAME>); publish_registers writes
    the registers to the holding registers named in options."""
    name = "modbus_tcp"
    PROTOCOL = "Modbus TCP"

    def publish(self, signals: dict) -> None:
        raise NotImplementedError(_PLACEHOLDER.format(
            cls=type(self).__name__, proto=self.PROTOCOL, method="publish(signals)",
            mapping="READY / BUSY / PASS / FAIL / ALARM / PULL / KICKOUT / PROMPT / "
                    "SENSOR_MISMATCH / PERMIT_<NAME> -> coils 'coil:N'"))

    def publish_registers(self, registers: dict) -> None:
        raise NotImplementedError(_PLACEHOLDER.format(
            cls=type(self).__name__, proto=self.PROTOCOL, method="publish_registers(registers)",
            mapping="result_code / state_code / station_config / profile / work_state / "
                    "work_state_code -> holding registers (options result_register, "
                    "state_register, station_config_register, profile_register, "
                    "work_state_register, work_state_code_register)"))


class OpcUaInput(InputBackend):
    """PLACEHOLDER: OPC UA client subscribed to the PLC's nodes (options: endpoint
    'opc.tcp://host:4840', security, station_config_node, profile_node). Mapping node id
    ('ns=2;s=Cell.Start') -> command on a false -> true change; signals node id -> sensor
    claim (its value); options.station_config_node / profile_node (Int) select the station
    configuration / the profile."""
    name = "opc_ua"
    PROTOCOL = "OPC UA"

    def start(self, bus: CommandBus) -> None:
        raise NotImplementedError(_PLACEHOLDER.format(
            cls=type(self).__name__, proto=self.PROTOCOL, method="start(bus)",
            mapping="Boolean nodes 'ns=..;s=..' -> START / ABORT / HOME / TRIGGER / STEP / "
                    "ACK / ARM (data-change subscription) and sensor claims (signals), "
                    "station_config_node -> SELECT_CONFIG, profile_node -> SELECT_PROFILE"))


class OpcUaOutput(OutputBackend):
    """PLACEHOLDER: OPC UA writes (options: endpoint, result_node, state_node,
    station_config_node, profile_node, work_state_node, work_state_code_node). Mapping signal
    (PERMIT_<NAME> included) -> Boolean node id; registers -> Int nodes."""
    name = "opc_ua"
    PROTOCOL = "OPC UA"

    def publish(self, signals: dict) -> None:
        raise NotImplementedError(_PLACEHOLDER.format(
            cls=type(self).__name__, proto=self.PROTOCOL, method="publish(signals)",
            mapping="READY / BUSY / PASS / FAIL / ALARM / PULL / KICKOUT / PROMPT / "
                    "SENSOR_MISMATCH / PERMIT_<NAME> -> Boolean nodes"))

    def publish_registers(self, registers: dict) -> None:
        raise NotImplementedError(_PLACEHOLDER.format(
            cls=type(self).__name__, proto=self.PROTOCOL, method="publish_registers(registers)",
            mapping="result_code / state_code / station_config / profile / work_state / "
                    "work_state_code -> Int nodes (options result_node, state_node, "
                    "station_config_node, profile_node, work_state_node, "
                    "work_state_code_node)"))


class EtherNetIpInput(InputBackend):
    """PLACEHOLDER: EtherNet/IP (CIP) - the scanner as an explicit-messaging client reading
    PLC tags, or an adapter with an implicit I/O assembly (options: host, slot, poll_ms,
    station_config_tag, profile_tag). Mapping tag or assembly bit ('Cell_Start', 'in:0.3') ->
    command on the rising edge; signals tag / bit -> sensor claim (its level);
    options.station_config_tag / profile_tag (DINT) select the station configuration / the
    profile."""
    name = "ethernet_ip"
    PROTOCOL = "EtherNet/IP"

    def start(self, bus: CommandBus) -> None:
        raise NotImplementedError(_PLACEHOLDER.format(
            cls=type(self).__name__, proto=self.PROTOCOL, method="start(bus)",
            mapping="BOOL tags / input-assembly bits -> START / ABORT / HOME / TRIGGER / "
                    "STEP / ACK / ARM and sensor claims (signals), station_config_tag (DINT) -> "
                    "SELECT_CONFIG, profile_tag (DINT) -> SELECT_PROFILE"))


class EtherNetIpOutput(OutputBackend):
    """PLACEHOLDER: EtherNet/IP writes (options: host, slot, result_tag, state_tag,
    station_config_tag, profile_tag, work_state_tag, work_state_code_tag). Mapping signal
    (PERMIT_<NAME> included) -> BOOL tag or output-assembly bit; registers -> DINT tags."""
    name = "ethernet_ip"
    PROTOCOL = "EtherNet/IP"

    def publish(self, signals: dict) -> None:
        raise NotImplementedError(_PLACEHOLDER.format(
            cls=type(self).__name__, proto=self.PROTOCOL, method="publish(signals)",
            mapping="READY / BUSY / PASS / FAIL / ALARM / PULL / KICKOUT / PROMPT / "
                    "SENSOR_MISMATCH / PERMIT_<NAME> -> BOOL tags / output-assembly bits"))

    def publish_registers(self, registers: dict) -> None:
        raise NotImplementedError(_PLACEHOLDER.format(
            cls=type(self).__name__, proto=self.PROTOCOL, method="publish_registers(registers)",
            mapping="result_code / state_code / station_config / profile / work_state / "
                    "work_state_code -> DINT tags (options result_tag, state_tag, "
                    "station_config_tag, profile_tag, work_state_tag, work_state_code_tag)"))


INPUTS = {c.name: c for c in (SimulatedInput, CameraGpioInput, SerialIOInput, DigitalIOInput,
                              ModbusTcpInput, OpcUaInput, EtherNetIpInput)}
OUTPUTS = {c.name: c for c in (SimulatedOutput, CameraGpioOutput, DigitalIOOutput,
                               ModbusTcpOutput, OpcUaOutput, EtherNetIpOutput)}


def load_io_config(path) -> tuple:
    """stations/io_example.json -> ([InputBackend], [OutputBackend]); entries with
    "enabled": false are skipped. Unknown kinds / commands / signals are refused. An input's
    "map" names commands, its optional "signals" the sensor claims (input -> signal name);
    an output's "map" names READY ... SENSOR_MISMATCH and PERMIT_<NAME> signals."""
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
        bad = [c for c in spec.get("map", {}).values()
               if c not in COMMANDS and c not in DEPRECATED_COMMANDS]
        if bad:
            raise ValueError(f"input {kind}: unknown commands {bad}")
        sigs = spec.get("signals") or {}
        if not isinstance(sigs, dict) or not all(
                isinstance(v, str) and _SIGNAL.fullmatch(v) for v in sigs.values()):
            raise ValueError(f"input {kind}: signals must map an input to a sensor signal "
                             f"name (letters, digits, _ . -), got {sigs!r}")
        ins.append(INPUTS[kind](spec.get("map"), sigs, **spec.get("options", {})))
    for spec in doc.get("outputs", []):
        if not spec.get("enabled", True):
            continue
        kind = spec.get("kind")
        if kind not in OUTPUTS:
            raise ValueError(f"output kind {kind!r} must be one of {sorted(OUTPUTS)}")
        bad = [s for s in spec.get("map", {}) if not is_signal(s)]
        if bad:
            raise ValueError(f"output {kind}: unknown signals {bad} (one of "
                             f"{', '.join(SIGNALS)} or PERMIT_<NAME>)")
        outs.append(OUTPUTS[kind](spec.get("map"), **spec.get("options", {})))
    return ins, outs
