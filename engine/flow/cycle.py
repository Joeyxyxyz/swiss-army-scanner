"""
The cycle: the work states of a station configuration, run once for one part.

Each work state is a claim about the line ("a tube is loaded") made by the line's own
sensors. The DLP checks against the sensor and CONFIRMS or CORRECTS the work state: is that
part really there, or not? is that dent there, or not? Per signal three values meet:

    expected   what the work state needs               (the flow's `expects`)
    sensor     what the line's sensor / PLC input says (the I/O layer, read when the scan starts)
    DLP        what the scan measured                   (the verify's observing check)

Outcome of one verify (classify):

    CONFIRMED  sensor and DLP agree, it is what the work state expects, every check passes
               with no doubt -> the permission is granted -> on_confirm (the next work state)
    CORRECTED  the DLP disagrees with a sensor (sensor says present, the DLP says absent, or
               the reverse) -> `sensor_mismatch` names the signal and both values -> the
               permission is withheld -> on_mismatch
    NOT_MET    sensor and DLP agree, on something the work state does not expect (no part
               where one is needed, a part where none may be) -> withheld -> on_not_met
    DEFECT     the part is there, sensor and DLP agree, and a check definitely fails
               (misaligned, dented) -> withheld -> on_defect
    DOUBT      the DLP could not tell (INVALID, unmeasured, a value in the guard band, a
               sensor that never reported or changed during the scan, a crashed or aborted
               check) -> withheld -> on_doubt (default: kickout PULL). Never a permission,
               never a reject, and never passed over: the work state is verified again
               (hold / retry) or the cycle ends PULL.

A definite sensor disagreement outranks everything else the same scan could not measure
(with the part gone the other checks have nothing to measure: the finding is the missing
part); then NOT_MET; then any doubt; then DEFECT; CONFIRMED is what is left.

Actions run in order after a verify that did not confirm:
    project_fault  paint the fault / missing-part zone on the part (the recipe feedback;
                   recorded as `requested`: the driver hands it to the projector)
    prompt         a message to the operator / system; the list waits for ACK
    hold           wait for ACK (the prompt before it, else its own), then verify again
    retry          verify again at once, up to the work state's `retries`; then go on
    kickout        end the cycle REJECT or PULL (REJECT only on a definite finding: under
                   DOUBT it is a PULL)
    goto           make another work state the current one (never out of a DOUBT)
A required work state whose list ends without confirming ends the cycle PULL (fail closed:
never CONTINUE, never a hang). A work state that is not required may be left unconfirmed on
a definite finding the sensor agrees with (NOT_MET, DEFECT) and the flow goes on - but one
left in DOUBT or CORRECTED (the sensor and the DLP still disagree) ends the cycle PULL,
required or not: the line never continues past an open question.

Permissions: one named permission per work state, true only while that work state is
CONFIRMED in this cycle AND every required work state before it is CONFIRMED and none before
it is left CORRECTED / in DOUBT (a flow that jumps ahead grants nothing on the way). They
all drop when the cycle ends anything but CONTINUE, on ABORT, and when a work state becomes
current again (goto / on_confirm back to it: it and every later one); `revoke` drops chosen
ones when a later work state is not confirmed. A new cycle starts with none. A row's
`permission_granted` says whether its permission stands NOW; `permission_note` says why a
CONFIRMED work state's permission is not (or no longer) granted.

The cycle ends CONTINUE (result PASS) only when every required work state was CONFIRMED, at
least one was, and none is left CORRECTED or in DOUBT. Any other ending is a kickout: the
record's `kickout` says where and why (a kickout action, or the cycle itself: unconfirmed,
timeout, abort ...).

Pure: no engine, no hardware, no thread, no clock of its own (the driver passes `clock`).
The driver (the command bus, `scan run`) calls begin / step / verified / ack / retry / skip
/ abort / tick and acts on the effects they return:
    ("verify", {step, index, attempt})   run the DLP check of the current work state, then
                                         call verified(dlp)
    ("project", {step, run_id, ...})     paint that step's fault on the part
    ("wait", "step" | "ack")             nothing to do until STEP / ACK
    ("done", None)                       the cycle ended: record() is its result
"""

from __future__ import annotations

import copy
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from .. import disposition as D
from .schema import StationConfiguration, Step, action_text

OUTCOME_CODES = {"PENDING": 0, "CONFIRMED": 1, "CORRECTED": 2, "NOT_MET": 3, "DEFECT": 4,
                 "DOUBT": 5, "SKIPPED": 6, "ABORTED": 7}
DEFINITE = ("CORRECTED", "NOT_MET", "DEFECT")       # a definite DLP finding: REJECT allowed
UNSETTLED = ("DOUBT", "CORRECTED")                  # an open question: never passed over
HOLD_PROMPT = "work state on hold: put it right, then acknowledge (ACK) to check again"
MAX_VERIFIES = 200                                  # per cycle: a looping flow ends PULL


class CycleError(RuntimeError):
    """The call does not fit the cycle's state (the command is refused)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tri(v) -> Optional[bool]:
    """True / False, anything else is unknown (None): only a real boolean is a claim."""
    return v if isinstance(v, bool) else None


def job_outcome(result: Optional[str], disposition: Optional[str]) -> str:
    """The outcome of a single job's one work state, from the job's own verdict and
    disposition: a doubt-free PASS is CONFIRMED, a definite FAIL is DEFECT, ABORTED as such,
    anything else (INVALID, a PULL, an INVALID the line rejects) is DOUBT."""
    if result == "ABORTED":
        return "ABORTED"
    if result == "PASS" and disposition == "CONTINUE":
        return "CONFIRMED"
    if result == "FAIL" and disposition == "REJECT":
        return "DEFECT"
    return "DOUBT"


def _say(v: Optional[bool]) -> str:
    return "unknown" if v is None else ("true" if v else "false")


def reason(code: str, message: str, step: Optional[str] = None) -> dict:
    """A pull reason of a cycle: engine/disposition.reason + the work state it belongs to."""
    r = D.reason(code, message)
    if step is not None:
        r["step"] = str(step)
    return r


def classify(step: Step, sensors: dict, dlp: Optional[dict],
             sensors_after: Optional[dict] = None, changed=()) -> dict:
    """One verify of a work state -> {outcome, observed, sensors, sensor_mismatch, reasons,
    message}. `sensors`: the sensor inputs read when the scan started; `sensors_after`: read
    again when it ended (a claim that changed meanwhile is a doubt); `changed`: signals the
    driver saw change during the scan even when they ended where they started (true ->
    false -> true); `dlp`: the DLP check's answer {result, disposition, observed: {signal:
    bool | None}, pull_reasons, error}."""
    dlp = dlp if isinstance(dlp, dict) else {}
    exp = step.expects
    seen = dlp.get("observed") if isinstance(dlp.get("observed"), dict) else {}
    observed = {s: tri(seen.get(s)) for s in exp}
    said = {s: tri((sensors or {}).get(s)) for s in exp}
    out = {"outcome": "DOUBT", "observed": observed, "sensors": said, "sensor_mismatch": [],
           "reasons": [], "message": ""}
    res, disp = dlp.get("result"), dlp.get("disposition")
    # every recorded doubt counts, whatever its shape: one that is not a {reason, message}
    # mapping (or a pull_reasons that is not a list) is a doubt too, never dropped
    raw = dlp.get("pull_reasons") or []
    doubts = [{**r, "step": step.id} if isinstance(r, dict) else
              reason("invalid", f"an unreadable pull reason: {r!r}"[:200], step.id)
              for r in (raw if isinstance(raw, (list, tuple)) else [raw])]

    def doubt(code: str, message: str) -> dict:
        out["reasons"] = [reason(code, message, step.id), *doubts]
        out["message"] = message
        return out

    if dlp.get("error"):
        return doubt("error", f"the DLP check did not finish: {dlp['error']}")
    if res not in ("PASS", "FAIL", "INVALID"):
        return doubt("aborted" if res == "ABORTED" else "invalid",
                     f"the DLP check gave no verdict (result {res!r})")
    if exp and all(v is not None for v in observed.values()):
        moved = [s for s in exp if s in (changed or ()) or (
            sensors_after is not None and tri(sensors_after.get(s)) != said[s])]
        if moved:
            def path(s: str) -> str:
                end = said[s] if sensors_after is None else tri(sensors_after.get(s))
                return f"{_say(said[s])} -> " + ("... -> " if end == said[s] else "") + _say(end)
            return doubt("sensor_unknown", "sensor " + ", ".join(
                f"{s} changed during the scan ({path(s)})" for s in moved)
                + ": the DLP can neither confirm nor correct it")
        mism = [s for s in exp if said[s] is not None and said[s] != observed[s]]
        if mism:
            out["outcome"] = "CORRECTED"
            out["sensor_mismatch"] = [{"signal": s, "sensor": said[s], "dlp": observed[s],
                                       "expected": exp[s]} for s in mism]
            out["message"] = "; ".join(f"sensor {s} says {_say(said[s])}, the DLP says "
                                       f"{_say(observed[s])}" for s in mism)
            out["reasons"] = [reason("sensor_mismatch", out["message"], step.id)]
            return out
        silent = [s for s in exp if said[s] is None]
        if silent:
            return doubt("sensor_unknown", f"sensor {', '.join(silent)} has not reported "
                         "for this cycle: the DLP has no claim to confirm or correct (POST "
                         "/api/io/inputs, or the line's I/O backend; claims are forgotten when "
                         "a cycle ends)")
        unmet = [s for s in exp if observed[s] != exp[s]]
        if unmet:
            out["outcome"] = "NOT_MET"
            out["message"] = "; ".join(
                f"{s} is {_say(observed[s])} (sensor and DLP agree), this work state expects "
                f"{_say(exp[s])}" for s in unmet)
            out["reasons"] = [reason("not_confirmed", out["message"], step.id)]
            return out
    elif exp:
        blind = [s for s in exp if observed[s] is None]
        return doubt(D.classify(dlp.get("invalid_reason"), "invalid"),
                     f"the DLP could not tell {', '.join(blind)}"
                     + (f": {dlp['invalid_reason']}" if dlp.get("invalid_reason") else ""))
    if res == "PASS" and disp == "CONTINUE" and not doubts:
        out["outcome"] = "CONFIRMED"
        out["message"] = "confirmed" + (": " + ", ".join(
            f"{s} {_say(observed[s])}" for s in exp) if exp else "")
        return out
    if res == "FAIL" and disp == "REJECT":
        out["outcome"] = "DEFECT"
        out["message"] = dlp.get("message") or "a check failed"
        out["reasons"] = [reason("not_confirmed", out["message"], step.id)]
        return out
    why = dlp.get("invalid_reason") or dlp.get("message") or f"result {res}, disposition {disp}"
    out["reasons"] = doubts or [reason(D.classify(why), why, step.id)]
    out["message"] = "the DLP check is in doubt: " + "; ".join(
        str(r.get("message") or r.get("reason")) for r in out["reasons"][:3])
    return out


class Cycle:
    """One cycle of a station configuration's flow with the active profile."""

    def __init__(self, cfg: StationConfiguration, profile: Optional[str] = None, *,
                 clock: Callable = time.monotonic, meta: Optional[dict] = None):
        self.cfg = cfg
        self.profile = cfg.default_profile if profile is None else profile
        self.legacy = bool(cfg.legacy)
        self.steps: list = list(cfg.flow)
        self.clock = clock
        self.meta = dict(meta or {})            # the driver's own (cycle id, folder)
        self.rows = [self._row(s) for s in self.steps]
        self.permissions = {p: False for p in cfg.permissions}
        self.index = 0
        self.state = "NEW"                      # NEW | WAIT_STEP | VERIFYING | WAIT_ACK | DONE
        self.prompt: Optional[dict] = None
        self.result: Optional[str] = None       # PASS | FAIL | INVALID | ABORTED
        self.disposition: Optional[str] = None  # CONTINUE | REJECT | PULL
        self.pull_reasons: list = []
        self.kickout: Optional[dict] = None
        self.mismatches: list = []              # every sensor mismatch seen, fixed or not
        self.ended_by: Optional[str] = None
        self.started, self.finished = _now(), None
        self.events: list = []                  # drained by the driver into the cycle log
        self._actions: list = []
        self._ai = 0
        self._acked = False
        self._wait: Optional[str] = None        # 'prompt' | 'hold' while WAIT_ACK
        self._deadline: Optional[float] = None
        self._verifies = 0

    # ------------------------------------------------------------ records
    @staticmethod
    def _row(s: Step) -> dict:
        return {"id": s.id, "title": s.title or s.id, "index": s.index, "required": s.required,
                "expects": dict(s.expects), "verify": s.verify_text(), "permit": s.permit,
                "trigger": s.trigger, "allow_skip": s.allow_skip,
                "outcome": "PENDING", "sensor_inputs": {}, "dlp_result": None, "observed": {},
                "sensor_mismatch": [], "permission_granted": False, "permission_note": None,
                "actions": [], "attempts": 0, "retries_used": 0, "disposition": None,
                "message": "", "reasons": [], "earlier": [], "started": None,
                "finished": None}

    def _event(self, event: str, level: str = "info", **fields) -> None:
        self.events.append({"event": event, "level": level, **fields})

    def drain_events(self) -> list:
        ev, self.events = self.events, []
        return ev

    @property
    def active(self) -> bool:
        return self.state not in ("NEW", "DONE")

    @property
    def step(self) -> Optional[Step]:
        return self.steps[self.index] if 0 <= self.index < len(self.steps) else None

    @property
    def row(self) -> Optional[dict]:
        return self.rows[self.index] if 0 <= self.index < len(self.rows) else None

    @property
    def waiting(self) -> Optional[str]:
        return {"WAIT_STEP": "step", "WAIT_ACK": "ack"}.get(self.state)

    @property
    def sensor_mismatch(self) -> bool:
        """A sensor and the DLP disagree NOW (a work state is CORRECTED); `mismatches` keeps
        every one the cycle saw, also those put right since."""
        return any(r["outcome"] == "CORRECTED" for r in self.rows)

    @property
    def unconfirmed(self) -> list:
        """The work states that are not CONFIRMED: [{step, outcome, required}]."""
        return [{"step": r["id"], "outcome": r["outcome"], "required": r["required"]}
                for r in self.rows if r["outcome"] != "CONFIRMED"]

    def drop_permissions(self, why: str) -> list:
        """Every permission off (the driver: ABORT, FAULT, HOME, another profile ...) ->
        the names that were on."""
        on = [n for n, v in self.permissions.items() if v]
        for n in self.permissions:
            self._drop(n, why)
        return on

    def record(self) -> dict:
        """The whole cycle as plain data (the result document, the bus snapshot)."""
        r = self.row
        return {"cycle_id": self.meta.get("cycle_id"), "station": self.cfg.station_id,
                "station_config": self.cfg.source.name if self.cfg.source else None,
                "profile": self.profile, "legacy": self.legacy, "state": self.state,
                "active": self.active, "waiting": self.waiting,
                "index": (r["index"] if r is not None else len(self.rows)),
                "step": r["id"] if r is not None else None,
                "prompt": copy.deepcopy(self.prompt),
                "steps": copy.deepcopy(self.rows), "permissions": dict(self.permissions),
                "result": self.result, "disposition": self.disposition,
                "pull_reasons": copy.deepcopy(self.pull_reasons),
                "kickout": copy.deepcopy(self.kickout), "ended_by": self.ended_by,
                "sensor_mismatch": self.sensor_mismatch,
                "sensor_mismatches": copy.deepcopy(self.mismatches),
                "unconfirmed": self.unconfirmed,
                "started": self.started, "finished": self.finished, **self.meta}

    # ------------------------------------------------------------ driving
    def _need(self, *states) -> None:
        if self.state not in states:
            raise CycleError(f"the cycle is {self.state}, not {' / '.join(states)}")

    def begin(self, force: bool = False) -> list:
        """Start at the first work state. force: verify it at once whatever its trigger (a
        TRIGGER started the cycle: the part is in place)."""
        self._need("NEW")
        self._event("cycle_start", station=self.cfg.station_id, profile=self.profile,
                    work_states=[s.id for s in self.steps])
        return self._enter(0, force=force)

    def step_now(self) -> list:
        """STEP / TRIGGER: verify the current work state."""
        self._need("WAIT_STEP")
        return self._verify()

    def request(self, sensors: Optional[dict]) -> dict:
        """The driver is about to run the DLP check: record what the sensors say NOW (the
        claim the DLP confirms or corrects) -> the verify request."""
        self._need("VERIFYING")
        row, step = self.row, self.step
        row["sensor_inputs"] = {s: tri((sensors or {}).get(s)) for s in step.expects}
        return {"config": self.cfg.source.name if self.cfg.source else None,
                "station": self.cfg.station_id, "profile": self.profile, "step": step.id,
                "index": step.index, "attempt": row["attempts"],
                "inputs": dict(row["sensor_inputs"]), **self.meta}

    def verified(self, dlp: Optional[dict], sensors_after: Optional[dict] = None,
                 changed=()) -> list:
        """The DLP check of the current work state answered. `changed`: the signals the
        driver saw change during the scan (classify)."""
        self._need("VERIFYING")
        row, step = self.row, self.step
        dlp = dlp if isinstance(dlp, dict) else None
        c = classify(step, row["sensor_inputs"], dlp, sensors_after, changed)
        row.update(outcome=c["outcome"], observed=c["observed"], dlp_result=_public_dlp(dlp),
                   sensor_mismatch=c["sensor_mismatch"], reasons=c["reasons"],
                   message=c["message"], finished=_now(), actions=[], disposition=None,
                   permission_granted=False, permission_note=None)
        for m in c["sensor_mismatch"]:
            self.mismatches.append({"step": step.id, "attempt": row["attempts"], **m})
        self._event("work_state", level="info" if c["outcome"] == "CONFIRMED" else "warning",
                    step=step.id, outcome=c["outcome"], attempt=row["attempts"],
                    sensor_inputs=row["sensor_inputs"], dlp=c["observed"],
                    sensor_mismatch=c["sensor_mismatch"], message=c["message"],
                    step_run_id=(dlp or {}).get("run_id"))
        if c["outcome"] == "CONFIRMED":
            return self._confirmed()
        self._revoke(step)
        self._actions, self._ai, self._acked = step.actions_for(c["outcome"]), 0, False
        return self._run_actions()

    def ack(self, source: str = "") -> list:
        """ACK: the prompt was seen / the operator did what it asked (confirmed removal)."""
        self._need("WAIT_ACK")
        p, self.prompt = self.prompt, None
        self._acked = True
        self._event("ack", step=self.row["id"], source=source, message=(p or {}).get("message"))
        for a in reversed(self.row["actions"]):
            if a.get("action") in ("prompt", "hold") and a.get("acknowledged") is False:
                a.update(acknowledged=True, acknowledged_at=_now(), acknowledged_by=source)
                break
        if self._wait == "prompt":
            self._ai += 1                       # past the prompt; a hold stays to re-verify
        self._wait = None
        return self._run_actions()

    def retry(self, source: str = "") -> list:
        """RETRY: scan the current work state again without acknowledging anything; uses
        the work state's retry budget."""
        self._need("WAIT_ACK")
        row, step = self.row, self.step
        if row["retries_used"] >= step.retries:
            raise CycleError(f"work state {step.id} has no retry left ({step.retries} allowed): "
                             "ACK or ABORT")
        row["retries_used"] += 1
        row["actions"].append({"action": "retry", "n": row["retries_used"], "of": step.retries,
                               "by": source or "command"})
        self.prompt, self._wait = None, None
        return self._verify()

    def skip(self, source: str = "") -> list:
        """SKIP: leave the current work state unconfirmed (only one that allows it). Never
        grants its permission."""
        self._need("WAIT_STEP", "WAIT_ACK")
        row, step = self.row, self.step
        if not step.allow_skip:
            raise CycleError(f"work state {step.id} does not allow SKIP")
        if row["outcome"] in UNSETTLED:
            raise CycleError(f"work state {step.id} is {row['outcome']}: an open question is "
                             "never skipped (ACK / RETRY to verify it again, or ABORT)")
        self._keep_earlier(row)
        row.update(outcome="SKIPPED", message=f"skipped ({source or 'command'})",
                   finished=_now(), permission_granted=False, disposition=None,
                   sensor_mismatch=[], reasons=[])
        self._revoke(step)
        self.prompt, self._wait = None, None
        self._event("work_state", level="warning", step=step.id, outcome="SKIPPED",
                    source=source)
        return self._enter(self.index + 1)

    def abort(self, why: str = "aborted by the operator") -> list:
        """ABORT: the cycle ends ABORTED (PULL), every permission drops."""
        if self.state == "DONE":
            return []
        row = self.row
        found: list = []
        if row is not None and row["outcome"] != "CONFIRMED":
            if row["outcome"] != "PENDING":     # what the DLP had found stays on record:
                found = list(row.get("reasons") or [])      # `earlier` + the pull reasons
                self._keep_earlier(row)
            row.update(outcome="ABORTED", message=why, finished=_now(),
                       permission_granted=False, sensor_mismatch=[], disposition="PULL",
                       actions=[{"action": "abort", "message": why}])
        return self._finish("ABORTED", "PULL",
                            [reason("aborted", why, row and row["id"]), *found],
                            ended_by="abort")

    def fail(self, why: str, code: str = "error") -> list:
        """The current work state could not be verified at all (the check could not start,
        its job was lost): DOUBT, the cycle ends PULL."""
        if self.state == "DONE":
            return []
        row = self.row
        why_r = [reason(code, why, row and row["id"])]
        if row is not None:
            row.update(outcome="DOUBT", message=why, finished=_now(), permission_granted=False,
                       reasons=why_r, disposition="PULL")
        return self._finish("INVALID", "PULL", why_r, ended_by="error")

    def due(self, now: Optional[float] = None) -> bool:
        return (self._deadline is not None and self.state in ("WAIT_STEP", "WAIT_ACK")
                and (self.clock() if now is None else now) >= self._deadline)

    def tick(self, now: Optional[float] = None) -> list:
        """A work state that waited longer than its timeout_s for STEP / ACK: DOUBT, the
        cycle ends PULL (nobody confirmed it)."""
        if not self.due(now):
            return []
        row, step = self.row, self.step
        why = (f"work state {step.id} was not confirmed within {step.timeout_s:g} s (waiting "
               f"for {'ACK' if self.state == 'WAIT_ACK' else 'STEP'})")
        if row["outcome"] == "PENDING":         # never verified: nothing is known about it
            row.update(outcome="DOUBT", message=why)
        row.update(finished=_now(), permission_granted=False, disposition="PULL")
        row["actions"].append({"action": "timeout", "timeout_s": step.timeout_s,
                               "message": why})
        return self._finish("INVALID", "PULL",
                            [reason("timeout", why, step.id), *(row.get("reasons") or [])],
                            ended_by="timeout")

    def give_up(self, why: str, ended_by: str = "unacknowledged",
                code: str = "timeout") -> list:
        """Nobody will acknowledge the open prompt (an unattended run), or acknowledging it
        again would change nothing: the cycle ends PULL, the work state keeps what the DLP
        found."""
        self._need("WAIT_ACK")
        row = self.row
        row["actions"].append({"action": ended_by, "message": why})
        row.update(finished=_now(), permission_granted=False, disposition="PULL")
        return self._finish("INVALID", "PULL",
                            [reason(code, why, row["id"]), *(row.get("reasons") or [])],
                            ended_by=ended_by)

    def legacy_done(self, result: Optional[str], disposition: Optional[str]) -> None:
        """A single-job configuration: its one work state ends as the job's own disposition
        says (the bus keeps the job's result and disposition as they are)."""
        row = self.rows[0]
        row.update(outcome=job_outcome(result, disposition),
                   dlp_result={"result": result, "disposition": disposition},
                   finished=_now(), disposition=disposition, message=f"job result {result}")
        self.result, self.disposition = result, disposition
        self.state, self.finished, self.ended_by = "DONE", _now(), "job"

    # ------------------------------------------------------------ internals
    def _keep_earlier(self, row: dict) -> None:
        if row["outcome"] != "PENDING":
            row["earlier"].append({k: copy.deepcopy(row[k]) for k in (
                "outcome", "sensor_inputs", "observed", "sensor_mismatch", "dlp_result",
                "actions", "message", "finished")})

    def _verify(self) -> list:
        row, step = self.row, self.step
        self._verifies += 1
        if self._verifies > MAX_VERIFIES:       # a flow that loops (goto / auto) for ever
            why = (f"the flow verified {MAX_VERIFIES} times in one cycle without ending "
                   f"(at work state {step.id}): check its goto / on_confirm / hold loops")
            row.update(outcome="DOUBT", message=why, permission_granted=False,
                       disposition="PULL")
            return self._finish("INVALID", "PULL", [reason("not_confirmed", why, step.id)],
                                ended_by="loop")
        self._keep_earlier(row)
        row["attempts"] += 1
        row.update(outcome="PENDING", started=row["started"] or _now(), finished=None,
                   message="", sensor_mismatch=[], observed={}, dlp_result=None, reasons=[])
        self.prompt, self._wait, self._acked = None, None, False
        self._deadline = None                   # not waiting: the DLP is verifying
        self.state = "VERIFYING"
        return [("verify", {"step": step.id, "index": step.index, "attempt": row["attempts"]})]

    def _wait_from_now(self, step: Step) -> None:
        """timeout_s counts each wait (for STEP, for ACK) from when it starts."""
        self._deadline = (self.clock() + float(step.timeout_s)) if step.timeout_s else None

    def _drop(self, name: Optional[str], why: str) -> None:
        """One permission off: the output, and the row that had granted it says why."""
        if not name:
            return
        if self.permissions.get(name):
            self.permissions[name] = False
            self._event("permission", level="warning", permission=name, granted=False, why=why)
        for r in self.rows:
            if r["permit"] == name and r["permission_granted"]:
                r["permission_granted"] = False
                r["permission_note"] = f"dropped: {why}"

    def _revoke(self, step: Step) -> None:
        for name in [step.permit, *step.revoke]:
            self._drop(name, f"work state {step.id} is not confirmed")

    def _confirmed(self) -> list:
        row, step = self.row, self.step
        row["disposition"] = "CONTINUE"
        if step.permit:
            # a permission lets the line do the NEXT thing: never while a required work
            # state before this one is unconfirmed, or one before it is an open question
            before = [r["id"] for r in self.rows[:self.index]
                      if (r["required"] and r["outcome"] != "CONFIRMED")
                      or r["outcome"] in UNSETTLED]
            if before:
                row["permission_note"] = ("withheld: work state(s) " + ", ".join(before)
                                          + " before it are not confirmed")
                self._event("permission", level="warning", permission=step.permit,
                            granted=False, step=step.id, why=row["permission_note"])
            else:
                self.permissions[step.permit] = True
                row["permission_granted"] = True
                self._event("permission", permission=step.permit, granted=True, step=step.id)
        nxt = step.on_confirm
        if nxt == "next":
            return self._enter(self.index + 1)
        if nxt == "done":
            return self._enter(len(self.steps))
        return self._enter(self._index_of(nxt))

    def _index_of(self, step_id: str) -> int:
        return next(i for i, s in enumerate(self.steps) if s.id == step_id)

    def _enter(self, i: int, force: bool = False) -> list:
        """Work state i becomes the current one (i past the end: the flow is through). One
        entered again is verified again: it and every later one lose their permission."""
        self.prompt, self._wait, self._deadline = None, None, None
        if i >= len(self.steps):
            return self._through()
        if self.rows[i]["outcome"] != "PENDING" or i < self.index:
            for j in range(i, len(self.steps)):
                r, s = self.rows[j], self.steps[j]
                self._drop(s.permit, f"work state {self.steps[i].id} is verified again")
                self._keep_earlier(r)
                r.update(outcome="PENDING", permission_granted=False, permission_note=None,
                         disposition=None, message="", retries_used=0, actions=[],
                         sensor_mismatch=[], observed={}, dlp_result=None, started=None,
                         finished=None, reasons=[])
        self.index = i
        step = self.steps[i]
        self.state = "WAIT_STEP"
        self._wait_from_now(step)
        if step.trigger == "auto" or force:
            return self._verify()
        return [("wait", "step")]

    def _run_actions(self) -> list:
        row, step = self.row, self.step
        effects: list = []
        while self._ai < len(self._actions):
            a = self._actions[self._ai]
            when = a.get("when_dlp")
            if when and any(row["observed"].get(k) is not v for k, v in when.items()):
                self._ai += 1
                continue
            kind = a["action"]
            if kind == "project_fault":
                dlp = row["dlp_result"] or {}
                ok = (bool(dlp.get("run_id")) and bool(dlp.get("feedback_rendered"))
                      and dlp.get("result") != "PASS")
                why = "" if ok else (
                    "every check passed: there is no fault on the part to paint"
                    if dlp.get("result") == "PASS" else
                    dlp.get("feedback_refused") or "the check rendered nothing to paint")
                row["actions"].append({"action": "project_fault", "requested": ok,
                                       **({"why": why} if why else {})})
                if ok:
                    effects.append(("project", {"step": step.id, "run_id": dlp.get("run_id"),
                                                "run_folder": dlp.get("run_folder"),
                                                "outcome": row["outcome"]}))
                self._ai += 1
            elif kind == "prompt":
                return effects + self._ask(a["message"], "prompt")
            elif kind == "hold":
                if not self._acked:
                    return effects + self._ask(HOLD_PROMPT, "hold")
                last = row["actions"][-1] if row["actions"] else {}
                if last.get("action") == "hold":
                    last["verify_again"] = True
                else:
                    row["actions"].append({"action": "hold", "verify_again": True})
                return effects + self._verify()
            elif kind == "retry":
                if row["retries_used"] < step.retries:
                    row["retries_used"] += 1
                    row["actions"].append({"action": "retry", "n": row["retries_used"],
                                           "of": step.retries})
                    return effects + self._verify()
                row["actions"].append({"action": "retry", "exhausted": True,
                                       "of": step.retries})
                self._ai += 1
            elif kind == "kickout":
                return effects + self._kickout(a)
            elif kind == "goto":
                if row["outcome"] == "DOUBT":   # a doubt is verified again or pulled: the
                    row["actions"].append({     # flow never walks away from it
                        "action": "goto", "step": a["step"], "refused": True,
                        "why": "a work state in DOUBT is never left by goto"})
                    break
                row["actions"].append({"action": "goto", "step": a["step"]})
                return effects + self._enter(self._index_of(a["step"]))
        if not step.required and row["outcome"] not in UNSETTLED:
            # a definite finding the sensor agrees with, on a work state the cycle does not
            # need: left unconfirmed (its permission withheld), the flow goes on
            row["actions"].append({"action": "continue", "why": "not a required work state"})
            return effects + self._enter(self.index + 1)
        # fail closed: nothing confirmed it and nothing else was configured (an open
        # question - DOUBT, an uncorrected sensor - is never passed over, required or not)
        why = f"work state {step.id} was not confirmed ({row['outcome']}): {row['message']}"
        row["disposition"] = "PULL"
        return effects + self._finish("FAIL" if row["outcome"] in DEFINITE else "INVALID",
                                      "PULL", self._reasons(row, why), ended_by="unconfirmed")

    def _ask(self, message: str, kind: str) -> list:
        row, step = self.row, self.step
        self.prompt = {"step": step.id, "title": step.title or step.id, "index": step.index,
                       "message": message, "detail": row["message"], "outcome": row["outcome"],
                       "kind": kind, "since": _now()}
        row["actions"].append({"action": kind, "message": message, "acknowledged": False})
        self._wait = kind
        self.state = "WAIT_ACK"
        self._wait_from_now(step)
        self._event("prompt", level="warning", step=step.id, message=message, kind=kind,
                    detail=row["message"])
        return [("wait", "ack")]

    def _reasons(self, row: dict, why: str) -> list:
        """The pull reasons of a cycle this work state ended: what it is (a sensor mismatch,
        a work state not confirmed) + the doubts its DLP check recorded."""
        code = {"CORRECTED": "sensor_mismatch", "DOUBT": None}.get(row["outcome"],
                                                                 "not_confirmed")
        own = [reason(code, why, row["id"])] if code else []
        rest = [r for r in row.get("reasons") or [] if r.get("reason") != code]
        return (own + rest) or [reason("not_confirmed", why, row["id"])]

    def _kickout(self, a: dict) -> list:
        row, step = self.row, self.step
        disp, note = a["disposition"], ""
        if disp == "REJECT" and row["outcome"] not in DEFINITE:
            disp, note = "PULL", "REJECT asked on a doubt: a doubt is never a reject"
        why = a.get("reason") or f"work state {step.id}: {row['outcome']} ({row['message']})"
        row["actions"].append({"action": "kickout", "disposition": disp, "reason": why,
                               **({"note": note} if note else {})})
        row["disposition"] = disp
        self.kickout = {"step": step.id, "disposition": disp, "reason": why,
                        "detail": row["message"], "by": "kickout"}
        definite = row["outcome"] in DEFINITE
        reasons = self._reasons(row, why) if disp == "PULL" else []
        return self._finish("FAIL" if definite else "INVALID", disp, reasons,
                            ended_by="kickout")

    def _through(self) -> list:
        """Past the last work state: CONTINUE only when every required one is CONFIRMED, at
        least one work state is, and none is left an open question (DOUBT / CORRECTED)."""
        open_ = [r for r in self.rows if (r["required"] and r["outcome"] != "CONFIRMED")
                 or r["outcome"] in UNSETTLED]
        if not open_ and any(r["outcome"] == "CONFIRMED" for r in self.rows):
            return self._finish("PASS", "CONTINUE", [], ended_by="confirmed")
        if not open_:
            return self._finish("INVALID", "PULL", [reason(
                "not_confirmed", "the flow ended with no work state CONFIRMED")],
                ended_by="unconfirmed")
        self.index = self.rows.index(open_[0])  # the work state that stopped the cycle
        reasons: list = []
        for r in open_:
            what = "required work state" if r["required"] else "work state"
            reasons += self._reasons(r, f"{what} not confirmed: {r['id']} ({r['outcome']})")
        return self._finish("INVALID", "PULL", reasons, ended_by="unconfirmed")

    def _finish(self, result: str, disposition: str, reasons: list, ended_by: str) -> list:
        self.result, self.disposition = result, disposition
        self.pull_reasons = [r for i, r in enumerate(reasons) if r not in reasons[:i]]
        self.ended_by = ended_by
        if disposition != "CONTINUE":           # a cycle that is not CONTINUE releases nothing
            self.drop_permissions(f"the cycle ended {disposition}")
            if self.kickout is None:            # ... and its part leaves the line: say where
                row = self.row                  # and why, also without a kickout action
                self.kickout = {
                    "step": row["id"] if row is not None else None, "disposition": disposition,
                    "reason": str((self.pull_reasons[0].get("message") if self.pull_reasons
                                   else "") or f"the cycle ended {disposition} ({ended_by})"),
                    "detail": row["message"] if row is not None else "", "by": ended_by}
        self.prompt, self._wait, self._deadline = None, None, None
        self.state, self.finished = "DONE", _now()
        self._event("cycle_end", level="info" if disposition == "CONTINUE" else "warning",
                    result=result, disposition=disposition, ended_by=ended_by,
                    pull_reasons=self.pull_reasons, permissions=dict(self.permissions),
                    work_states={r["id"]: r["outcome"] for r in self.rows})
        return [("done", None)]


def _public_dlp(dlp: Optional[dict]) -> Optional[dict]:
    """The DLP answer as the step records it (plain data; no pictures, no arrays)."""
    if dlp is None:
        return None
    keep = ("result", "disposition", "observed", "run_id", "run_folder", "recipe", "preset",
            "scan_id", "message", "invalid_reason", "pull_reasons", "zones", "error",
            "feedback_rendered", "feedback_refused", "observe", "warnings", "exit_code")
    return {k: copy.deepcopy(dlp[k]) for k in keep if k in dlp}


def actions_text(step: Step) -> dict:
    """The action lists of a work state as short texts (docs, the operator page)."""
    return {k: [action_text(a) for a in getattr(step, k)]
            for k in ("on_mismatch", "on_not_met", "on_defect", "on_doubt")}
