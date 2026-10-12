"""
Dispositions: what happens to the part, decided on top of the verdict.

    CONTINUE   the verdict is PASS and nothing about it is in doubt: the part goes on
    REJECT     a definite measured FAIL: a trusted measurement found a bad part
    PULL       doubt: the part is PULLED for quality inspection (a person looks at it);
               never a pass, never a reject on numbers nobody can trust

Precedence (one run): any definite measured FAIL -> REJECT (even when another zone is
INVALID); else any doubt -> PULL; else CONTINUE. A doubt never becomes CONTINUE. A FAIL is
definite only when nothing about it is in doubt: a check FAIL carrying a `pull` reason, or
a run-level FAIL with a recorded doubt (a metric FAIL on a placement that did not converge),
is a PULL. A run refused or crashed before it finished (`refused`, `error`) is a PULL
whatever else it recorded. The verdict (`result`: PASS / FAIL / INVALID / ABORTED) is
unchanged: the disposition is a second key.

Doubt (`pull_reasons`, [{zone?, check?, reason, message}]; `reason` is one of REASONS):
  uncalibrated       no stereo / mm calibration at these rig settings
  stale_calibration  the scan was made at other rig settings or with another calibration
  px_not_mm          the deviation came out in px, not mm
  too_few_points     too few points, too little of a zone or roi measured (coverage)
  zone_unresolved    a recipe zone could not be resolved on the scan
  golden             the golden (good-part) scan is missing or does not match this setup
  ambiguous_faces    the end faces of a station scene are ambiguous
  incomplete         INCOMPLETE: partial data, a placement that did not converge, ...
  invalid            any other INVALID
  check_crashed      a check raised (a bug is never a verdict)
  code_unreadable    an expected code could not be read at all (unreadable is not wrong)
  rig_check          the rig check run before the cycle said MOVED / UNKNOWN
  refused            the run was refused before a verdict (config, rule, confirmation);
                     recorded as such, never keyword-classified
  error              the run crashed (its message is never keyword-classified: a crash
                     text is not engine wording)
  aborted            the cycle was ABORTED
  guard_band         a PASS value within guard_band_pct of a tolerance limit
  sensor_mismatch    a work state: a line sensor and the DLP disagree (engine/flow/cycle.py)
  sensor_unknown     a work state: a sensor it expects never reported, or changed mid-scan
  not_confirmed      a work state was not confirmed (not met, a defect the line pulls,
                     skipped, never reached)
  timeout            a work state was not confirmed in time (nobody acknowledged)

A PASS check whose own tolerances do not hold it up is a doubt (`invalid`), guard band or
not: a value that is missing or not a finite number, or outside its own limits (the verdict
and the number disagree: a bug, or a rounded value), or a malformed tolerance list.

Guard band (recipe pull.guard_band_pct, station metric job.guard_band_pct; 0 = off): a
PASSing value whose distance to a tolerance limit is at most pct % of the tolerance is
PULLed (inclusive: a value on its limit is always in the band). The tolerance is the
tolerance's `span` when it gives one (a bounded share: part_pct >= 92 of at most 100 has
span 8), else the half width of a two-sided band [lo, hi] (from each limit), else |limit|
of a one-sided one. A band that comes out 0 (a zero-width [x, x], a one-sided limit at 0
with no span) cannot clear any value: every PASS of that tolerance is PULLed.

recipe pull.invalid_reasons_to_reject (default []: every doubt is a PULL): reasons from
REJECTABLE the line rejects instead of pulling (e.g. code_unreadable: no label, no part).
`aborted`, `guard_band`, `refused` and `error` are never rejectable (they say nothing
about the part), nor are the work-state reasons (a cycle's own kickout action decides), and
a run that was refused or crashed is never rejected through the list.

Exit codes (run, recipe-run, station-check, face-check): CONTINUE 0, REJECT 1, PULL 5
(refused 2, crashed 4 and ABORTED 1 keep their documented codes; their disposition is
PULL). Registers: the result code of the verdict, except that a PULL whose verdict is PASS
or FAIL writes the PULL code and a REJECT writes the FAIL code (register_key): the PASS
code is only ever written for CONTINUE.

Pure: no engine, no hardware (the command bus imports it).
"""

from __future__ import annotations

import math
from typing import Optional

DISPOSITIONS = ("CONTINUE", "REJECT", "PULL")
PULL_TEXT = "PULL (quality inspection)"
EXIT_CODES = {"CONTINUE": 0, "REJECT": 1, "PULL": 5}
REASONS = {
    "uncalibrated": "no calibration at these rig settings: no mm",
    "stale_calibration": "made at other rig settings or with another calibration",
    "px_not_mm": "the deviation is in px, not mm",
    "too_few_points": "too few points / too little measured",
    "zone_unresolved": "a zone could not be resolved on the scan",
    "golden": "the golden scan is missing or does not match",
    "ambiguous_faces": "the end faces are ambiguous",
    "incomplete": "incomplete data",
    "invalid": "the numbers cannot be trusted",
    "check_crashed": "a check crashed",
    "code_unreadable": "an expected code could not be read",
    "rig_check": "the rig check before the cycle said MOVED / UNKNOWN",
    "refused": "the run was refused before a verdict",
    "error": "the run crashed",
    "aborted": "the cycle was aborted",
    "guard_band": "a passing value within the guard band of its limit",
    "sensor_mismatch": "a line sensor and the DLP disagree",
    "sensor_unknown": "a sensor the work state expects did not report",
    "not_confirmed": "a work state was not confirmed",
    "timeout": "a work state was not confirmed in time",
}
# never rejectable through a recipe's list: they say nothing definite about the part (the
# work-state reasons belong to a cycle, whose own kickout action decides REJECT / PULL)
NOT_REJECTABLE = ("aborted", "guard_band", "refused", "error", "sensor_mismatch",
                  "sensor_unknown", "not_confirmed", "timeout")
REJECTABLE = tuple(r for r in REASONS if r not in NOT_REJECTABLE)
UNFINISHED = ("refused", "error")       # the run never finished: PULL, whatever it recorded
MAX_GUARD_BAND_PCT = 100.0          # exclusive: 100 % would pull every pass


# ------------------------------------------------------------------ config

def check_pct(v, what: str = "guard_band_pct") -> float:
    """A guard band in % of the tolerance: a finite number, 0 <= pct < 100 (0 = off).
    -> float, or ValueError naming `what`."""
    if (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            or not 0 <= v < MAX_GUARD_BAND_PCT):
        raise ValueError(f"{what} must be a finite number 0 <= pct < {MAX_GUARD_BAND_PCT:g} "
                         f"(% of the tolerance; 0 = no guard band), got {v!r}")
    return float(v)


def check_reject_list(v, what: str = "invalid_reasons_to_reject") -> list:
    """The doubt reasons a line rejects instead of pulling: a list of REJECTABLE names.
    -> list (order kept, duplicates dropped), or ValueError naming `what`."""
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ValueError(f"{what} must be a list of reasons ({', '.join(REJECTABLE)}), got {v!r}")
    bad = [x for x in v if x not in REJECTABLE]
    if bad:
        never = [x for x in bad if x in NOT_REJECTABLE]
        raise ValueError(f"{what}: {bad} " + (
            f"({', '.join(never)} never reject: they say nothing about the part) "
            if never else "") + f"- one of {', '.join(REJECTABLE)}")
    return list(dict.fromkeys(v))


# ------------------------------------------------------------------ reasons

def reason(code: str, message: str = "", zone: Optional[str] = None,
           check: Optional[str] = None) -> dict:
    """One pull reason {zone?, check?, reason, message}; an unknown code is 'invalid'."""
    out = {}
    if zone is not None:
        out["zone"] = str(zone)
    if check is not None:
        out["check"] = str(check)
    out["reason"] = code if code in REASONS else "invalid"
    out["message"] = " ".join(str(message or REASONS[out["reason"]]).split())
    return out


def add(res: dict, code: str, message: str = "", zone: Optional[str] = None,
        check: Optional[str] = None) -> dict:
    """Record a pull reason the flow knows (a rig check MOVED, no calibration) in res."""
    res.setdefault("pull_reasons", [])
    if res["pull_reasons"] is None:
        res["pull_reasons"] = []
    r = reason(code, message, zone, check)
    if r not in res["pull_reasons"]:
        res["pull_reasons"].append(r)
    return r


def classify(message, default: str = "invalid") -> str:
    """The reason code of an INVALID message (engine wording), else `default`."""
    m = " ".join(str(message or "").split()).lower()
    if not m:
        return default
    if m.startswith("the check failed:"):
        return "check_crashed"
    if "rig check" in m:
        return "rig_check"
    if (m.startswith("golden") or "golden scan" in m or "no golden" in m
            or "names no golden" in m or "golden metric scan" in m):
        return "golden"
    if "not mm" in m or "deviation in px" in m:
        return "px_not_mm"
    if ("no stereo calibration" in m or "no calibration" in m or "uncalibrated" in m
            or "no mm calibration" in m):
        return "uncalibrated"
    if ("other rig settings" in m or "triangulated with" in m or "geometrymismatch" in m
            or "re-triangulate" in m or "another calibration" in m):
        return "stale_calibration"
    if "ambiguous" in m and "face" in m:
        return "ambiguous_faces"
    if "could not be resolved" in m:
        return "zone_unresolved"
    if any(k in m for k in ("too few", "only ", "unseen", "unmeasured", "covers no camera px",
                            "no extent", "nothing left to judge", "sampling of it is unknown",
                            "no reference plane", "too little")):
        return "too_few_points"
    if any(k in m for k in ("incomplete", "can't pass", "can't be compared", "did not converge",
                            "can't be trusted", "partial")):
        return "incomplete"
    return default


# ------------------------------------------------------------------ guard band

def _num(v) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def tolerance(name: str, value, lo=None, hi=None, span=None) -> dict:
    """One tolerance a check held a value to: PASS means lo <= value <= hi (None = no
    limit on that side). Units as the name says (_mm, _deg, _pct). Give the value the check
    judged (not a rounded copy). span (optional): the size of the tolerance the guard band
    is a % of, when the default does not fit (two-sided: the half width; one-sided:
    |limit|): e.g. a share bounded at 100 %, part_pct >= 92 -> span 8."""
    t = {"name": str(name), "value": value, "lo": lo, "hi": hi}
    if span is not None:
        t["span"] = span
    return t


def check_tolerances(tols) -> list:
    """A check's `tolerances` as given (None, or a list of tolerance mappings) -> the list,
    or ValueError: a malformed list is a bug, never silently no guard band."""
    if tols is None:
        return []
    if not isinstance(tols, (list, tuple)):
        raise ValueError(f"tolerances must be a list of {{name, value, lo, hi}}, got {tols!r}")
    out = []
    for t in tols:
        if not isinstance(t, dict) or not isinstance(t.get("name"), str):
            raise ValueError(f"tolerance {t!r} must be {{name, value, lo, hi}}")
        lo, hi, span = t.get("lo"), t.get("hi"), t.get("span")
        for k, v in (("lo", lo), ("hi", hi)):
            if v is not None and _num(v) is None:
                raise ValueError(f"tolerance {t['name']}: {k} must be a finite number or null, "
                                 f"got {v!r}")
        if lo is None and hi is None:
            raise ValueError(f"tolerance {t['name']}: give lo, hi or both")
        if lo is not None and hi is not None and float(lo) > float(hi):
            raise ValueError(f"tolerance {t['name']}: lo {lo} > hi {hi}")
        if span is not None and (_num(span) is None or span < 0):
            raise ValueError(f"tolerance {t['name']}: span must be a finite number >= 0 or "
                             f"null, got {span!r}")
        out.append(tolerance(t["name"], t.get("value"), lo, hi, span))
    return out


def tolerances_of(check: dict) -> list:
    """A check result's tolerances: its own `tolerances` when it gives them (built-ins do),
    else its headline key / limit in deviation semantics (docs/CHECKS.md: |value| <= limit;
    the value as given, so a missing or NaN one is caught by unsound). ValueError when its
    own list is malformed."""
    if check.get("tolerances") is not None:
        return check_tolerances(check["tolerances"])
    k, lim = check.get("key"), _num(check.get("limit"))
    if k is None or lim is None:
        return []
    return [tolerance(k, (check.get("values") or {}).get(k), -lim, lim)]


def unsound(tols: list) -> list:
    """The tolerances a PASS cannot stand on: no finite value, or a value outside its own
    limits (the verdict and the number disagree) -> [{name, value, why}]."""
    out = []
    for t in tols:
        v, lo, hi = _num(t.get("value")), _num(t.get("lo")), _num(t.get("hi"))
        if v is None:
            out.append({"name": t["name"], "value": t.get("value"),
                        "why": "has no finite value"})
        elif (lo is not None and v < lo) or (hi is not None and v > hi):
            out.append({"name": t["name"], "value": v,
                        "why": f"is outside its own limits [{_g(lo)}, {_g(hi)}]"})
    return out


def _g(v) -> str:
    return "-" if v is None else f"{v:g}"


def _band(t: dict, f: float) -> float:
    span, lo, hi = _num(t.get("span")), _num(t.get("lo")), _num(t.get("hi"))
    if span is not None:
        return f * span
    if lo is not None and hi is not None:
        return f * (hi - lo) / 2.0              # % of the half width, from each limit
    return f * abs(hi if hi is not None else lo)


def guard_band(tols: list, pct: float) -> list:
    """The tolerances whose value sits inside its limits but within pct % of the tolerance
    of one of them (inclusive) -> [{name, value, limit, side, distance, band, unsized}]; []
    when pct is 0. A band of 0 (zero-width tolerance, one-sided limit at 0 without a span)
    clears nothing: every value inside is a hit (unsized). A value with no finite number or
    outside its limits is unsound's (a doubt of its own), not a guard-band hit."""
    if not pct:
        return []
    f = float(pct) / 100.0
    hits = []
    for t in tols:
        v, lo, hi = _num(t.get("value")), _num(t.get("lo")), _num(t.get("hi"))
        if v is None or (lo is not None and v < lo) or (hi is not None and v > hi):
            continue                            # unsound(): its own doubt
        band = _band(t, f)
        near = [(abs(v - lim), side, lim) for side, lim in (("lo", lo), ("hi", hi))
                if lim is not None]
        if not near:
            continue
        d, side, lim = min(near)
        if d <= band or band <= 0:
            hits.append({"name": t["name"], "value": v, "limit": lim, "side": side,
                         "distance": round(d, 6), "band": round(max(band, 0.0), 6),
                         "unsized": band <= 0 and d > band})
    return hits


def _gb_reason(hit: dict, pct: float, zone=None, check=None) -> dict:
    where = f"{'upper' if hit['side'] == 'hi' else 'lower'} limit {hit['limit']:g}"
    if hit.get("unsized"):
        return reason("guard_band",
                      f"{hit['name']} {hit['value']:g}: its tolerance has no size to take "
                      f"{pct:g} % of ({where}, no span), so no value can be cleared of the "
                      f"guard band", zone, check)
    return reason("guard_band",
                  f"{hit['name']} {hit['value']:g} is within {pct:g} % of the tolerance of its "
                  f"{where} ({hit['distance']:g} from it, guard band {hit['band']:g})",
                  zone, check)


def pass_doubts(check: dict, pct: float, zone=None, check_id=None) -> list:
    """The pull reasons of a PASS (a check result, or {"tolerances": [...]} for a run's
    own limits): its tolerances unsound or malformed -> `invalid`; a value in the guard band
    -> `guard_band`."""
    try:
        tols = tolerances_of(check)
    except ValueError as e:
        return [reason("invalid", f"PASS, but its tolerances are malformed: {e}", zone,
                       check_id)]
    out = [reason("invalid", f"PASS, but {u['name']} {u['value']!r} {u['why']}: the verdict "
                             "and its number disagree", zone, check_id) for u in unsound(tols)]
    return out + [_gb_reason(h, pct, zone, check_id) for h in guard_band(tols, pct)]


# ------------------------------------------------------------------ the decision

def zone_doubts(zones: list, pct: float = 0.0) -> tuple:
    """Recipe zones -> (definite FAIL found, [pull reasons]). A FAIL check is definite unless
    it carries a `pull` reason (an unreadable code); an INVALID (or unknown) check is doubt;
    a PASS check is doubt when it carries a `pull` reason, when its own tolerances do not
    hold it up (unsound, malformed) or when a value sits in the guard band."""
    definite, out = False, []
    for z in zones or []:
        for ch in z.get("checks") or []:
            zid, cid = z.get("id"), ch.get("id")
            v, pull = ch.get("verdict"), ch.get("pull")
            msg = ch.get("message") or ""
            if v == "FAIL" and not pull:
                definite = True
            elif v in ("FAIL", "PASS") and pull:
                out.append(reason(pull, msg, zid, cid))
            elif v == "PASS":
                out += pass_doubts(ch, pct, zid, cid)
            else:                               # INVALID, or anything that is not a verdict
                out.append(reason(pull or classify(msg), msg or f"verdict {v!r}", zid, cid))
    return definite, out


def decide(result: str, definite_fail: bool, reasons: list, reject=()) -> str:
    """The precedence: ABORTED -> PULL; a definite measured FAIL -> REJECT; a run refused or
    crashed (a `refused` / `error` reason) -> PULL; a doubt the line rejects (reject) ->
    REJECT; else any doubt (or a verdict that is not PASS) -> PULL; else CONTINUE."""
    if result == "ABORTED":
        return "PULL"
    if result == "FAIL" and definite_fail:
        return "REJECT"
    codes = {r.get("reason") for r in reasons}
    if codes & set(UNFINISHED):         # it never finished: nothing it recorded is a reject
        return "PULL"
    if codes & set(reject):
        return "REJECT"
    if result == "PASS" and not reasons:
        return "CONTINUE"
    return "PULL"


def _dedupe(items: list) -> list:
    out = []
    for r in items:
        if r not in out:
            out.append(r)
    return out


def settle(res: dict, guard_band_pct: float = 0.0, reject=(), tolerances=None) -> dict:
    """Decide res["disposition"] and res["pull_reasons"] from the run's verdict, its recipe
    zones (when it has checked zones), the reasons the flow recorded (add) and, for a run
    without zones, `tolerances` (its own limits: the guard band and the soundness of a PASS;
    a FAIL with a recorded doubt is not definite). Idempotent. -> res"""
    result = res.get("result")
    reasons = list(res.get("pull_reasons") or [])
    reject = tuple(reject or ())
    zones = [z for z in res.get("zones") or [] if z.get("checks")]
    if zones:
        definite, zr = zone_doubts(zones, guard_band_pct)
        reasons += zr
    else:
        # a run-level FAIL is definite unless the flow recorded a doubt about it (the numbers
        # it rests on are not trusted: a placement that did not converge, partial data)
        definite = result == "FAIL" and not any(r.get("reason") != "guard_band"
                                                for r in reasons)
        if result == "PASS":
            reasons += pass_doubts({"tolerances": tolerances}, guard_band_pct)
    if result == "ABORTED":
        if not any(r.get("reason") == "aborted" for r in reasons):
            reasons.append(reason("aborted", "the cycle was aborted before its verdict"))
    elif result != "PASS" and not (result == "FAIL" and definite) and not any(
            r.get("reason") != "guard_band" for r in reasons):
        # INVALID (or not a verdict) with no reason recorded: its invalid_reason says why
        why = res.get("invalid_reason") or f"result {result!r}"
        reasons.append(reason(classify(why), why))
    reasons = _dedupe(reasons)
    disp = decide(result, definite, reasons, reject)
    res["disposition"], res["pull_reasons"] = disp, reasons
    return res


def ensure(res: dict) -> dict:
    """Fail closed before a result is written: a missing disposition is settled (from the
    verdict, no guard band), an unknown one is a PULL, and so is a CONTINUE that is not a
    doubt-free PASS."""
    d = res.get("disposition")
    if d is None:
        settle(res)
    elif d not in DISPOSITIONS:
        res["disposition"] = "PULL"
        add(res, "invalid", f"unknown disposition {d!r}")
    elif d == "CONTINUE" and (res.get("result") != "PASS" or res.get("pull_reasons")):
        res["disposition"] = "PULL"
        if not res.get("pull_reasons"):
            add(res, classify(res.get("invalid_reason")),
                res.get("invalid_reason") or f"result {res.get('result')!r}")
    if not isinstance(res.get("pull_reasons"), list):
        res["pull_reasons"] = []
    return res


def of_result(result: Optional[str], disposition: Optional[str] = None,
              error: bool = False) -> Optional[str]:
    """A disposition for a verdict reported by someone else (the command bus): the given one
    when it is consistent with the verdict, else PASS -> CONTINUE, FAIL -> REJECT, anything
    else -> PULL. An error or ABORTED is always PULL; an unknown disposition is PULL."""
    if result is None and disposition is None and not error:
        return None
    if error or result == "ABORTED":
        return "PULL"
    if disposition is not None:
        if disposition not in DISPOSITIONS:
            return "PULL"
        if disposition == "CONTINUE" and result != "PASS":
            return "PULL"
        if disposition == "REJECT" and result not in ("FAIL", "INVALID"):
            return "PULL"
        return disposition
    return {"PASS": "CONTINUE", "FAIL": "REJECT"}.get(result, "PULL")


def register_key(result: Optional[str], disposition: Optional[str]) -> Optional[str]:
    """Which result code a register shows: PASS only for CONTINUE, FAIL for REJECT, INVALID /
    ABORTED as such, PULL for a PULL whose verdict is PASS or FAIL; None = no result yet."""
    if result is None:
        return None
    if disposition == "CONTINUE" and result == "PASS":
        return "PASS"
    if disposition == "REJECT":
        return "FAIL"
    if result in ("INVALID", "ABORTED"):
        return result
    return "PULL"


def exit_code(res: dict) -> int:
    """CONTINUE 0, REJECT 1, PULL 5 (anything else: 5, never 0)."""
    return EXIT_CODES.get(res.get("disposition"), EXIT_CODES["PULL"])


def summary(res: dict, n: int = 3) -> str:
    """One line: 'CONTINUE', 'REJECT', or 'PULL (quality inspection): zone/check: reason ...'."""
    d = res.get("disposition") or "PULL"
    if d != "PULL":
        return d
    bits = []
    for r in (res.get("pull_reasons") or [])[:n]:
        where = "/".join(x for x in (r.get("zone"), r.get("check")) if x)
        bits.append(f"{where + ': ' if where else ''}{r.get('reason')} ({r.get('message')})")
    more = len(res.get("pull_reasons") or []) - n
    return PULL_TEXT + (": " + "; ".join(bits) if bits else "") + (f" (+{more} more)"
                                                                   if more > 0 else "")
