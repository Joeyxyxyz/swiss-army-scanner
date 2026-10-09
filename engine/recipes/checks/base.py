"""
The check plugin interface: one class per check type, registered by id.

    from engine.recipes import register
    from engine.recipes.checks.base import Check, CheckResult, Param

    @register
    class MaxHeight(Check):
        id = "max_height"
        params = {"limit_mm": Param("number", unit="mm", min=0, doc="FAIL above this height")}

        def run(self, ctx) -> CheckResult:
            ...                                     # docs/CHECKS.md has the whole example

A check never touches the rig: it reads the scan data the runner hands it (CheckContext:
the metric cloud in mm, the decoded camera -> DMD correspondences, the flood pair, the part
frame, the zone's camera pixels, the golden scan when the recipe names one) and answers
PASS / FAIL / INVALID with its numbers (units in every value name: _mm, _deg, _mm2, _pct)
and the fault points (camera px, or xyz mm in the camera frame) the projector marks.
INVALID = the check could not measure (too few points, no golden, no image): never a pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

VERDICTS = ("PASS", "FAIL", "INVALID")
_REQUIRED = object()
KINDS = ("number", "int", "bool", "str", "choice", "pair", "list", "any")


class ParamError(ValueError):
    pass


@dataclass(frozen=True)
class Param:
    """One check parameter: kind number | int | bool | str | choice | pair (two numbers, lo <=
    hi) | list | any; unit (always say it: mm, deg, mm2, pct, px, count); default (omitted =
    required); bounds min / max (inclusive; open_min: > min)."""
    kind: str
    default: Any = _REQUIRED
    unit: Optional[str] = None
    doc: str = ""
    choices: tuple = ()
    min: Optional[float] = None
    max: Optional[float] = None
    open_min: bool = False

    @property
    def required(self) -> bool:
        return self.default is _REQUIRED

    def describe(self) -> dict:
        d = {"kind": self.kind, "unit": self.unit, "doc": self.doc,
             "required": self.required}
        if not self.required:
            d["default"] = self.default
        if self.choices:
            d["choices"] = list(self.choices)
        if self.min is not None:
            d["min" if not self.open_min else "greater_than"] = self.min
        if self.max is not None:
            d["max"] = self.max
        return d

    def coerce(self, name: str, v):
        """-> the validated value, or ParamError with a clear message."""
        unit = f" ({self.unit})" if self.unit else ""
        if v is None:
            if self.required:
                raise ParamError(f"{name} is required{unit}: {self.doc}")
            return None
        k = self.kind
        if k in ("number", "int"):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ParamError(f"{name} must be a number{unit}, got {v!r}")
            if k == "int" and float(v) != int(v):
                raise ParamError(f"{name} must be a whole number{unit}, got {v!r}")
            v = int(v) if k == "int" else float(v)
            self._bounds(name, v, unit)
            return v
        if k == "bool":
            if not isinstance(v, bool):
                raise ParamError(f"{name} must be true or false, got {v!r}")
            return v
        if k == "str":
            if not isinstance(v, str):
                raise ParamError(f"{name} must be text, got {v!r}")
            return v
        if k == "choice":
            if v not in self.choices:
                raise ParamError(f"{name} must be one of {', '.join(map(str, self.choices))}, "
                                 f"got {v!r}")
            return v
        if k == "pair":
            if (not isinstance(v, (list, tuple)) or len(v) != 2
                    or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                               for x in v) or v[0] > v[1]):
                raise ParamError(f"{name} must be [lo, hi]{unit} with lo <= hi, got {v!r}")
            return [float(v[0]), float(v[1])]
        if k == "list":
            if not isinstance(v, list):
                raise ParamError(f"{name} must be a list, got {v!r}")
            return list(v)
        return v

    def _bounds(self, name, v, unit):
        if self.min is not None and (v <= self.min if self.open_min else v < self.min):
            raise ParamError(f"{name} must be {'>' if self.open_min else '>='} {self.min:g}{unit}, "
                             f"got {v:g}")
        if self.max is not None and v > self.max:
            raise ParamError(f"{name} must be <= {self.max:g}{unit}, got {v:g}")


@dataclass
class CheckResult:
    """verdict PASS | FAIL | INVALID; values {name: number} with the unit in the name (and
    in `units`); fault_points [{"px": [x, y]} camera px | {"xyz_mm": [x, y, z]} camera frame
    mm, optional "label"]; key / limit: the headline value and its limit (same unit).
    tolerances: every value the check held to a limit, [tolerance(name, value, lo, hi)]
    (PASS means lo <= value <= hi; the recipe's guard band reads them; None = key / limit
    in deviation semantics). pull: a reason (engine/disposition.REASONS) when the verdict is
    in doubt, e.g. "code_unreadable" on a FAIL that read nothing at all: the part is PULLed
    for quality inspection instead of rejected."""
    verdict: str
    values: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    fault_points: list = field(default_factory=list)
    message: str = ""
    key: Optional[str] = None
    limit: Optional[float] = None
    warnings: list = field(default_factory=list)
    details: dict = field(default_factory=dict)
    tolerances: Optional[list] = None
    pull: Optional[str] = None

    def __post_init__(self):
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {self.verdict!r}")
        if not self.units:
            self.units = {k: unit_of(k) for k in self.values}

    @classmethod
    def invalid(cls, message: str, **kw) -> "CheckResult":
        return cls(verdict="INVALID", message=message, **kw)

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "values": dict(self.values), "units": dict(self.units),
                "key": self.key, "limit": self.limit, "message": self.message,
                "fault_points": [dict(f) for f in self.fault_points],
                "warnings": list(self.warnings), "details": dict(self.details),
                "tolerances": (None if self.tolerances is None
                               else [dict(t) for t in self.tolerances]),
                "pull": self.pull}


def unit_of(name: str) -> str:
    """The unit a value name carries (its suffix)."""
    for suf, u in (("_mm2", "mm2"), ("_mm", "mm"), ("_deg", "deg"), ("_pct", "%"),
                   ("_px", "camera px"), ("_dmd_px", "DMD px"), ("_z", "sigma")):
        if name.endswith(suf):
            return u
    if name.startswith("n_") or name.endswith("_count"):
        return "count"
    return ""


class Check:
    """Base class of a recipe check. Subclasses set `id`, `title`, `params` and implement
    run(ctx) -> CheckResult. `needs` names what the check reads beyond the cloud: "golden"
    (the recipe's golden scan), "flood" (the scan's flood pair), "corr_y" (both decoded
    axes); needs_for(params) refines it per instance (e.g. only with a golden tolerance)."""
    id: str = ""
    title: str = ""
    params: dict = {}
    needs: tuple = ()
    shapes: Optional[tuple] = None          # allowed zone shapes; None = any

    def __init__(self, params: Optional[dict] = None, check_id: Optional[str] = None):
        self.p = self.validate(params or {})
        self.check_id = check_id or self.id

    @classmethod
    def validate(cls, raw: dict) -> dict:
        if not isinstance(raw, dict):
            raise ParamError(f"params must be a mapping, got {raw!r}")
        unknown = sorted(set(raw) - set(cls.params))
        if unknown:
            raise ParamError(f"unknown params {unknown} (known: {', '.join(sorted(cls.params))})")
        out, errs = {}, []
        for name, spec in cls.params.items():
            try:
                out[name] = spec.coerce(name, raw.get(name, None if spec.required
                                                     else spec.default))
            except ParamError as e:
                errs.append(str(e))
        if not errs:
            try:
                cls.check_params(out)
            except ParamError as e:
                errs.append(str(e))
        if errs:
            raise ParamError("; ".join(errs))
        return out

    @classmethod
    def check_params(cls, p: dict) -> None:
        """Cross-parameter rules (raise ParamError)."""

    def needs_for(self) -> set:
        return set(self.needs)

    @classmethod
    def describe(cls) -> dict:
        return {"id": cls.id, "title": cls.title, "needs": list(cls.needs),
                "shapes": list(cls.shapes) if cls.shapes else None,
                "params": {k: v.describe() for k, v in cls.params.items()},
                "doc": (cls.__doc__ or "").strip()}

    def run(self, ctx) -> CheckResult:          # pragma: no cover - abstract
        raise NotImplementedError(f"check {self.id} has no run()")
