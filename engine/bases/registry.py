"""
Basis registry: name -> PatternBasis class.

KNOWN lists every basis the architecture defines, so presets can be
validated now; a name is usable for capture once its class is registered
(graycode / phaseshift / hybrid land in build step 2, checkerboard /
diagonal / hadamard in step 9).
"""

from __future__ import annotations

from numbers import Real

from .base import PatternBasis

# gray-code options shared by the graycode spec and the hybrid composite:
# coarse_bits = decode only the N coarsest planes; min_bit_contrast = TI's
# normal-vs-inverse floor (grey levels) for complementary decode;
# fix_period_jumps = opt-in local-median fringe-order correction
_GRAY_OPTS = {"coarse_bits", "min_bit_contrast", "fix_period_jumps"}
# phase-shift options: sat_level = Mono8 counts at which a fringe sample is
# clipped (dropped, phase re-solved from the rest); null = off
_PHASE_OPTS = {"sat_level"}

KNOWN = {
    "graycode":     {"bits", "complementary", "axis"} | _GRAY_OPTS,
    "phaseshift":   {"steps", "period_px", "axis"} | _PHASE_OPTS,
    "hybrid":       ({"bits", "steps", "period_px", "complementary", "axis"}
                     | _GRAY_OPTS | _PHASE_OPTS),
    "checkerboard": {"square_px", "shifts"},
    "diagonal":     {"bits", "steps", "period_px"},
    "hadamard":     {"order", "fov_px"},
    # N-step x several periods, hierarchical temporal unwrap (bases/multifreq.py)
    "multifreq":    {"steps", "periods_px", "axis", "diamond_parity", "mod_min"},
}

_AXES = ("x", "y")

_REGISTRY: dict[str, type[PatternBasis]] = {}


def register(cls: type[PatternBasis]) -> type[PatternBasis]:
    if cls.name not in KNOWN:
        raise ValueError(f"basis {cls.name!r} is not in the architecture's KNOWN list")
    _REGISTRY[cls.name] = cls
    return cls


def validate_spec(spec: dict) -> None:
    """Names always; values of the stripe options too, so a bad preset fails
    at load_preset instead of halfway through a capture."""
    name = spec.get("name")
    if name not in KNOWN:
        raise ValueError(f"unknown basis {name!r}; known: {sorted(KNOWN)}")
    extra = set(spec) - {"name"} - KNOWN[name]
    if extra:
        raise ValueError(f"basis {name!r}: unknown params {sorted(extra)}")
    axes = _AXES + ("xy",) if name == "multifreq" else _AXES
    if "axis" in spec and spec["axis"] not in axes:
        raise ValueError(f"basis {name!r}: axis must be one of {list(axes)}, "
                         f"got {spec['axis']!r}")
    if name == "multifreq":
        _validate_multifreq(spec)
    if "complementary" in spec and not isinstance(spec["complementary"], bool):
        raise ValueError(f"basis {name!r}: complementary must be true/false, "
                         f"got {spec['complementary']!r}")
    cb = spec.get("coarse_bits")
    if cb is not None:
        if isinstance(cb, bool) or not isinstance(cb, int) or cb < 1:
            raise ValueError(f"basis {name!r}: coarse_bits must be a positive integer "
                             f"(or null = all bits), got {cb!r}")
        if isinstance(spec.get("bits"), int) and cb > spec["bits"]:
            raise ValueError(f"basis {name!r}: coarse_bits {cb} > bits {spec['bits']}")
    mbc = spec.get("min_bit_contrast")
    if "min_bit_contrast" in spec and (isinstance(mbc, bool) or not isinstance(mbc, Real)
                                       or mbc < 0):
        raise ValueError(f"basis {name!r}: min_bit_contrast must be >= 0 grey levels, "
                         f"got {mbc!r}")
    if "fix_period_jumps" in spec and not isinstance(spec["fix_period_jumps"], bool):
        raise ValueError(f"basis {name!r}: fix_period_jumps must be true/false, "
                         f"got {spec['fix_period_jumps']!r}")
    sl = spec.get("sat_level")
    if sl is not None and (isinstance(sl, bool) or not isinstance(sl, Real)
                           or not 0 < sl <= 255):
        raise ValueError(f"basis {name!r}: sat_level must be in (0, 255] Mono8 counts "
                         f"(or null = no clipped-sample handling), got {sl!r}")


def _validate_multifreq(spec: dict) -> None:
    st = spec.get("steps")
    if "steps" in spec and (isinstance(st, bool) or not isinstance(st, int) or st < 3):
        raise ValueError(f"basis 'multifreq': steps must be an integer >= 3, got {st!r}")
    per = spec.get("periods_px")
    if "periods_px" in spec and (not isinstance(per, (list, tuple)) or len(per) < 2 or not all(
            isinstance(p, Real) and not isinstance(p, bool) and p >= 3 for p in per)
            or list(per) != sorted(per)):
        raise ValueError("basis 'multifreq': periods_px must be >= 2 increasing periods "
                         f"(DMD px, each >= 3), got {per!r}")
    dp = spec.get("diamond_parity")
    if dp not in (None, 0, 1) or isinstance(dp, bool):
        raise ValueError(f"basis 'multifreq': diamond_parity must be null, 0 or 1, got {dp!r}")
    mm = spec.get("mod_min")
    if mm is not None and (isinstance(mm, bool) or not isinstance(mm, Real) or mm < 0):
        raise ValueError(f"basis 'multifreq': mod_min must be >= 0 grey levels, got {mm!r}")


def _load_builtin():
    from . import hybrid  # noqa: F401  (registers itself)
    from . import multifreq  # noqa: F401
    from . import checkerboard  # noqa: F401


def is_implemented(name: str) -> bool:
    _load_builtin()
    return name in _REGISTRY


def get(spec: dict) -> PatternBasis:
    validate_spec(spec)
    _load_builtin()
    name = spec["name"]
    if name not in _REGISTRY:
        raise NotImplementedError(
            f"basis {name!r} is specified but not implemented yet "
            "(graycode/phaseshift/hybrid: build step 2)")
    params = {k: v for k, v in spec.items() if k != "name"}
    return _REGISTRY[name](**params)
