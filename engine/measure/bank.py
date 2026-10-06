"""
Good memory bank — flag anything not seen in known-good parts.

A bank is a set of known-good scans of the same part type at the same
geometry. From them the bank builds a per-pixel reference (median relief)
and spread (robust sigma, floored by the scan noise). `check` compares a
new scan: pixels deviating by more than k_sigma * spread, in connected
regions >= min_area_px, are anomalies -> "FLAG". This catches defect types
nobody wrote a rule for. Placement must be repeatable (same jig/zone); the
bank records which zone it was built for.

Storage: library/banks/<geometry_id>/<bank>/members.jsonl (append-only scan refs),
         model_<n>_<members-hash>.npz (reference/spread for that member set), meta.json
Outputs: library/output/anomalies/<ts>_<label>.json + .png

Design notes (why):
  * A bank is geometry-bound like a base (rules 5/6): a bank name that
    exists only at another geometry is refused, not silently re-started —
    after the rig moves, trash the old bank or use a new name.
  * Membership is append-only: members.jsonl gets {"op": "add"} lines and
    `remove()` appends {"op": "remove"} lines; a member whose scan folder
    was moved to trash/ is skipped with a warning.
  * The model file name carries the member count and a hash of the member
    ids, so any change to the member set builds a new file (never an
    overwrite) and a cached model can never be stale.
  * spread = max(per-pixel 1.4826*MAD across members, noise floor). With
    few members the per-pixel MAD is itself noisy and part edges shift by
    a pixel between placements, so the MAD map is max-filtered over 3x3.
    The floor is the members' robust sigma on the base plane (relief ~ 0),
    never below MIN_FLOOR_PX; it includes base-capture noise, so it errs
    on the conservative side. A zone scan of a part that fills the zone has
    no base plane in view: the floor is then estimated from the differences
    between good scans (robust sigma of (R_i - R_j) / sqrt(2), which also
    holds part-to-part variation) and the check says so. Below
    MIN_MEMBERS_SPREAD members the MAD is meaningless and only the floor is
    used (with a warning).
  * Memory: the model is built from each member's valid bounding box and
    in row chunks (the old full-frame stack + two sorted copies needed
    ~22 MB per member), and from at most the newest MAX_MODEL_MEMBERS
    members (a warning says when older ones are left out).
  * Only pixels valid in the scan AND in >= 2/3 of the members are judged;
    candidates get a 3x3 opening so 1-2 px edge slivers from sub-pixel
    placement differences never FLAG a good part. Thin (1-4 px) lines of
    unjudged px inside the judged area (raised surfaces decode with thin
    invalid lines) neither shred an anomaly before the opening nor split it
    into several regions after it.
  * Regions valid in every member but missing in the scan (new shadow,
    occluder, glare) are reported in `missing` + a warning, not FLAGged:
    validity flickers in dark areas on the real bench.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage

from .. import rules
from ..library.store import safe_label
from ..results import TIER_DEPTH, TIER_RELIEF
from .segment import BRIDGE_PX

MIN_FLOOR_PX = 0.05         # the spread never drops below this (relief px)
MIN_MEMBERS_SPREAD = 3      # per-pixel spread needs at least this many members
MAX_MODEL_MEMBERS = 64      # the model uses the newest this-many members (memory bound)
CHUNK_ELEMS = 8_000_000     # member x pixel elements per build chunk (~32 MB float32)
FLOOR_MAX_PAIRS = 8         # member pairs used for the difference-based noise floor
MEMBER_FRAC = 2.0 / 3.0     # a pixel is judged if valid in this fraction of members
OVERLAP_WARN = 0.80         # scan valid-mask vs reference coverage below this -> warning
_EIGHT = np.ones((3, 3), bool)


@dataclass
class BankInfo:
    name: str
    geometry_id: str
    n_scans: int
    scan_ids: list
    path: Path
    zone: Optional[str] = None
    warnings: list = field(default_factory=list)


@dataclass
class BankVerdict:
    result: str                 # "PASS" | "FLAG" | "INDETERMINATE"
    bank: str
    n_ref: int
    anomalies: list             # [{area_px, bbox, centroid, max_z, mean_dev, units}]
    k_sigma: float
    units: str                  # deviation units ("mm" | "px")
    max_z: float                # worst |deviation| / spread, sustained over a 3x3 patch
                                # (the detector's support: 1-2 px edge slivers don't count)
    json_path: Optional[Path] = None
    png_path: Optional[Path] = None
    warnings: list = field(default_factory=list)
    tier: str = TIER_RELIEF
    n_compared_px: int = 0      # pixels judged (valid in scan and >= 2/3 of members)
    overlap_pct: Optional[float] = None   # % of the reference's valid area valid in the scan
    noise_floor: Optional[float] = None   # spread floor (units)
    zone: Optional[str] = None  # zone the bank was built for
    missing: list = field(default_factory=list)   # [{area_px, bbox, centroid}] valid in all
                                                  # members, invalid in the scan
    max_z_pixel: Optional[float] = None   # raw single-pixel worst |z| (diagnostic only)


# ------------------------------------------------------------ bank storage

def _bank_dir(store, geometry_id: str, name: str) -> Path:
    return store.root / "banks" / geometry_id / name


def _other_geometries(store, name: str, geometry_id: str) -> list:
    return [g for g in store.geometry_ids("banks")
            if g != geometry_id and _bank_dir(store, g, name).is_dir()]


def _refuse_foreign(store, name: str, geometry_id: str, what: str) -> None:
    """Rules 5/6: a bank is geometry-bound; never mix scans across rigs."""
    if _bank_dir(store, geometry_id, name).is_dir():
        return
    others = _other_geometries(store, name, geometry_id)
    if others:
        raise rules.GeometryMismatch(
            f"{what}: bank '{name}' was built at geometry {', '.join(others)}, but the scan is "
            f"from geometry '{geometry_id}'. A bank is valid only at the geometry its good "
            "scans came from - after moving the rig, trash the old bank (api.delete) or build "
            "a new bank under a different name.")


def _rel(store, p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(store.root.resolve())).replace("\\", "/")
    except ValueError:
        return str(p)


def _resolve(store, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else store.root / q


def _read_lines(d: Path) -> list:
    f = d / "members.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _members(store, d: Path, check_exists: bool = True):
    """Active members in add order -> (list of member dicts, warnings)."""
    active = {}
    for rec in _read_lines(d):
        sid = rec.get("scan_id")
        if rec.get("op", "add") == "remove":
            active.pop(sid, None)
        elif sid not in active:
            active[sid] = rec
    warnings, out = [], []
    for rec in active.values():
        if check_exists and not (_resolve(store, rec["path"]) / "relief.npz").exists():
            warnings.append(f"member {rec['scan_id']} is missing from the library "
                            "(trashed?); skipped")
            continue
        out.append(rec)
    return out, warnings


def _meta(d: Path) -> dict:
    f = d / "meta.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _info(store, d: Path, geometry_id: str, warnings=None) -> BankInfo:
    members, w = _members(store, d)
    zone = _meta(d).get("zone")
    return BankInfo(name=d.name, geometry_id=geometry_id, n_scans=len(members),
                    scan_ids=[m["scan_id"] for m in members], path=d, zone=zone,
                    warnings=list(warnings or []) + w)


def _valid(relief, mask):
    return np.asarray(mask, bool) & np.isfinite(relief)


# --------------------------------------------------------------- the verbs

def add(session, scan, bank: str = "default") -> BankInfo:
    """Append a known-good scan to banks/<geometry_id>/<bank>/members.jsonl."""
    store = session.store
    name = safe_label(bank)
    _refuse_foreign(store, name, scan.geometry_id, "bank_add")
    relief, mask = scan.relief, scan.relief_mask
    valid = _valid(relief, mask)
    if not valid.any():
        raise ValueError(f"scan {scan.id} has no valid relief pixels; it cannot be a "
                         "known-good reference (re-shoot it)")
    d = _bank_dir(store, scan.geometry_id, name)
    zone = (scan.meta or {}).get("zone")
    warnings = []
    if not d.is_dir():
        d.mkdir(parents=True)
        store.write_json(d / "meta.json", {
            "name": name, "geometry_id": scan.geometry_id, "zone": zone,
            "preset": (scan.meta or {}).get("preset"),
            "created": datetime.now().isoformat(timespec="seconds"),
            "note": "good memory bank: known-good scans of one part type at one geometry "
                    "and placement; members.jsonl is append-only"})
    meta = _meta(d)
    if meta.get("geometry_id", scan.geometry_id) != scan.geometry_id:     # hand-copied folder
        rules.require_same_geometry(scan.geometry_id, meta["geometry_id"], "bank_add")
    members, mw = _members(store, d)
    warnings += mw
    if any(m["scan_id"] == scan.id for m in members):
        warnings.append(f"scan {scan.id} is already in bank '{name}'; not added twice "
                        "(it would bias the reference)")
        return _info(store, d, scan.geometry_id, warnings)
    if members:
        with np.load(_resolve(store, members[0]["path"]) / "relief.npz") as z:
            first = _valid(z["relief"], z["mask"])
        if first.shape != valid.shape:
            raise rules.GeometryMismatch(
                f"bank_add: scan relief is {valid.shape[1]}x{valid.shape[0]} but bank '{name}' "
                f"holds {first.shape[1]}x{first.shape[0]} scans (camera settings changed?)")
        ov = (valid & first).sum() / max(first.sum(), 1)
        if ov < OVERLAP_WARN:
            warnings.append(f"only {100 * ov:.0f}% of the bank's valid area is valid in this "
                            "scan - is the part placed the same way (same jig / zone)?")
    bank_zone = meta.get("zone")
    if bank_zone != zone:
        warnings.append(f"bank '{name}' was built for zone {bank_zone!r} but this scan was "
                        f"taken in zone {zone!r}; placement must be repeatable")
    rec = {"op": "add", "scan_id": scan.id, "label": scan.label, "path": _rel(store, scan.path),
           "time": datetime.now().isoformat(timespec="seconds"), "zone": zone,
           "geometry_id": scan.geometry_id, "valid_px": int(valid.sum())}
    with open(d / "members.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    info = _info(store, d, scan.geometry_id, warnings)
    if info.n_scans < MIN_MEMBERS_SPREAD:
        info.warnings.append(f"bank '{name}' has {info.n_scans} good scan(s); add at least "
                             f"{MIN_MEMBERS_SPREAD} so per-pixel spread can be learned (until "
                             "then checks use the noise floor only)")
    return info


def remove(session, scan, bank: str = "default", reason: str = "") -> BankInfo:
    """Drop a member (appends a 'remove' line; nothing is deleted)."""
    store = session.store
    name = safe_label(bank)
    _refuse_foreign(store, name, scan.geometry_id, "bank_remove")
    d = _bank_dir(store, scan.geometry_id, name)
    members, _ = _members(store, d, check_exists=False)
    if not any(m["scan_id"] == scan.id for m in members):
        raise ValueError(f"scan {scan.id} is not a member of bank '{name}' "
                         f"at geometry {scan.geometry_id}")
    with open(d / "members.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"op": "remove", "scan_id": scan.id, "reason": reason,
                            "time": datetime.now().isoformat(timespec="seconds")}) + "\n")
    return _info(store, d, scan.geometry_id)


def list_banks(session, geometry_id: Optional[str] = None) -> list:
    store = session.store
    geos = [geometry_id] if geometry_id else store.geometry_ids("banks")
    return [_info(store, d, g) for g in geos for d in store.entries("banks", g)]


def check(session, scan, bank: str = "default", k_sigma: float = 4.0, calibration=None,
          min_area_px: int = 25) -> BankVerdict:
    if not k_sigma > 0:
        raise ValueError("k_sigma must be > 0 (deviation threshold in spreads)")
    if int(min_area_px) < 1:
        raise ValueError("min_area_px must be >= 1")
    store = session.store
    name = safe_label(bank)
    _refuse_foreign(store, name, scan.geometry_id, "bank_check")
    d = _bank_dir(store, scan.geometry_id, name)
    warnings = []

    # rule 6: deviations in mm only with a depth calibration for this geometry
    cal = calibration
    if cal is not None and not rules.calibration_matches(cal, scan.geometry_id):
        warnings.append(f"calibration {cal.id} is for geometry {cal.geometry_id}, "
                        f"not {scan.geometry_id}; ignored (rule 6)")
        cal = None
    zs = cal.mm_per_px if (cal is not None and cal.has_z) else None
    xy = cal.lateral_px_per_mm if (cal is not None and cal.has_xy) else None
    units, tier = ("mm", TIER_DEPTH) if zs else ("px", TIER_RELIEF)
    if not zs:
        warnings.append(f"no depth calibration for geometry {scan.geometry_id}; "
                        "deviations in relief px (rule 6)")
    to_u = (lambda v: v * zs) if zs else (lambda v: v)       # intercept cancels in a difference

    members, mw = _members(store, d) if d.is_dir() else ([], [])
    warnings += mw
    zone = _meta(d).get("zone") if d.is_dir() else None
    label = f"{scan.label}_vs_{name}"
    v = BankVerdict(result="INDETERMINATE", bank=name, n_ref=len(members), anomalies=[],
                    k_sigma=float(k_sigma), units=units, max_z=float("nan"),
                    warnings=warnings, tier=tier, zone=zone)
    if not members:
        warnings.append(f"bank '{name}' has no good scans at geometry {scan.geometry_id}; "
                        "add some with bank_add first")
        _write(session, scan, v, label, None)
        return v

    model = _load_or_build(store, d, members)
    relief = np.asarray(scan.relief, np.float64)
    valid = _valid(relief, scan.relief_mask)
    ref, sig, n_valid = model["reference"], model["sigma"], model["n_valid"]
    if relief.shape != ref.shape:
        raise rules.GeometryMismatch(
            f"bank_check: scan relief is {relief.shape[1]}x{relief.shape[0]} but bank "
            f"'{name}' was built from {ref.shape[1]}x{ref.shape[0]} scans")
    n = int(model["n_members"])           # members in the model (newest MAX_MODEL_MEMBERS)
    if n < len(members):
        warnings.append(f"bank '{name}' has {len(members)} good scans; the model uses the "
                        f"newest {n} (memory bound, MAX_MODEL_MEMBERS)")
    floor = float(model["noise_floor"])
    src = str(model["floor_source"]) if "floor_source" in model else "base_plane"
    if src == "member_diff":
        warnings.append(f"no base plane in view in the good scans (part fills the zone?): the "
                        f"noise floor {floor:.3f} px was estimated from differences between "
                        "good scans, which include part-to-part variation")
    elif src == "minimum":
        warnings.append(f"no base plane in view and no second good scan: the noise floor is "
                        f"the minimum {MIN_FLOOR_PX} px, which is likely optimistic - add "
                        "more good scans")
    if n < MIN_MEMBERS_SPREAD:
        warnings.append(f"bank '{name}' has only {n} good scan(s); using the noise floor "
                        f"({floor:.3f} px) as the spread everywhere - add at least "
                        f"{MIN_MEMBERS_SPREAD} for a learned per-pixel spread")
        spread = np.full(ref.shape, floor)
    else:
        spread = np.maximum(sig.astype(np.float64), floor)
    if any(m["scan_id"] == scan.id for m in members):
        warnings.append(f"scan {scan.id} is itself a member of bank '{name}'; the check "
                        "is optimistic")
    scan_zone = (scan.meta or {}).get("zone")
    if zone != scan_zone:
        warnings.append(f"bank '{name}' was built for zone {zone!r} but this scan was taken "
                        f"in zone {scan_zone!r}")

    ref_ok = (n_valid >= MEMBER_FRAC * n - 1e-9) & np.isfinite(ref)
    comp = valid & ref_ok
    n_cmp = int(comp.sum())
    overlap = float(comp.sum() / max(ref_ok.sum(), 1))
    v.n_compared_px, v.overlap_pct = n_cmp, round(100 * overlap, 2)
    v.noise_floor = round(abs(to_u(floor)), 5)
    if overlap < OVERLAP_WARN:
        warnings.append(f"only {100 * overlap:.0f}% of the reference's valid area is valid in "
                        "this scan - placement not repeatable (same jig / zone?), or "
                        "something new shadows the part")
    if n_cmp == 0:
        warnings.append("scan and bank share no valid pixels; cannot judge")
        _write(session, scan, v, label, None)
        return v

    dev = np.where(comp, relief - np.where(ref_ok, ref, 0.0), np.nan)
    az = np.where(comp, np.abs(np.nan_to_num(dev / spread)), 0.0)
    v.max_z_pixel = round(float(az.max()), 3)
    # Thin gaps (<= 2*BRIDGE_PX px) of unjudged px INSIDE the judged area take
    # their nearest judged value before the opening: raised surfaces decode with
    # 1-2 px invalid lines (clipped fringe samples), and where the scan's height
    # differs from the members' - i.e. on an anomaly - its lines and theirs do
    # not coincide, so the judged px there are 2-4 px strips the 3x3 opening
    # would erase. Outer edges (shadows, the field edge) are not filled, so
    # 1-2 px placement slivers along the part outline still open away.
    gap = ~comp & ndimage.binary_closing(comp, structure=_EIGHT, iterations=BRIDGE_PX)
    if gap.any():
        near = ndimage.distance_transform_edt(~comp, return_distances=False,
                                              return_indices=True)
        az = np.where(gap, az[tuple(near)], az)
        del near
    # grey opening == binary opening of every threshold, so `azo > k` IS the
    # 3x3-opened candidate mask and max(azo) < k guarantees a PASS
    azo = np.where(comp, ndimage.grey_opening(az, size=(3, 3)), 0.0)
    v.max_z = round(float(azo.max()), 3)

    cand = azo > k_sigma
    v.anomalies = _regions(cand, int(min_area_px), dev=dev, az=azo, to_u=to_u, units=units,
                           xy=xy, bridge=~comp)
    miss = ndimage.binary_opening((n_valid >= n) & ~valid, structure=np.ones((3, 3)))
    v.missing = _regions(miss, max(4 * int(min_area_px), 100))
    if v.missing:
        tot = sum(r["area_px"] for r in v.missing)
        warnings.append(f"{len(v.missing)} region(s), {tot} px, valid in every good scan are "
                        f"missing in this one (largest at bbox {v.missing[0]['bbox']}): new "
                        "shadow / occluder, glare or a dark spot? Not judged.")
    v.result = "FLAG" if v.anomalies else "PASS"
    _write(session, scan, v, label, dict(relief=relief, comp=comp, cand=cand))
    return v


# ------------------------------------------------------------------ model

def _member_key(members) -> str:
    ids = "\n".join(sorted(m["scan_id"] for m in members))
    return hashlib.sha1(ids.encode("utf-8")).hexdigest()[:10]


def _load_or_build(store, d: Path, members) -> dict:
    f = d / f"model_{len(members)}_{_member_key(members)}.npz"
    if f.exists():
        with np.load(f) as z:
            return {k: z[k] for k in z.files}
    use = members[-MAX_MODEL_MEMBERS:]
    model = build_model(_load_member(store, d, m) for m in use)   # one full frame at a time
    model["member_ids"] = np.array([m["scan_id"] for m in use])
    store.write_npz(f, **model)
    return model


def _load_member(store, d: Path, m: dict):
    p = _resolve(store, m["path"])
    meta_f = p / "meta.json"
    bank_geo = _meta(d).get("geometry_id")
    if meta_f.exists() and bank_geo:
        geo = json.loads(meta_f.read_text(encoding="utf-8")).get("geometry_id")
        rules.require_same_geometry(geo, bank_geo, f"bank member {m['scan_id']}")
    with np.load(p / "relief.npz") as z:
        return np.where(z["mask"], z["relief"], np.nan).astype(np.float32)


def _nan_median0(stack):
    """Median along axis 0 ignoring NaN (np.sort puts NaN last), vectorised."""
    cnt = np.isfinite(stack).sum(axis=0)
    s = np.sort(stack, axis=0)
    lo = np.clip((cnt - 1) // 2, 0, None)[None]
    hi = np.clip(cnt // 2, 0, stack.shape[0] - 1)[None]
    med = 0.5 * (np.take_along_axis(s, lo, 0)[0] + np.take_along_axis(s, hi, 0)[0])
    return np.where(cnt > 0, med, np.nan)


def base_plane_sigma(relief) -> Optional[float]:
    """Robust sigma of the relief on the base plane (the datum, relief ~ 0).

    Starts from pixels within 1 px of the datum and widens the band to 5
    sigma, so a noisy rig is not truncated to an optimistic floor."""
    v = np.asarray(relief, np.float64)
    v = v[np.isfinite(v)]
    band, sig = 1.0, None
    for _ in range(3):
        s = v[np.abs(v) < band]
        if s.size < 1000:
            return sig
        sig = 1.4826 * float(np.median(np.abs(s - np.median(s))))
        band = max(1.0, 5.0 * sig)
    return sig


def _crop(relief):
    """-> (float32 crop of the relief's valid bounding box, (y0, x0)) or (None, None)."""
    fin = np.isfinite(relief)
    rows, cols = np.flatnonzero(fin.any(axis=1)), np.flatnonzero(fin.any(axis=0))
    if rows.size == 0:
        return None, None
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    return np.array(relief[y0:y1, x0:x1], dtype=np.float32), (y0, x0)


def _pair_sigma(crops) -> Optional[float]:
    """Noise from member-to-member differences: median over consecutive pairs of
    1.4826 * MAD(R_i - R_j) / sqrt(2) on their common valid px (None if no pair)."""
    items = [c for c in crops if c[0] is not None][-(FLOOR_MAX_PAIRS + 1):]
    sig = []
    for (a, (ay, ax)), (b, (by, bx)) in zip(items, items[1:]):
        y0, x0 = max(ay, by), max(ax, bx)
        y1, x1 = min(ay + a.shape[0], by + b.shape[0]), min(ax + a.shape[1], bx + b.shape[1])
        if y1 <= y0 or x1 <= x0:
            continue
        dv = a[y0 - ay:y1 - ay, x0 - ax:x1 - ax] - b[y0 - by:y1 - by, x0 - bx:x1 - bx]
        dv = dv[np.isfinite(dv)]
        if dv.size < 1000:
            continue
        sig.append(1.4826 * float(np.median(np.abs(dv - np.median(dv)))) / np.sqrt(2.0))
    return float(np.median(sig)) if sig else None


def build_model(reliefs) -> dict:
    """Member reliefs (NaN where invalid; any iterable) -> reference, sigma map,
    counts, floor. Members are kept as valid-bbox crops and the median / MAD
    run in row chunks, so memory is ~ the crops plus one CHUNK_ELEMS stack."""
    crops, floors, shape = [], [], None
    for r in reliefs:
        r = np.asarray(r, np.float32)
        if shape is None:
            shape = r.shape
        elif r.shape != shape:
            raise ValueError(f"bank member relief is {r.shape[1]}x{r.shape[0]}, the others "
                             f"are {shape[1]}x{shape[0]}")
        sb = base_plane_sigma(r)
        if sb is not None:
            floors.append(sb)
        crops.append(_crop(r))
    if shape is None:
        raise ValueError("build_model needs at least one member relief")
    n = len(crops)
    ref = np.full(shape, np.nan, np.float32)
    sig = np.zeros(shape, np.float32)
    n_valid = np.zeros(shape, np.uint16)
    boxes = [(o[0], o[0] + c.shape[0], o[1], o[1] + c.shape[1]) for c, o in crops if c is not None]
    if boxes:
        Y0, Y1 = min(b[0] for b in boxes), max(b[1] for b in boxes)
        X0, X1 = min(b[2] for b in boxes), max(b[3] for b in boxes)
        Wb = X1 - X0
        step = max(1, CHUNK_ELEMS // max(n * Wb, 1))
        for ya in range(Y0, Y1, step):
            yb = min(ya + step, Y1)
            S = np.full((n, yb - ya, Wb), np.nan, np.float32)
            for i, (c, o) in enumerate(crops):
                if c is None:
                    continue
                ta, tb = max(ya, o[0]), min(yb, o[0] + c.shape[0])
                if ta < tb:
                    S[i, ta - ya:tb - ya, o[1] - X0:o[1] - X0 + c.shape[1]] = c[ta - o[0]:tb - o[0]]
            n_valid[ya:yb, X0:X1] = np.isfinite(S).sum(axis=0)
            med = _nan_median0(S).astype(np.float32)
            ref[ya:yb, X0:X1] = med
            if n >= MIN_MEMBERS_SPREAD:
                np.subtract(S, med[None], out=S)
                np.abs(S, out=S)
                sig[ya:yb, X0:X1] = np.nan_to_num(1.4826 * _nan_median0(S), nan=0.0)
            del S
    if n >= MIN_MEMBERS_SPREAD:
        sig = ndimage.maximum_filter(sig, size=3)           # +-1 px placement tolerance
    if floors:
        floor, source = float(np.median(floors)), "base_plane"
    else:
        # no base plane in view (a zone scan the part fills): member-to-member noise
        pair = _pair_sigma(crops) if n >= 2 else None
        floor, source = (pair, "member_diff") if pair is not None else (0.0, "minimum")
    floor = max(floor, MIN_FLOOR_PX)
    return {"reference": ref, "sigma": sig.astype(np.float32),
            "n_valid": n_valid, "noise_floor": np.float64(floor), "n_members": np.int64(n),
            "floor_source": np.array(source)}


# ------------------------------------------------------------ regions/out

def _regions(mask, min_area: int, dev=None, az=None, to_u=None, units=None, xy=None,
             bridge=None) -> list:
    """Connected regions of `mask`. `bridge`: px (not judged / invalid) through
    which pieces within 2*BRIDGE_PX of each other still count as one region -
    raised surfaces come back gridded by thin invalid lines. Regions hold only
    mask px (area, bbox, centroid)."""
    link = mask
    if bridge is not None:
        link = mask | (bridge & ndimage.binary_dilation(mask, iterations=BRIDGE_PX))
    lab, k = ndimage.label(link, structure=_EIGHT)
    if k == 0:
        return []
    lab[~mask] = 0
    idx = np.arange(1, k + 1)
    area = ndimage.sum(mask, lab, idx)
    keep = [i for i, a in zip(idx, area) if a >= min_area]
    if not keep:
        return []
    slices = ndimage.find_objects(lab)
    cms = ndimage.center_of_mass(mask, lab, keep)
    out = []
    for i, (cy, cx) in zip(keep, cms):
        sy, sx = slices[i - 1]
        r = {"area_px": int(area[i - 1]),
             "bbox": [int(sx.start), int(sy.start), int(sx.stop - 1), int(sy.stop - 1)],
             "centroid": [round(float(cx), 1), round(float(cy), 1)]}
        if dev is not None:
            m = lab[sy, sx] == i
            dv = dev[sy, sx][m]
            peak = dv[np.argmax(np.abs(dv))]
            r.update(max_z=round(float(az[sy, sx][m].max()), 2),
                     mean_dev=round(float(to_u(np.mean(dv))), 5),
                     peak_dev=round(float(to_u(peak)), 5), units=units)
            if xy:
                r["area_mm2"] = round(r["area_px"] / xy ** 2, 3)
        out.append(r)
    out.sort(key=lambda r: -r["area_px"])
    return out


def _json_safe(o):
    """NaN/inf -> null: the output bin is read by browsers too (JSON.parse rejects NaN)."""
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _json_safe(x) for k, x in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(x) for x in o]
    return o


def _write(session, scan, v: BankVerdict, label: str, maps: Optional[dict]) -> None:
    store = session.store
    out = store.output_path("anomalies", label, ".json")
    store.write_json(out, _json_safe({"scan": scan.id, "scan_label": scan.label,
                                      "geometry_id": scan.geometry_id,
                                      **{k: val for k, val in vars(v).items()
                                         if k not in ("json_path", "png_path")}}))
    v.json_path = out
    store.record_output("anomalies", out, label=label, geometry_id=scan.geometry_id,
                        source=scan.path)
    png = annotate_png(v, **maps) if maps else None
    if png is not None:
        p = out.with_suffix(".png")
        rules.ensure_new_path(p)
        with open(p, "xb") as f:
            f.write(png)
        v.png_path = p
        store.record_output("anomalies", p, label=label, geometry_id=scan.geometry_id,
                            source=scan.path)


def annotate_png(v: BankVerdict, relief, comp, cand, max_boxes: int = 12) -> Optional[bytes]:
    """Scan relief (grey), pixels not judged (dark), |z| > k (red), anomaly boxes."""
    try:
        import cv2
    except ImportError:
        return None
    vals = relief[comp]
    lo, hi = (np.percentile(vals, 1), np.percentile(vals, 99)) if vals.size else (0.0, 1.0)
    g = np.clip((np.nan_to_num(relief, nan=lo) - lo) / (hi - lo + 1e-12), 0, 1)
    g8 = (40 + 180 * g).astype(np.uint8)
    vis = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)
    vis[~comp] = (vis[~comp] * 0.25).astype(np.uint8)
    vis[cand] = (0.4 * vis[cand] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, a in enumerate(v.anomalies[:max_boxes]):
        x0, y0, x1, y1 = a["bbox"]
        cv2.rectangle(vis, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (0, 220, 255), 2)
        txt = f"#{i + 1} {a['mean_dev']:+.3f}{a['units']} z{a['max_z']:.0f} {a['area_px']}px"
        (tw, th), bl = cv2.getTextSize(txt, font, 0.5, 1)
        tx, ty = min(x0, vis.shape[1] - tw - 4), max(th + 40, y0 - 8)
        cv2.rectangle(vis, (tx - 3, ty - th - 4), (tx + tw + 3, ty + bl + 2), (0, 0, 0), -1)
        cv2.putText(vis, txt, (tx, ty), font, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    for m in v.missing[:max_boxes]:
        x0, y0, x1, y1 = m["bbox"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 160, 0), 1)
    col = {"PASS": (0, 200, 0), "FLAG": (0, 0, 255)}.get(v.result, (0, 200, 255))
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(vis, f"{v.result}  bank '{v.bank}' n={v.n_ref}  k={v.k_sigma:g} sigma  "
                f"max_z {v.max_z:.1f}  anomalies {len(v.anomalies)}  units {v.units}",
                (10, 24), font, 0.65, col, 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", vis)
    return buf.tobytes() if ok else None
