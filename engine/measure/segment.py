"""
Object segmentation on a relief map — ported from calibrate_depth.py
(the blob-fix revision) with SystemExit turned into exceptions.

Shadow/occlusion appears as thin, scattered outliers of the OPPOSITE
sign to the object; a naive |relief| > k*MAD grabs both and the median
washes to ~0 (the "1.63 px" bug). Instead:
  1. threshold each sign separately at k*MAD,
  2. keep the LARGEST connected component per sign,
  3. pick the sign whose blob is bigger (or honour `sign`),
  4. return that blob and its median relief.

Reference level. A normal scan's relief is datum-removed (the field median,
i.e. the base plane when the part is a minority, sits at 0), so the blob is
thresholded around the median (`ref=None`). A zone scan is NOT: only the
zone is lit, relief is the raw object - base difference and the base plane
sits at 0 whatever the part covers (api.capture_object, datum=False). When
the part fills most of the zone the median is the part TOP, and thresholding
around it picks the base ring as "the object" (H ~ 0, W/L = the zone). So
datum-free scans are segmented around `ref=0.0` (scan_blob does this):
  * the base level and its noise come from the pixels near ref only;
  * if (almost) no pixel is near ref, no base is in view: the part fills the
    zone and the whole lit zone (on the part's side) is taken as its top -
    H is its median, W/L are the zone's extent, and a warning says so.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

BASE_BAND_PX = 1.0          # base-plane search band around ref (relief px)
BASE_MIN_PX = 100           # fewer base px than this (or ...
BASE_MIN_FRAC = 0.03        # ... this fraction of the valid px) -> no base in view
EDGE_SLACK_PX = 3           # blob bbox this close to the zone's bbox -> clipped by the zone
BRIDGE_PX = 2               # object pieces split by invalid px lines this thin (x2) stay one


class SegmentationError(ValueError):
    pass


def _largest_blob(mask, finite=None):
    """Largest connected component of the 3x3-opened mask. With `finite`, pieces
    split only by thin lines of INVALID px (<= 2*BRIDGE_PX wide - zone scans'
    part tops come back gridded by 1-2 px NaN lines) count as one component;
    valid non-object px never bridge. The blob itself holds only mask px."""
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    link = mask
    if finite is not None:
        link = mask | (~finite & ndimage.binary_dilation(mask, iterations=BRIDGE_PX))
    lab, n = ndimage.label(link)
    if n == 0 or not mask.any():
        return None, 0
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    return (lab == k) & mask, int(sizes.max())


def support(blob, gap_px: int = 2):
    """The blob with gaps up to ~2*gap_px wide filled in (thin NaN lines, and the
    lone valid px in them that the opening dropped), for region operations such
    as erosion that those gaps would otherwise shred."""
    closed = ndimage.binary_closing(blob, structure=np.ones((3, 3)), iterations=gap_px)
    return blob | closed


def is_datum_free(scan) -> bool:
    """True when the scan's relief was NOT median-datum'd (zone scans): the
    base plane is at relief 0 whatever fraction of the view the part covers."""
    meta = getattr(scan, "meta", None) or {}
    if meta.get("zone"):
        return True
    d = meta.get("datum_px")
    return d is not None and float(d) == 0.0 and int(meta.get("overlap_px") or 0) > 0


def scan_blob(scan, **kw):
    """object_blob on a scan, with the right reference level for its datum.
    -> (blob mask, relief px over the blob vs the base plane, info dict)."""
    ref = 0.0 if is_datum_free(scan) else None
    blob, h, info = object_blob(scan.relief, ref=ref, **kw)
    if ref is not None:
        info["warnings"] += _zone_edge_warning(blob, np.isfinite(np.asarray(scan.relief)),
                                               info)
    return blob, h, info


def _zone_edge_warning(blob, valid, info) -> list:
    """Datum-free (zone) scans: a blob reaching the zone's edge is clipped by
    it, so its W/L are the zone's, not the part's."""
    if not info.get("base_in_view", True):
        return []                       # already said: the whole zone is the part
    vy, vx = np.nonzero(valid)
    by, bx = np.nonzero(blob)
    s = EDGE_SLACK_PX
    hit = (bx.min() <= vx.min() + s or bx.max() >= vx.max() - s or
           by.min() <= vy.min() + s or by.max() >= vy.max() - s)
    info["clipped_by_zone"] = bool(hit)
    if not hit:
        return []
    return ["the part reaches the edge of the scanned zone: W/L (and any profile across it) "
            "are clipped by the zone - scan a larger zone to measure the full footprint"]


def _base_level(v, ref: float, band_max: float):
    """Base plane near `ref`: -> (level, raw MAD, n px in band) or (None, None, n)."""
    center, band = ref, band_max
    level = mad = None
    n = 0
    for _ in range(3):
        s = v[np.abs(v - center) < band]
        n = int(s.size)
        if n < BASE_MIN_PX:
            return None, None, n
        level = float(np.median(s))
        mad = float(np.median(np.abs(s - level)))
        # re-centre on the base (stays within band_max of ref) and fit the band
        # to its noise, so part edges / a thin part do not inflate the MAD
        center = float(np.clip(level, ref - band_max, ref + band_max))
        band = min(band_max, max(5.0 * 1.4826 * mad, 0.05))
    return level, mad, n


def object_blob(relief, sign: str = "auto", k_mad: float = 4.0, min_px: int = 50,
                ref=None):
    """-> (blob mask, median relief px over blob, info dict).

    ref=None: threshold around the median of the valid px (datum-removed
    relief, part a minority of the field). ref=<number>: the base plane is at
    that level (datum-free zone scans, ref=0.0); the returned height is the
    blob's median relief minus ref. info["warnings"] lists anything the
    caller should pass on (no base in view, blob within noise of the base)."""
    r = np.asarray(relief, float)
    finite = np.isfinite(r)
    v = r[finite]
    if v.size < 100:
        raise SegmentationError("too few valid pixels in relief")
    med = float(np.median(v))
    warnings = []
    info = {"p1_p50_p99_px": [float(np.percentile(v, q)) for q in (1, 50, 99)]}

    if ref is None:
        level = med
        mad = float(np.median(np.abs(v - med))) + 1e-9
        h0 = 0.0
    else:
        ref = float(ref)
        m = med - ref
        # keep the base search band on the base's side of the median, so a
        # thin part covering most of the view is not taken for the base
        band_max = BASE_BAND_PX if abs(m) < 0.2 else min(BASE_BAND_PX, abs(m) / 2.0)
        level, mad, n_base = _base_level(v, ref, band_max)
        info.update(ref_px=ref, base_px=n_base)
        if level is None or n_base < max(BASE_MIN_PX, BASE_MIN_FRAC * v.size):
            return _no_base(r, finite, ref, med, min_px, info, warnings)
        mad += 1e-9
        h0 = ref
        info.update(base_in_view=True, base_level_px=round(level, 4))
        if abs(level - ref) > max(3.0 * 1.4826 * mad, 0.1):
            warnings.append(f"the base plane in view sits at {level - ref:+.3f} px relief, "
                            "not 0: did the camera, projector or base move since the base "
                            "capture? Heights include that offset")

    thr = k_mad * mad
    pos_mask, pos_n = _largest_blob(finite & (r - level > thr), finite)
    neg_mask, neg_n = _largest_blob(finite & (level - r > thr), finite)
    if sign == "pos":
        chosen, cn = pos_mask, pos_n
    elif sign == "neg":
        chosen, cn = neg_mask, neg_n
    else:
        chosen, cn = (pos_mask, pos_n) if pos_n >= neg_n else (neg_mask, neg_n)
    if chosen is None or cn < min_px:
        raise SegmentationError(
            "no clear raised object blob found — is the object a solid minority "
            "of the field, flat-topped, and lit?")
    h = float(np.median(r[chosen])) - h0
    if abs(h) < thr:
        # the blob sits at the relief's zero, i.e. on the base plane: typical of a
        # datum-free (zone) relief segmented around its median
        warnings.append(f"the object blob's median relief ({h:+.3f} px) is within noise of "
                        "the base plane - segmentation probably picked the base, not the part "
                        "(does the part fill the view?)")
    info.update(pos_blob_px=pos_n, neg_blob_px=neg_n, mad_px=float(mad), warnings=warnings)
    return chosen, h, info


def _no_base(r, finite, ref, med, min_px, info, warnings):
    """Datum-free relief with (almost) no pixel near the base level: the
    part fills the lit zone. Its top is the valid px on the median's side of
    ref (opposite-sign shadow / unwrap outliers and edge transitions out)."""
    m = med - ref
    if m == 0.0:
        raise SegmentationError("no base plane and no part level in view")
    side = np.sign(m) * (r - ref) > 0.5 * abs(m)
    blob, n = _largest_blob(finite & side, finite)
    if blob is None or n < min_px:
        raise SegmentationError(
            "no base plane in view and no coherent part top either - is the zone lit "
            "and on the part?")
    h = float(np.median(r[blob])) - ref
    top = r[blob]
    warnings.append("no base plane in view: the part fills the scanned zone, so the whole "
                    f"lit zone is taken as the part top (H = its median relief, {h:+.3f} px). "
                    "W/L are the zone's extent, not the part's - scan a larger zone for W/L")
    info.update(base_in_view=False, base_level_px=None, pos_blob_px=n if m > 0 else 0,
                neg_blob_px=0 if m > 0 else n,
                mad_px=float(np.median(np.abs(top - np.median(top)))),
                clipped_by_zone=True, warnings=warnings)
    return blob, h, info
