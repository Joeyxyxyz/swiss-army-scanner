"""
Defect detection on a relief map — ported from inspect_relief.py.

Same pipeline:
  1. local background model (large-window median) -> residual
  2. noise floor from residual MAD -> threshold max(n_sigma*sigma, min_thr)
  3. threshold + 3x3 opening -> connected blobs >= min_area
  4. blob table (area, bbox, height, SNR) + verdict

Changes from the script:
  * the background median runs on a `step`-subsampled grid (a 121x121
    median over 1.5 Mpx takes minutes; the legacy script paid that)
  * the region is the part's interior (segment.object_blob, eroded) when
    a part is present, so the part's own outline is not reported as a
    "defect"; outside-region pixels are filled with their nearest in-region
    value before filtering so the background does not bleed across edges
  * heights in mm when a depth calibration for the scan's geometry exists
    (rule 6: otherwise px, and the mm verdict is INDETERMINATE)
  * verdict against the preset trigger (tolerance.ch_fail_mm, 0.7 mm):
    FAIL if any blob's |height| >= trigger. Blobs above the noise
    threshold but below the trigger are still reported (sub-trigger
    detections, when the noise allows them).
  * PASS only when the scan could have SEEN a trigger-sized defect: a blob
    is only detected where |residual| > threshold (max(n_sigma*sigma,
    floor)), so a defect of exactly the trigger height needs the threshold
    plus DETECT_MARGIN_SIGMA noise sigmas to fit under the trigger. When it
    does not, "no defect found" is not evidence of a good part and the
    verdict is INDETERMINATE with a warning naming the numbers (a FAIL
    stands: a blob over the trigger was seen whatever the threshold).
  * with a depth calibration and a trigger, the floor is
    min(min_thr_px, TRIGGER_FLOOR_FRAC * trigger): the px floor alone
    (0.3 px ~ 0.67 mm at 0.45 px/mm) would otherwise sit at the trigger.
  * zone scans (datum-free relief) find the part around the base level 0
    (segment.scan_blob), so a part filling the zone is inspected, not the
    base ring around it.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .. import rules
from ..decode.unwrap import _fill_invalid
from ..results import TIER_DEPTH, TIER_RELIEF, Defect, Inspection
from .segment import BRIDGE_PX, SegmentationError, scan_blob, support

DETECT_MARGIN_SIGMA = 2.0   # a trigger-high defect must clear the threshold by this
TRIGGER_FLOOR_FRAC = 0.5    # calibrated floor: at most this fraction of the trigger


def residual(relief, region, bg_window: int = 121, step: int = 4):
    filled = _fill_invalid(np.where(region, relief, 0.0), region)
    small = filled[::step, ::step]
    k = max(3, (bg_window // step) | 1)
    bg = ndimage.median_filter(small, size=k)
    bg = np.repeat(np.repeat(bg, step, 0), step, 1)[:relief.shape[0], :relief.shape[1]]
    return np.where(region, relief - bg, np.nan)


def inspect(scan, calibration=None, *, trigger_mm=None, n_sigma: float = 6.0,
            min_thr_px: float = 0.3, min_area: int = 50, bg_window: int = 121,
            erode_px: int = 6, whole_field: bool = False) -> Inspection:
    warnings = []
    relief = scan.relief.astype(np.float64)
    valid = scan.relief_mask & np.isfinite(relief)

    if calibration is not None and not rules.calibration_matches(calibration, scan.geometry_id):
        calibration = None
    z = calibration.mm_per_px if (calibration is not None and calibration.has_z) else None
    xy = calibration.lateral_px_per_mm if (calibration is not None and calibration.has_xy) else None
    units, tier = ("mm", TIER_DEPTH) if z else ("px", TIER_RELIEF)
    s = abs(z) if z else 1.0            # sign of the relief->height slope is rig-specific
    if not z:
        warnings.append(f"no depth calibration for geometry {scan.geometry_id}; "
                        "heights in px, no mm verdict (rule 6)")
    elif not trigger_mm:
        warnings.append("no trigger (preset tolerance.ch_fail_mm); defects are listed "
                        "but there is no PASS/FAIL verdict")

    if whole_field:
        region = valid
    else:
        try:
            blob, _, info = scan_blob(scan)
            warnings += info.get("warnings", [])
            # erode the blob with its NaN gaps filled: thin invalid lines (zone scans)
            # would otherwise shred it to nothing
            region = ndimage.binary_erosion(support(blob), iterations=erode_px) & valid
        except SegmentationError:
            region = valid
            warnings.append("no part found; inspecting the whole field")
    if region.sum() < 200:
        return Inspection("INDETERMINATE", [], "px", TIER_RELIEF, float("nan"),
                          float("nan"), trigger_mm, int(region.sum()),
                          warnings=warnings + ["inspection region too small"])

    resid = residual(relief, region, bg_window)
    v = resid[region]
    sigma = 1.4826 * float(np.median(np.abs(v - np.median(v))))
    floor = float(min_thr_px)
    if z and trigger_mm:
        floor = min(floor, TRIGGER_FLOOR_FRAC * float(trigger_mm) / s)
    thr = max(n_sigma * sigma, floor)
    cand = region & (np.abs(np.nan_to_num(resid)) > thr)
    cand = ndimage.binary_opening(cand, structure=np.ones((3, 3)))
    # one defect split by thin lines of invalid px is still one defect (its blob
    # holds only the candidate px); valid non-candidate px never bridge
    link = cand | (~valid & ndimage.binary_dilation(cand, iterations=BRIDGE_PX))
    labels, n = ndimage.label(link)
    labels[~cand] = 0

    defects = []
    for i in range(1, n + 1):
        m = labels == i
        area = int(m.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(m)
        h = float(np.nanmedian(resid[m]))
        h_u = h * (z if z else 1.0)
        peak = float(np.nanmax(np.abs(resid[m]))) * s
        defects.append(Defect(
            area_px=area, bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            centroid=[int(xs.mean()), int(ys.mean())], height=round(h_u, 4),
            peak=round(peak, 4), snr=round(abs(h) / max(sigma, 1e-9), 1), units=units,
            area_mm2=round(area / xy ** 2, 3) if xy else None,
            exceeds_trigger=(abs(h_u) >= trigger_mm) if (z and trigger_mm) else None))
    defects.sort(key=lambda d: -abs(d.height))

    if z and trigger_mm:
        reach = (thr + DETECT_MARGIN_SIGMA * sigma) * s     # smallest surely-seen defect
        if any(d.exceeds_trigger for d in defects):
            verdict = "FAIL"
        elif reach <= trigger_mm:
            verdict = "PASS"
        else:
            verdict = "INDETERMINATE"
            warnings.append(
                f"cannot PASS: the detection threshold {thr * s:.3f} mm (+{DETECT_MARGIN_SIGMA:g} "
                f"sigma margin = {reach:.3f} mm; noise sigma {sigma * s:.3f} mm, n_sigma "
                f"{n_sigma:g}) is not below the {trigger_mm:g} mm trigger, so a defect at the "
                "trigger could go unseen. Lower the noise (more camera.average; more phase "
                "steps if the noise is fringe ripple) or lower n_sigma, then re-inspect")
    else:
        verdict = "INDETERMINATE"
    return Inspection(verdict=verdict, defects=defects, units=units, tier=tier,
                      noise_sigma=round(sigma * s, 5), threshold=round(thr * s, 5),
                      trigger=trigger_mm, region_px=int(region.sum()), warnings=warnings)


def annotate_png(scan, insp: Inspection, max_blobs: int = 10):
    """PNG bytes of the residual with defect boxes (None without cv2)."""
    try:
        import cv2
    except ImportError:
        return None
    relief = scan.relief.astype(np.float64)
    img = np.clip((np.nan_to_num(relief, nan=np.nanmedian(relief)) - np.nanpercentile(relief, 1))
                  / (np.nanpercentile(relief, 99) - np.nanpercentile(relief, 1) + 1e-9), 0, 1)
    vis = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    vis[~scan.relief_mask] = 0
    for i, d in enumerate(insp.defects[:max_blobs]):
        x0, y0, x1, y1 = d.bbox
        col = (0, 0, 255) if d.exceeds_trigger else (0, 200, 255)
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
        cv2.putText(vis, f"#{i+1} {d.height:+.3f}{d.units} SNR{d.snr:.0f}", (x0, max(15, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", vis)
    return buf.tobytes() if ok else None
