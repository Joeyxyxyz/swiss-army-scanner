"""
Result types returned by the verbs in engine/api.py.

Callers (UI or agent) read these dataclasses — never PNGs. Every result
that carries a number also carries `units` ("px" | "mm") and `tier`
(§8) so relative values are never mistaken for metric ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# §8 reconstruction tiers
TIER_RELIEF = "relief"            # relative height, projector px
TIER_DEPTH = "depth_calibrated"   # Z in mm
TIER_LATERAL = "lateral_calibrated"  # Z + XY in mm
TIER_STEREO = "stereo"            # future


@dataclass
class Calibration:
    id: str
    geometry_id: str
    mm_per_px: Optional[float] = None       # Z: height_mm = mm_per_px*relief + intercept
    intercept_mm: float = 0.0
    lateral_px_per_mm: Optional[float] = None  # XY: camera px per mm on the base plane
    r2: Optional[float] = None
    n_samples: int = 0
    source: str = "library"                 # "library" | "preset"
    path: Optional[Path] = None
    fingerprint: Optional[dict] = None      # geometry-bound settings it was fitted at

    @property
    def has_z(self) -> bool:
        return self.mm_per_px is not None

    @property
    def has_xy(self) -> bool:
        return self.lateral_px_per_mm is not None


@dataclass
class SignalReport:
    gc_conf: float          # median fringe modulation over lit pixels
    valid_pct: float        # % camera px with contrast AND modulation above floors
    contrast_mean: float
    focus_ratio: float      # modulation / (contrast/2); ~1 = sharp fringes
    focus_hint: str
    quality: str            # unusable | marginal | usable | good | excellent
    ok: bool
    geometry_id: str
    mock: bool
    warnings: list = field(default_factory=list)
    coded_axis: str = ""        # camera axis the fringes vary along ("x" | "y")
    axis_ok: Optional[bool] = None   # coded axis matches rig.projector_side?
    saturated_pct: Optional[float] = None   # % of lit px >= 250 counts in the bright flood
    rig_axis: str = ""          # axis depth shifts run along (from rig.projector_side)
    basis_axis: str = ""        # axis the preset's patterns code ("x" | "y")


@dataclass
class BaseRef:
    """Handle to a stored base reference. The field loads lazily."""
    id: str
    label: str
    geometry_id: str
    path: Path
    created: str
    meta: dict = field(default_factory=dict, repr=False)
    warnings: list = field(default_factory=list)
    _field: object = field(default=None, repr=False, compare=False)

    @property
    def field(self):
        if self._field is None:
            from .decode.deformation import DeformationField
            self._field = DeformationField.load(self.path / "field.npz")
        return self._field


@dataclass
class ObjectScan:
    """Handle to a stored object scan (field + relief vs its base)."""
    id: str
    label: str
    geometry_id: str
    path: Path
    created: str
    base_id: str
    known_dims: Optional[dict] = None       # {"H": mm, "W": mm, "L": mm}
    calibration_id: Optional[str] = None    # calibration bound at capture time
    meta: dict = field(default_factory=dict, repr=False)
    _field: object = field(default=None, repr=False, compare=False)
    _relief: Optional[tuple] = field(default=None, repr=False, compare=False)

    @property
    def field(self):
        if self._field is None:
            from .decode.deformation import DeformationField
            self._field = DeformationField.load(self.path / "field.npz")
        return self._field

    @property
    def relief(self) -> np.ndarray:
        """Relief vs base in projector px (NaN where invalid)."""
        return self._load_relief()[0]

    @property
    def relief_mask(self) -> np.ndarray:
        return self._load_relief()[1]

    def _load_relief(self):
        if self._relief is None:
            with np.load(self.path / "relief.npz") as z:
                self._relief = (z["relief"], z["mask"])
        return self._relief


@dataclass
class Dimensions:
    H: Optional[float]
    W: Optional[float]
    L: Optional[float]
    units: str                  # "mm" | "px" | "mixed"
    unit_by_dim: dict           # {"H": "mm", "W": "px", ...}
    tier: str
    error: Optional[dict] = None       # measured - known, per dim (same units)
    error_pct: Optional[dict] = None
    n_px: int = 0
    relief_px: Optional[float] = None  # raw object relief (median over blob)
    warnings: list = field(default_factory=list)


@dataclass
class ToleranceBand:
    fail_mm: Optional[float] = None    # |deviation| above this fails (calibrated)
    fail_px: Optional[float] = None    # fallback band in projector px (uncalibrated)
    min_defect_px: int = 25            # connected area needed to call a defect
    smooth_px: int = 5                 # box-average deviation before thresholding


@dataclass
class CADRef:
    """Scan-to-CAD reference (build step 7)."""
    path: Path
    units: str = "mm"


@dataclass
class Verdict:
    result: str                 # "PASS" | "FAIL" | "INDETERMINATE"
    units: str
    tier: str
    max_abs_dev: Optional[float]
    p99_abs_dev: Optional[float]
    n_fail_px: int
    n_compared_px: int
    tolerance: Optional[float]
    deviation: Optional[np.ndarray] = field(default=None, repr=False)
    warnings: list = field(default_factory=list)


@dataclass
class Code:
    kind: str                   # "QR", "EAN_13", "CODE_128", ...
    data: str
    corners: Optional[list]     # [[x, y], ...] camera px, or None
    decoder: str                # which library/variant read it
    illumination: str = ""      # which DMD illumination it was read under
    raw: str = ""               # decoder's own reading when it differs (an EAN-13 with a
                                # leading 0 is reported as the UPC-A printed on the label)


@dataclass
class CodeReading:
    codes: list
    illuminations: list         # tried, in order
    polarity_inverted: bool
    path: Optional[Path] = None # library entry (frames + codes.json)
    warnings: list = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.codes)


@dataclass
class Defect:
    area_px: int
    bbox: list                  # [x0, y0, x1, y1] camera px
    centroid: list
    height: float               # median deviation over the blob (units below)
    peak: float
    snr: float
    units: str                  # "mm" | "px"
    area_mm2: Optional[float] = None
    exceeds_trigger: Optional[bool] = None   # |height| >= tolerance trigger (mm only)


@dataclass
class Inspection:
    verdict: str                # "PASS" | "FAIL" | "INDETERMINATE"
    defects: list
    units: str
    tier: str
    noise_sigma: float          # residual noise floor (units)
    threshold: float            # detection threshold (units)
    trigger: Optional[float]    # FAIL trigger (mm), from preset tolerance
    region_px: int
    path: Optional[Path] = None
    warnings: list = field(default_factory=list)


@dataclass
class MetricScan:
    """Handle to a stored metric scan (library/metric/<geometry>/<entry>):
    both-axis field + triangulated cloud in mm (stereo calibration, rule 6)."""
    id: str
    label: str
    geometry_id: str
    path: Path
    created: str
    calibration_id: str
    units: str = "mm"
    meta: dict = field(default_factory=dict, repr=False)
    _cloud: object = field(default=None, repr=False, compare=False)
    _field: object = field(default=None, repr=False, compare=False)

    @property
    def cloud(self):
        if self._cloud is None:
            from .reconstruct.triangulate import MetricCloud
            self._cloud = MetricCloud.load(self.path / "cloud.npz", self.meta)
        return self._cloud

    @property
    def field(self):
        if self._field is None:
            from .decode.deformation import DeformationField
            self._field = DeformationField.load(self.path / "field.npz")
        return self._field


@dataclass
class DentSet:
    """Handle to a stored dent set (library/dentsets/<geometry>/<entry>, or a
    legacy offline folder with dataset.json): two-exposure pattern frames of
    one pose + the metric scan(s) taken with it."""
    id: str
    label: str
    geometry_id: str
    path: Path
    created: str
    zone: Optional[str] = None
    metric_scan_ids: list = field(default_factory=list)
    legacy: bool = False
    meta: dict = field(default_factory=dict, repr=False)
    warnings: list = field(default_factory=list)


@dataclass
class DentReport:
    """detect_dents result. Positions in camera px AND cell coordinates
    (theta_deg about the axis; s_mm along it from the axis point level with
    the camera's optical axis - not a cell end - and s_from_top_mm from the
    top end of the decoded cell); depth / size in mm (rule 6: only with the
    stereo calibration the metric scans were made with); score = fused
    evidence / the good cell's maximum (null calibration); a detection needs
    score > thresholds['thr'] (default 1.25, a margin over the one good
    cell's own maximum)."""
    verdict: str                    # "FLAG" (>= 1 detection) | "CLEAR"
    candidates: list                # dicts, strongest first (see units)
    n_detections: int
    max_score: Optional[float]      # strongest fused score inside the face
    units: dict
    dentset_id: str
    metric_scan_ids: list
    null_calibration_id: str
    cell_diameter_mm: float
    subset: str = "all"
    geometry: dict = field(default_factory=dict)
    finder_seconds: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    finders: list = field(default_factory=list)        # finders fused (the null's set)
    thresholds: dict = field(default_factory=dict)     # thr, watch, edge_px, depth_sig_factor
    fixture: Optional[str] = None                      # the fixture the null was matched on


# ------------------------------------------------------------ end faces
# (engine/measure/endface: a cylinder's circular end face toward the camera)

@dataclass
class FaceTarget:
    """Handle to a located end face (library/faces/<geometry>/<entry>): the face plane,
    centre and fixed-radius rim circle in mm (camera frame, from the locate metric scan's
    stereo calibration), the camera face mask / ellipse (px) and the camera -> DMD affine
    fitted on the decoded face pixels, with the aim ring in native DMD (col, row) px."""
    id: str
    label: str
    geometry_id: str
    path: Path
    created: str
    face_diameter_mm: float
    calibration_id: str
    source_scan: str
    centre_mm: list                 # face centre, camera frame (x right, y down, z forward)
    normal: list                    # unit face normal, toward the camera
    tilt_deg: float                 # face normal vs the camera ray to the centre
    rim_rms_mm: float               # fixed-radius circle fit residual on the rim points
    n_rim_points: int
    camera_ellipse: dict            # cx, cy, full axes a, b (px), angle_deg
    dmd_centre: list                # native DMD (col, row)
    dmd_ellipse: dict
    affine: dict                    # camera -> DMD: A (2x3), rms_col_px, rms_row_px, n_px
    mm_per_camera_px: Optional[float] = None
    units: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict, repr=False)
    warnings: list = field(default_factory=list)


@dataclass
class RingShot:
    """aim_ring / check_ring: the ring + cross drawn for a face target, its photo and (check)
    whether the face is still where it was located: a flood through the DMD face ellipse
    must light the whole located camera face mask (lit_pct) and nothing much outside it
    (spill_pct: the face slid or tilted), and the ring must land on the located rim (centre
    offset: the face moved toward / away from the camera). Camera px; mm via the face's
    mm per px."""
    face_id: str
    mode: str                       # "aim" | "check"
    photo: Optional[Path]
    overlay: Optional[Path]
    dmd_centre: list
    offset_dmd: list
    hold_s: float = 0.0
    held_s: float = 0.0
    hold_cancelled: bool = False
    verdict: Optional[str] = None   # check: "ON_FACE" | "OFF" | "NOT_FOUND"
    lit_pct: Optional[float] = None     # check: % of the located face lit by the face flood
    spill_pct: Optional[float] = None   # check: face-flood light outside the located face, %
    centre_offset_px: Optional[list] = None
    centre_offset_mm: Optional[float] = None
    size_ratio: Optional[float] = None
    tol_px: Optional[float] = None
    measurement: dict = field(default_factory=dict)
    polarity_inverted: bool = False
    units: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


@dataclass
class FaceScan:
    """Handle to a targeted end-face scan (library/facescans/<geometry>/<entry>): N repeat
    metric scans lit through the face ellipse mask (ray-plane, column axis), the face target
    they were aimed with, and (optional) the polar pattern frames (rings, spokes, floods)."""
    id: str
    label: str
    geometry_id: str
    path: Path
    created: str
    face_id: str
    metric_scan_ids: list = field(default_factory=list)
    repeats: list = field(default_factory=list)
    polar: bool = False
    meta: dict = field(default_factory=dict, repr=False)
    warnings: list = field(default_factory=list)


@dataclass
class FaceProfile:
    """face_profile: height map of one placement in the face frame (u camera right, v camera
    down, h + toward the camera; mm on a grid_mm grid), radial profile, sections, flatness and
    repeatability across the repeat scans. The arrays are in files['npz']."""
    label: str
    geometry_id: str
    calibration_id: str
    face_id: str
    facescan_id: Optional[str]
    metric_scan_ids: list
    face_diameter_mm: float
    grid_mm: float
    stats: dict
    radial: list
    centre_mm: list
    normal: list
    units: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    geometry_fp: Optional[dict] = field(default=None, repr=False)
    _height: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _grid: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    # per camera pixel + repeat data and the flood pair (measure.endface.mfcfar.Pixels):
    # what the MF-CFAR detector and its reflectance FIR work on
    _pixels: Optional[object] = field(default=None, repr=False, compare=False)


@dataclass
class FaceGolden:
    """Handle to a golden end face (library/face_golden/<geometry>/<entry>) from good
    placements, bound to the geometry + fingerprint, the face diameter, the grid and the stereo
    calibration. It holds every detector's model (MF-CFAR: radial profile + the reflectance
    FIR's coefficients; band: median map + tolerance band); params['detector'] is the default
    one, params its settings. leave_one_out = false triggers on the good placements (each vs
    the others), sensitivity = the synthetic-dent battery."""
    id: str
    label: str
    geometry_id: str
    path: Path
    created: str
    face_diameter_mm: float
    calibration_id: str
    n_placements: int
    params: dict
    leave_one_out: list = field(default_factory=list)
    sensitivity: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict, repr=False)
    warnings: list = field(default_factory=list)


@dataclass
class FaceCheck:
    """check_face: triggers = dents found by the detector (mf_fir / mf: z >= z_thresh, with
    depth_mm (+ = a dent), width_mm, z; band: outside the tolerance band) with zone face |
    rim, r / theta / u / v in the face frame, dev_mm signed (- = below the golden), area_mm2.
    FAIL when any trigger; INCOMPLETE when too little of the face was measured to pass it."""
    verdict: str                    # "PASS" | "FAIL" | "INCOMPLETE"
    triggers: list
    n_triggers: int
    golden_id: str
    placement: dict
    stats: dict
    params: dict
    detector: str = "band"
    units: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


@dataclass
class StationMeasurement:
    """station_measure: one metric scan of a station (cells with their end faces toward the
    camera + the cooling tube (CT) between them). cells: per cell (C1..Cn, image order) the
    camera circle, face centre / normal (mm, camera frame), height vs the mean face plane (mm,
    + toward the camera), flatness; pairs: 'Cj-Ci' height difference along the common face
    normal, centre spacing in the face plane; ct: the edge height band, pixels, trapezoid wave;
    zones: T1..Tn part-fixed zones (index k vs the cell-pair centre) with the CT edge height
    vs the mean face plane (median / p5 / p95, mm; - = behind the faces)."""
    label: str
    scan_id: str
    geometry_id: str
    calibration_id: str
    cells: list
    pairs: list
    ct: Optional[dict]
    zones: list
    flood_source: str
    params: dict
    frame: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict, repr=False)
    warnings: list = field(default_factory=list)


@dataclass
class StationGolden:
    """Handle to a station golden (library/station_golden/<geometry>/<entry>): the nominal cell
    height differences, spacings and CT zone heights (mean of the good measurements), the zone
    model (pitch + phase: the zones every check uses) and numbering, the station parameters and
    the user-set limits (tolerances around nominal and / or absolute bands), bound to the
    geometry + fingerprint and the stereo calibration."""
    id: str
    label: str
    geometry_id: str
    path: Path
    created: str
    calibration_id: str
    n_measurements: int
    limits: dict
    nominal: dict
    zone_model: Optional[dict] = None
    params: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict, repr=False)
    warnings: list = field(default_factory=list)


@dataclass
class StationCheck:
    """station_check: a measurement vs a station golden. items = every compared quantity (kind
    cell_height | spacing | ct_zone, value, nominal, band [lo, hi] mm, status ok | trigger |
    missing); triggers = the items outside their band. FAIL when any trigger; INCOMPLETE when a
    golden item could not be measured (never passed); else PASS."""
    verdict: str                    # "PASS" | "FAIL" | "INCOMPLETE"
    triggers: list
    n_triggers: int
    items: list
    golden_id: str
    measurement: dict
    units: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
