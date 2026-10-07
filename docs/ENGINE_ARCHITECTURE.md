# Swiss Army Scanner — Engine Architecture Spec
### The bones for Claude Code / Grok to build the working codebase (GitHub-ready)

**Purpose:** define a pattern-agnostic scanner ENGINE with a tiny verb API,
so any front-end — a human UI, a CLI, or an AI agent — drives the scanner
by calling a handful of simple functions. The engine owns all complexity
(hardware, patterns, decode, reconstruction, libraries, calibration,
logging, safety rules). Callers never touch hardware directly.

Target: a clean GitHub repo. Existing proven code (`swiss-army-sensor` package,
now in-repo as `swiss_army_sensor/`,
`first_scan_v10.py`, `calibrate_depth.py`, `inspect_relief.py`) is the
seed — refactor it INTO this structure, don't rewrite the proven math.

---

## 0. Design principles (non-negotiable)

1. **The pattern IS the sensor.** Every measurement is: project a coded
   basis → capture → decode the deformation. The engine is agnostic to
   WHICH basis. New patterns plug in without touching the core.
2. **One universal intermediate: the deformation field.** Whatever pattern
   was used, decode produces a per-camera-pixel *correspondence /
   deformation field* + a confidence map. Everything downstream
   (reconstruction, measurement, comparison, libraries) consumes ONLY this.
3. **Base is the datum.** A base reference (empty scene) is captured once
   per geometry and stored. Objects are reconstructed as deformation
   RELATIVE to a base. Libraries are built from these fields.
4. **Geometry-bound calibration.** Every base + calibration is tagged with a
   geometry id. Move the rig → new geometry → new base + calibration.
5. **Simple verbs.** The public API is ~8 functions. A UI or an AI agent
   uses the same API. If a caller needs to know about HDMI, USB, mirrors,
   or thresholds, the engine has failed.
6. **Safety rules live IN the engine** (see §9) so no caller can violate
   them. Most important: never drive the DLPC350 over USB in the scan path.
7. **Everything logged.** Every action → structured record. Nothing overwrites.

---

## 1. Repo structure

```
swiss_army_scanner/
├── README.md                    # what it is, quickstart, the verb API
├── pyproject.toml
├── docs/                        # design notes; TI primary sources (DLPU010G, DLPU011E,
│                                #   DLPA036/037, TIDA-00254) are NOT in the repo —
│                                #   download them from ti.com
├── engine/                      # THE ENGINE (this spec)
│   ├── __init__.py
│   ├── api.py                   # public verbs (§2) — the ONLY thing callers import
│   ├── session.py               # ScanSession: orchestrates a capture, logs it
│   ├── config.py                # presets, geometry ids, calibration binding (§6)
│   ├── rules.py                 # safety invariants enforced at runtime (§9)
│   ├── bases/                   # PATTERN BASES — pluggable (§3)
│   │   ├── base.py              #   abstract PatternBasis interface
│   │   ├── graycode.py          #   gray code (+ complementary/mirrored)
│   │   ├── phaseshift.py        #   N-step sinusoidal fringes
│   │   ├── checkerboard.py      #   checkerboard (calibration / corner detect)
│   │   ├── diagonal.py          #   diagonal stripes (off-axis correspondence)
│   │   ├── hadamard.py          #   single-pixel / compressive basis
│   │   └── registry.py          #   name -> basis class
│   ├── capture/                 # hardware I/O (wraps swiss_army_sensor drivers)
│   │   ├── projector.py         #   HDMITransport ONLY in scan path (§9)
│   │   ├── camera.py            #   FLIR/PySpin + mock
│   │   └── sync.py              #   free-run now; hardware-trigger mode later
│   ├── decode/
│   │   ├── deformation.py       #   DeformationField dataclass (§4)
│   │   └── fuse.py              #   fuse multi-basis results into one field
│   ├── reconstruct/
│   │   ├── relief.py            #   object − base → relief (height)
│   │   ├── metric.py            #   apply calibration → mm (Z and XY)
│   │   └── pointcloud.py        #   → .ply / .xyz export
│   ├── library/                 # PERSISTENT LIBRARIES (§5)
│   │   ├── store.py             #   filesystem store, timestamped, never overwrite
│   │   ├── bases.py             #   base-reference library
│   │   └── objects.py           #   object library (+ known dims, results)
│   ├── measure/
│   │   ├── dimensions.py        #   measure H/W/L of an object vs known dims
│   │   ├── inspect.py           #   defect blobs (from inspect_relief.py)
│   │   └── golden.py            #   scan-to-CAD / scan-to-golden (from golden.py)
│   ├── calibrate/
│   │   ├── depth.py             #   Z: relief px → mm (from calibrate_depth.py)
│   │   ├── lateral.py           #   XY: pixels → mm from known width/length
│   │   └── stereo.py            #   FUTURE: full intrinsics/extrinsics (§8)
│   └── log/
│       └── records.py           #   structured session/scan records
├── ui/                          # thin front-ends — ALL call engine.api only
│   ├── cli.py                   # guided CLI wizard
│   ├── web/                     # local web UI (live signal meter, 3D viewer)
│   └── agent.md                 # how an AI agent drives the engine (§7)
├── presets/                     # geometry-bound configs (§6)
├── tests/                       # mock-mode tests; physics ground-truth tests
```

---

## 2. The public API — `engine/api.py` (the verbs)

This is the ENTIRE surface a UI or agent uses. Keep it this small.

```python
def load_preset(name: str) -> Preset
    # loads geometry + settings + bound calibration

def check_signal() -> SignalReport
    # live: projects a test pattern, returns gc_conf, valid%, focus hint.
    # UI shows a meter; agent reads numbers. Used to position/tune.

def capture_base(preset, label: str) -> BaseRef
    # empty scene → deformation field → stored in base library (geometry-tagged)

def capture_object(preset, base: BaseRef, label: str,
                   known_dims: dict | None = None) -> ObjectScan
    # object scene → deformation field → relief vs base → stored in object
    # library. known_dims (H/W/L mm) optional: enables measure + calibration.

def reconstruct(scan: ObjectScan, fmt="ply") -> Path
    # deformation field (+calibration) → 3D file (.ply/.xyz)

def measure(scan: ObjectScan) -> Dimensions
    # H/W/L in mm (needs calibration); if known_dims given, reports error vs known

def compare(scan: ObjectScan, golden: ObjectScan | CADRef,
            tol: ToleranceBand) -> Verdict
    # deviation map + PASS/FAIL (mm after calibration)

def calibrate(preset, samples: list[ObjectScan]) -> Calibration
    # fits Z (and XY if known_dims present) from known-dimension scans;
    # binds result to the preset's geometry id. Reports R^2.

def list_library(kind="bases"|"objects") -> list
```

Every verb: validates preconditions (via rules.py), logs a record, never
overwrites prior data. Every verb works in `mock=True` for tests/agents
without hardware.

---

## 3. Pattern bases — `engine/bases/` (the agnostic part)

Every basis implements ONE interface. The engine never special-cases a pattern.

```python
class PatternBasis(ABC):
    name: str
    axes: set[str]            # which correspondence axes it yields: {"x"}, {"y"}, {"x","y"}, {"diag"}
    def generate(self, dmd_shape, **params) -> list[np.ndarray]   # patterns to project (uint8)
    def decode(self, frames: list[np.ndarray], base_frames=None, **params) -> DeformationField
    def n_frames(self, **params) -> int
    def complementary(self) -> bool   # project pattern + inverse (ambient-robust)
```

Bases to implement (seed from existing `patterns.py`/`decoders.py`):
- **graycode** — coarse absolute correspondence; `complementary=True` option
  (pattern + inverse = "mirrored/alternating"); axes x or y.
- **phaseshift** — fine sub-stripe phase; N steps, period; axes x or y.
- **hybrid** — graycode coarse + phaseshift fine (the proven 3D path) —
  implemented as a *composite* that runs two bases and fuses.
- **checkerboard** — corner/feature detection; for lateral calibration and
  future stereo calibration; NOT a 3D correspondence basis.
- **diagonal** — stripes at 45°; yields diagonal-axis correspondence,
  useful when features align with x or y and a single axis is degenerate.
- **hadamard** — compressive/single-pixel basis (barcode/QR modes, later).

**Fusion** (`decode/fuse.py`): combine multiple bases' fields (e.g. x from
graycode+phase, y from a second run, diag as a check) into one field with a
per-pixel confidence = min over sources. Agnostic: any set of bases in.

---

## 4. The universal intermediate — `DeformationField`

```python
@dataclass
class DeformationField:
    corr_x: np.ndarray | None    # per camera px: projector column (float)
    corr_y: np.ndarray | None    # per camera px: projector row (float)
    corr_diag: np.ndarray | None
    conf: np.ndarray             # per px confidence (gc_conf-style)
    mask: np.ndarray             # valid pixels
    geometry_id: str
    basis_names: list[str]
    meta: dict                   # capture settings, timestamps, preset
```

- **Relief** = object.corr − base.corr (per axis), masked by both.
- **Metric** = relief × calibration.mm_per_px (Z); px extent × lateral
  scale (XY).
- Libraries store DeformationFields (npz) + PNG previews + meta.json.

---

## 5. Libraries — `engine/library/`

**Filesystem store, timestamped, append-only. Nothing overwrites.**

```
library/
  bases/
    <geometry_id>/<timestamp>_<label>/   field.npz, preview.png, meta.json
  objects/
    <timestamp>_<label>/                 field.npz, relief.npz, preview.png,
                                         known_dims.json (if given),
                                         measured.json (if measured),
                                         meta.json (preset, base_ref, calib_id)
  calibrations/
    <geometry_id>/<timestamp>/           depth_calibration.json, lateral.json, samples/
  sessions/
    <timestamp>/                         log.jsonl (every verb call + result)
```

- A base is reusable by any object scan at the SAME geometry_id.
- An object with `known_dims` is automatically a **calibration candidate**.
- `list_library()` lets a UI/agent browse and pick bases/objects/golden refs.

---

## 6. Config & presets — `engine/config.py`

A preset = one JSON, geometry-bound. ALL tunables live here (nothing
hardcoded, no sed-edits):

```json
{
  "name": "cell_wall_2p7ft",
  "geometry_id": "wall_2p7ft_v1",
  "camera":     {"exposure_us": 40000, "gain_db": 0, "average": 8},
  "projection": {"monitor_offset": [1920,0], "settle_s": 0.0, "scheme": "bw", "gamma": 1.0},
  "bases":      [{"name":"graycode","bits":10,"complementary":true},
                 {"name":"phaseshift","steps":4,"period_px":24}],
  "decode":     {"conf_min": 8},
  "calibration": {"id": null, "mm_per_px": null, "lateral_px_per_mm": null},
  "tolerance":  {"ch_fail_mm": 0.7, "dent_fail_um": 25}
}
```
`load_preset` binds the latest calibration for that geometry_id if present.

---

## 7. AI-agent driving — `ui/agent.md`

An agent uses the SAME verbs, in mock or real. Contract:
- Agent may call `check_signal()`, `capture_*`, `reconstruct`, `measure`,
  `compare`, `calibrate`, `list_library` freely.
- Agent must NOT bypass the API to touch hardware/files.
- **Physical steps are human-only** (place object, focus, gauge blocks,
  enter true dimensions). The agent PROMPTS for them and waits; it reads
  `check_signal()` to know if the human's setup is good. It never assumes
  physical state.
- Agent reads results from returned dataclasses, never from parsing PNGs.

This is what makes the UI "simple even for AI": the agent's whole world is
~8 verbs + returned numbers.

---

## 8. Reconstruction tiers (be honest about which is built)

| Tier | Needs | Gives | Status |
|---|---|---|---|
| Relief vs base | base + object, same geometry | relative height (px) | **built** |
| Depth-calibrated | + known-height samples | Z in mm at that geometry | tool built, cal in progress |
| Lateral-calibrated | + known width/length | XY in mm | to build (`calibrate/lateral.py`) |
| Full stereo | checkerboard intrinsics + projector-as-camera + extrinsics | absolute XYZ, single-shot, distance-agnostic within volume | **future** (`calibrate/stereo.py`) |

The engine exposes the same verbs at every tier; results carry a
`units` field ("px" | "mm") and a `tier` field so callers never mistake
relative for metric.

---

## 9. Safety rules — `engine/rules.py` (enforced, not advisory)

1. **Scan path is HDMI-only.** `capture/projector.py` uses HDMITransport.
   The DLPC350 USB driver is NOT imported in the scan path. Any USB
   display-mode command (set_mode/video) knocks the DMD off 912×1140 HDMI
   lock → internal blue-lines pattern → dead scan. The engine raises if a
   caller tries. (A future `sync.hardware_trigger` mode will need USB
   pattern-mode — it must be a separate, explicitly-entered mode with its
   own guard, never the default.)
2. **Window stays painted during holds** (pump cv2.waitKey; no bare sleeps
   in the projector transport).
3. **LED thermal limits** enforced in config validation: short-burst
   ceiling R=180 G=255 B=230; average R=120 G=180 B=160.
4. **Mono camera ⇒ bw scheme** default; color schemes flagged as
   signal-reducing.
5. **Base and object must share geometry_id**; `capture_object` refuses a
   base from a different geometry.
6. **Calibration is geometry-bound**; `measure/compare` refuse to report mm
   without a calibration matching the scan's geometry_id (they report px
   + a warning instead).
7. **Never overwrite** library data; sessions are append-only.

---

## 10. Build order for the agents (each step ships working + tested in mock)

1. **Skeleton + API stubs + rules.py + config.py** — the verbs exist, mock
   mode runs end-to-end on synthetic scenes. (Seed from `swiss_army_sensor`
   mocks.)
2. **Bases**: port graycode (+complementary) and phaseshift from
   `patterns.py`/`decoders.py`; implement `DeformationField` + hybrid
   composite + fuse. Reproduce `first_scan_v10` results in mock.
3. **Capture**: wrap HDMITransport + PySpin camera; real-hardware path with
   rules enforced. Reproduce the bench scan (gc_conf ~90, valid ~38%).
4. **Library store** (timestamped, append-only) + `list_library`.
5. **Reconstruct**: relief + `.ply` export. View in Blender/MeshLab.
6. **Calibrate**: port `calibrate_depth.py` (Z); add `lateral.py` (XY from
   known W/L). `measure()` reports H/W/L vs known dims with error.
7. **Inspect/compare**: port `inspect_relief.py` + `golden.py`; verdicts in
   mm; tolerances from preset (0.7mm CH, 25µm dent).
8. **UI**: CLI wizard first (guided base→object→measure flow), then local
   web UI (live `check_signal` meter, 3D viewer, preset picker).
9. **Add bases**: checkerboard (lateral cal / corners), diagonal, hadamard.
10. **Later**: hardware-trigger sync mode (guarded), full stereo calibration.

Definition of done for each step: mock tests green; if hardware-touching,
one real bench run reproduces prior known-good numbers.

---

## 11. Validation object on hand (use it)

A known object exists on the bench: **H 106.95 mm, W 56.50 mm, L 21.90 mm**,
already scanned against a base at geometry "wall ~2.6–2.8 ft" (gc_conf ~89,
valid ~38%, relief range ≈ ±48 px). Use it as the first `known_dims`
object: it seeds Z calibration (height) and lateral calibration (W, L),
and `measure()` must reproduce ~107/56.5/21.9 within tolerance. If the
engine can correctly dimension THIS, it can dimension cylindrical cans.
Caveats: 107 mm is tall for the current depth-of-field/focus — expect the
top surface to be softer; prefer measuring W/L from the base footprint and
Z from the well-resolved region. Add small known steps (0.5–5 mm) later for
the 0.7 mm CH regime.
