"""
Filesystem library (§5): timestamped, append-only. Nothing overwrites.

    <root>/bases/<geometry_id>/<ts>_<label>/        field.npz preview.png meta.json
    <root>/objects/<ts>_<label>/                    field.npz relief.npz preview.png
                                                    known_dims.json measured.json meta.json
    <root>/calibrations/<geometry_id>/<ts>/         depth_calibration.json lateral.json
                                                    (or stereo_calibration.json - also the
                                                    re-homed copies -, parity.json,
                                                    dent_null.json, rigref.json)
    <root>/dentsets/<geometry_id>/<ts>_<label>/     dent-detector frames (two exposures) +
                                                    meta.json
    <root>/faces/<geometry_id>/<ts>_<label>/        located end faces: face.npz (camera mask,
                                                    rim, DMD ring) + meta.json
    <root>/facescans/<geometry_id>/<ts>_<label>/    targeted end-face scans: masks.npz, polar
                                                    frames + meta.json naming the metric scans
    <root>/face_golden/<geometry_id>/<ts>_<label>/  golden end faces: golden.npz + meta.json
    <root>/station_golden/<geometry_id>/<ts>_<label>/  station goldens: golden.json + meta.json
    <root>/recipe_golden/<geometry_id>/<ts>_<recipe>/  recipe goldens: meta.json naming the
                                                    GOOD part's metric scan
    <root>/presets/<name>/                          user presets: preset.json + meta.json;
                                                    presets/active.jsonl = the selected one
    <root>/zones/<geometry_id>/<name>/              painted zone sets (engine/zones): zones.json
                                                    (shapes in projector px), masks.npz,
                                                    preview.png
    <root>/sessions/<ts>/                           log.jsonl
    <root>/readings/<ts>_<label>/                   code-reading frames + codes.json
    <root>/output/<kind>/<ts>_<label>.<ext>         THE OUTPUT BIN — every derived result,
                                                    by kind: pointclouds, defects,
                                                    measurements, codes, spectral,
                                                    previews; output/index.csv lists them
    <root>/trash/<kind>/<entry>[-N]/                "deleted" entries (moved, recoverable;
                                                    -N when that name was trashed before)

Every file is opened in exclusive-create mode ('x'), so an existing path
raises instead of being replaced (rule 7). "Delete" moves an entry to
trash/ — nothing is destroyed by the engine; emptying trash/ is the
operator's decision. The move is one atomic rename of one whole entry
(never copy-then-delete); an entry with a file held open is left untouched
(rules.EntryInUse, retry). Entries record their outputs in exports.jsonl
(append-only). meta.json is written last: an entry without a readable one
is a half-written save, skipped by listings (LibraryStore.incomplete_entries).
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .. import rules
from ..results import Calibration

_LABEL_RE = re.compile(r"[^A-Za-z0-9_.\-]+")

OUTPUT_KINDS = ("pointclouds", "defects", "measurements", "codes", "spectral", "previews",
                "profiles", "anomalies", "tuning", "comparisons", "sensors", "calibration",
                "flashpacks", "flashscans")
INDEX_FIELDS = ["time", "kind", "label", "geometry_id", "file", "source"]
ENTRY_KINDS = ("bases", "objects", "calibrations", "readings", "spectral", "banks", "maps",
               "sensors", "atlases", "calib_poses", "metric", "dentsets", "faces", "facescans",
               "face_golden", "station_golden", "recipe_golden", "presets", "recipes", "zones")

LIBRARY_README = """# Swiss Army Scanner library

Written by the engine. Nothing here is overwritten; "delete" moves entries to trash/.

| folder | what |
|---|---|
| bases/<geometry_id>/ | BASE LIBRARY - empty-scene references, one folder per rig geometry |
| objects/ | 3D scans: field.npz (correspondence), relief.npz (height vs base), preview.png, meta.json |
| calibrations/<geometry_id>/ | depth (Z) + lateral (XY) calibration fits, stereo (camera + projector) calibrations and their re-homed copies (rig-check --rehome: a new entry, the original kept), diamond row parity, rig-check references (rigref), dent-detector null calibrations (dent_null.json, per cell diameter + fixture) |
| calib_poses/<geometry_id>/ | ChArUco board poses for calibrate_stereo: flood.png, field.npz (both axes), meta.json (corners found, corners with a projector position, the fringe periods used) |
| metric/<geometry_id>/ | metric scans: field.npz, cloud.npz (xyz mm + consistency), meta.json |
| dentsets/<geometry_id>/ | dent-detector pattern frames of one pose (long + short exposure, cropped to the zone), meta.json names its metric scan |
| faces/<geometry_id>/ | located end faces (locate_face): face plane, centre, rim circle, camera face mask, camera -> DMD affine + aim ring |
| facescans/<geometry_id>/ | targeted end-face scans: the face DMD mask, polar pattern frames, meta.json names the repeat metric scans |
| face_golden/<geometry_id>/ | golden end faces (face_golden): golden.npz (MF radial profile, reflectance FIR coefficients, golden + band maps) + meta.json (detector settings; bound to geometry, face diameter, stereo calibration) |
| station_golden/<geometry_id>/ | station goldens (station_golden): golden.json (nominal end-face positions and height differences, spacings, tube zone heights, the zone model + numbering, user-set limits) + meta.json (bound to geometry, stereo calibration) |
| recipe_golden/<geometry_id>/ | recipe goldens (scan recipe-golden): meta.json names the GOOD part's metric scan, its label, recipe, rig fingerprint and stereo calibration; only these (or an exact scan id) can be a recipe's golden |
| readings/ | barcode / QR reading frames + codes.json |
| spectral/ | spectral captures (cube.npz R/G/B signal, meta.json), materials/<name>/<ts>.json signature versions, light/ direct-global separations |
| output/ | THE OUTPUT BIN - every result by kind; output/index.csv lists them all |
| output/pointclouds/ | .ply / .xyz 3D point clouds (open in MeshLab / CloudCompare / Blender) |
| output/defects/ | defect inspection .json + annotated .png |
| output/measurements/ | H/W/L measurements .json |
| output/codes/ | decoded barcodes / QR .json |
| output/spectral/ | material classification maps .png + .json |
| output/previews/ | relief previews, grid / zone photos |
| output/profiles/ | 2D height profiles .csv + .png |
| output/anomalies/ | good-memory-bank check results .json + .png |
| output/comparisons/ | scan-vs-golden compare verdicts .json + deviation .png |
| output/tuning/ | scan-assistant / gamma reports |
| output/calibration/ | ChArUco boards to print (PNG / PDF), stereo calibration and row-parity reports |
| output/flashpacks/ | DLPC350 pattern-mode packs (tools/make_flash_pack.py --fastscan): 24-bit flash BMPs, manifest.json, procedure.txt - loaded by a human in the 4500 MKII GUI |
| output/flashscans/ | flash soft-sync recordings (free-running camera frames + timestamps + the loop layout, .npz) and offline sorts (per-pattern images + report) |
| banks/<geometry_id>/<bank>/ | GOOD MEMORY BANKS - known-good scans per part type |
| maps/<geometry_id>/ | projector maps (camera <-> DMD pixel), used by grid zones |
| sensors/<geometry_id>/<name>/ | taught virtual sensors (part present, alignment) |
| output/sensors/ | sensor checks: spot photos + .json verdicts |
| presets/ | USER PRESETS (part recipes): presets/<name>/preset.json + meta.json (author, description, copied from); presets/active.jsonl records the selected preset (last line wins) |
| recipes/ | USER RECIPES (the checks that verify a work state: preset -> zones -> checks -> projected feedback): recipes/<name>/recipe.json + meta.json; recipes/active.jsonl records the selected recipe (last line wins) |
| zones/<geometry_id>/<name>/ | PAINTED ZONE SETS (scan zone-paint: shapes drawn on the part with the mouse, in projector px, valid at this rig placement): zones.json + masks.npz (union + one DMD mask per zone) + preview.png (+ camera.png, a photo of the zones on the part) |
| runs/ | recipe run folders (api.recipe_run): result.json + run.log.jsonl + overlay / feedback images, one NEW folder per run |
| sessions/ | log.jsonl - every action, success or failure |
| trash/ | deleted entries (recoverable - move back to restore) |
"""


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S_%f")


def safe_label(label: str) -> str:
    s = _LABEL_RE.sub("-", label.strip()).strip("-")
    if not s:
        raise ValueError("label must contain at least one [A-Za-z0-9_.-] character")
    return s[:64]


class LibraryStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def ensure_readme(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.root / "README.md", "x", encoding="utf-8") as f:
                f.write(LIBRARY_README)
        except FileExistsError:
            pass

    # ------------------------------------------------------- output bin
    def output_path(self, kind: str, label: str, ext: str) -> Path:
        """A fresh file path in output/<kind>/ (never an existing one)."""
        if kind not in OUTPUT_KINDS:
            raise ValueError(f"output kind must be one of {OUTPUT_KINDS}")
        d = self.root / "output" / kind
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{timestamp()}_{safe_label(label)}"
        for n in range(1000):
            p = d / (f"{stem}{ext}" if n == 0 else f"{stem}-{n}{ext}")
            if not p.exists():
                return p
        raise rules.OverwriteRefused(f"could not allocate a new output under {d}")

    def record_output(self, kind: str, path: Path, *, label: str, geometry_id: str = "",
                      source: Optional[Path] = None) -> None:
        """Append one row to output/index.csv, and a pointer in the source
        entry's exports.jsonl (both append-only)."""
        idx = self.root / "output" / "index.csv"
        new = not idx.exists()
        with open(idx, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
            if new:
                w.writeheader()
            w.writerow({"time": datetime.now().isoformat(timespec="seconds"), "kind": kind,
                        "label": label, "geometry_id": geometry_id,
                        "file": self._rel(path), "source": self._rel(source) if source else ""})
        if source is not None and Path(source).is_dir():
            with open(Path(source) / "exports.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"kind": kind, "file": self._rel(path),
                                    "time": datetime.now().isoformat(timespec="seconds")}) + "\n")

    def outputs(self, kind: Optional[str] = None) -> list[dict]:
        idx = self.root / "output" / "index.csv"
        if not idx.exists():
            return []
        with open(idx, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return [r for r in rows if kind is None or r["kind"] == kind]

    def _rel(self, p) -> str:
        try:
            return str(Path(p).resolve().relative_to(self.root.resolve())).replace("\\", "/")
        except ValueError:
            return str(p)

    # ------------------------------------------------------------ trash
    def trash(self, entry: Path, reason: str = "") -> Path:
        """Move ONE entry directory to trash/<kind>/... — recoverable, never destroyed.

        Only a whole entry at its kind's depth moves (not a geometry folder,
        not a file inside an entry), and only by one atomic rename: if a file
        in it is open (a preview being served, a viewer, an antivirus scan)
        the entry is left exactly as it was and EntryInUse says to retry.
        Never copy-then-delete, which would leave a half-deleted entry.
        An entry whose name is already in trash/ (a material re-learned and
        trashed again, a re-taught sensor) goes to <name>-1, <name>-2, ...:
        nothing in trash/ is ever replaced. -> the entry's path in trash/."""
        root = self.root.resolve()
        entry = Path(entry).resolve()
        try:
            rel = entry.relative_to(root)
        except ValueError:
            raise ValueError(f"{entry} is outside the library {root}") from None
        _check_entry(entry, rel)
        first = root / "trash" / rel
        first.parent.mkdir(parents=True, exist_ok=True)
        dest = _move_to_trash(entry, first)
        record = {"from": str(rel).replace("\\", "/"), "reason": reason,
                  "time": datetime.now().isoformat(timespec="seconds")}
        if dest != first:
            record["trashed_as"] = dest.name   # same name trashed before (re-learned, re-taught)
        _write_trash_record(dest, record)
        # same form as the store's other paths
        return self.root / "trash" / rel.parent / dest.name

    @contextmanager
    def filling(self, d: Path):
        """Wrap the writes that fill a fresh entry: if they fail (disk full,
        Ctrl+C, a bad array) the half-written entry is moved to trash/ so it
        never sits in the library looking like a real one."""
        try:
            yield d
        except BaseException as e:
            try:
                self.trash(d, reason=f"incomplete save: {type(e).__name__}: {e}"[:300])
            except Exception:           # noqa: BLE001 - listings skip it anyway
                pass
            raise

    def incomplete_entries(self) -> list[dict]:
        """Base / object entries without a readable meta.json (an interrupted
        save). Listings skip them; api.delete(path) moves them to trash/."""
        out = []
        dirs = [("bases", d) for g in self.geometry_ids("bases") for d in self.entries("bases", g)]
        dirs += [("objects", d) for d in self.entries("objects")]
        for kind, d in dirs:
            if read_meta(d) is None:
                out.append({"kind": kind, "id": d.name, "path": d,
                            "reason": "no readable meta.json (interrupted save?)"})
        return out

    # ---------------------------------------------------------- writing
    def new_dir(self, *parts: str, label: Optional[str] = None) -> Path:
        """Create a fresh, never-before-used entry directory."""
        parent = self.root.joinpath(*parts)
        parent.mkdir(parents=True, exist_ok=True)
        stem = timestamp() + (f"_{safe_label(label)}" if label else "")
        for n in range(1000):
            d = parent / (stem if n == 0 else f"{stem}-{n}")
            try:
                d.mkdir()           # exist_ok=False: never reuse a directory
                return d
            except FileExistsError:
                continue
        raise rules.OverwriteRefused(f"could not allocate a new entry under {parent}")

    @staticmethod
    def write_json(path: Path, obj) -> Path:
        rules.ensure_new_path(path)
        with open(path, "x", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=_json_default)
        return path

    @staticmethod
    def write_npz(path: Path, **arrays) -> Path:
        rules.ensure_new_path(path)
        with open(path, "xb") as f:
            np.savez_compressed(f, **arrays)
        return path

    @staticmethod
    def write_field(path: Path, fld) -> Path:
        rules.ensure_new_path(path)
        with open(path, "xb") as f:
            fld.save(f)
        return path

    @staticmethod
    def write_preview(path: Path, arr: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[Path]:
        """1..99 percentile-stretched PNG (v10 save_png). Skipped without cv2."""
        try:
            import cv2
        except ImportError:
            return None
        a = np.asarray(arr, dtype=np.float64)
        vals = a[mask] if mask is not None else a[np.isfinite(a)]
        lo, hi = (np.percentile(vals, 1), np.percentile(vals, 99)) if vals.size else (0.0, 1.0)
        img = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo + 1e-12), 0, 1)
        img8 = (img * 255).astype(np.uint8)
        if mask is not None:
            img8[~mask] = 0
        rules.ensure_new_path(path)
        ok, buf = cv2.imencode(".png", img8)
        if not ok:
            return None
        with open(path, "xb") as f:
            f.write(buf.tobytes())
        return path

    # ---------------------------------------------------------- reading
    def entries(self, *parts: str) -> list[Path]:
        root = self.root.resolve()
        d = root.joinpath(*parts)
        try:
            d.resolve().relative_to(root)
        except ValueError:
            raise ValueError(f"{'/'.join(parts)!r} points outside the library") from None
        if any(p in ("", ".", "..") for part in parts for p in re.split(r"[\\/]", str(part))):
            raise ValueError(f"library path parts must be plain names, got {parts!r}")
        if not d.is_dir():
            return []
        return sorted((self.root.joinpath(*parts) / p.name) for p in d.iterdir() if p.is_dir())

    def geometry_ids(self, kind: str) -> list[str]:
        return [p.name for p in self.entries(kind)]

    def latest_calibration(self, geometry_id: str,
                           fingerprint: Optional[dict] = None) -> Optional[Calibration]:
        """Newest calibration for the geometry; with `fingerprint`, the newest
        one fitted at the same geometry-bound settings (rules.fingerprint_diffs)."""
        dirs = self.entries("calibrations", geometry_id)
        for d in reversed(dirs):
            f = d / "depth_calibration.json"
            if not f.exists():
                continue
            try:
                cal = calibration_from_json(json.loads(f.read_text(encoding="utf-8")), d)
            except (OSError, ValueError, KeyError, TypeError):
                continue                    # unreadable / half-written: never used for mm
            if not rules.fingerprint_diffs(cal.fingerprint, fingerprint):
                return cal
        return None


def read_meta(d: Path) -> Optional[dict]:
    """An entry's meta.json, or None when it is missing or unreadable."""
    try:
        m = json.loads((Path(d) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return m if isinstance(m, dict) else None


def calibration_from_json(c: dict, path: Path) -> Calibration:
    return Calibration(id=c["id"], geometry_id=c["geometry_id"],
                       mm_per_px=c.get("mm_per_px"),
                       intercept_mm=c.get("intercept_mm") or 0.0,
                       lateral_px_per_mm=c.get("lateral_px_per_mm"),
                       r2=c.get("r2"), n_samples=c.get("n_samples", 0),
                       source="library", path=path, fingerprint=c.get("geometry_fp"))


def json_safe(o):
    """A JSON-safe copy: NaN / inf -> None (null), numpy -> Python, Paths ->
    str. json.dump would otherwise write bare NaN tokens, which JSON.parse
    and strict parsers reject (the output bin is read by browsers/agents)."""
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, Path):
        return str(o)
    return o


# Entry depth under the library root, per kind (<kind>/.../<entry>).
_ENTRY_DEPTH = {"bases": 3, "objects": 2, "calibrations": 3, "readings": 2,
                "spectral": 2, "banks": 3, "maps": 3, "sensors": 3, "atlases": 3,
                "calib_poses": 3, "metric": 3, "dentsets": 3, "faces": 3, "facescans": 3,
                "face_golden": 3, "station_golden": 3, "recipe_golden": 3, "presets": 2,
                "recipes": 2, "zones": 3}
_SPECTRAL_SUBDIRS = ("materials", "light")      # spectral/<sub>/<entry>
_ENTRY_MARKERS = {"bases": ("meta.json", "field.npz"),
                  "objects": ("meta.json", "field.npz", "relief.npz"),
                  "calibrations": ("depth_calibration.json", "stereo_calibration.json",
                                   "parity.json", "dent_null.json", "rigref.json"),
                  "calib_poses": ("meta.json", "field.npz"),
                  "metric": ("meta.json", "cloud.npz"),
                  "dentsets": ("meta.json", "frames_long.npy", "frames_short.npy"),
                  "faces": ("meta.json", "face.npz"),
                  "facescans": ("meta.json", "masks.npz"),
                  "face_golden": ("meta.json", "golden.npz"),
                  "station_golden": ("meta.json", "golden.json"),
                  "recipe_golden": ("meta.json",),
                  "presets": ("preset.json", "meta.json"),
                  "recipes": ("recipe.json", "meta.json"),
                  "zones": ("zones.json", "masks.npz"),
                  "readings": ("codes.json", "full_on.png"),
                  "spectral": ("meta.json", "cube.npz", "*.json"),
                  "banks": ("members.jsonl", "meta.json"),
                  "maps": ("map.npz", "meta.json"),
                  "sensors": ("*.json",),
                  "atlases": ("*.json",)}          # sensors/<geometry_id>/<name>/<ts>.json
_TS_NAME = re.compile(r"^\d{8}T\d{6}_\d{6}")     # timestamp() — a new_dir entry


def _check_entry(entry: Path, rel: Path) -> None:
    """Refuse anything but one whole entry directory at its kind's depth."""
    parts = rel.parts
    kind = parts[0] if parts else ""
    if kind not in ENTRY_KINDS:
        raise ValueError(f"{entry} is not a library entry ({'/'.join(ENTRY_KINDS)})")
    depth = _ENTRY_DEPTH[kind]
    if kind == "spectral" and len(parts) >= 2 and parts[1] in _SPECTRAL_SUBDIRS:
        depth = 3
    if len(parts) != depth:
        want = {2: f"{kind}/<entry>", 3: f"{kind}/<geometry_id or group>/<entry>"}[depth]
        raise ValueError(f"{rel.as_posix()} is not one library entry: {kind} entries live at "
                         f"{want}. Delete entries one at a time.")
    if not entry.is_dir():
        raise ValueError(f"{rel.as_posix()} is not an entry folder (a file inside an entry "
                         "can't be deleted on its own)")
    kids = list(entry.iterdir())
    if not kids or _TS_NAME.match(entry.name):
        return                                  # empty or a timestamped (maybe partial) entry
    if any(any(entry.glob(m)) for m in _ENTRY_MARKERS[kind]):
        return
    raise ValueError(f"{rel.as_posix()} does not look like a {kind} entry "
                     f"(none of {', '.join(_ENTRY_MARKERS[kind])} in it)")


# os.rename retries while a file in the entry is briefly held open (Windows
# sharing violation: a preview being streamed, Defender scanning a new file)
TRASH_RETRY_DELAYS_S = (0.05, 0.1, 0.2, 0.4)


TRASH_MAX_SUFFIX = 1000


def _move_to_trash(entry: Path, first: Path) -> Path:
    """Rename `entry` to `first`, or — when an entry of that name was trashed
    before (a re-learned material, a re-taught sensor) — to first-1,
    first-2, ... Never onto an existing path: each attempt is one atomic
    os.rename that fails if its target exists, so a concurrent trash of the
    same name just takes the next suffix. -> the destination used."""
    for n in range(TRASH_MAX_SUFFIX):
        dest = first if n == 0 else first.with_name(f"{first.name}-{n}")
        if dest.exists():
            continue
        try:
            _rename_entry(entry, dest)
        except FileExistsError:
            continue                        # taken between the check and the rename
        return dest
    raise rules.OverwriteRefused(f"trash already holds {first.name} and {TRASH_MAX_SUFFIX - 1} "
                                 f"numbered copies of it under {first.parent}")


def _write_trash_record(dest: Path, record: dict) -> None:
    """TRASHED.json in the trashed entry; an entry restored from trash and
    trashed again already has one, so later records get TRASHED-1.json..."""
    for n in range(TRASH_MAX_SUFFIX):
        f = dest / ("TRASHED.json" if n == 0 else f"TRASHED-{n}.json")
        try:
            with open(f, "x", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2)
            return
        except FileExistsError:
            continue


def _rename_entry(src: Path, dest: Path) -> None:
    for delay in (*TRASH_RETRY_DELAYS_S, None):
        try:
            os.rename(src, dest)            # atomic on one volume; never copy+delete
            return
        except PermissionError as e:
            if delay is None:
                raise rules.EntryInUse(
                    f"{src.name} is in use (a file in it is open: {e.strerror or e}); nothing "
                    "was moved. Close whatever has it open (image viewer, Explorer preview, "
                    "a download in the UI) and retry the delete.") from e
            time.sleep(delay)


def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")
