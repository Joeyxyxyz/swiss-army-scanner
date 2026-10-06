"""Object library: object fields + relief vs base, known dims, results."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np

from ..results import ObjectScan
from .store import LibraryStore, read_meta

_DIM_ALIASES = {"h": "H", "height": "H", "w": "W", "width": "W",
                "l": "L", "length": "L"}


def normalize_known_dims(known: Optional[dict]) -> Optional[dict]:
    """{'height': 106.95, 'W': 56.5} -> {'H': 106.95, 'W': 56.5} (mm, > 0)."""
    if not known:
        return None
    out = {}
    for k, v in known.items():
        key = _DIM_ALIASES.get(str(k).lower())
        if key is None:
            raise ValueError(f"known_dims key {k!r}; use H/W/L (mm)")
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError(f"known_dims[{k!r}] must be a positive number of mm")
        out[key] = float(v)
    # W is the major footprint extent, L the minor (measure reports it that way)
    if "W" in out and "L" in out and out["W"] < out["L"]:
        out["W"], out["L"] = out["L"], out["W"]
    return out


def save_object(store: LibraryStore, fld, relief: np.ndarray, relief_mask: np.ndarray,
                label: str, base_id: str, known_dims: Optional[dict],
                calibration_id: Optional[str], meta: dict) -> ObjectScan:
    d = store.new_dir("objects", label=label)
    with store.filling(d):          # a failed save goes to trash/, not into the listing
        store.write_field(d / "field.npz", fld)
        store.write_npz(d / "relief.npz", relief=relief.astype(np.float32), mask=relief_mask)
        store.write_preview(d / "preview.png", relief, relief_mask)
        if known_dims:
            store.write_json(d / "known_dims.json", {**known_dims, "units": "mm"})
        created = datetime.now().isoformat(timespec="seconds")
        meta = {**meta, "id": d.name, "label": label, "geometry_id": fld.geometry_id,
                "created": created, "base_ref": base_id, "calib_id": calibration_id,
                "known_dims": known_dims, "valid_pct": round(fld.valid_pct, 2),
                "basis_names": fld.basis_names}
        store.write_json(d / "meta.json", meta)     # written last: marks the entry complete
    return ObjectScan(id=d.name, label=label, geometry_id=fld.geometry_id, path=d,
                      created=created, base_id=base_id, known_dims=known_dims,
                      calibration_id=calibration_id, meta=meta, _field=fld,
                      _relief=(relief, relief_mask))


def list_objects(store: LibraryStore, geometry_id: Optional[str] = None) -> list[ObjectScan]:
    """Complete scans (entries with a readable meta.json), oldest first.
    Half-written entries are skipped: see LibraryStore.incomplete_entries."""
    out = []
    for d in store.entries("objects"):
        m = read_meta(d)
        if m is None or "geometry_id" not in m:
            continue
        if geometry_id and m["geometry_id"] != geometry_id:
            continue
        out.append(ObjectScan(id=d.name, label=m.get("label", d.name),
                              geometry_id=m["geometry_id"],
                              path=d, created=m.get("created", ""),
                              base_id=m.get("base_ref", ""),
                              known_dims=m.get("known_dims"),
                              calibration_id=m.get("calib_id"), meta=m))
    return out
