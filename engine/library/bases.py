"""Base-reference library: empty-scene fields, stored per geometry_id."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from ..results import BaseRef
from .store import LibraryStore, read_meta


def save_base(store: LibraryStore, fld, label: str, meta: dict) -> BaseRef:
    d = store.new_dir("bases", fld.geometry_id, label=label)
    with store.filling(d):          # a failed save goes to trash/, not into the listing
        store.write_field(d / "field.npz", fld)
        store.write_preview(d / "preview.png", getattr(fld, fld.axes[0]), fld.mask)
        created = datetime.now().isoformat(timespec="seconds")
        meta = {**meta, "id": d.name, "label": label, "geometry_id": fld.geometry_id,
                "created": created, "valid_pct": round(fld.valid_pct, 2),
                "basis_names": fld.basis_names}
        store.write_json(d / "meta.json", meta)     # written last: marks the entry complete
    return BaseRef(id=d.name, label=label, geometry_id=fld.geometry_id, path=d,
                   created=created, meta=meta, _field=fld)


def list_bases(store: LibraryStore, geometry_id: Optional[str] = None) -> list[BaseRef]:
    """Complete bases (entries with a readable meta.json), oldest first.
    Half-written entries are skipped: see LibraryStore.incomplete_entries."""
    geos = [geometry_id] if geometry_id else store.geometry_ids("bases")
    out = []
    for g in geos:
        for d in store.entries("bases", g):
            m = read_meta(d)
            if m is None:
                continue
            rej = d / "REJECTED.json"
            if rej.exists():
                try:
                    reason = json.loads(rej.read_text(encoding="utf-8")).get("reason", "rejected")
                except (OSError, ValueError, AttributeError):
                    reason = "rejected"
                m["rejected"] = reason or "rejected"
            out.append(BaseRef(id=d.name, label=m.get("label", d.name),
                               geometry_id=m.get("geometry_id", g),
                               path=d, created=m.get("created", ""), meta=m))
    return sorted(out, key=lambda b: b.id)
