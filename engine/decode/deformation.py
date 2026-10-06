"""
DeformationField — the one universal intermediate (§4).

Whatever basis was projected, decode produces this: per-camera-pixel
projector correspondence (+ confidence + valid mask). Everything
downstream consumes ONLY this.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

AXES = ("corr_x", "corr_y", "corr_diag")


@dataclass
class DeformationField:
    corr_x: Optional[np.ndarray]     # per camera px: projector column (float)
    corr_y: Optional[np.ndarray]     # per camera px: projector row (float)
    corr_diag: Optional[np.ndarray]
    conf: np.ndarray                 # per px confidence (gc_conf-style)
    mask: np.ndarray                 # valid pixels
    geometry_id: str
    basis_names: list
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if all(getattr(self, a) is None for a in AXES):
            raise ValueError("DeformationField needs at least one correspondence axis")
        shape = self.mask.shape
        for a in AXES:
            v = getattr(self, a)
            if v is not None and v.shape != shape:
                raise ValueError(f"{a} shape {v.shape} != mask shape {shape}")
        if self.conf.shape != shape:
            raise ValueError(f"conf shape {self.conf.shape} != mask shape {shape}")
        self.mask = self.mask.astype(bool)

    @property
    def shape(self):
        return self.mask.shape

    @property
    def axes(self) -> list:
        return [a for a in AXES if getattr(self, a) is not None]

    @property
    def valid_pct(self) -> float:
        return 100.0 * float(self.mask.mean())

    def save(self, f) -> None:
        """Write npz to an open binary file handle (the store opens it 'xb')."""
        arrays = {a: getattr(self, a) for a in self.axes}
        np.savez_compressed(
            f, conf=self.conf.astype(np.float32), mask=self.mask, **arrays,
            _header=np.array(json.dumps({
                "geometry_id": self.geometry_id,
                "basis_names": list(self.basis_names),
                "meta": self.meta})))

    @classmethod
    def load(cls, path: Path) -> "DeformationField":
        with np.load(path) as z:
            hdr = json.loads(str(z["_header"]))
            axes = {a: (z[a] if a in z.files else None) for a in AXES}
            return cls(conf=z["conf"], mask=z["mask"],
                       geometry_id=hdr["geometry_id"],
                       basis_names=hdr["basis_names"], meta=hdr["meta"], **axes)
