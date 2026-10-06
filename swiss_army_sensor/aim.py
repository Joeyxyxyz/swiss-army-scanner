"""Aim / hold-field helpers for Swiss Army Scanner.

Numbered-grid callout -> flood verify -> atlas freeze -> targeted scan window.
Label contract matches patterns.GridIndex (DMD top-left origin).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from .patterns import DMD_H, DMD_W, GridIndex, flood, numbered_grid

Rect = Tuple[int, int, int, int]  # x1,y1,x2,y2 half-open


@dataclass
class HoldField:
    """Frozen DMD hold for an object face (and optional floor strip)."""
    held_cell: Rect
    held_floor: Optional[Rect] = None
    step_px: int = 48
    grid_labels: dict = field(default_factory=dict)
    origin: str = "DMD (0,0) top-left of 912x1140"
    object_id: str = "standing_cell"
    size: Tuple[int, int] = (DMD_W, DMD_H)
    notes: str = ""

    @classmethod
    def from_grid(cls, cell_spec: str, floor_spec: Optional[str] = None,
                  step_px: int = 48, object_id: str = "standing_cell",
                  notes: str = "") -> "HoldField":
        gi = GridIndex(step_px=step_px)
        cell = gi.parse(cell_spec)
        floor = gi.parse(floor_spec) if floor_spec else None
        labels = {"cell": cell_spec.upper()}
        if floor_spec:
            labels["floor"] = floor_spec.upper()
        return cls(held_cell=cell, held_floor=floor, step_px=step_px,
                   grid_labels=labels, object_id=object_id, notes=notes)

    def to_atlas(self) -> dict:
        return {
            "source": "swiss_army_sensor.aim.HoldField",
            "object_id": self.object_id,
            "step_px": self.step_px,
            "origin": self.origin,
            "grid_labels": self.grid_labels,
            "dmd": {
                "held_cell": list(self.held_cell),
                "held_floor": list(self.held_floor) if self.held_floor else None,
                "size": list(self.size),
            },
            "notes": self.notes,
        }

    def save_atlas(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_atlas(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_atlas(cls, path: Union[str, Path]) -> "HoldField":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        dmd = data.get("dmd", {})
        cell = tuple(dmd["held_cell"])
        floor = dmd.get("held_floor")
        floor_t = tuple(floor) if floor else None
        return cls(
            held_cell=cell,  # type: ignore[arg-type]
            held_floor=floor_t,  # type: ignore[arg-type]
            step_px=int(data.get("step_px", 48)),
            grid_labels=dict(data.get("grid_labels") or {}),
            origin=str(data.get("origin", cls.origin)),
            object_id=str(data.get("object_id", "object")),
            size=tuple(dmd.get("size") or (DMD_W, DMD_H)),  # type: ignore[arg-type]
            notes=str(data.get("notes") or ""),
        )

    def dual_grid_image(self) -> np.ndarray:
        return numbered_grid(step_px=self.step_px, held=self.held_cell, label=True)

    def flood_cell(self, value: int = 255) -> np.ndarray:
        img = np.zeros((self.size[1], self.size[0]), dtype=np.uint8)
        x1, y1, x2, y2 = self.held_cell
        img[y1:y2, x1:x2] = int(value)
        return img


def window_pattern(pattern: np.ndarray, rect: Rect, amp: int = 255) -> np.ndarray:
    """Keep pattern only inside rect (for targeted phase / shine)."""
    img = np.zeros_like(pattern)
    x1, y1, x2, y2 = rect
    sl = pattern[y1:y2, x1:x2]
    if sl.size == 0:
        return img
    if sl.dtype != np.uint8:
        sl = np.clip(sl, 0, 255).astype(np.uint8)
    if amp != 255:
        sl = (sl.astype(np.float32) / 255.0 * amp).astype(np.uint8)
    img[y1:y2, x1:x2] = sl
    return img


# Standing cell preset from 2026-09-23 flood-verify (column P, rows 17-22; 23=floor)
STANDING_CELL_P17_P22 = HoldField.from_grid(
    "P17-P22", floor_spec="P23", object_id="standing_cell",
    notes="Joey flood-verify: column P covers cell; row 23 is floor not face",
)
