"""Structured records: one JSON line per verb call, append-only."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class Record:
    verb: str
    args: dict
    ok: bool = True
    result: Any = None
    error: Optional[str] = None
    duration_s: float = 0.0
    ts: str = field(default_factory=lambda: datetime.now().isoformat(timespec="milliseconds"))

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=summarize)


def summarize(o):
    """JSON-safe, compact view of verb args/results for the log."""
    if isinstance(o, np.ndarray):
        return f"<ndarray {o.dtype} {list(o.shape)}>"
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if is_dataclass(o):
        out = {}
        for k, v in vars(o).items():
            if k.startswith("_") or k == "meta":
                continue
            out[k] = v
        return {"type": type(o).__name__, **out}
    if isinstance(o, (set, tuple)):
        return list(o)
    return repr(o)


def append(path: Path, rec: Record) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(rec.to_json() + "\n")
