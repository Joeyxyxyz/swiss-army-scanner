"""
PatternBasis — the one interface every projected basis implements (§3).

The engine never special-cases a pattern: it asks a basis for frames to
project, captures them, and hands them back to the same basis to decode
into a DeformationField.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..decode.deformation import DeformationField


class PatternBasis(ABC):
    name: str = ""
    axes: set = set()     # subset of {"x", "y", "diag"}

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def generate(self, dmd_shape: tuple) -> list[np.ndarray]:
        """Patterns to project, uint8 arrays of dmd_shape (H, W)."""

    @abstractmethod
    def decode(self, frames: list[np.ndarray], geometry_id: str,
               base_frames=None) -> DeformationField:
        """Captured frames (same order as generate) -> DeformationField."""

    @abstractmethod
    def n_frames(self) -> int:
        ...

    def complementary(self) -> bool:
        """True if each pattern is followed by its inverse (ambient-robust)."""
        return bool(self.params.get("complementary", False))
