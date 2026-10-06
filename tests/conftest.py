import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import api  # noqa: E402
from engine.capture.mock_bench import Box, MockBench, VALIDATION_OBJECT  # noqa: E402,F401

# frozen copies: bench tuning of presets/ must not change test expectations
FIXTURE_PRESETS = ROOT / "tests" / "fixtures" / "presets"
PRESET = "cell_wall_2p7ft_vstripe"   # fringes vertical in camera = correct for a right-side projector


@pytest.fixture
def eng(tmp_path):
    """Engine in mock mode on a throwaway library."""
    session = api.configure(tmp_path / "library", mock=True, bench=MockBench(seed=1),
                            presets_dir=FIXTURE_PRESETS)
    yield session
    api._session = None


@pytest.fixture
def preset(eng):
    return api.load_preset(PRESET)
