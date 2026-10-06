"""Zone-lit bases: a zone scan should be compared with a base lit the same way."""

import numpy as np

from engine import api

from conftest import Box


def test_zone_lit_base_matches_zone_scan(preset):
    preset.camera.average = 1
    api.capture_map()
    api.project_grid()
    full = api.capture_base(preset, "full")
    zoned = api.capture_base(preset, "zoned_E6", zone="E6")
    assert zoned.meta["zone"] == "E6" and zoned.meta["zone_cells"]
    assert zoned.field.valid_pct < full.field.valid_pct          # only the zone was lit

    api.mock_bench().place(Box(15.0, 40.0, 40.0))
    a = api.capture_object(preset, full, "vs_full", zone="E6")
    b = api.capture_object(preset, zoned, "vs_zoned", zone="E6")
    assert any("stray light differs" in w for w in a.meta["warnings"])
    assert not any("stray light differs" in w for w in b.meta["warnings"])
    rb = b.relief[b.relief_mask]
    assert rb.size > 100 and np.isfinite(rb).all()


def test_read_codes_still_reads_after_split(preset):
    from engine.capture.mock_bench import Label
    preset.camera.average = 1
    api.mock_bench().place(Label("zone-base-test", center_mm=(0, 0), size_mm=45))
    r = api.read_codes()
    assert [c.data for c in r.codes] == ["zone-base-test"]
