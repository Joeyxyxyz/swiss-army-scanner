"""Barcode/QR reading and defect detection, end to end in mock mode."""

import json

import pytest

from engine import api
from engine.capture.mock_bench import Box, Label, VALIDATION_OBJECT
from engine.measure import inspect as inspect_mod

from conftest import PRESET


@pytest.fixture
def fast(eng):
    p = api.load_preset(PRESET)
    p.camera.average = 1          # speed; decode quality is not the subject here
    return p


def test_read_qr_codes(fast):
    bench = api.mock_bench()
    bench.place(Label("SAS-0001 cell can lot 42", center_mm=(-60, 0), size_mm=45))
    bench.place(Label("https://example.com/part/7", center_mm=(70, 20), size_mm=45))
    r = api.read_codes()
    got = {c.data for c in r.codes}
    assert got == {"SAS-0001 cell can lot 42", "https://example.com/part/7"}, r
    assert all(c.kind == "QR" and c.corners for c in r.codes)
    assert r.polarity_inverted                      # bench-like inverted path
    assert {"red", "white", "green", "blue"} <= set(r.illuminations)
    saved = json.loads((r.path / "codes.json").read_text())
    assert len(saved["codes"]) == 2 and (r.path / "full_on.png").exists()


def test_read_codes_none_in_view(fast):
    r = api.read_codes()
    assert not r.found and r.warnings


@pytest.fixture
def avg4(fast):
    fast.camera.average = 4       # defect SNR needs some averaging (like the bench)
    return fast


def test_inspect_finds_bump_and_applies_trigger(avg4):
    fast = avg4
    base = api.capture_base(fast, "empty")
    bench = api.mock_bench()
    bench.place(VALIDATION_OBJECT)
    cal = api.capture_object(fast, base, "cal", known_dims={"H": 106.95})
    api.calibrate(fast, [cal])

    bench.clear()
    bench.place(Box(30.0, 90.0, 70.0))                           # flat part
    good = api.capture_object(fast, base, "part_good")
    ig = api.inspect(good)
    assert ig.units == "mm" and ig.verdict != "FAIL", ig
    assert not any(d.exceeds_trigger for d in ig.defects)
    # PASS only when a trigger-high defect would have been seen: threshold plus a
    # 2-sigma margin under the trigger. The mock's in-part noise (phase ripple)
    # keeps it just above 0.7 mm, which used to be reported as PASS anyway.
    reach = ig.threshold + inspect_mod.DETECT_MARGIN_SIGMA * ig.noise_sigma
    assert (ig.verdict == "PASS") == (reach <= ig.trigger + 1e-4), ig
    if ig.verdict == "INDETERMINATE":
        assert any("cannot PASS" in w for w in ig.warnings), ig

    bench.place(Box(34.0, 12.0, 12.0, center_mm=(15, 10)))       # 4 mm bump on top
    bad = api.capture_object(fast, base, "part_bump")
    ib = api.inspect(bad)
    assert ib.verdict == "FAIL"
    top = ib.defects[0]
    assert top.exceeds_trigger and top.height == pytest.approx(4.0, abs=0.5)
    assert top.area_px > 400
    assert ib.path.exists() and ib.path.with_suffix(".png").exists()


def test_inspect_uncalibrated_is_px_and_indeterminate(avg4):
    fast = avg4
    base = api.capture_base(fast, "empty")
    api.mock_bench().place(Box(30.0, 90.0, 70.0))
    api.mock_bench().place(Box(34.0, 12.0, 12.0, center_mm=(15, 10)))
    i = api.inspect(api.capture_object(fast, base, "p"))
    assert i.units == "px" and i.verdict == "INDETERMINATE"
    assert i.defects and any("rule 6" in w for w in i.warnings)


# -------------------------------------- read_codes: capture / decode split

def test_codes_capture_only_shoots_and_decode_needs_no_rig(fast, monkeypatch):
    """capture() is the rig part (shoot calls only, nothing slow), decode()
    runs on its frames after the rig is released; read() = both."""
    from engine.sense import codes as codes_mod
    bench = api.mock_bench()
    bench.place(Label("SAS-0001 cell can lot 42", center_mm=(-60, 0), size_mm=45))
    s = api._engine()
    decoded_in_rig = []
    real_decode_image = codes_mod.decode_image
    in_rig = {"on": False}

    def spy(img, illumination=""):
        decoded_in_rig.append(in_rig["on"])
        return real_decode_image(img, illumination)
    monkeypatch.setattr(codes_mod, "decode_image", spy)
    with s.rig(fast) as sync:
        in_rig["on"] = True
        frames = codes_mod.capture(sync, (255, 110), ("red", "white"))
        in_rig["on"] = False
    assert list(frames) == ["dmd_black", "flood_255", "flood_110", "red", "white"]
    assert not decoded_in_rig                          # nothing decoded while the rig is open
    found, full_on, inverted = codes_mod.decode(frames, (255, 110))
    assert {c.data for c in found} == {"SAS-0001 cell can lot 42"}
    assert inverted and full_on == "dmd_black"         # bench-like inverted path
    assert len(decoded_in_rig) == 5 and not any(decoded_in_rig)
    assert found[0].illumination in frames
    with pytest.raises(ValueError, match="flood_170"):
        codes_mod.decode(frames, (255, 170))           # levels must match the capture
    with pytest.raises(ValueError, match="colour"):
        with s.rig(fast) as sync:
            codes_mod.capture(sync, (255,), ("infrared",))

    # read() keeps its 4-tuple for existing callers
    with s.rig(fast) as sync:
        codes, frames2, full_on2, inv2 = codes_mod.read(sync, (255, 110), ("red",))
    assert {c.data for c in codes} == {"SAS-0001 cell can lot 42"}
    assert list(frames2) == ["dmd_black", "flood_255", "flood_110", "red"]
    assert full_on2 == "dmd_black" and inv2
