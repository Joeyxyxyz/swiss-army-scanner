import copy
import shutil
import json

import pytest

from engine import config, rules
from engine.config import ConfigError, preset_from_dict

from conftest import FIXTURE_PRESETS, PRESET


def raw():
    return json.loads(config.resolve_preset_path(PRESET, FIXTURE_PRESETS).read_text())


def test_shipped_preset_loads_real_and_mock():
    real = config.load_preset(PRESET, presets_dir=FIXTURE_PRESETS)
    assert real.geometry_id == "wall_2p7ft_vstripe_v1" and not real.mock
    assert real.camera.average == 8 and real.projection.duty == 0.55
    assert real.projection.monitor_offset == (1920, 0)
    mock = config.load_preset(PRESET, presets_dir=FIXTURE_PRESETS, mock=True)
    assert mock.geometry_id == "mock-wall_2p7ft_vstripe_v1"
    assert mock.rig_geometry_id == "wall_2p7ft_vstripe_v1"


@pytest.mark.parametrize("path,value", [
    (("camera", "exposure_ms"), 40),           # typo'd key
    (("projection", "gama"), 2.2),
    (("bogus",), 1),
])
def test_unknown_keys_rejected(path, value):
    d = raw()
    tgt = d
    for k in path[:-1]:
        tgt = tgt[k]
    tgt[path[-1]] = value
    with pytest.raises(ConfigError, match="unknown keys"):
        preset_from_dict(d)


@pytest.mark.parametrize("proj", [
    {"transport": "usb"},
    {"set_mode": "video"},
    {"display_mode": "video"},
    {"dlp_usb": True},
])
def test_usb_projection_config_refused(proj):
    d = raw()
    d["projection"].update(proj)
    with pytest.raises(rules.SafetyViolation):
        preset_from_dict(d)


def test_led_thermal_limits():
    d = raw()
    d["projection"]["led"] = {"mode": "burst", "r": 180, "g": 255, "b": 230}
    preset_from_dict(d)                                   # at the ceiling: ok
    d["projection"]["led"] = {"mode": "burst", "r": 181}
    with pytest.raises(rules.SafetyViolation, match="R=181"):
        preset_from_dict(d)
    d["projection"]["led"] = {"mode": "continuous", "g": 200}   # avg limit 180
    with pytest.raises(rules.SafetyViolation, match="G=200"):
        preset_from_dict(d)


def test_color_scheme_on_mono_is_flagged():
    d = raw()
    d["projection"]["scheme"] = "green"
    p = preset_from_dict(d)
    assert any("MONO" in w for w in p.warnings)
    # (the real-mode preset now also carries LED / duty warnings, tested below)
    assert not any("MONO" in w for w in preset_from_dict(raw()).warnings)
    assert not preset_from_dict(raw(), mock=True).warnings


def test_bases_validated():
    d = raw()
    d["bases"] = [{"name": "moire"}]
    with pytest.raises(ConfigError, match="unknown basis"):
        preset_from_dict(d)
    d["bases"] = [{"name": "graycode", "bitz": 10}]
    with pytest.raises(ConfigError, match="unknown params"):
        preset_from_dict(d)


def test_preset_calibration_binds_real_only():
    d = raw()
    d["calibration"]["mm_per_px"] = 2.2
    assert preset_from_dict(d).calibration.mm_per_px == 2.2
    m = preset_from_dict(copy.deepcopy(d), mock=True)
    assert m.calibration is None and any("ignored in mock" in w for w in m.warnings)


# ------------------------------------------------ review fixes (core area)

def test_real_preset_warns_led_unrecorded_and_duty_floor():
    """Rule 3 can't run without an led block; invert + duty<1 on a DMD that
    shows what is sent is a 45 % light floor, not a software ND."""
    w = preset_from_dict(raw()).warnings
    assert any("projection.led is not recorded" in x for x in w)
    assert any("115..255" in x and "45%" in x for x in w)
    d = raw()
    d["projection"]["led"] = {"mode": "continuous", "r": 100, "g": 150, "b": 150}
    d["projection"]["duty"] = 1.0
    d["camera"]["exposure_us"] = 16667          # a whole 60 Hz video frame (see below)
    assert not preset_from_dict(d).warnings
    d["projection"].update(invert=False, duty=0.55)       # a real ND: peak lowered, floor 0
    assert not preset_from_dict(d).warnings


def test_exposure_not_frame_locked_is_warned_not_refused():
    """HDMI video mode draws each 60 Hz frame as R/G/B + PWM bit-planes; a
    free-running exposure that is not a whole number of frames flickers.
    Warn (real rig), never refuse; whole frames within 0.5 % are clean."""
    d = raw()
    d["projection"]["led"] = {"mode": "continuous", "r": 100, "g": 150, "b": 150}
    d["projection"]["duty"] = 1.0
    for us in (8000, 2000, 28000, 20000):
        d["camera"]["exposure_us"] = us
        w = [x for x in preset_from_dict(d).warnings if "camera.exposure_us" in x]
        assert len(w) == 1, (us, w)
        assert f"camera.exposure_us {us}" in w[0] and "16667 or 33333 us" in w[0]
        assert "R/G/B" in w[0] and "PWM" in w[0] and "iris" in w[0] and "duty" in w[0]
    for us in (16667, 33333, 50000, 16700):          # 16700: within the 0.5 % tolerance
        d["camera"]["exposure_us"] = us
        assert not preset_from_dict(d).warnings, us
    d["camera"]["exposure_us"] = 8000
    assert not preset_from_dict(d, mock=True).warnings   # the simulated bench has no flicker
    assert config.frame_locked(16667) and not config.frame_locked(8000)
    assert config.VIDEO_FRAME_US == pytest.approx(1e6 / 60)
    from engine.capture import assist
    assert assist.frame_locked is config.frame_locked            # one definition
    assert assist.VIDEO_FRAME_US == config.VIDEO_FRAME_US


def test_fullrange_variant_is_same_geometry_at_duty_1():
    std = config.load_preset(PRESET, presets_dir=FIXTURE_PRESETS)
    full = config.load_preset("cell_wall_2p7ft_vstripe_fullrange")
    assert full.projection.duty == 1.0 and full.geometry_id == std.geometry_id
    assert config.geometry_fingerprint(full) == config.geometry_fingerprint(std)
    assert not any("115..255" in w for w in full.warnings)


def test_gain_db_range_validated():
    for bad in (-1, 60, "6", True):
        d = raw()
        d["camera"]["gain_db"] = bad
        with pytest.raises(ConfigError, match="gain_db"):
            preset_from_dict(d)
    d = raw()
    d["camera"]["gain_db"] = 12.5
    assert preset_from_dict(d).camera.gain_db == 12.5


def test_ids_refuse_dot_segments():
    for bad in (".", "..", "../x", "a/b"):
        d = raw()
        d["geometry_id"] = bad
        with pytest.raises(ConfigError, match="geometry_id"):
            preset_from_dict(d)


def test_geometry_fingerprint_tracks_geometry_bound_settings():
    d = raw()
    base = config.geometry_fingerprint(preset_from_dict(d))
    assert base["basis_axis"] == "x" and base["rotate90"] is False
    for path, value in ((("projection", "rotate90"), True), (("rig", "projector_side"), "left"),
                        (("rig", "standoff_mm"), 900.0), (("projection", "size"), [800, 600])):
        e = raw()
        e[path[0]][path[1]] = value
        assert rules.fingerprint_diffs(base, config.geometry_fingerprint(preset_from_dict(e)))
    e = raw()
    e["camera"]["exposure_us"], e["projection"]["duty"] = 1234, 1.0      # not geometry
    e["rig"]["baseline_mm"] = None                                        # unknown: no diff
    assert not rules.fingerprint_diffs(base, config.geometry_fingerprint(preset_from_dict(e)))


def test_save_preset_refuses_geometry_change_under_same_id(tmp_path):
    """Finding: overwrite=True with rotate90 / rig side changed kept the
    geometry_id, so old bases and calibrations were silently reused."""
    shutil.copy(config.resolve_preset_path(PRESET, FIXTURE_PRESETS), tmp_path / f"{PRESET}.json")
    d = raw()
    d["projection"]["rotate90"] = True
    with pytest.raises(ConfigError, match="geometry_id wall_2p7ft_vstripe_v1.*rotate90"):
        config.save_preset(d, presets_dir=tmp_path, overwrite=True)
    d["name"] = "other_name"                     # a new name reusing the id: same refusal
    with pytest.raises(ConfigError, match="new geometry_id"):
        config.save_preset(d, presets_dir=tmp_path)
    d["geometry_id"] = "wall_2p7ft_vstripe_v2"   # the right way: a new geometry
    assert config.save_preset(d, presets_dir=tmp_path).exists()
    ok = raw()
    ok["camera"]["exposure_us"] = 2000           # not geometry: overwrite is fine
    config.save_preset(ok, presets_dir=tmp_path, overwrite=True)
    assert json.loads((tmp_path / f"{PRESET}.json").read_text())["camera"]["exposure_us"] == 2000
