"""Code decoding on synthetic symbols (kind names, UPC-A, contrast variants for
every decoder), the experimental anti-glare pipeline, and the CLI's exit codes
and one-line errors around codes / end faces / a missing camera SDK."""

import subprocess
import sys
import types

import numpy as np
import pytest

from engine.sense import antiglare as ag
from engine.sense import codes as codes_mod
from ui import cli

from conftest import ROOT

zxingcpp = pytest.importorskip("zxingcpp")
cv2 = pytest.importorskip("cv2")


def symbol(text, fmt, scale=4, pad=40) -> np.ndarray:
    """A synthetic code, black on white, with a quiet zone (float32 0..255)."""
    im = np.asarray(zxingcpp.write_barcode_to_image(zxingcpp.create_barcode(text, fmt),
                                                    scale=scale)).astype(np.float32)
    return cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def read(img):
    return [(c.kind, c.data) for c in codes_mod.decode_image(img)]


# ------------------------------------------------------------ decode_image

def test_kind_names_are_one_per_symbology():
    for a, b in (("EAN_13", "EAN13"), ("QRCode", "QR"), ("QRCODE", "QR"),
                 ("CODE_128", "Code128"), ("DataMatrix", "DATA_MATRIX"), ("UPCA", "UPC_A")):
        assert codes_mod.normalise_kind(a) == codes_mod.normalise_kind(b)
    assert codes_mod.normalise_code("EAN13", "0036000291452") == \
        ("UPC_A", "036000291452", "0036000291452")
    assert codes_mod.normalise_code("EAN_13", "4006381333931") == ("EAN_13", "4006381333931", "")


def test_qr_and_code128():
    assert read(symbol("https://example.com/p/7", zxingcpp.BarcodeFormat.QRCode).astype(np.uint8)) \
        == [("QR", "https://example.com/p/7")]
    assert read(symbol("ABC-12345", zxingcpp.BarcodeFormat.Code128).astype(np.uint8)) \
        == [("CODE_128", "ABC-12345")]


def test_upc_a_is_reported_as_the_printed_12_digits_once():
    img = symbol("03600029145", zxingcpp.BarcodeFormat.UPCA).astype(np.uint8)
    found = codes_mod.decode_image(img)
    assert len(found) == 1, found                    # opencv + zxing, all variants: one entry
    c = found[0]
    assert (c.kind, c.data) == ("UPC_A", "036000291452") and len(c.data) == 12
    assert c.raw == "0036000291452"


def test_dim_noisy_data_matrix():
    rng = np.random.default_rng(7)
    dm = symbol("LOT-42 SN 0001", zxingcpp.BarcodeFormat.DataMatrix, scale=3)
    dim = np.clip(0.35 * dm + 40 + rng.normal(0, 8, dm.shape), 0, 255).astype(np.uint8)
    assert read(dim) == [("DATA_MATRIX", "LOT-42 SN 0001")]


def test_zxing_runs_on_the_contrast_variants(monkeypatch):
    """zxing sees the same raw / CLAHE / CLAHE x2 images the OpenCV path gets."""
    seen = []
    real = zxingcpp.read_barcodes
    monkeypatch.setattr(zxingcpp, "read_barcodes", lambda im, *a, **k: seen.append(im.shape)
                        or real(im, *a, **k))
    img = np.full((60, 80), 128, np.uint8)
    codes_mod.decode_image(img)
    assert seen == [(60, 80), (60, 80), (120, 160)]


# ------------------------------------------------------------ anti-glare (experimental)

def glare_bracket(seed=1):
    """A QR on a glossy label: reflectance 0.15..0.9 plus an ADDITIVE specular
    stripe (vertical, up to 6x the white level), shot at the anti-glare levels."""
    rng = np.random.default_rng(seed)
    qr = symbol("SAS-0001 lot 42", zxingcpp.BarcodeFormat.QRCode, scale=4)
    refl = 0.15 + 0.75 * qr / 255.0
    h, w = refl.shape
    x = np.arange(w)
    stripe = 6.0 * np.exp(-0.5 * ((x - w / 2) / (w * 0.06)) ** 2) * \
        (1 + 0.3 * np.sin(np.arange(h) / 40.0))[:, None]
    frames = {"dmd_black": np.clip(5 + rng.normal(0, 1.5, refl.shape), 0, 255)}
    for lv in ag.LEVELS:
        frames[f"flood_{lv}"] = np.clip(5 + lv * (refl + stripe) + rng.normal(0, 1.5, refl.shape),
                                        0, 255).round()
    return frames, refl, stripe


def test_plain_decode_fails_and_anti_glare_reads_through_the_stripe():
    frames, _, _ = glare_bracket()
    assert all(not codes_mod.decode_image(f) for f in frames.values())   # every single frame
    found, img, used = ag.decode(frames)
    assert [c.data for c in found] == ["SAS-0001 lot 42"] and used == "hdr"
    assert found[0].illumination.startswith("anti_glare_hdr:deveil")
    # either polarity: an optically inverted path delivers the same frames under other names
    inv = {"dmd_black": frames["flood_255"], **{f"flood_{lv}": f for lv, f in
                                                zip(ag.LEVELS, list(frames.values())[::-1])}}
    assert [c.data for c in ag.decode(inv)[0]] == ["SAS-0001 lot 42"]


def test_hdr_fuse_recovers_radiance_where_bright_frames_saturate():
    frames, refl, stripe = glare_bracket()
    hdr = ag.hdr_fuse([f for n, f in frames.items() if n != "dmd_black"], frames["dmd_black"])
    truth = 255.0 * (refl + stripe)                   # units of the brightest frame
    m = truth > 300                                    # saturated in flood_255
    assert m.any() and np.median(np.abs(hdr[m] / truth[m] - 1)) < 0.05


def test_strip_own_drops_bloom_and_deveil_removes_offset():
    S = np.zeros((5, 4, 5), np.float32)
    for k in range(5):
        S[k, :, k] = 100.0                             # strip k lights column k
        S[k] += 30.0                                   # bloom everywhere
    own = ag.strip_own(S, half=0)
    assert np.allclose(own, 130.0)
    tot = S.sum(0)
    assert np.allclose(tot, 250.0)                     # the flood-like sum carries 4x the bloom
    pats = ag.strip_patterns(10, 16, 3, 2, bright=0, dark=255, shape=(8, 20))
    assert [n for n, _ in pats] == ["strip_10", "strip_12", "strip_14"]
    assert (pats[0][1][:, 10:13] == 0).all() and (pats[0][1][:, 13:] == 255).all()
    img = np.tile(np.array([0, 1] * 10, np.float32) * 50, (30, 1)) + \
        np.linspace(0, 120, 20)[None]                  # print + an additive ramp
    c = ag.deveil(img, (9, 3))
    assert c[:, 5:15:2].mean() > 0.8 and c[:, 6:16:2].mean() < 0.2


# ------------------------------------------------------------ CLI

def run_cli(tmp_path, *argv):
    return subprocess.run([sys.executable, "-m", "ui.cli", "--mock", "--average", "1",
                           "--library", str(tmp_path / "lib"), *argv], cwd=ROOT,
                          capture_output=True, text=True, timeout=600)


def test_codes_exit_codes_through_the_entry_point(tmp_path):
    found = run_cli(tmp_path, "--mock-part", "qr:SAS-0001 lot 42", "codes")
    assert found.returncode == 0, found.stdout + found.stderr
    assert "SAS-0001 lot 42" in found.stdout and "Traceback" not in found.stderr
    none = run_cli(tmp_path, "codes")
    assert none.returncode == cli.EXIT_NOT_OK, none.stdout + none.stderr
    assert "0 found" in none.stdout


def test_strip_cols_needs_anti_glare(tmp_path, capsys):
    code = cli.main(["--mock", "--library", str(tmp_path), "codes", "--strip-cols", "100,200"])
    assert code == cli.EXIT_REFUSED and "--anti-glare" in capsys.readouterr().err


def test_face_locate_on_an_x_only_scan_names_the_rescan(tmp_path, capsys, monkeypatch):
    ms = types.SimpleNamespace(id="m1", field=types.SimpleNamespace(corr_x=np.zeros(1), corr_y=None))
    monkeypatch.setattr(cli, "_find_metric", lambda key: ms)
    code = cli.main(["--mock", "--library", str(tmp_path), "face-locate", "m1"])
    err = capsys.readouterr().err
    assert code == cli.EXIT_REFUSED and "scan-fast --axis xy" in err and "Traceback" not in err


def test_auto_intensity_on_an_inverted_preset_is_one_line(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_find_face", lambda key: object())
    called = []
    monkeypatch.setattr(cli.api, "auto_intensity", lambda *a, **k: called.append(a))
    code = cli.main(["--mock", "--library", str(tmp_path), "--preset", "bench_9in_metric",
                     "face-scan", "c1", "--auto-intensity"])
    err = capsys.readouterr().err
    assert code == cli.EXIT_REFUSED and not called
    assert "projection.invert false" in err and "bench_9in_metric_bright" in err
    assert len(err.strip().splitlines()) == 1 and "Traceback" not in err


def test_missing_pyspin_is_one_actionable_line(monkeypatch):
    from engine.capture import camera
    monkeypatch.setitem(sys.modules, "PySpin", None)          # "cannot import PySpin"
    with pytest.raises(ModuleNotFoundError) as ei:
        camera.require_pyspin()
    code, line = cli._classify_error(ei.value, "scan")
    assert code == cli.EXIT_NOT_AVAILABLE and "unexpected" not in line
    assert "PySpin (FLIR Spinnaker SDK)" in line and "SCAN_PYTHON" in line and "--mock" in line
    code, line = cli._classify_error(ModuleNotFoundError("No module named 'PySpin'",
                                                         name="PySpin"), "scan")
    assert "SCAN_PYTHON" in line


def test_read_codes_anti_glare_on_the_mock_rig(eng):
    """The opt-in path shoots the normal bracket + the anti-glare set (+ strips)
    through the rig and still reads a plain label with the normal decoder."""
    from engine import api
    from engine.capture.mock_bench import Label
    from conftest import PRESET
    p = api.load_preset(PRESET)
    p.camera.average = 1
    api.mock_bench().place(Label("SAS-0001 cell can lot 42", center_mm=(-60, 0), size_mm=45))
    r = api.read_codes(p, anti_glare=True, strip_cols=(400, 412))
    assert {c.data for c in r.codes} == {"SAS-0001 cell can lot 42"}
    names = set(r.illuminations)
    assert {f"ag_flood_{lv}" for lv in ag.LEVELS} <= names and "ag_strip_dark" in names
    assert sum(n.startswith("ag_strip_") for n in names) == 1 + 12
