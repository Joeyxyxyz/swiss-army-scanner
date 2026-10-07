"""api.project_capture / `project-capture`: the caller's own DMD images go through the mock
rig (HDMI window path, never USB) one frame per image, in order, stored under
library/readings/ (frames.npz + meta.json). Shapes, dtypes and the count are refused up front,
before the rig. No real hardware."""

import json

import cv2
import numpy as np
import pytest

from engine import api
from engine.capture import sync as sync_mod
from ui import cli

H, W = 1140, 912
LEVELS = [0, 255, 64, 192, 128]


def _pats(levels=LEVELS):
    return [np.full((H, W), v, np.uint8) for v in levels]


def test_frames_come_back_in_order(eng, preset, monkeypatch):
    seen = []
    shoot = sync_mod.FreeRunSync.shoot

    def spy(self, pattern, raw=False):
        seen.append((int(np.asarray(pattern).mean()), raw))
        return shoot(self, pattern, raw=raw)
    monkeypatch.setattr(sync_mod.FreeRunSync, "shoot", spy)
    r = api.project_capture(_pats(), preset, label="ladder", raw=True, average=2, hold_s=0.0)
    assert [lv for lv, _ in seen] == LEVELS and all(raw for _, raw in seen)
    assert r["n"] == len(LEVELS) and r["path"].parent.name == "readings"
    with np.load(r["frames_npz"]) as z:
        fr, pats = z["frames"], z["patterns"]
    assert fr.shape[0] == len(LEVELS) and fr.dtype == np.float32
    assert [int(p.mean()) for p in pats] == LEVELS
    means = fr.reshape(len(LEVELS), -1).mean(1)
    assert list(np.argsort(means)) == list(np.argsort(LEVELS))      # brighter pattern, brighter frame
    meta = json.loads(r["meta_json"].read_text())
    assert meta["kind"] == "custom_capture" and meta["average"] == 2 and meta["raw"] is True
    assert any(e.get("kind") == "custom_capture" for e in api.list_library("readings"))


def test_refused_before_the_rig(eng, preset, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the rig must not open")
    monkeypatch.setattr(type(api._engine()), "rig", boom)
    for bad, what in (([np.zeros((H, W), np.float32)], "uint8"),
                      ([np.zeros((W, H), np.uint8)], "shape"),
                      ([np.zeros((H, W, 4), np.uint8)], "shape"),
                      ([], "1..256"),
                      (np.zeros((257, H, W), np.uint8), "1..256")):
        with pytest.raises(ValueError, match=what):
            api.project_capture(bad, preset)
    with pytest.raises(ValueError, match="average"):
        api.project_capture(_pats([1]), preset, average=0)
    with pytest.raises(ValueError, match="hold_s"):
        api.project_capture(_pats([1]), preset, hold_s=-1)


def test_one_colour_image_and_a_stack(eng, preset):
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[..., 1] = 200
    assert api.project_capture(rgb, preset, raw=True)["n"] == 1
    assert api.project_capture(np.stack(_pats([10, 20, 30])), preset)["n"] == 3


def test_cli_npz_and_png_folder(tmp_path, capsys):
    lib = tmp_path / "library"
    npz = tmp_path / "p.npz"
    np.savez(npz, patterns=np.stack(_pats([0, 255])))
    folder = tmp_path / "pngs"
    folder.mkdir()
    for i, v in enumerate((255, 0, 128)):
        cv2.imwrite(str(folder / f"p{i:02d}.png"), np.full((H, W), v, np.uint8))
    base = ["--mock", "--library", str(lib), "--json"]
    for src, n in ((npz, 2), (folder, 3)):
        rc = cli.main([*base, "project-capture", str(src), "--label", "cli", "--average", "1",
                       "--hold", "0", "--raw"])
        out = json.loads(capsys.readouterr().out)
        assert rc == 0 and out["project_capture"]["n"] == n
    rc = cli.main([*base, "project-capture", str(tmp_path / "nope.txt")])
    assert rc == 2
    api._session = None
