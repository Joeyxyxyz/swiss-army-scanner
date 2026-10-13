"""Painted zones on the operator page (/quick, Lights panel) and the web server, in MOCK mode:
the Paint zones button starts the painter job on the rig, the page lists the zone sets of
the loaded preset's geometry and shows the saved preview, the picked set is the region of
MEASURE. The page logic runs in node on the real quick.js with a tiny fake DOM."""

import json
import re
import shutil
import subprocess
import threading
import types

import pytest

from engine import api
from conftest import ROOT
from test_ui_server import Client

from ui.web import server as ui_server

STATIC = ROOT / "ui" / "web" / "static"
PRESET = "bench_9in_metric"
GEO = "mock-bench_9in_v1"


@pytest.fixture(scope="module")
def web(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("zui")
    lib, stations = tmp / "library", tmp / "stations"
    stations.mkdir()
    httpd, app = ui_server.make_server(mock=True, library=lib, port=0, stations_dir=stations,
                                       preset=PRESET)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    try:
        yield Client(app.port), app, lib
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()
        api._session = None


def test_zone_verbs_are_wired():
    for v in ("zone_paint", "zone_show", "zone_delete"):
        assert v in ui_server.VERBS and callable(getattr(ui_server.App, f"_v_{v}", None)), v
    # the painter and the projection hold the rig: never rig-free, and an ABORT ends them
    assert {"zone_paint", "zone_show"} <= ui_server.CANCELLABLE_VERBS
    assert not {"zone_paint", "zone_show"} & ui_server.RIG_FREE_VERBS
    assert "zone_delete" in ui_server.RIG_FREE_VERBS
    assert {"name", "source", "replace", "events"} <= set(ui_server.VERBS["zone_paint"][1])
    assert "zones" in ui_server.VERBS["scan_fast"][1]
    assert "zoneset" in ui_server.VERBS["measure_run"][1]["zone"]
    assert any(r[1] == r"/api/zonesets" for r in ui_server.ROUTES)


def test_page_has_the_paint_button_the_picker_and_no_inline_handlers():
    html = (STATIC / "quick.html").read_text(encoding="utf-8")
    js = (STATIC / "quick.js").read_text(encoding="utf-8")
    lights = html[html.index('id="qLights"'):html.index('id="qCamera"')]
    for el in ('<button id="qPaint"', '<select id="qZoneSet"', '<input id="qZoneName"',
               '<button id="qZoneShow"', '<img id="qZoneImg"'):
        assert el in lights, el
    assert "Paint zones" in lights and "Space" in lights and "Esc" in lights
    assert not re.search(r"\son[a-z]+\s*=", html), "inline event handler in quick.html"
    for word in ("paintZones", "showZonesOnPart", "loadZoneSets", "'/api/zonesets'",
                 "'zone_paint'", "'zone_show'", "zoneset:${"):
        assert word in js, word
    node = shutil.which("node")
    if node is not None:
        r = subprocess.run([node, "--check", str(STATIC / "quick.js")], capture_output=True,
                           text=True, timeout=60)
        assert r.returncode == 0, r.stderr


def test_paint_job_saves_a_zone_set_and_the_page_gets_its_preview(web):
    c, app, lib = web
    status, d = c.get("/api/zonesets")
    assert status == 200 and d["sets"] == [] and d["geometry_id"] == GEO and d["preset"] == PRESET
    # the Paint zones button: one job on the rig (MOCK: the built-in demo session is replayed)
    j = c.done("zone_paint", {"name": "ui_set"})
    r = j["result"]
    assert r["saved"] is True and r["n_zones"] == 4 and "demo session" in r["note"]
    assert r["preview"] == f"zones/{GEO}/ui_set/preview.png"
    assert j["links"]["preview"] == f"zones/{GEO}/ui_set/camera.png"
    status, d = c.get("/api/zonesets")
    (z,) = d["sets"]
    assert z["name"] == "ui_set" and z["valid"] and z["n_zones"] == 4 and d["other_geometries"] == []
    assert [q["id"] for q in z["zones"]] == ["Z1", "Z2", "Z3", "Z4"] and z["zones"][0]["name"] == "part"
    assert z["preview"] == f"zones/{GEO}/ui_set/preview.png" and z["camera"].endswith("camera.png")
    for link in (z["preview"], z["camera"]):                # what the page's <img> loads
        status, raw, hdrs = c.request("GET", f"/files/{link}")
        assert status == 200 and hdrs.get("Content-Type") == "image/png" and raw[:4] == b"\x89PNG"
    # the same name again: refused (nothing is replaced unless asked); editing it replaces
    again = c.job("zone_paint", {"name": "ui_set"})
    assert again["status"] == "error" and "exists" in again["error"]
    script = [{"type": "click", "at": [300, 365]}, {"type": "key", "key": "delete"},
              {"type": "key", "key": "space"}]
    j = c.done("zone_paint", {"name": "ui_set", "source": "ui_set", "replace": True,
                              "events": script, "polarity": "normal"})
    assert j["result"]["saved"] and j["result"]["n_zones"] == 3 and "note" not in j["result"]
    assert (lib / "trash" / "zones" / GEO / "ui_set" / "zones.json").is_file()
    # a session that is not confirmed: the job is done, nothing is saved, the page says why
    j = c.done("zone_paint", {"name": "other", "polarity": "normal",
                              "events": [{"type": "key", "key": "escape"}]})
    assert j["result"]["saved"] is False and "nothing saved" in j["result"]["reason"]
    assert [s["name"] for s in c.get("/api/zonesets")[1]["sets"]] == ["ui_set"]
    # bad input is refused
    assert c.post("/api/jobs", {"verb": "zone_paint", "args": {"name": "x", "colour": 1}})[0] == 400
    bad = c.job("zone_paint", {"name": "bad name"})
    assert bad["status"] == "error" and "name must be" in bad["error"]
    bad = c.job("zone_paint", {"name": "x", "events": "space"})
    assert bad["status"] == "error" and "list of painter events" in bad["error"]


def test_zone_show_and_delete_jobs(web):
    c, app, lib = web
    j = c.done("zone_show", {"zones": "ui_set:Z2", "hold_s": 0})
    r = j["result"]
    assert r["all_seen"] and [z["id"] for z in r["zones"]] == ["Z2"] and r["zones"][0]["seen"]
    assert j["links"]["preview"].endswith("_overlay.png")
    status, raw, hdrs = c.request("GET", f"/files/{j['links']['preview']}")
    assert status == 200 and raw[:4] == b"\x89PNG"
    bad = c.job("zone_show", {"zones": "ui_set", "hold_s": 9999})
    assert bad["status"] == "error" and "hold_s" in bad["error"]
    bad = c.job("zone_show", {"zones": "nope"})
    assert bad["status"] == "error" and "no zone set 'nope'" in bad["error"]
    # delete: only with the operator's confirmation, and to the trash
    bad = c.job("zone_delete", {"name": "ui_set", "reason": "test"})
    assert bad["status"] == "error" and "confirm" in bad["error"]
    j = c.done("zone_delete", {"name": "ui_set", "reason": "test", "confirm": True})
    assert j["result"]["trashed"].startswith(f"trash/zones/{GEO}/ui_set")
    assert c.get("/api/zonesets")[1]["sets"] == []


def test_real_server_never_takes_a_scripted_session():
    app = types.SimpleNamespace(mock=False)
    with pytest.raises(ui_server.ArgError, match="MOCK ONLY"):
        ui_server.App._v_zone_paint(app, {"name": "x", "events": []})


# ------------------------------------------------ quick.js logic (node, fake DOM)

_HARNESS = r"""
const fs = require('fs'), vm = require('vm');
class El {
  constructor(tag, id) { this.tagName = String(tag).toUpperCase(); this.id = id; this.children = [];
    this.dataset = {}; this.className = ''; this._text = ''; this._value = ''; this.checked = false;
    this.disabled = false; this.hidden = false; this._sel = undefined; this.src = ''; this.open = false;
    this.classList = { toggle() {}, add() {}, remove() {} }; }
  set textContent(t) { this._text = String(t); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(''); }
  appendChild(c) { this.children.push(c); return c; }
  get options() { return this.children.filter((c) => c.tagName === 'OPTION'); }
  get value() {
    if (this.tagName !== 'SELECT') return this._value;
    const o = this.options;
    if (this._sel === null) return '';
    if (this._sel && o.includes(this._sel)) return this._sel.value;
    return o.length ? o[0].value : '';
  }
  set value(v) {
    if (this.tagName !== 'SELECT') { this._value = String(v); return; }
    this._sel = this.options.find((x) => x.value === String(v)) || null;
  }
  addEventListener() {}
}
const TAGS = { qZoneSet: 'select', qZoneName: 'input', qPaint: 'button', qZoneShow: 'button',
  qZoneImg: 'img', qMeasWhat: 'select', qMeasScan: 'select', qRigPreset: 'select' };
const byId = {};
const el = (id) => byId[id] || (byId[id] = new El(TAGS[id] || 'div', id));
const sent = [];
const srv = { sets: [], paint: { saved: true, name: 'lid', n_zones: 2 } };
const sandbox = {
  console, setTimeout, clearTimeout, setInterval, clearInterval, encodeURIComponent,
  document: { addEventListener() {}, querySelector: (s) => el(s.slice(1)), querySelectorAll: () => [],
              createElement: (t) => new El(t), body: new El('body') },
  fetch: async (path, opt) => {
    const body = opt && opt.body ? JSON.parse(opt.body) : undefined;
    sent.push({ path, body });
    let data = {};
    if (path === '/api/zonesets') data = { geometry_id: 'g', sets: srv.sets };
    else if (path === '/api/jobs') data = { id: 'j1', status: 'done',
      result: body.verb === 'zone_paint' ? srv.paint
        : { zones: [{ id: 'Z1', seen: true }, { id: 'Z2', seen: false }], warnings: ['zone Z2 does not land'],
            overlay_png: 'output/previews/x_overlay.png' } };
    return { ok: true, status: 200, json: async () => data };
  },
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), sandbox, { filename: 'quick.js' });
const run = (code) => vm.runInContext(code, sandbox);
const jobs = () => sent.filter((s) => s.path === '/api/jobs').map((s) => s.body);
const out = {};
(async () => {
  run("state.mode = 'real'");
  // nothing painted yet: the picker says so, Paint zones asks for a name
  await run('loadZoneSets()');
  out.empty = { options: el('qZoneSet').options.map((o) => o.textContent), img: el('qZoneImg').hidden };
  await run('paintZones()');
  out.noName = { msg: el('qSetupMsg').textContent, jobs: jobs().length };
  // a new name typed: one painter job; afterwards the new set is listed, picked and shown
  el('qZoneName').value = 'lid';
  srv.sets = [{ name: 'lid', n_zones: 2, valid: true, created: '2026-01-02T03:04:05',
                zones: [{ id: 'Z1', name: 'part' }, { id: 'Z2', name: '' }],
                preview: 'zones/g/lid/preview.png', camera: 'zones/g/lid/camera.png' },
              { name: 'old', n_zones: 1, valid: false, error: 'outside the DMD', zones: [],
                preview: null, camera: null }];
  await run('paintZones()');
  out.painted = { job: jobs().pop(), msg: el('qSetupMsg').textContent, picked: el('qZoneSet').value,
                  img: el('qZoneImg').src, hidden: el('qZoneImg').hidden, info: el('qZoneInfo').textContent,
                  typed: el('qZoneName').value, options: el('qZoneSet').options.map((o) => o.value) };
  // nothing typed, a set picked: Paint zones edits it (its zones loaded, the old one to trash)
  await run('paintZones()');
  out.edit = jobs().pop();
  // a session that was not confirmed: said, and the list is not re-read
  srv.paint = { saved: false, reason: 'cancelled: nothing saved' };
  const n = sent.filter((s) => s.path === '/api/zonesets').length;
  await run('paintZones()');
  out.cancelled = { msg: el('qSetupMsg').textContent,
                    reread: sent.filter((s) => s.path === '/api/zonesets').length - n };
  // Show zones: the picked set projected, the overlay shown
  await run('showZonesOnPart()');
  out.show = { job: jobs().pop(), msg: el('qSetupMsg').textContent, img: el('qZoneImg').src };
  // MEASURE: the picked set is the lit + measured region; none picked: the whole field
  const opt = (id, v) => { const o = new El('option'); o.value = v; el(id).appendChild(o); };
  opt('qMeasWhat', 'flatness'); opt('qMeasScan', 'metric');
  el('qMeasRepeats').value = '1';
  out.measure = run('measureArgs()');
  out.measureSays = el('qMeasZone').textContent;
  el('qZoneSet').value = '';
  run('showZoneSet()');
  out.measureNone = run('measureArgs()');
  out.measureSaysNone = el('qMeasZone').textContent;
  out.noneShown = { hidden: el('qZoneImg').hidden, info: el('qZoneInfo').textContent };
  el('qZoneSet').value = 'old';
  run('showZoneSet()');
  out.invalid = el('qZoneInfo').textContent;
  el('qZoneSet').value = '';
  await run('showZonesOnPart()');
  out.showNone = el('qSetupMsg').textContent;
  // the buttons follow the bus like the other setup buttons
  run("state.bus = { state: 'RUNNING' }; refreshButtons()");
  out.busy = [el('qPaint').disabled, el('qZoneShow').disabled];
  run("state.bus = { state: 'READY' }; refreshButtons()");
  out.idle = [el('qPaint').disabled, el('qZoneShow').disabled];
  // another preset is loaded: its geometry's sets are read, and none stays picked
  el('qZoneSet').value = 'lid';
  run("state.opMode = 'station'; state.shownPreset = 'p1'");
  const before = sent.filter((s) => s.path === '/api/zonesets').length;
  run("syncRigPreset({ active_preset: 'p1' })");
  out.samePreset = sent.filter((s) => s.path === '/api/zonesets').length - before;
  run("syncRigPreset({ active_preset: 'p2' })");
  await new Promise((r) => setTimeout(r, 20));
  out.otherPreset = { reread: sent.filter((s) => s.path === '/api/zonesets').length - before,
                      picked: el('qZoneSet').value };
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_quick_js_paints_picks_and_measures_with_zone_sets(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed; quick.js logic not checked")
    h = tmp_path / "zones_harness.js"
    h.write_text(_HARNESS, encoding="utf-8")
    r = subprocess.run([node, str(h), str(STATIC / "quick.js")], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["empty"] == {"options": ["(none painted yet)"], "img": True}
    assert "name the zone set" in out["noName"]["msg"] and out["noName"]["jobs"] == 0
    p = out["painted"]
    assert p["job"] == {"verb": "zone_paint", "args": {"name": "lid"}}
    assert "lid saved: 2 zone(s)" in p["msg"] and p["picked"] == "lid" and p["typed"] == ""
    assert p["options"] == ["", "lid", "old"] and p["hidden"] is False
    assert p["img"].startswith("/files/zones/g/lid/camera.png")
    assert "Z1 part, Z2" in p["info"] and "MEASURE lights and measures only these zones" in p["info"]
    assert out["edit"] == {"verb": "zone_paint",
                           "args": {"name": "lid", "source": "lid", "replace": True}}
    assert "not saved: cancelled" in out["cancelled"]["msg"] and out["cancelled"]["reread"] == 0
    assert out["show"]["job"] == {"verb": "zone_show", "args": {"zones": "lid", "hold_s": 5}}
    assert "1 of 2 zone(s) land" in out["show"]["msg"] and "Z2 does not land" in out["show"]["msg"]
    assert out["show"]["img"] == "/files/output/previews/x_overlay.png"
    assert out["measure"]["zone"] == "zoneset:lid" and out["measure"]["what"] == "flatness"
    assert "Zone set lid is picked" in out["measureSays"] and out["measureSaysNone"] == ""
    assert "zone" not in out["measureNone"]
    assert out["samePreset"] == 0 and out["otherPreset"] == {"reread": 1, "picked": ""}
    assert out["noneShown"] == {"hidden": True, "info": ""}
    assert "not usable: outside the DMD" in out["invalid"]
    assert "pick a zone set first" in out["showNone"]
    assert out["busy"] == [True, True] and out["idle"] == [False, False]
