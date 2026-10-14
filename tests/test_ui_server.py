"""Web UI server, driven over HTTP exactly like the browser does, in MOCK mode.

One server per module (the worker thread owns the engine), on a free port,
with a throwaway library and a copy of the presets folder so saving presets
never touches the repo. No hardware: mock=True throughout.
"""

import ast
import http.client
import os
import json
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from engine import api, config, rules

from conftest import PRESET, ROOT

from ui.web import server as ui_server

WEB = ROOT / "ui" / "web"
STATIC = WEB / "static"


class Client:
    """http.client, not urllib: paths go out byte-for-byte (traversal tests)
    and headers are ours to set (origin / content-type guards)."""

    def __init__(self, port: int):
        self.port = port

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=180)
        hdrs = dict(headers or {})
        data = None
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            hdrs.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=data, headers=hdrs)
        r = conn.getresponse()
        raw = r.read()
        out = (r.status, raw, dict(r.getheaders()))
        conn.close()
        return out

    def get(self, path):
        status, raw, _ = self.request("GET", path)
        return status, json.loads(raw) if raw else None

    def post(self, path, body=None, headers=None):
        status, raw, _ = self.request("POST", path, body if body is not None else {}, headers)
        return status, json.loads(raw) if raw else None

    def job(self, verb, args=None, timeout=180.0) -> dict:
        status, j = self.post("/api/jobs", {"verb": verb, "args": args or {}})
        assert status == 200, j
        t0 = time.monotonic()
        while j["status"] in ("queued", "running"):
            assert time.monotonic() - t0 < timeout, f"{verb} did not finish: {j}"
            time.sleep(0.1)
            _, j = self.get(f"/api/jobs/{j['id']}")
        return j

    def done(self, verb, args=None) -> dict:
        j = self.job(verb, args)
        assert j["status"] == "done", f"{verb}: {j['error']}\n{j.get('trace')}"
        return j


@pytest.fixture(scope="module")
def ui(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ui")
    presets = tmp / "presets"
    presets.mkdir()
    for p in (ROOT / "presets").glob("*.json"):
        shutil.copy(p, presets / p.name)
    httpd, app = ui_server.make_server(mock=True, library=tmp / "library", presets_dir=presets,
                                       port=0)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    try:
        yield Client(app.port), app, presets
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()
        api._session = None


@pytest.fixture(scope="module")
def fast(ui):
    """Save + load a copy of the correct-orientation preset with camera.average=1,
    through the same endpoints the Setup tab uses (a USER preset in the library)."""
    c, app, presets = ui
    _, d = c.get(f"/api/presets/{PRESET}")
    p = d["preset"]
    p["name"], p["camera"]["average"] = "ui_fast", 1
    status, r = c.post("/api/presets", {"preset": p, "base": PRESET})
    assert status == 200, r
    assert r["tier"] == "user" and not (presets / "ui_fast.json").exists()
    assert (config.user_presets_dir(app.root) / "ui_fast" / config.USER_PRESET_FILE).is_file()
    status, j = c.post("/api/presets/ui_fast/load", {})
    assert status == 200 and j["status"] == "done", j
    assert j["result"]["camera"]["average"] == 1
    return c


@pytest.fixture(scope="module")
def scene(fast):
    """Empty-scene base, then the §11 validation object with known dims."""
    c = fast
    base = c.done("capture_base", {"label": "empty"})["result"]
    c.done("mock_place", {"object": "validation"})
    obj = c.done("capture_object", {"label": "validation", "base_id": base["id"],
                                    "known_dims": {"H": 106.95, "W": 56.5, "L": 21.9}})["result"]
    return c, base, obj


# ------------------------------------------------------------------ page

def test_index_and_static_are_served(ui):
    c, _, _ = ui
    status, raw, hdrs = c.request("GET", "/")
    assert status == 200 and hdrs["Content-Type"].startswith("text/html")
    html = raw.decode()
    assert '<script src="/static/app.js"' in html and "/static/style.css" in html
    assert "cdn" not in html.lower() and "https://" not in html      # the bench may be offline
    for name, ctype in (("app.js", "text/javascript"), ("style.css", "text/css")):
        status, raw, hdrs = c.request("GET", f"/static/{name}")
        assert status == 200 and hdrs["Content-Type"].startswith(ctype) and raw
    assert c.request("GET", "/static/nope.js")[0] == 404
    assert c.request("GET", "/static/..%2Fserver.py")[0] in (403, 404)


def test_page_ids_used_by_the_script_exist():
    """Every static element app.js looks up by id is in index.html (typos
    would only show up as a dead button in the browser)."""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    used = set(re.findall(r"\$\('#([\w-]+)'\)", js)) | set(re.findall(r"on\('#([\w-]+)'", js))
    made_in_js = {"ledOn", "ledMode", "ledr", "ledg", "ledb", "ledLimr", "ledLimg", "ledLimb",
                  "ledFs", "derivedNote"}
    defined = set(re.findall(r'\bid="([\w-]+)"', html))
    assert not (used - defined - made_in_js), sorted(used - defined - made_in_js)


def test_app_js_has_no_syntax_errors():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed; app.js syntax not checked")
    r = subprocess.run([node, "--check", str(STATIC / "app.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_every_response_forbids_framing(ui):
    """Clickjacking: the Origin/Host guards pass for a framed page (its posts
    are genuinely same-origin), so the page must refuse to be framed at all."""
    c, _, _ = ui
    for path in ("/", "/static/app.js", "/api/status", "/api/nope"):
        status, _, hdrs = c.request("GET", path)
        assert hdrs.get("X-Frame-Options") == "DENY", (path, status)
        csp = hdrs.get("Content-Security-Policy", "")
        assert "frame-ancestors 'none'" in csp and "script-src 'self'" in csp, path
        assert hdrs.get("X-Content-Type-Options") == "nosniff", path
    # the CSP forbids inline handlers: the page must not rely on any
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"\son[a-z]+\s*=", html), "inline event handler in index.html"


def test_every_engine_verb_is_exposed():
    """The VERBS whitelist is 'every engine.api verb except configure /
    mock_bench and the CLI-only hardware-trigger / raw-pattern verbs'
    (bank_remove was once missing, so a wrong bank member could only be
    fixed by trashing the whole bank). trigger_test / scan_trigger need the
    projector put in pattern mode by a human first, and project_capture
    projects arbitrary images: they stay off the web UI / agent surface."""
    verbs = {n for n, f in vars(api).items()
             if not n.startswith("_") and callable(f) and getattr(f, "__wrapped__", None)
             and getattr(f, "__module__", "") == api.__name__}
    assert "bank_remove" in verbs and "list_library" in verbs
    cli_only = {"trigger_test", "scan_trigger", "project_capture"}
    missing = verbs - set(ui_server.VERBS) - {"configure", "mock_bench"} - cli_only
    assert not cli_only & set(ui_server.VERBS)
    assert not missing, f"engine verbs not exposed to the web UI / agents: {sorted(missing)}"
    assert set(ui_server.RIG_FREE_VERBS) <= set(ui_server.VERBS)
    for v in ("capture_base", "capture_object", "check_signal", "project_grid", "load_preset",
              "delete", "live_view"):
        assert v not in ui_server.RIG_FREE_VERBS, v


# --------------------------------------------------------- app.js logic (node)

_JS_HARNESS = r"""
const fs = require('fs'), vm = require('vm');
class Node {}
class El extends Node {
  constructor(tag) { super(); this.tagName = String(tag).toUpperCase(); this.children = []; this.attrs = {};
    this.dataset = {}; this.style = {}; this.className = ''; this._text = ''; this._value = undefined; this._sel = undefined;
    this.classList = { add() {}, remove() {}, toggle() {}, contains() { return false; } }; }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === 'value') this._value = String(v); }
  getAttribute(k) { return this.attrs[k]; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener() {}
  append(...k) { this.children.push(...k); }
  replaceChildren(...k) { this.children = k; this._sel = undefined; }
  set textContent(t) { this._text = String(t); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent || '').join(''); }
  get options() { return this.children.filter((c) => c.tagName === 'OPTION'); }
  get value() {
    if (this.tagName !== 'SELECT') return this._value == null ? '' : this._value;
    const o = this.options;
    if (this._sel === null) return '';
    if (this._sel && o.includes(this._sel)) return this._sel.value;
    return o.length ? o[0].value : '';
  }
  set value(v) {
    if (this.tagName !== 'SELECT') { this._value = String(v); return; }
    this._sel = this.options.find((x) => x.value === String(v)) || null;
  }
}
const byId = {};
const objectSelects = [];
const el = (id, tag) => byId[id] || (byId[id] = Object.assign(new El(tag || 'div'), { id }));
for (const id of ['measScan', 'inspScan', 'cmpScan', 'cmpGolden', 'bankScan', 'profScan']) objectSelects.push(el(id, 'select'));
const store = {};
const sandbox = {
  console, Node,
  document: {
    addEventListener() {}, documentElement: { dataset: {}, style: { setProperty() {} } },
    createElement: (t) => new El(t), createTextNode: (t) => Object.assign(new Node(), { textContent: String(t), nodeType: 3 }),
    querySelector: (s) => (s.startsWith('#') ? el(s.slice(1)) : null),
    querySelectorAll: (s) => (s === 'select[data-objects]' ? objectSelects : []),
  },
  localStorage: { getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); } },
  location: { port: '8765', hash: '' }, window: { addEventListener() {} }, navigator: {},
  getComputedStyle: () => ({ getPropertyValue: () => '' }), setTimeout, clearTimeout,
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), sandbox, { filename: 'app.js' });
const run = (code) => vm.runInContext(code, sandbox);
run("Prof.picker = { shapes: [], clear() {}, set() {}, draw() {} };");
const sel = () => Object.fromEntries(objectSelects.map((s) => [s.id, s.value]));
const scan = (id, t) => `{ id: '${id}', label: 'L${id}', created: '2026-09-22T10:0${t}:00' }`;
const out = {};

// pure picker rule
out.pure = run(`[
  objectSelectValue({ selId: 'measScan', current: 'B', touched: false, ids: ['C', 'B'], fresh: 'C', forced: null }),
  objectSelectValue({ selId: 'measScan', current: 'B', touched: true, ids: ['C', 'B'], fresh: 'C', forced: null }),
  objectSelectValue({ selId: 'measScan', current: 'B', touched: true, ids: ['D', 'C', 'B'], fresh: 'D', forced: 'D' }),
  objectSelectValue({ selId: 'cmpGolden', current: 'A', touched: false, ids: ['D', 'A'], fresh: 'D', forced: 'D' }),
  objectSelectValue({ selId: 'cmpGolden', current: '', touched: false, ids: ['B', 'A'], fresh: null, forced: null }),
  objectSelectValue({ selId: 'measScan', current: '', touched: false, ids: [], fresh: null, forced: null }),
]`);

// the real fillObjectSelects over a scan list that grows
run(`S.objects = [${scan('B', 2)}, ${scan('A', 1)}]; fillObjectSelects();`);
out.first = sel();
byId.inspScan.value = 'A'; byId.inspScan.dataset.touched = '1';          // operator picks by hand
run(`S.objects = [${scan('C', 3)}, ${scan('B', 2)}, ${scan('A', 1)}]; fillObjectSelects();`);
out.agentScan = sel();                                                      // another client captured C
run(`S.objects = [${scan('C', 3)}, ${scan('B', 2)}, ${scan('A', 1)}]; fillObjectSelects();`);
out.noChange = sel();
run(`S.selectScan = 'D'; S.objects = [${scan('D', 4)}, ${scan('C', 3)}, ${scan('B', 2)}, ${scan('A', 1)}]; fillObjectSelects();`);
out.ownCapture = sel();
out.touchedAfterOwn = byId.inspScan.dataset.touched || null;
out.pending = run('S.selectScan');

// zones are pinned to the grid this page shows
out.zones = run(`[zoneForEngine('e6', { grid: { spacing_px: 96 } }), zoneForEngine('C4@64', { grid: { spacing_px: 96 } }),
  zoneForEngine('c4:d6', null), zoneForEngine('B2,E5', { grid: { spacing_px: 64 } }), zoneForEngine('', { grid: { spacing_px: 96 } })]`);
out.atlasZone = run(`[zoneForEngine(' atlas: cell/cell ', { grid: { spacing_px: 96 } }), ATLAS_RE.test('atlas:cell/cell'), regionArg('720, 768, 768, 1056')]`);
out.paintedZone = run(`[zoneForEngine(' zoneset: lid_Zones ', { grid: { spacing_px: 96 } }), zoneForEngine('ZoneSet:lid.a-1: Z1, Z3', { grid: { spacing_px: 64 } }),
  regionArg('zoneset:lid_Zones/Z2'), ZONESET_RE.test('zoneset:lid_Zones'), ZONESET_RE.test('zoneset:lid_Zones:Z1,Z3'), ZONESET_RE.test('zoneset:lid_Zones/Z2'),
  ZONESET_RE.test('zoneset:two words'), ZONESET_RE.test('zoneset:lid:C4'), ZONE_RE.test('zoneset:lid_Zones')]`);

// virtual sensors: where a spot / sensor goes, as the operator types it
out.at = run(`[parseAt('h8', { grid: { spacing_px: 96 } }), parseAt('dmd 456,570', null), parseAt('cam 700 500', null),
  parseAt('Camera: (12.5, 3)', null), parseAt('g7:h9', { grid: { spacing_px: 64 } }), parseAt('  ', null)]`);
out.atText = run(`[atText({ cell: 'H8', spacing_px: 96 }), atText({ dmd: [456, 570.5] }),
  atText({ camera: [700, 500], dmd: [455.2, 560], map_id: 'm' }), atText({ zone: 'G7:H9', spacing_px: 64 }), atText('H8@96')]`);
out.why = run(`[verdictWhy('INDETERMINATE', ['other', 'cannot PASS: the detection threshold is above the trigger']).textContent,
  verdictWhy('INDETERMINATE', ['INDETERMINATE: no spot returned']).textContent,
  verdictWhy('PASS', ['cannot PASS: x']), verdictWhy('INDETERMINATE', ['other'])]`);
out.badges = run(`['PRESENT', 'ABSENT', 'INDETERMINATE', 'NOT_ALIGNED'].map((v) => { const b = sensorBadge(v); return [b.className, b.textContent]; })`);
out.cells = run(`(() => {
  const g = { grid: { spacing_px: 96, n_cols: 9, n_rows: 11, col_labels: 'ABCDEFGHI'.split(''),
    row_labels: Array.from({ length: 11 }, (_, i) => String(i + 1)) }, cell_centers: { H8: [1050, 650], A1: [200, 150] } };
  const r = (x) => x.cell || 'error';
  return [r(cellAt([700, 700], g, 'dmd')), r(cellAt([1040, 640], g, 'camera')), r(cellAt([5000, 10], g, 'dmd')),
    r(cellAt([1, 1], { grid: g.grid }, 'camera')), r(cellAt([1, 1], null, 'dmd'))];
})()`);

(async () => {
  run(`runJob = async (verb, args) => { globalThis.__sent = { verb, args }; return { status: 'error' }; };`);
  await run(`S.grid = { grid: { spacing_px: 96 } }; captureObject({ label: 'p', zone: 'e6', out: document.querySelector('#zoneOut') })`);
  out.sent = run('globalThis.__sent');
  // the agent brief: zone pinned to the spacing, and the base the operator chose
  run(`S.status = { server: { port: 8765 }, mock: true, preset: { name: 'x' }, geometry_id: 'g', paths: {} };
       S.geo = 'g'; S.bases = [{ id: 'base_1', label: 'empty' }]; LS.set('base.g', 'base_1');
       S.grid = { grid: { spacing_px: 96, n_cols: 9, n_rows: 11, col_labels: ['A'], row_labels: ['1'] } };`);
  el('zoneText', 'input')._value = 'e6';
  run('updateAgentBrief()');
  out.brief = el('agentBrief').value;
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def _run_harness(tmp_path) -> dict:
    """The real app.js in node with a tiny fake DOM -> the harness's `out`."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed; app.js logic not checked")
    h = tmp_path / "harness.js"
    h.write_text(_JS_HARNESS, encoding="utf-8")
    r = subprocess.run([node, str(h), str(STATIC / "app.js")], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_app_js_scan_pickers_follow_new_scans_and_zones_pin_the_grid(tmp_path):
    """Regressions, run on the real app.js in node with a tiny fake DOM:
    - after a capture, every scan picker (Measure, Inspect, Compare, Bank,
      Profile) moved to the new scan, not the one that was newest before
      (Measure used to measure the previous scan); the golden stays put, and
      a picker set by hand keeps its choice when another client adds a scan;
    - zone names go out pinned to the grid this page shows ('E6@96'), in
      captures and in the agent brief, which also names the chosen base."""
    out = _run_harness(tmp_path)
    assert out["pure"] == ["C", "B", "D", "A", "A", ""]
    assert out["first"] == {"measScan": "B", "inspScan": "B", "cmpScan": "B", "cmpGolden": "A",
                            "bankScan": "B", "profScan": "B"}
    assert out["agentScan"] == {"measScan": "C", "inspScan": "A", "cmpScan": "C",
                                "cmpGolden": "A", "bankScan": "C", "profScan": "C"}
    assert out["noChange"] == out["agentScan"]
    assert out["ownCapture"] == {"measScan": "D", "inspScan": "D", "cmpScan": "D",
                                 "cmpGolden": "A", "bankScan": "D", "profScan": "D"}
    assert out["touchedAfterOwn"] is None and out["pending"] is None
    assert out["zones"] == ["E6@96", "C4@64", "C4:D6", "B2,E5@64", ""]
    # Surfer atlas regions go out as typed (never upper-cased or pinned to a grid)
    assert out["atlasZone"] == ["atlas:cell/cell", True, [720, 768, 768, 1056]]
    # ... and so does a painted zone set (its name keeps its case; no '@96' is added)
    assert out["paintedZone"] == ["zoneset:lid_Zones", "zoneset:lid.a-1:Z1,Z3", "zoneset:lid_Zones/Z2",
                                  True, True, True, False, False, False]
    assert out["sent"]["verb"] == "capture_object" and out["sent"]["args"]["zone"] == "E6@96"
    brief = out["brief"]
    assert '"zone":"E6@96"' in brief and '"base_id":"base_1"' in brief, brief
    assert 'zone="E6@96"' in brief


def test_app_js_sensor_helpers(tmp_path):
    """The Sensors tab's pure helpers, on the real app.js: a typed / clicked
    target becomes the engine's `at` (cells and zones pinned to the grid shown,
    DMD and camera points as objects); the reason for an INDETERMINATE (inspect's
    'cannot PASS', a sensor's 'INDETERMINATE: ...') sits next to its badge."""
    out = _run_harness(tmp_path)
    assert out["at"] == ["H8@96", {"dmd": [456, 570]}, {"camera": [700, 500]},
                         {"camera": [12.5, 3]}, "G7:H9@64", None]
    assert out["atText"] == ["H8@96", "dmd 456, 570.5", "cam 700, 500 (dmd 455.2, 560)",
                             "zone G7:H9@64", "H8@96"]
    assert out["why"][0].startswith("cannot PASS") and out["why"][1] == "no spot returned"
    assert out["why"][2:] == [None, None]
    assert out["badges"] == [["verdict big ok", "PRESENT"], ["verdict big info", "ABSENT"],
                             ["verdict big warn", "INDETERMINATE"],
                             ["verdict big bad", "NOT ALIGNED"]]
    assert out["cells"] == ["H8", "H8", "error", "error", "error"]


def test_sensor_verbs_are_documented_for_agents():
    """GET /api/verbs is the agents' manual: every virtual-sensor verb says what
    each argument is (and the operator's EMPTY confirmation is an explicit arg)."""
    for v in ("spot_size", "project_spot", "teach_presence", "check_presence",
              "teach_alignment", "check_alignment"):
        doc, args = ui_server.VERBS[v]
        assert doc and args and all(isinstance(d, str) and d.strip() for d in args.values()), v
    assert "empty_confirmed" in ui_server.VERBS["teach_presence"][1]
    assert "object_id" in ui_server.VERBS["check_alignment"][1]
    assert "sensors" in ui_server.LIST_KINDS and "sensors" in ui_server.DELETABLE
    assert {"spot_size", "teach_alignment", "check_alignment"} <= ui_server.RIG_FREE_VERBS
    for v in ("project_spot", "teach_presence", "check_presence"):      # these light the rig
        assert v not in ui_server.RIG_FREE_VERBS, v


def test_ui_never_imports_usb_code():
    """Rule 1 applies to the UI too: no swiss_army_sensor / hid / dmd_control."""
    for py in WEB.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            for n in names:
                assert n != "hid" and not n.startswith("swiss_army_sensor") \
                    and n.split(".")[-1] != "dmd_control", f"{py}: imports {n}"


# ---------------------------------------------------------------- status

def test_status_reports_mock_mode_and_paths(ui):
    c, app, _ = ui
    status, st = c.get("/api/status")
    assert status == 200
    assert st["mode"] == "mock" and st["mock"] is True
    assert st["server"]["bind"] == "127.0.0.1"
    assert st["preset"]["name"] in (PRESET, "ui_fast")
    assert st["geometry_id"] == "mock-wall_2p7ft_vstripe_v1"
    rig = st["preset"]["rig"]
    assert rig["coded_axis"] == "x" and rig["nominal_mm_per_relief_px"] > 0
    assert 0 < rig["triangulation_angle_deg"] < 45
    assert Path(st["paths"]["library"]) == app.root
    assert "previews" in st["output_kinds"]
    _, verbs = c.get("/api/verbs")
    names = {v["verb"] for v in verbs["verbs"]}
    assert "configure" not in names and "mock_bench" not in names
    assert {"capture_object", "project_grid", "profile", "mock_place"} <= names
    zone = next(v for v in verbs["verbs"] if v["verb"] == "capture_object")
    assert "zone" in zone["args"]


# --------------------------------------------------------------- presets

def test_presets_list_and_get(ui):
    c, _, _ = ui
    status, r = c.get("/api/presets")
    assert status == 200 and {PRESET, "cell_wall_2p7ft"} <= set(r["presets"])
    status, d = c.get(f"/api/presets/{PRESET}")
    assert status == 200
    assert d["preset"]["name"] == PRESET and d["preset"]["camera"]["exposure_us"] > 0
    assert d["preset"]["projection"]["monitor_offset"] == [1920, 0]
    assert d["derived"]["coded_axis"] == "x"
    assert d["geometry_id"] == "mock-wall_2p7ft_vstripe_v1"
    assert c.get("/api/presets/no_such_preset")[0] == 404
    # names only, never paths: resolve_preset_path would otherwise open any file
    assert c.get("/api/presets/..%2F..%2Fpyproject")[0] == 404


def test_preset_save_returns_validation_errors_verbatim(ui):
    c, app, presets = ui
    _, d = c.get(f"/api/presets/{PRESET}")
    p = d["preset"]

    hot = json.loads(json.dumps(p))
    hot["name"] = "ui_hot"
    hot["projection"]["led"] = {"mode": "continuous", "r": 200}
    status, r = c.post("/api/presets", {"preset": hot, "base": PRESET})
    assert status == 400 and "SafetyViolation" in r["error"] and "thermal limit" in r["error"]

    typo = json.loads(json.dumps(p))
    typo["name"] = "ui_typo"
    typo["camera"]["exposure"] = 5000
    status, r = c.post("/api/presets", {"preset": typo, "base": PRESET})
    assert status == 400 and "unknown keys" in r["error"] and "exposure" in r["error"]

    status, r = c.post("/api/presets", {"preset": p})          # a built-in's own name
    assert status == 400 and "built-in" in r["error"]
    users = config.user_presets_dir(app.root)
    assert not (presets / "ui_hot.json").exists() and not (presets / "ui_typo.json").exists()
    assert not any((users / n).exists() for n in ("ui_hot", "ui_typo", PRESET))
    assert c.post("/api/presets", {"nope": 1})[0] == 400
    new = json.loads(json.dumps(p))
    new["name"] = "ui_nobase"
    status, r = c.post("/api/presets", {"preset": new})        # a new name needs its base
    assert status == 400 and "base" in r["error"]
    status, r = c.post("/api/presets", {"preset": new, "base": PRESET, "overwrite": "yes"})
    assert status == 400 and "overwrite" in r["error"]


def test_setup_save_writes_a_user_preset_never_the_builtin_folder(ui):
    """The Setup editor's Save used to write <presets>/<name>.json - the tracked built-in
    folder (a part / line name ending up in the repo) - and 'overwrite' rewrote a built-in.
    It now saves a USER preset in the library, like preset_add: built-ins are read-only."""
    c, app, presets = ui
    before = {f.name: f.read_bytes() for f in presets.glob("*.json")}
    _, d = c.get(f"/api/presets/{PRESET}")
    p = d["preset"]
    p["name"], p["camera"]["exposure_us"] = "ui_line_part", p["camera"]["exposure_us"] + 1000
    status, r = c.post("/api/presets", {"preset": p, "base": PRESET})
    assert status == 200 and r["saved"] == "ui_line_part" and r["tier"] == "user", r
    users = config.user_presets_dir(app.root)
    saved = users / "ui_line_part" / config.USER_PRESET_FILE
    assert Path(r["path"]) == saved and saved.is_file() and not r["overwrote"]
    assert json.loads(saved.read_text())["camera"]["exposure_us"] == p["camera"]["exposure_us"]
    _, lst = c.get("/api/presets")
    assert {t["name"]: t["tier"] for t in lst["tiers"]}["ui_line_part"] == "user"
    status, r = c.post("/api/presets", {"preset": p, "base": PRESET})     # exists, no replace
    assert status == 400 and "exists" in r["error"]
    p["camera"]["exposure_us"] += 1000
    status, r = c.post("/api/presets", {"preset": p, "base": "ui_line_part", "overwrite": True})
    assert status == 200 and r["overwrote"] and r["replaced"], r       # old one -> trash
    assert json.loads(saved.read_text())["camera"]["exposure_us"] == p["camera"]["exposure_us"]
    assert Path(r["replaced"]).is_dir() and "trash" in Path(r["replaced"]).parts
    # 'overwrite' never reaches a built-in
    _, d = c.get(f"/api/presets/{PRESET}")
    b = d["preset"]
    b["camera"]["exposure_us"] += 1000
    status, r = c.post("/api/presets", {"preset": b, "base": PRESET, "overwrite": True})
    assert status == 400 and "built-in" in r["error"]
    assert {f.name: f.read_bytes() for f in presets.glob("*.json")} == before


def test_load_preset_makes_it_active(fast):
    _, st = fast.get("/api/status")
    assert st["preset"]["name"] == "ui_fast" and st["preset"]["camera"]["average"] == 1
    _, r = fast.get("/api/presets")
    assert r["active"] == "ui_fast"


# ------------------------------------------------------------------ jobs

def test_check_signal_job(fast):
    j = fast.done("check_signal")
    r = j["result"]
    assert r["__type__"] == "SignalReport" and r["mock"] is True
    assert r["gc_conf"] > 30 and r["axis_ok"] is True and r["coded_axis"] == "x"
    assert j["started"] and j["finished"] and j["duration_s"] >= 0


def test_base_capture_listing_and_preview(scene):
    c, base, _ = scene
    assert base["geometry_id"] == "mock-wall_2p7ft_vstripe_v1"
    _, lst = c.get("/api/library/bases?geometry_id=mock-wall_2p7ft_vstripe_v1")
    assert [b["id"] for b in lst["items"]] == [base["id"]] and not lst["stale"]
    rel = lst["items"][0]["preview"]
    assert rel.startswith("bases/") and not Path(rel).is_absolute()
    status, raw, hdrs = c.request("GET", "/files/" + rel)
    assert status == 200 and hdrs["Content-Type"] == "image/png" and raw[:4] == b"\x89PNG"
    _, st = c.get("/api/status")
    assert st["latest_base"]["id"] == base["id"]


def test_capture_object_result_is_json_safe(scene):
    c, base, obj = scene
    assert obj["__type__"] == "ObjectScan" and obj["base_id"] == base["id"]
    assert obj["known_dims"] == {"H": 106.95, "W": 56.5, "L": 21.9}
    assert obj["path"].startswith("objects/")                  # library-relative
    assert "_relief" not in obj and "_field" not in obj         # arrays never shipped
    assert obj["meta"]["overlap_px"] > 1000
    _, lst = c.get("/api/library/objects")
    assert obj["id"] in [o["id"] for o in lst["items"]]


def test_reconstruct_and_pointcloud(scene):
    c, _, obj = scene
    j = c.done("reconstruct", {"object_id": obj["id"], "fmt": "ply"})
    rel = j["result"]
    assert rel.startswith("output/pointclouds/") and rel.endswith(".ply")
    assert j["links"]["file"] == rel
    status, raw, hdrs = c.request("GET", "/files/" + rel + "?download=1")
    assert status == 200 and raw.startswith(b"ply") and "attachment" in hdrs["Content-Disposition"]

    status, pc = c.get(f"/api/pointcloud/{obj['id']}?max=5000")
    assert status == 200
    assert pc["n"] == len(pc["points"]) == 5000 < pc["n_total"]
    assert pc["units"] == {"xy": "px", "z": "px"} and pc["tier"] == "relief"
    zs = [p[2] for p in pc["points"]]
    assert 40 < max(zs) < 55                     # §11 object: ~48 px of relief
    assert c.get("/api/pointcloud/nope")[0] == 404


def test_files_traversal_is_blocked(scene):
    c, _, obj = scene
    for bad in ("/files/../README.md", "/files/%2e%2e/%2e%2e/pyproject.toml",
                "/files/..%5c..%5cpyproject.toml", "/files/C:/Windows/win.ini",
                "/files/objects/../../../pyproject.toml"):
        assert c.request("GET", bad)[0] == 403, bad
    # inside the library but not a served type (raw fields stay private)
    assert c.request("GET", f"/files/{obj['path']}/field.npz")[0] == 403
    assert c.request("GET", "/files/output/nope.png")[0] == 404


def test_outputs_index(scene):
    c, _, _ = scene
    status, r = c.get("/api/outputs")
    assert status == 200
    kinds = {o["kind"] for o in r["items"]}
    assert {"previews", "pointclouds"} <= kinds
    assert r["paths"]["output"].endswith("output")
    _, only = c.get("/api/outputs?kind=pointclouds")
    assert only["items"] and {o["kind"] for o in only["items"]} == {"pointclouds"}
    assert c.get("/api/outputs?kind=nope")[0] == 400


def test_measure_and_calibrate_jobs(scene):
    c, _, obj = scene
    d = c.done("measure", {"object_id": obj["id"]})["result"]
    assert d["units"] == "px" and any("rule 6" in w for w in d["warnings"])
    cal = c.done("calibrate", {"object_ids": [obj["id"]]})["result"]
    assert cal["geometry_id"] == "mock-wall_2p7ft_vstripe_v1" and cal["mm_per_px"] > 0
    _, st = c.get("/api/status")
    assert st["calibration"]["id"] == cal["id"]
    d = c.done("measure", {"object_id": obj["id"]})["result"]
    assert d["units"] == "mm" and abs(d["H"] - 106.95) < 0.7


def test_delete_moves_entry_to_trash(scene):
    c, base, _ = scene
    tmp = c.done("capture_object", {"label": "to-delete", "base_id": base["id"]})["result"]
    j = c.done("delete", {"kind": "objects", "id": tmp["id"], "reason": "test"})
    dest = j["result"]["trashed_to"]
    assert dest.startswith("trash/objects/")
    _, st = c.get("/api/status")
    lib = Path(st["paths"]["library"])
    assert (lib / dest / "TRASHED.json").exists() and not (lib / tmp["path"]).exists()
    _, lst = c.get("/api/library/objects")
    assert tmp["id"] not in [o["id"] for o in lst["items"]]
    again = c.job("delete", {"kind": "objects", "id": tmp["id"]})
    assert again["status"] == "error" and again["error_type"] == "EntryNotFound"


def test_compare_px_band_alone_keeps_the_preset_mm_band(scene):
    """A px band alone used to drop the mm band, so a calibrated geometry (it
    judges in mm) always answered INDETERMINATE. Like the CLI, the preset's
    CH band now applies; the deviation PNG comes back as a link."""
    c, _, obj = scene
    _, st = c.get("/api/status")
    assert st["calibration"] and st["calibration"]["mm_per_px"], "needs the calibration above"
    trig = st["preset"]["tolerance"]["ch_fail_mm"]
    j = c.done("compare", {"object_id": obj["id"], "golden_id": obj["id"], "fail_px": 2})
    v = j["result"]
    assert v["units"] == "mm" and v["result"] in ("PASS", "FAIL"), v
    assert v["tolerance"] == pytest.approx(trig)
    png = j["links"]["png"]
    assert png.startswith("output/comparisons/") and png.endswith(".png")
    assert j["links"]["json"].endswith(".json")
    status, raw, _ = c.request("GET", "/files/" + png)
    assert status == 200 and raw[:4] == b"\x89PNG"
    j = c.done("compare", {"object_id": obj["id"], "golden_id": obj["id"], "fail_mm": 0.25})
    assert j["result"]["tolerance"] == pytest.approx(0.25)


def test_bank_member_can_be_removed(scene):
    """A wrong scan added to a bank can be dropped on its own (append-only
    remove record); before, only the whole bank could be trashed."""
    c, _, obj = scene
    info = c.done("bank_add", {"object_id": obj["id"], "bank": "ui_rm"})["result"]
    assert obj["id"] in info["scan_ids"]
    info = c.done("bank_remove", {"object_id": obj["id"], "bank": "ui_rm",
                                  "reason": "wrong part"})["result"]
    assert info["n_scans"] == 0 and obj["id"] not in info["scan_ids"]
    _, banks = c.get("/api/library/banks?geometry_id=mock-wall_2p7ft_vstripe_v1")
    b = next(b for b in banks["items"] if b["name"] == "ui_rm")
    assert b["scan_ids"] == []
    _, st = c.get("/api/status")
    members = Path(st["paths"]["library"]) / b["path"] / "members.jsonl"
    assert [json.loads(x)["op"] for x in members.read_text().splitlines()] == ["add", "remove"]
    again = c.job("bank_remove", {"object_id": obj["id"], "bank": "ui_rm"})
    assert again["status"] == "error" and "not a member" in again["error"]


def test_material_can_be_trashed(scene):
    """A material learned from a bad ROI is used by every default classify:
    it can now be moved to trash (all its versions) from the UI."""
    c, _, _ = scene
    rd = c.done("spectral_capture", {"label": "ui_spec"})["result"]
    m = c.done("learn_material", {"reading_id": rd["id"], "name": "ui_wall",
                                  "roi": [700, 520, 740, 560]})["result"]
    assert m["name"] == "ui_wall"
    _, mats = c.get("/api/library/materials")
    assert "ui_wall" in [x["name"] for x in mats["items"]]
    j = c.done("delete", {"kind": "materials", "id": "ui_wall", "reason": "bad ROI"})
    assert j["result"]["trashed_to"].startswith("trash/spectral/materials/")
    _, mats = c.get("/api/library/materials")
    assert "ui_wall" not in [x["name"] for x in mats["items"]]
    # the reading itself is untouched and can be trashed separately
    j = c.done("delete", {"kind": "spectral", "id": rd["id"]})
    assert j["result"]["trashed_to"].startswith("trash/spectral/")


def _done_or_pending(j):
    """Grid / zone / map verbs are built by other modules; until they land the
    job must fail cleanly with NotImplementedError, never with a UI arg error."""
    assert j["status"] == "done" or j["error_type"] == "NotImplementedError", \
        f"{j['verb']}: {j['error']}\n{j.get('trace')}"
    return j["status"] == "done"


def test_grid_photo_and_scan_this_zone(scene):
    """The operator's flow: map the scene, project the lettered grid, read the
    photo, then scan only the zone the part is in."""
    c, base, _ = scene
    if _done_or_pending(c.job("capture_map", {"label": "map"})):
        _, maps = c.get("/api/library/maps?geometry_id=mock-wall_2p7ft_vstripe_v1")
        assert maps["items"]
    j = c.job("project_grid", {"spacing_px": 96, "label": "grid"})
    if _done_or_pending(j):
        shot = j["result"]
        assert shot["kind"] == "grid" and shot["grid"]["col_labels"][0] == "A"
        status, raw, _ = c.request("GET", "/files/" + shot["camera_png"])
        assert status == 200 and raw[:4] == b"\x89PNG"
    j = c.job("capture_object", {"label": "zone-part", "base_id": base["id"], "zone": "F4:G5"})
    if _done_or_pending(j):
        assert j["result"]["meta"]["zone"] == "F4:G5"
    j = c.job("project_zones", {"label": "zones", "zones": [
        {"name": "Z1", "polygon": [[850, 350], [1000, 350], [1000, 460], [850, 460]],
         "space": "camera", "pattern": "outline", "level": 255}]})
    _done_or_pending(j)
    bad = c.job("project_zones", {"zones": [{"name": "Z1", "polygon": [[1, 2], [3, 4]], "shape": "x"}]})
    assert bad["status"] == "error" and bad["error_type"] == "ArgError"


def test_status_shows_the_grid_the_engine_uses(scene):
    """Zone names resolve against the LAST grid projected for the geometry, by
    anyone. /api/status exposes that grid so every browser can show it (and
    warn when its own photo is older); a zone sent as 'F4@96' keeps meaning
    the 96 px cell even after an agent projected a 64 px grid."""
    c, base, _ = scene
    first = c.done("project_grid", {"spacing_px": 96, "label": "grid96"})["result"]
    _, st = c.get("/api/status")
    assert st["grid"]["json_path"] == first["json_path"]
    assert st["grid"]["grid"]["spacing_px"] == 96
    agent = c.done("project_grid", {"spacing_px": 64, "label": "agent64"})["result"]
    _, st = c.get("/api/status")
    g = st["grid"]
    assert g["json_path"] == agent["json_path"] != first["json_path"]
    assert g["grid"]["spacing_px"] == 64 and g["grid"]["col_labels"][0] == "A"
    assert g["camera_png"] == agent["camera_png"] and not Path(g["camera_png"]).is_absolute()
    assert g["cell_centers"] == agent["cell_centers"]
    assert c.request("GET", "/files/" + g["camera_png"])[0] == 200
    j = c.done("capture_object", {"label": "zone96", "base_id": base["id"], "zone": "F4@96"})
    assert j["result"]["meta"]["zone"] == "F4@96"


def test_grid_hold_is_capped(fast):
    """A pumped hold can't be interrupted and keeps the machine-wide rig lock."""
    j = fast.job("project_grid", {"spacing_px": 96, "hold_s": 3600})
    assert j["status"] == "error" and j["error_type"] == "ArgError" and "hold_s" in j["error"]
    j = fast.job("project_zones", {"hold_s": -1, "zones": [
        {"name": "Z1", "polygon": [[0, 0], [10, 10]], "space": "projector"}]})
    assert j["status"] == "error" and j["error_type"] == "ArgError"


# --------------------------------------------------------- virtual sensors

GEO = "mock-wall_2p7ft_vstripe_v1"
SPOT_AT = {"dmd": [456, 570]}          # mock bench: lands at camera (720, 540), base-plane (0, 0)


def _restore_scene(c):
    """Put the module's scene back: the §11 validation object alone on the bench."""
    c.done("mock_clear")
    c.done("mock_place", {"object": "validation"})


def test_spot_size_nominal_vs_measured(fast):
    nom = fast.done("spot_size", {"diameter_mm": 6.5})["result"]
    assert nom["__type__"] == "SpotSpec" and nom["source"] == "nominal"
    assert nom["diameter_dmd_rows"] == pytest.approx(2 * nom["diameter_dmd_px"], abs=1e-3)
    assert any("nominal" in w for w in nom["warnings"])
    meas = fast.done("spot_size", {"diameter_mm": 6.5, "mm_per_dmd_px": 0.25})["result"]
    assert (meas["source"], meas["diameter_dmd_px"], meas["diameter_dmd_rows"]) == \
        ("measured", 26.0, 52.0)
    j = fast.job("spot_size", {})
    assert j["status"] == "error" and j["error_type"] == "ArgError" and "diameter_mm" in j["error"]


def test_project_spot_at_a_grid_cell(scene):
    c, _, _ = scene
    j = c.done("project_spot", {"at": "H8@96", "diameter_mm": 6.5, "label": "ui_h8"})
    r = j["result"]
    assert r["__type__"] == "SpotReading" and r["found"] is True
    assert r["target_dmd"] == [719.5, 719.5] and r["at"] == {"cell": "H8", "spacing_px": 96}
    assert r["centroid_px"] and r["diameter_px"] > 0 and r["spec"]["source"] == "nominal"
    for key in ("photo_png", "overlay_png"):
        assert r[key].startswith("output/") and c.request("GET", "/files/" + r[key])[0] == 200
    zone = c.job("project_spot", {"at": "G7:H9", "diameter_mm": 6.5})
    assert zone["status"] == "error" and "zone" in zone["error"]         # a spot is one cell
    j = c.job("project_spot", {"diameter_mm": 6.5})
    assert j["status"] == "error" and j["error_type"] == "ArgError" and "'at'" in j["error"]
    j = c.job("project_spot", {"at": {"dmd": [1, "x"]}, "diameter_mm": 6.5})
    assert j["status"] == "error" and j["error_type"] == "ArgError"


def test_presence_sensor_teach_check_list_and_trash(scene):
    """Teach on the EMPTY bench (only with the operator's confirmation), place a
    darker box: PRESENT; take it away: ABSENT. Listed, and trashable."""
    c, _, _ = scene
    try:
        c.done("mock_clear")                     # the simulated human clears the field
        args = {"name": "ui_pe", "at": SPOT_AT, "diameter_mm": 6.5}
        j = c.job("teach_presence", args)
        assert j["status"] == "error" and "empty_confirmed" in j["error"], j
        t = c.done("teach_presence", {**args, "empty_confirmed": True})["result"]
        assert t["__type__"] == "PresenceTeach" and t["method"] == "photoelectric"
        assert t["reference_signal"] > 100 and t["path"] == f"sensors/{GEO}/ui_pe"

        c.done("mock_place", {"object": "box", "h_mm": 10, "w_mm": 40, "l_mm": 40,
                              "albedo": 0.2})
        r = c.done("check_presence", {"name": "ui_pe"})["result"]
        assert r["__type__"] == "PresenceResult" and r["verdict"] == "PRESENT"
        assert r["present"] is True and r["relative_change"] < -0.25
        assert c.request("GET", "/files/" + r["photo_png"])[0] == 200
        c.done("mock_clear")
        r = c.done("check_presence", {"name": "ui_pe", "label": "ui_pe_empty"})["result"]
        assert r["verdict"] == "ABSENT" and r["present"] is False

        _, lst = c.get(f"/api/library/sensors?geometry_id={GEO}")
        row = next(s for s in lst["items"] if s["name"] == "ui_pe")
        assert row["kind"] == "presence" and row["path"] == f"sensors/{GEO}/ui_pe"
        _, lst = c.post("/api/jobs", {"verb": "list_library", "args": {"kind": "sensors"},
                                      "wait": 30})
        assert "ui_pe" in [s["name"] for s in lst["result"]]

        j = c.done("delete", {"kind": "sensors", "id": "ui_pe", "reason": "test"})
        assert j["result"]["trashed_to"].startswith(f"trash/sensors/{GEO}/ui_pe")
        _, lst = c.get(f"/api/library/sensors?geometry_id={GEO}")
        assert "ui_pe" not in [s["name"] for s in lst["items"]]
        j = c.job("check_presence", {"name": "ui_pe"})
        assert j["status"] == "error" and "no sensor named" in j["error"]
    finally:
        _restore_scene(c)


def test_alignment_sensor_teach_and_check(scene):
    c, base, obj = scene
    try:
        ref = c.done("teach_alignment", {"name": "ui_fix", "object_id": obj["id"]})["result"]
        assert ref["__type__"] == "AlignmentRef" and ref["scan_id"] == obj["id"]
        assert ref["axis_ratio"] > 2 and ref["area_px"] > 1000
        # tol_px judges the position when the geometry is uncalibrated, tol_mm when it is
        args = {"name": "ui_fix", "tol_px": 3, "tol_mm": 1}
        same = c.done("check_alignment", {**args, "object_id": obj["id"]})["result"]
        assert same["verdict"] == "ALIGNED" and abs(same["dtheta_deg"]) < 0.3

        c.done("mock_clear")                     # the same part, 5 mm to the right
        c.done("mock_place", {"object": "box", "h_mm": 106.95, "w_mm": 56.5, "l_mm": 21.9,
                              "x_mm": -15, "y_mm": 0, "albedo": 0.7})
        moved = c.done("capture_object", {"label": "ui_moved", "base_id": base["id"]})["result"]
        r = c.done("check_alignment", {**args, "object_id": moved["id"]})["result"]
        assert r["verdict"] == "NOT_ALIGNED" and r["position_ok"] is False and r["dx_px"] > 5
        if r["position_units"] == "mm":
            assert r["dx_mm"] == pytest.approx(5.0, abs=0.5)
        assert c.request("GET", "/files/" + r["overlay_png"])[0] == 200

        _, lst = c.get(f"/api/library/sensors?geometry_id={GEO}")
        assert next(s for s in lst["items"] if s["name"] == "ui_fix")["kind"] == "alignment"
        j = c.job("check_presence", {"name": "ui_fix"})
        assert j["status"] == "error" and "presence" in j["error"]
        j = c.job("check_alignment", {"name": "ui_fix", "object_id": "no-such-scan"})
        assert j["status"] == "error" and j["error_type"] == "EntryNotFound"
    finally:
        _restore_scene(c)


def test_live_view_duration_is_capped_and_checked_against_led_limits(fast):
    """The live view holds the rig and the flood: max_s is capped server-side at
    30 min, and rule 3 over time refuses burst LED currents for it."""
    c = fast
    j = c.job("live_view", {"max_s": 3600})
    assert j["status"] == "error" and j["error_type"] == "ArgError" and "1800" in j["error"]
    _, d = c.get("/api/presets/ui_fast")
    p = d["preset"]
    p["name"] = "ui_burst"
    p["projection"]["led"] = {"mode": "burst", "r": 170, "g": 200, "b": 200}
    status, r = c.post("/api/presets", {"preset": p, "base": "ui_fast"})
    assert status == 200, r
    try:
        assert c.post("/api/presets/ui_burst/load", {})[1]["status"] == "done"
        j = c.job("live_view", {"max_s": 60})
        assert j["status"] == "error" and j["error_type"] == "SafetyViolation", j
        assert "live view" in j["error"] and not c.get("/api/live/state")[1]["running"]
    finally:
        assert c.post("/api/presets/ui_fast/load", {})[1]["status"] == "done"


def test_rig_holder_names_this_web_ui(ui):
    """RigBusy messages in another program (CLI, second UI) name who holds the rig."""
    _, app, _ = ui
    assert app.session.holder == f"web UI (port {app.port})"


def test_main_reports_only_bind_failures_as_port_in_use(monkeypatch, capsys):
    """An engine start-up failure is reported as itself; only the socket bind
    means 'is another UI already running?'. No real engine / socket here."""
    class FakeApp:
        closed = False

        def __init__(self, **kw):
            self.session, self.port, self.mock = type("Session", (), {})(), None, True

        def start(self):
            return self

        def close(self):
            FakeApp.closed = True

    def refuse(*a, **k):
        raise OSError("[WinError 10048] only one usage of each socket address")

    monkeypatch.setattr(ui_server, "App", FakeApp)
    monkeypatch.setattr(ui_server, "UIServer", refuse)
    assert ui_server.main(["--mock", "--port", "8765"]) == 2
    assert FakeApp.closed and "another UI already running" in capsys.readouterr().out

    class Broken(FakeApp):
        def start(self):
            raise OSError("library folder is read-only")

    monkeypatch.setattr(ui_server, "App", Broken)
    with pytest.raises(OSError, match="read-only"):
        ui_server.main(["--mock", "--port", "8765"])
    assert "another UI" not in capsys.readouterr().out


def test_failing_verbs_become_job_errors_not_crashes(fast, monkeypatch):
    c = fast

    def pending(*a, **k):
        raise NotImplementedError("profile: pending implementation")

    monkeypatch.setattr(api, "profile", pending)       # a mode still being built
    j = c.job("profile", {"object_id": "whatever"})
    assert j["status"] == "error"
    # the id resolves first; use a real one so the verb itself is reached
    _, lst = c.get("/api/library/objects")
    if lst["items"]:
        j = c.job("profile", {"object_id": lst["items"][0]["id"], "axis": "auto"})
        assert j["status"] == "error" and j["error_type"] == "NotImplementedError"
        assert "still being built" in j["hint"]

    j = c.job("list_library", {"kind": "nope"})
    assert j["status"] == "error" and j["error_type"] == "ValueError"
    j = c.job("capture_object", {"label": "x", "base_id": "no-such-base"})
    assert j["status"] == "error" and j["error_type"] == "EntryNotFound"
    j = c.job("reconstruct", {"object_id": "x", "fmt": "stl"})
    assert j["status"] == "error" and "fmt" in j["error"]

    assert c.post("/api/jobs", {"verb": "configure", "args": {"mock": False}})[0] == 400
    assert c.post("/api/jobs", {"verb": "rm_rf"})[0] == 400
    status, r = c.post("/api/jobs", {"verb": "check_signal", "args": {"bogus": 1}})
    assert status == 400 and "bogus" in r["error"]
    assert c.get("/api/jobs/j999999")[0] == 404

    assert c.done("list_library", {"kind": "bases"})     # the worker is still alive
    _, jobs = c.get("/api/jobs?n=5")
    assert len(jobs["jobs"]) == 5 and jobs["jobs"][0]["verb"] == "list_library"


def test_live_view_streams_jpeg_and_focus_until_stopped(fast):
    c = fast
    status, j = c.post("/api/live/start", {"pattern": "stripes", "level": 200})
    assert status == 200 and j["verb"] == "live_view"
    t0 = time.monotonic()
    st = {}
    while time.monotonic() - t0 < 60:
        _, st = c.get("/api/live/state")
        if st["frame"] >= 3:
            break
        time.sleep(0.1)
    assert st["running"] and st["frame"] >= 3, st
    assert st["metrics"]["focus"] > 0 and st["focus_peak"] >= st["metrics"]["focus"]
    assert st["settings"]["pattern"] == "stripes" and st["settings"]["level"] == 200
    status, raw, hdrs = c.request("GET", "/api/live.jpg")
    assert status == 200 and hdrs["Content-Type"] == "image/jpeg" and raw[:2] == b"\xff\xd8"
    assert c.post("/api/live/settings", {"pattern": "laser"})[0] == 400
    assert c.post("/api/live/settings", {"level": 90})[1]["settings"]["level"] == 90

    c.post("/api/live/stop", {})
    _, done = c.get(f"/api/jobs/{j['id']}")
    t0 = time.monotonic()
    while done["status"] == "running" and time.monotonic() - t0 < 30:
        time.sleep(0.1)
        _, done = c.get(f"/api/jobs/{j['id']}")
    assert done["status"] == "done" and done["result"]["frames"] >= 3
    assert not c.get("/api/live/state")[1]["running"]


def test_post_guards_and_log(fast):
    c = fast
    status, raw, _ = c.request("POST", "/api/jobs", b'{"verb": "check_signal"}',
                               {"Content-Type": "text/plain"})
    assert status == 415                          # forms / no-cors posts can't send JSON
    status, _ = c.post("/api/jobs", {"verb": "check_signal"},
                       {"Origin": "http://evil.example"})
    assert status == 403
    status, _, _ = c.request("GET", "/api/status", headers={"Host": "attacker.example:80"})
    assert status == 403                          # DNS rebinding
    _, log = c.get("/api/log?n=500")
    verbs = [r["verb"] for r in log["records"]]
    assert "load_preset" in verbs and "check_signal" in verbs
    assert any(r["ok"] is False for r in log["records"])     # failures are logged too


def _post_by_hand(port, headers: dict, body: bytes, pause_s: float) -> tuple:
    """A POST written on a socket: its headers, a pause, then its body (a slow client: the
    body is not in the server's buffer when it reads the headers) -> (status, JSON answer).
    The answer is read to its end, as a browser does."""
    s = socket.create_connection(("127.0.0.1", port), timeout=30)
    try:
        lines = ["POST /api/jobs HTTP/1.1", f"Content-Length: {len(body)}", "Connection: close"]
        lines += [f"{k}: {v}" for k, v in headers.items()]
        s.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
        time.sleep(pause_s)
        s.sendall(body)
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    finally:
        s.close()
    head, _, payload = data.partition(b"\r\n\r\n")
    return int(head.split()[1]), json.loads(payload)


def test_a_refused_post_is_answered_every_time(ui):
    """A POST refused before its body was read (a foreign Host or Origin, not JSON) used to
    lose its answer now and then on Windows: the server closed the connection with the body
    still unread, the socket was reset, and the client saw WinError 10053 / 10054 instead of
    the 403 / 415. The server reads (and drops) a refused request's body before it answers;
    what it refuses, and what it logs, are unchanged."""
    c, app, _ = ui
    port = app.port
    _, log = c.get("/api/log?n=500")
    before = log["records"]
    body = json.dumps({"verb": "check_signal", "args": {"pad": "x" * 2000}}).encode()
    refusals = (({"Origin": "http://evil.example"}, 403, "cross-origin request"),
                ({"Content-Type": "text/plain"}, 415, "must be application/json"),
                ({"Host": "attacker.example:80"}, 403, "unexpected Host header"))
    for hdrs, code, says in refusals:
        for _ in range(150):                     # http.client: headers and body sent apart
            status, raw, _h = c.request("POST", "/api/jobs", body,
                                        {"Content-Type": "application/json", **hdrs})
            assert status == code and says in json.loads(raw)["error"], (status, raw)
        for _ in range(5):                       # a slow client: the body comes later
            status, answer = _post_by_hand(
                port, {"Host": f"127.0.0.1:{port}", "Content-Type": "application/json", **hdrs},
                body, 0.2)
            assert status == code and says in answer["error"], (status, answer)
    # an accepted POST after them is read as before
    status, j = c.post("/api/jobs", {"verb": "check_signal"})
    assert status == 200 and j["verb"] == "check_signal", j
    _wait_job(c, j["id"])
    # nothing refused reached the log: only the accepted job was added
    _, log = c.get("/api/log?n=500")
    after = log["records"]
    i = max((k for k, r in enumerate(after) if before and r == before[-1]), default=-1)
    added = after[i + 1:]
    assert added and {r["verb"] for r in added} == {"check_signal"}, added


def test_log_filters_before_the_cut_and_browsing_is_compact(fast):
    """Browsing used to log every full listing and the Log tab filtered only
    after taking the last n, so real actions vanished from its window."""
    c = fast
    c.done("check_signal")
    for _ in range(12):
        assert c.get("/api/library/objects")[0] == 200
    _, log = c.get("/api/log?n=5&exclude=list_library")
    verbs = [r["verb"] for r in log["records"]]
    assert len(verbs) == 5 and "list_library" not in verbs and "check_signal" in verbs
    _, log = c.get("/api/log?n=5")
    recs = log["records"]
    assert [r["verb"] for r in recs] == ["list_library"] * 5
    # a count, not the listing itself
    assert all(isinstance(r["result"], dict) and "n" in r["result"] for r in recs), recs[-1]


# ------------------------------------------------ busy worker / live view

def _gate(monkeypatch):
    """check_signal that holds the worker until the test releases it
    (stands in for a long real-rig job)."""
    gate, started = threading.Event(), threading.Event()

    def slow(preset=None):
        started.set()
        gate.wait(60)
        return {"slow": True}

    monkeypatch.setattr(api, "check_signal", slow)
    return gate, started


def _wait_job(c, jid, timeout=60.0):
    t0 = time.monotonic()
    while True:
        _, j = c.get(f"/api/jobs/{jid}")
        if j["status"] not in ("queued", "running") or time.monotonic() - t0 > timeout:
            return j
        time.sleep(0.1)


def test_pointcloud_of_a_new_scan_while_the_worker_is_busy(scene, monkeypatch):
    """The 3D viewer used to answer 404 'no scan' for a just-captured scan
    whenever a job held the worker (stale, geometry-less listing cache)."""
    c, base, obj = scene
    assert c.get(f"/api/pointcloud/{obj['id']}?max=500")[0] == 200     # fills the cache
    new = c.done("capture_object", {"label": "newobj", "base_id": base["id"]})["result"]
    gate, started = _gate(monkeypatch)
    try:
        _, j = c.post("/api/jobs", {"verb": "check_signal"})
        assert started.wait(30)
        status, pc = c.get(f"/api/pointcloud/{new['id']}?max=500")
        assert status == 200 and pc["object_id"] == new["id"], pc
        status, r = c.get("/api/pointcloud/nope")
        assert status == 503 and "busy" in r["error"]      # not a misleading 'no such scan'
    finally:
        gate.set()
    assert _wait_job(c, j["id"])["status"] == "done"
    assert c.get("/api/pointcloud/nope")[0] == 404


def test_live_stop_while_queued_is_not_lost(fast, monkeypatch):
    """Stop pressed while the live view waits behind a job used to be cleared
    when it started, so it ran for 30 min holding the rig."""
    c = fast
    gate, started = _gate(monkeypatch)
    try:
        _, busy = c.post("/api/jobs", {"verb": "check_signal"})
        assert started.wait(30)
        _, live = c.post("/api/live/start", {"pattern": "flood"})
        assert live["status"] == "queued"
        status, r = c.post("/api/live/stop", {})
        assert status == 200 and r["stopped"] == 1
        _, lj = c.get(f"/api/jobs/{live['id']}")
        assert lj["status"] == "error" and lj["error_type"] == "Cancelled" and not lj["started"]
    finally:
        gate.set()
    assert _wait_job(c, busy["id"])["status"] == "done"
    time.sleep(0.3)
    _, st = c.get("/api/status")
    assert not st["live"]["running"] and not st["job"] and not st["queued"]


def test_rig_job_drops_a_queued_live_view(fast, monkeypatch):
    c = fast
    gate, started = _gate(monkeypatch)
    try:
        _, busy = c.post("/api/jobs", {"verb": "check_signal"})
        assert started.wait(30)
        _, live = c.post("/api/live/start", {})
        _, nxt = c.post("/api/jobs", {"verb": "check_signal"})
        assert nxt["note"] and "live view stopped" in nxt["note"]
        _, lj = c.get(f"/api/jobs/{live['id']}")
        assert lj["status"] == "error" and lj["error_type"] == "Cancelled"
    finally:
        gate.set()
    assert _wait_job(c, nxt["id"])["status"] == "done"
    _, st = c.get("/api/status")
    assert not st["live"]["running"]


def test_live_view_stops_when_nobody_watches(fast, monkeypatch):
    """An abandoned browser must not keep the rig, the flood and the rig lock."""
    monkeypatch.setattr(ui_server, "LIVE_IDLE_S", 1.0)
    c = fast
    _, live = c.post("/api/jobs", {"verb": "live_view", "args": {"pattern": "flood"}})
    j = _wait_job(c, live["id"], timeout=30)          # polls the JOB, not the live view
    assert j["status"] == "done", j
    assert "no viewer" in j["result"]["stopped_because"]


def test_rig_free_jobs_run_during_live_view(scene):
    """Measure / inspect don't need the rig: they run between live frames
    instead of stopping the live view (and a rig job still stops it)."""
    c, _, obj = scene
    _, live = c.post("/api/live/start", {"pattern": "stripes"})
    t0 = time.monotonic()
    st = {}
    while time.monotonic() - t0 < 60:
        _, st = c.get("/api/live/state")
        if st["running"] and st["frame"] >= 2:
            break
        time.sleep(0.1)
    assert st["running"], st
    try:
        m = c.done("measure", {"object_id": obj["id"]})
        assert m["note"] is None and m["result"]["units"] in ("mm", "px")
        _, st = c.get("/api/live/state")
        assert st["running"] and st["job_id"] == live["id"]
        _, lj = c.get(f"/api/jobs/{live['id']}")
        assert lj["status"] == "running"
    finally:
        c.post("/api/live/stop", {})
    j = _wait_job(c, live["id"])
    assert j["status"] == "done" and j["result"]["stopped_because"] == "stopped by the operator"


def test_rig_busy_gets_its_own_hint():
    e = rules.RigBusy("the camera + DMD are in use by another process (pid 1 'x')")
    assert "Another program holds the camera" in ui_server.error_hint(e)


def test_no_usb_driver_loaded_by_the_ui_run(scene):
    rules.check_scan_path_usb_free()
    assert "hid" not in sys.modules


# ------------------------------------------------ cell & 3D, maps, checker, surfer

ATLAS_SRC = Path(os.environ.get("SCANNER_TEST_ATLAS", ""))   # a real held-cell atlas.json; skipped if unset


@pytest.fixture(scope="module")
def cell(scene):
    """A standing cell (mock Cylinder with one dent) scanned against the module's base."""
    c, base, _ = scene
    c.done("mock_clear")
    placed = c.done("mock_place", {"object": "cylinder", "radius_mm": 24.05, "height_mm": 90,
                                   "dents": [{"depth_mm": 1.5, "size_mm": 12, "angle_deg": 10,
                                              "z_mm": 45}]})
    assert placed["result"]["objects"][0]["radius_mm"] == 24.05
    try:
        scan = c.done("capture_object", {"label": "cell", "base_id": base["id"]})["result"]
    finally:
        _restore_scene(c)
    return c, scan


def test_mock_place_cylinder_rejects_bad_dents(fast):
    j = fast.job("mock_place", {"object": "cylinder", "dents": [{"depth_mm": 1}]})
    assert j["status"] == "error" and j["error_type"] == "ArgError"


def test_export_mesh_stl_obj_and_download(cell):
    """STL / OBJ meshes land in output/pointclouds with their meta JSON, and /files
    serves them as downloads only (never rendered inline)."""
    c, scan = cell
    for kind, fmt_ in (("cylinder", "stl"), ("relief", "obj"), ("base", "stl")):
        j = c.done("export_mesh", {"object_id": scan["id"], "kind": kind, "fmt": fmt_,
                                   "stride": 4, "radius_mm": 24.05 if kind == "cylinder" else None})
        r = j["result"]
        assert r["file"].endswith("." + fmt_) and r["file"].startswith("output/pointclouds/")
        assert r["meta"]["finite_vertices"] is True and r["meta"]["n_vertices"] > 0
        assert j["links"]["file"] == r["file"]
        status, raw, hdrs = c.request("GET", "/files/" + r["file"])
        assert status == 200 and len(raw) > 100
        assert hdrs["Content-Type"] == f"model/{fmt_}"
        assert hdrs["Content-Disposition"].startswith("attachment")
        assert hdrs["X-Content-Type-Options"] == "nosniff"
        if kind == "cylinder":
            assert r["meta"]["open_arc_only"] is True and r["meta"]["arc_deg"] < 180
    bad = c.job("export_mesh", {"object_id": scan["id"], "fmt": "ply"})
    assert bad["status"] == "error" and bad["error_type"] == "ArgError"
    # other mesh-like files stay unserved
    status, _, _ = c.request("GET", "/files/output/pointclouds/x.3mf")
    assert status == 403


def test_cell_unwrap_and_dents(cell):
    c, scan = cell
    u = c.done("cell_unwrap", {"object_id": scan["id"], "radius_mm": 24.05})
    r = u["result"]
    assert 60 < r["arc_deg"] < 180 and r["cell_height"] > 0 and r["units"] in ("mm", "px")
    assert "deviation" not in r and "dent_map" not in r          # no big arrays in JSON
    status, raw, _ = c.request("GET", "/files/" + u["links"]["png"])
    assert status == 200 and raw[:4] == b"\x89PNG"
    d = c.done("cell_dents", {"object_id": scan["id"], "radius_mm": 24.05})
    rep = d["result"]
    assert rep["verdict"] in ("PASS", "FAIL", "INDETERMINATE")
    if rep["units"] == "px":                          # uncalibrated: honest, never PASS
        assert rep["verdict"] == "INDETERMINATE"
        assert any("rule 6" in w for w in rep["warnings"]), rep["warnings"]
    assert rep["dents"], "the mock dent must be found"
    dent = rep["dents"][0]
    assert {"theta_deg", "z", "depth", "units"} <= set(dent) and abs(dent["theta_deg"] - 10) < 8
    assert d["links"]["png"].endswith(".png") and d["links"]["json"].endswith(".json")


def test_bake_maps_and_checker_scan(cell):
    c, scan = cell
    m = c.done("bake_maps", {"object_id": scan["id"]})
    png = m["result"]["png_paths"]
    assert {"height", "cavity", "normal", "validity", "density"} <= set(png)
    assert "height" not in {k for k, v in m["result"].items() if isinstance(v, list)}
    status, raw, _ = c.request("GET", "/files/" + png["normal"])
    assert status == 200 and raw[:4] == b"\x89PNG"
    bad = c.job("bake_maps", {"object_id": scan["id"], "light_id": "nope"})
    assert bad["status"] == "error" and bad["error_type"] == "EntryNotFound"
    k = c.done("checker_scan", {"square_px": 32, "shifts": 1})
    r = k["result"]
    assert r["n_corners"] > 0 and set(r["png_paths"]) >= {"stretch", "contrast"}
    assert r["median"]["contrast"] is not None
    assert len(json.dumps(r)) < 20000                   # a summary, not the corner arrays
    _, li = c.get("/api/library/lights")
    assert li["kind"] == "lights" and isinstance(li["items"], list)


def test_atlas_import_list_and_surfer_methods(scene, tmp_path):
    """Import a temp COPY of the real atlas (the source is never touched), then
    its regions work as zones: flood verify, anti-glare, multi-orientation, and
    a zone scan with 'atlas:cell/cell'."""
    c, base, _ = scene
    if not ATLAS_SRC.is_file():
        pytest.skip(f"{ATLAS_SRC} not on this machine")
    before = ATLAS_SRC.read_bytes()
    src = tmp_path / "cell_held_atlas.json"
    shutil.copy(ATLAS_SRC, src)
    j = c.done("import_atlas", {"path": str(src), "name": "cell"})
    assert j["result"]["file"].startswith("atlases/")
    specs = {g["spec"] for g in j["result"]["atlas"]["regions"]}
    assert {"atlas:cell/cell", "atlas:cell/floor"} <= specs
    _, la = c.get("/api/library/atlases")
    assert any(a["name"] == "cell" and a["regions"] for a in la["items"])
    assert c.done("list_atlases")["result"][0]["name"] == "cell"
    for bad in ({"path": str(tmp_path / "missing.json"), "name": "x"},
                {"path": str(src), "name": "../x"}):
        e = c.job("import_atlas", bad)
        assert e["status"] == "error" and e["error_type"] == "ArgError"

    f = c.done("flood_verify", {"region": "atlas:cell/cell"})
    r = f["result"]
    assert r["region"] == "atlas:cell/cell" and r["verdict"] and isinstance(r["lands"], bool)
    status, raw, _ = c.request("GET", "/files/" + r["camera_png"])
    assert status == 200 and raw[:4] == b"\x89PNG"
    g = c.done("anti_glare_sweep", {"region": "atlas:cell/cell",
                                    "carriers": [[0, 0], [0.3, -0.5]]})
    assert g["result"]["wording"] and len(g["result"]["scores"]) == 2
    bad = c.job("anti_glare_sweep", {"region": "atlas:cell/cell", "carriers": [[1]]})
    assert bad["status"] == "error" and bad["error_type"] == "ArgError"
    mo = c.done("multi_orientation_scan", {"region": "atlas:cell/cell",
                                           "orientations": ["v", "h"]})
    assert mo["result"]["orientations"] == ["v", "h"] and "quality" not in mo["result"]
    assert "quality.png" in mo["links"]
    assert any(k.endswith("fused_modulation.png") for k in mo["links"])
    z = c.job("capture_object", {"label": "atlas-zone", "base_id": base["id"],
                                 "zone": "atlas:cell/cell"})
    assert z["status"] == "done", z["error"]
    assert "atlas:cell/cell" in str(z["result"]["meta"].get("zone"))
    assert ATLAS_SRC.read_bytes() == before


def test_make_flash_pack_writes_files_only(fast, tmp_path, monkeypatch):
    import tools.make_flash_pack as fp
    monkeypatch.setattr(fp, "DEFAULT_OUT", tmp_path / "packs")
    # the pack tool reads the repo's presets folder: use a repo preset, then restore
    fast.done("load_preset", {"name": PRESET})
    try:
        j = fast.done("make_flash_pack", {"axes": ["x"]})
    finally:
        fast.post("/api/presets/ui_fast/load", {})
    r = j["result"]
    assert Path(r["folder"]).is_dir() and Path(r["folder"]).parent == tmp_path / "packs"
    assert "manifest.json" in r["files"] and r["doc"].endswith("FLASH_PATTERN_MODE.md")
    assert "pattern mode" in r["doc_summary"]
    bad = fast.job("make_flash_pack", {"axes": ["x", "y"]})
    assert bad["status"] == "error" and bad["error_type"] == "ArgError"
