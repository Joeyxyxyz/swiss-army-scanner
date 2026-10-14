"""The Scan zones panel of the operator page (/quick) and the web server, in MOCK mode: the
panel is a panel of its own above 1 Lights; the Paint zones button starts the painter job on
the rig and every page learns how it ended (saved as NAME with N zones, or NOT saved and
why); several zone sets can be switched on and off, and single zones inside them; the
selection is stored by the server (it survives a reload and a restart) and is stated beside
the run buttons together with what the run button does with it. The page logic runs in node
on the real quick.js with a tiny fake DOM and a fake server."""

import json
import re
import shutil
import subprocess
import threading
import time
import types

import pytest

from engine import api, config
from engine.zones import Zone, ZoneSet
from engine.zones import selection as sel
from engine.zones import store as zstore
from conftest import ROOT
from test_ui_server import Client

from ui.web import server as ui_server

STATIC = ROOT / "ui" / "web" / "static"
PRESET = "bench_9in_metric"
GEO = "mock-bench_9in_v1"


def _serve(lib, stations, **kw):
    httpd, app = ui_server.make_server(mock=True, library=lib, port=0, stations_dir=stations,
                                       preset=PRESET, **kw)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    return httpd, app


@pytest.fixture(scope="module")
def web(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("zui")
    lib, stations = tmp / "library", tmp / "stations"
    stations.mkdir()
    httpd, app = _serve(lib, stations)
    try:
        yield Client(app.port), app, lib
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()
        api._session = None


def _state(c) -> dict:
    return c.get("/api/cmd/state")[1]


def _select(c, sets, was=None, **more):
    return c.post("/api/zones/selection", {"sets": sets, "was": was, "source": "quick", **more})


def test_zone_verbs_are_wired():
    for v in ("zone_paint", "zone_show", "zone_delete", "zone_select"):
        assert v in ui_server.VERBS and callable(getattr(ui_server.App, f"_v_{v}", None)), v
    # the painter and the projection hold the rig: never rig-free, and an ABORT ends them
    assert {"zone_paint", "zone_show"} <= ui_server.CANCELLABLE_VERBS
    assert not {"zone_paint", "zone_show"} & ui_server.RIG_FREE_VERBS
    assert {"zone_delete", "zone_select"} <= ui_server.RIG_FREE_VERBS       # files only
    assert "zone_select" not in ui_server.CANCELLABLE_VERBS
    # what "the selected zones" are never changes while a station is armed or in a cycle
    assert ui_server.SELECTION_CHANGE_VERBS == {"zone_select", "zone_paint", "zone_delete"}
    assert {"name", "source", "replace", "events", "select"} <= set(ui_server.VERBS["zone_paint"][1])
    assert {"sets", "zones", "on", "off", "clear", "expect"} <= set(ui_server.VERBS["zone_select"][1])
    assert "zones" in ui_server.VERBS["scan_fast"][1]
    for verb, arg in (("measure_run", "zone"), ("quick_check", "zone"), ("zone_show", "zones"),
                      ("scan_fast", "zones")):
        text = ui_server.VERBS[verb][1][arg]
        assert "selection" in text and ("zoneset" in text or "+" in text), (verb, text)
    routes = {(m, p) for m, p, _ in ui_server.ROUTES}
    assert ("GET", r"/api/zonesets") in routes
    assert {("GET", r"/api/zones/selection"), ("POST", r"/api/zones/selection")} <= routes


def test_page_has_the_scan_zones_panel_its_ids_and_no_inline_handlers():
    html = (STATIC / "quick.html").read_text(encoding="utf-8")
    js = (STATIC / "quick.js").read_text(encoding="utf-8")
    # a panel of its own, the first one, above 1 Lights - and always open (not a <details>)
    panels = html[html.index('<div class="qpanels">'):]
    assert panels.index('id="qScanZones"') < panels.index('id="qLights"') < panels.index('id="qCamera"')
    assert '<div class="qpanel qzones" id="qScanZones"' in html and "<b>Scan zones</b>" in html
    zones = html[html.index('id="qScanZones"'):html.index('id="qLights"')]
    # the switches come first and what the last zone action did (SAVED / NOT SAVED ...) sits
    # right under them: on a low screen both are in view before anything else of the panel
    assert zones.index('id="qZoneRows"') < zones.index('id="qZoneResult"') < zones.index('id="qZoneGeo"') \
        < zones.index('id="qZoneShow"') < zones.index('id="qPaint"') < zones.index('id="qZoneAsk"')
    lights = html[html.index('id="qLights"'):html.index('id="qCamera"')]
    for el in ('<button id="qPaint"', '<select id="qZoneSet"', '<input id="qZoneName"',
               '<button id="qZoneShow"', '<img id="qZoneImg"', '<button id="qZoneDelete"',
               'id="qZoneGeo"', 'id="qZoneRows"', 'id="qZoneResult"', 'id="qZoneInfo"',
               'id="qZoneAsk"', 'id="qZoneAskText"', 'id="qZoneReasonRow"', '<input id="qZoneReason"',
               '<button id="qZoneAskYes"', '<button id="qZoneAskNo"'):
        assert el in zones and el not in lights, el
    assert "Paint zones" in zones and "Show selected zones" in zones and "Delete set" in zones
    assert "Space" in zones and "Esc" in zones and "Set to view or edit" in zones
    for kept in ('id="qZone"', 'id="qZones"', 'id="qGrid"', 'id="qGridImg"'):      # still in Lights
        assert kept in lights, kept
    # what the run button inspects: stated between the notice and the run buttons, never hidden
    block = html[html.index('id="qRunBlock"'):html.index('id="qStationExtras"')]
    assert block.index('id="qNotice"') < block.index('id="qZoneLine"') < block.index('id="qRunScope"') \
        < block.index('id="qMeasure"') < block.index('id="qStart"')
    assert not re.search(r'id="q(RunZones|ZoneLine|RunScope)"[^>]*hidden', html)
    for el in ('id="qInspected"', '<ul id="qZoneCounts"', 'id="qZoneWarn"'):
        assert el in html[html.index('class="qright"'):], el
    assert "empty = all selected zones, or the whole field when none is on" in html
    assert not re.search(r"\son[a-z]+\s*=", html), "inline event handler in quick.html"
    for word in ("paintZones", "showZonesOnPart", "loadZoneSets", "zoneScope", "renderZoneRows",
                 "deleteZoneSet", "'/api/zonesets'", "'/api/zones/selection'", "'zone_paint'",
                 "'zone_show'", "'zone_delete'", "selection@${"):
        assert word in js, word
    css = (STATIC / "quick.css").read_text(encoding="utf-8")
    for rule in (".qzrows", "overflow-y: auto", ".qrunblock", "position: sticky", ".qthumb", ".qask"):
        assert rule in css, rule
    # the run block stays at the bottom only on a screen that is high enough, and has a height
    # limit there (its texts scroll in their own boxes): on a low screen it would cover the
    # Scan zones panel. What the page scrolls to stops above it
    plain = css[:css.index("@media (min-height: 900px)")]
    tall = css[css.index("@media (min-height: 900px)"):]
    tall = tall[:tall.index("\n}") + 2]
    assert not re.search(r"\.qrunblock\s*\{[^}]*position:\s*sticky", plain)
    assert re.search(r"\.qrunblock\s*\{[^}]*position:\s*sticky;\s*bottom:\s*0", tall)
    for rule in (r"\.qrunblock \.qrunzones\s*\{[^}]*max-height:[^}]*overflow-y:\s*auto",
                 r"\.qrunblock \.qnotice\s*\{[^}]*max-height:[^}]*overflow-y:\s*auto",
                 r"html\s*\{[^}]*scroll-padding-bottom"):
        assert re.search(rule, tall), rule
    node = shutil.which("node")
    if node is not None:
        r = subprocess.run([node, "--check", str(STATIC / "quick.js")], capture_output=True,
                           text=True, timeout=60)
        assert r.returncode == 0, r.stderr


def test_paint_job_saves_a_zone_set_and_every_page_is_told_how_it_ended(web):
    c, app, lib = web
    status, d = c.get("/api/zonesets")
    assert status == 200 and d["sets"] == [] and d["geometry_id"] == GEO and d["preset"] == PRESET
    assert d["selection"]["empty"] and d["selection"]["label"] == "whole field" and d["others"] == []
    b = _state(c)
    assert b["painter"] is None and b["last_paint"] is None and b["zone_selection"]["state"] == "ok"
    # the Paint zones button: one job on the rig (MOCK: the built-in demo session is replayed)
    j = c.done("zone_paint", {"name": "ui_set", "select": True})
    r = j["result"]
    assert r["saved"] is True and r["outcome"] == "saved" and r["n_zones"] == 4
    assert "demo session" in r["note"] and r["on"] is True and r["replaced"] is None
    assert r["preview"] == f"zones/{GEO}/ui_set/preview.png"
    assert j["links"]["preview"] == f"zones/{GEO}/ui_set/camera.png"
    # ... and the bus state says so to every page (also one that was reloaded meanwhile)
    b = _state(c)
    lp = b["last_paint"]
    assert (lp["name"], lp["outcome"], lp["n_zones"], lp["on"], lp["replaced"]) == \
        ("ui_set", "saved", 4, True, None)
    assert lp["zones"][0] == {"id": "Z1", "name": "part"} and lp["photo"].endswith("camera.png")
    assert lp["time"] and b["painter"] is None
    assert b["zone_selection"]["label"] == "ui_set (4)" and b["zone_selection"]["text"] == "zoneset:ui_set"
    status, d = c.get("/api/zonesets")
    (z,) = d["sets"]
    assert z["name"] == "ui_set" and z["valid"] and z["n_zones"] == 4 and d["other_geometries"] == []
    assert z["on"] is True and z["ids_on"] is None and all(q["on"] for q in z["zones"])
    assert z["used_by"] == {"recipes": [], "stations": []} and z["usable"] is True
    assert [q["id"] for q in z["zones"]] == ["Z1", "Z2", "Z3", "Z4"] and z["zones"][0]["name"] == "part"
    assert z["preview"] == f"zones/{GEO}/ui_set/preview.png" and z["camera"].endswith("camera.png")
    for link in (z["preview"], z["camera"]):                # what the page's <img> loads
        status, raw, hdrs = c.request("GET", f"/files/{link}")
        assert status == 200 and hdrs.get("Content-Type") == "image/png" and raw[:4] == b"\x89PNG"
    # the same name again: refused (nothing is replaced unless asked) - and said as that
    again = c.job("zone_paint", {"name": "ui_set"})
    assert again["status"] == "error" and "exists" in again["error"]
    lp = _state(c)["last_paint"]
    assert lp["outcome"] == "refused" and "exists" in lp["reason"] and lp["name"] == "ui_set"
    # painted over (asked for): the old one to the trash, its switch kept
    script = [{"type": "click", "at": [300, 365]}, {"type": "key", "key": "delete"},
              {"type": "key", "key": "space"}]
    j = c.done("zone_paint", {"name": "ui_set", "source": "ui_set", "replace": True, "select": True,
                              "events": script, "polarity": "normal"})
    assert j["result"]["saved"] and j["result"]["n_zones"] == 3 and "note" not in j["result"]
    assert j["result"]["on"] is True and j["result"]["replaced"] == f"trash/zones/{GEO}/ui_set"
    assert (lib / "trash" / "zones" / GEO / "ui_set" / "zones.json").is_file()
    lp = _state(c)["last_paint"]
    assert lp["outcome"] == "saved" and lp["replaced"] == f"trash/zones/{GEO}/ui_set" and lp["n_zones"] == 3
    # a session that is not confirmed: the job is done, nothing is saved, the page says why
    j = c.done("zone_paint", {"name": "other", "polarity": "normal",
                              "events": [{"type": "key", "key": "escape"}]})
    assert j["result"]["saved"] is False and "nothing saved" in j["result"]["reason"]
    assert j["result"]["outcome"] == "cancelled"
    lp = _state(c)["last_paint"]
    assert (lp["name"], lp["outcome"], lp["zones"]) == ("other", "cancelled", [])
    j = c.done("zone_paint", {"name": "other", "polarity": "normal",
                              "events": [{"type": "drag", "from": [100, 100], "to": [300, 300]}]})
    assert j["result"]["outcome"] == "script_ended" and _state(c)["last_paint"]["outcome"] == "script_ended"
    assert [s["name"] for s in c.get("/api/zonesets")[1]["sets"]] == ["ui_set"]
    # bad input is refused
    assert c.post("/api/jobs", {"verb": "zone_paint", "args": {"name": "x", "colour": 1}})[0] == 400
    bad = c.job("zone_paint", {"name": "bad name"})
    assert bad["status"] == "error" and "name must be" in bad["error"]
    assert _state(c)["last_paint"]["outcome"] == "refused"
    bad = c.job("zone_paint", {"name": "x", "events": "space"})
    assert bad["status"] == "error" and "list of painter events" in bad["error"]


def test_selection_over_http_two_sets_one_off_then_one_zone_off(web):
    c, app, lib = web
    p = app.session.preset
    zstore.save(app.session.store, ZoneSet(
        name="tube", geometry_id=GEO, preset=p.name, mock=True,
        fingerprint=config.geometry_fingerprint(p),
        zones=[Zone.rect("Z1", 240, 620, 460, 800, name="tube"), Zone.rect("Z2", 560, 640, 740, 820),
               Zone.rect("Z3", 600, 480, 720, 580)]))
    st = c.get("/api/zones/selection")[1]["selection"]
    assert st["label"] == "ui_set (3)" and [r["set"] for r in st["sets"]] == ["tube", "ui_set"]
    # both on
    status, r = _select(c, [{"set": "ui_set", "on": True, "ids": None},
                            {"set": "tube", "on": True, "ids": None}], was=st["stamp"])
    assert status == 200 and r["accepted"] and r["message"] == ""
    assert r["selection"]["label"] == "tube (3), ui_set (3)" and r["bus"]["zone_selection"] == r["selection"]
    assert r["selection"]["by"] == "operator page"
    # one set off
    status, r = _select(c, [{"set": "ui_set", "on": False, "ids": None},
                            {"set": "tube", "on": True, "ids": None}], was=r["selection"]["stamp"])
    assert status == 200 and r["selection"]["label"] == "tube (3)" and r["selection"]["text"] == "zoneset:tube"
    # one zone off inside the set that is on
    status, r = _select(c, [{"set": "ui_set", "on": False, "ids": None},
                            {"set": "tube", "on": True, "ids": ["Z1", "Z2"]}],
                        was=r["selection"]["stamp"])
    assert status == 200 and r["selection"]["label"] == "tube (2 of 3)"
    assert r["selection"]["text"] == "zoneset:tube:Z1,Z2"
    good = r["selection"]
    assert _state(c)["zone_selection"]["label"] == "tube (2 of 3)"           # near the run button
    d = c.get("/api/zonesets")[1]
    by = {z["name"]: z for z in d["sets"]}
    assert by["tube"]["on"] and by["tube"]["ids_on"] == ["Z1", "Z2"] and not by["ui_set"]["on"]
    assert [z["on"] for z in by["tube"]["zones"]] == [True, True, False]
    # refused: an unknown set, an unknown zone, no zone on (400); a stale `was` (409). Each
    # answer carries the STORED selection: a page never keeps a switch the server refused
    for sets, code, msg in (
            ([{"set": "nope", "on": True, "ids": None}], 400, "no zone set 'nope'"),
            ([{"set": "tube", "on": True, "ids": ["Z9"]}], 400, "no zone 'Z9'"),
            ([{"set": "tube", "on": True, "ids": []}], 400, "no zone is switched on"),
            ([{"set": "tube", "on": "yes", "ids": None}], 400, "must be true or false")):
        status, r = _select(c, sets, was=good["stamp"])
        assert status == code and not r["accepted"] and msg in r["message"], (sets, r)
        assert r["selection"] == good and r["bus"]["zone_selection"] == good
    status, r = _select(c, [{"set": "tube", "on": False, "ids": None}], was="0123456789ab")
    assert status == 409 and "changed meanwhile" in r["message"] and r["selection"] == good
    assert c.post("/api/zones/selection", {})[0] == 400
    # the other forms of the verb, and who changed it
    status, r = c.post("/api/zones/selection", {"on": "ui_set", "source": "a script"})
    assert status == 200 and r["selection"]["label"] == "tube (2 of 3), ui_set (3)"
    assert r["selection"]["by"] == "a script"
    # a run that asks for "the selection as shown" with another stamp is refused at once
    assert c.post("/api/cmd/mode", {"mode": "measure"})[0] == 200
    try:
        status, r = c.post("/api/measure", {"what": "flatness", "zone": f"selection@{good['stamp']}"})
        assert status == 409 and "the zone selection changed (it is now: tube (2 of 3), ui_set (3))" in r["error"]
        status, r = c.post("/api/quick-check", {"check": "flatness", "threshold_mm": 0.5,
                                                "zone": "selection@0123456789ab"})
        assert status == 409                                 # (and Quick check mode is not on)
    finally:
        assert c.post("/api/cmd/mode", {"mode": "station"})[0] == 200
    # delete: only a set that is switched off, with the operator's confirmation, to the trash
    bad = c.job("zone_delete", {"name": "tube", "reason": "test", "confirm": True})
    assert bad["status"] == "error" and "switched on in the zone selection" in bad["error"]
    bad = c.job("zone_delete", {"name": "tube", "reason": "test"})
    assert bad["status"] == "error" and "confirm" in bad["error"]
    _select(c, [{"set": "tube", "on": False, "ids": ["Z1", "Z2"]}, {"set": "ui_set", "on": True, "ids": None}])
    j = c.done("zone_delete", {"name": "tube", "reason": "test", "confirm": True})
    assert j["result"]["trashed"].startswith(f"trash/zones/{GEO}/tube") and not j["result"]["was_on"]
    st = _state(c)["zone_selection"]
    assert st["label"] == "ui_set (3)" and [r_["set"] for r_ in st["sets"]] == ["ui_set"]


def test_selection_survives_a_server_restart(web, tmp_path):
    c, app, lib = web
    before = _state(c)["zone_selection"]
    assert before["label"] == "ui_set (3)" and before["time"]
    # a second server on the same library (the first one's process is not needed for it)
    saved = api._session
    httpd, app2 = _serve(lib, tmp_path / "stations2")
    try:
        after = Client(app2.port).get("/api/cmd/state")[1]["zone_selection"]
        assert (after["label"], after["text"], after["stamp"], after["time"], after["by"]) == \
            (before["label"], before["text"], before["stamp"], before["time"], before["by"])
    finally:
        httpd.shutdown()
        httpd.server_close()
        app2.close()
        api._session = saved


def test_the_bus_snapshot_does_not_read_the_selection_on_every_poll(web, monkeypatch):
    c, app, lib = web
    calls = []
    real = api.zone_selection

    def counted(preset=None):
        calls.append(1)
        return real(preset)
    monkeypatch.setattr(api, "zone_selection", counted)
    app._zone_cache.clear()
    first = _state(c)["zone_selection"]
    for _ in range(5):
        assert _state(c)["zone_selection"] == first
    assert len(calls) == 1                                   # five polls, no file read
    # a change is seen at once (the selection file changed): never a stale answer
    status, r = _select(c, [{"set": "ui_set", "on": True, "ids": ["Z2"]}], was=first["stamp"])
    assert status == 200 and _state(c)["zone_selection"]["label"] == "ui_set (1 of 3)"
    n = len(calls)
    _state(c)
    assert len(calls) == n
    # a selection that cannot be read is NOT KNOWN in the snapshot, never "nothing is on"
    f = sel.file_of(app.session.store, GEO)
    keep = f.read_bytes()
    try:
        with open(f, "ab") as fh:
            fh.write(b'{"schema": "zone-selection-1", "sets": [{"se')
        st = _state(c)["zone_selection"]
        assert st["state"] == "unknown" and st["label"] == "NOT KNOWN" and not st["ok"] and not st["empty"]
    finally:
        f.write_bytes(keep)
    assert _state(c)["zone_selection"]["label"] == "ui_set (1 of 3)"


def test_zone_show_and_delete_jobs(web):
    c, app, lib = web
    j = c.done("zone_show", {"zones": "ui_set:Z2", "hold_s": 0})
    r = j["result"]
    assert r["all_seen"] and [z["id"] for z in r["zones"]] == ["Z2"] and r["zones"][0]["seen"]
    assert r["zones"][0]["key"] == "ui_set/Z2" and r["zoneset"] == "ui_set"
    assert j["links"]["preview"].endswith("_overlay.png")
    status, raw, hdrs = c.request("GET", f"/files/{j['links']['preview']}")
    assert status == 200 and raw[:4] == b"\x89PNG"
    # Show selected zones: the page asks for the selection exactly as it showed it
    st = _state(c)["zone_selection"]
    j = c.done("zone_show", {"zones": f"selection@{st['stamp']}", "hold_s": 0})
    assert [z["key"] for z in j["result"]["zones"]] == ["ui_set/Z2"] and j["result"]["spec"] == st["text"]
    stale = c.job("zone_show", {"zones": "selection@0123456789ab", "hold_s": 0})
    assert stale["status"] == "error" and "changed since it was shown" in stale["error"]
    bad = c.job("zone_show", {"zones": "ui_set", "hold_s": 9999})
    assert bad["status"] == "error" and "hold_s" in bad["error"]
    bad = c.job("zone_show", {"zones": "nope"})
    assert bad["status"] == "error" and "no zone set 'nope'" in bad["error"]
    # delete: only with the operator's confirmation, a set that is off, and to the trash
    bad = c.job("zone_delete", {"name": "ui_set", "reason": "test"})
    assert bad["status"] == "error" and "confirm" in bad["error"]
    bad = c.job("zone_delete", {"name": "ui_set", "reason": "test", "confirm": True})
    assert bad["status"] == "error" and "switch it off first" in bad["error"]
    assert _select(c, [{"set": "ui_set", "on": False, "ids": None}])[0] == 200
    j = c.done("zone_delete", {"name": "ui_set", "reason": "test", "confirm": True})
    assert j["result"]["trashed"].startswith(f"trash/zones/{GEO}/ui_set")
    d = c.get("/api/zonesets")[1]
    assert d["sets"] == [] and d["selection"]["empty"] and d["selection"]["sets"] == []


def test_a_queued_painter_dropped_by_abort_is_said_as_that(web):
    c, app, lib = web
    hold = threading.Event()
    th = threading.Thread(target=app.call, args=(lambda: hold.wait(30), 60), daemon=True)
    th.start()                                               # the worker is busy: jobs stay queued
    t0 = time.monotonic()
    while app._quick:
        assert time.monotonic() - t0 < 30
        time.sleep(0.01)
    try:
        status, j = c.post("/api/jobs", {"verb": "zone_paint", "args": {"name": "late"}})
        assert status == 200 and j["status"] == "queued"
        b = _state(c)
        assert b["painter"]["state"] == "queued" and b["painter"]["name"] == "late"
        status, r = c.post("/api/cmd/abort", {})
        assert status == 200, r
    finally:
        hold.set()
    b = _state(c)
    assert b["painter"] is None and b["last_paint"]["outcome"] == "dropped"
    assert b["last_paint"]["name"] == "late" and c.get("/api/zonesets")[1]["sets"] == []


def test_a_selection_change_is_never_left_in_the_queue_behind_another_job(web, monkeypatch):
    """A switch pressed while another job holds the worker is refused within seconds, and
    "not changed" stays true: the change is not left queued to be applied later, with nobody
    looking. A job ahead that ends at once (the preset load of a configuration just
    selected) is waited for."""
    c, app, lib = web
    p = app.session.preset
    zstore.save(app.session.store, ZoneSet(
        name="held", geometry_id=GEO, preset=p.name, mock=True,
        fingerprint=config.geometry_fingerprint(p), zones=[Zone.rect("Z1", 100, 100, 300, 300)]))
    hold, entered = threading.Event(), threading.Event()

    def slow(a):                                             # any job of another page
        entered.set()
        assert hold.wait(60), "the fake job was never released"
        return []
    monkeypatch.setattr(app, "_v_list_library", slow, raising=False)
    want = [{"set": "held", "on": True, "ids": None}]
    try:
        status, j = c.post("/api/jobs", {"verb": "list_library", "args": {}})
        assert status == 200 and entered.wait(30)
        before = c.get("/api/zones/selection")[1]["selection"]
        assert before["empty"]
        t0 = time.monotonic()
        status, r = _select(c, want, was=before["stamp"])
        assert time.monotonic() - t0 < ui_server.ZONE_SELECT_GRACE_S + 2.0   # not after the job
        assert status == 409 and not r["accepted"], r
        assert f"another job holds the scanner ({j['id']} list_library)" in r["message"]
        assert r["selection"]["stamp"] == before["stamp"] and r["selection"]["empty"]
        waiting = [x for x in c.get("/api/jobs")[1]["jobs"]
                   if x["verb"] == "zone_select" and x["status"] in ("queued", "running")]
        assert waiting == []                                 # nothing left to be applied later
    finally:
        hold.set()
    t0 = time.monotonic()
    while c.get(f"/api/jobs/{j['id']}")[1]["status"] != "done":
        assert time.monotonic() - t0 < 30
        time.sleep(0.05)
    time.sleep(0.3)                                          # (a change left queued would land now)
    assert c.get("/api/zones/selection")[1]["selection"]["stamp"] == before["stamp"]
    # a job ahead that ends within a moment is waited for: the change is taken
    hold.clear()
    entered.clear()
    status, j = c.post("/api/jobs", {"verb": "list_library", "args": {}})
    assert status == 200 and entered.wait(30)
    threading.Timer(0.5, hold.set).start()
    status, r = _select(c, want, was=before["stamp"])
    assert status == 200 and r["accepted"] and r["selection"]["label"] == "held (1)", r
    assert _select(c, [{"set": "held", "on": False, "ids": None}])[0] == 200
    c.done("zone_delete", {"name": "held", "reason": "test cleanup", "confirm": True})


def test_a_set_saved_before_the_rig_close_failed_is_told_as_saved(web, monkeypatch):
    """The painter saves while it holds the rig. A camera / projector close error after that
    must not be told as "NOT SAVED: the painter did not start": the set is in the library."""
    from engine import session as engine_session
    c, app, lib = web
    real, fails = engine_session._release_rig, []

    def release(camera, projector, lock, primary):
        real(camera, projector, lock, primary)
        if primary is None and not fails:
            fails.append(1)
            raise RuntimeError("camera close failed")
    monkeypatch.setattr(engine_session, "_release_rig", release)
    j = c.done("zone_paint", {"name": "closefail", "select": True, "polarity": "normal", "events": [
        {"type": "drag", "from": [100, 100], "to": [300, 300]}, {"type": "key", "key": "space"}]})
    assert fails == [1]
    r = j["result"]
    assert r["saved"] is True and r["outcome"] == "saved" and r["on"] is True
    assert r["rig_error"] == "RuntimeError: camera close failed"
    b = _state(c)
    lp = b["last_paint"]
    assert (lp["name"], lp["outcome"], lp["on"], lp["n_zones"]) == ("closefail", "saved", True, 1)
    assert lp["rig_error"] == "RuntimeError: camera close failed"
    assert any("rig was not given back cleanly" in w for w in lp["warnings"])
    # where and what: a page tells from this whether the message is about the set it lists
    (z,) = c.get("/api/zonesets")[1]["sets"]
    assert z["name"] == "closefail" and z["on"] is True
    assert (lp["geometry_id"], lp["preset"], lp["created"]) == (GEO, PRESET, z["created"])
    assert b["zone_selection"]["label"] == "closefail (1)"
    assert _select(c, [{"set": "closefail", "on": False, "ids": None}])[0] == 200
    c.done("zone_delete", {"name": "closefail", "reason": "test cleanup", "confirm": True})
    # a session that saved nothing says where it was, too
    bad = c.job("zone_paint", {"name": "bad name"})
    lp = _state(c)["last_paint"]
    assert bad["status"] == "error" and lp["outcome"] == "refused"
    assert (lp["geometry_id"], lp["preset"], lp["created"]) == (GEO, PRESET, None)


def test_real_server_never_takes_a_scripted_session():
    app = types.SimpleNamespace(mock=False)
    with pytest.raises(ui_server.ArgError, match="MOCK ONLY"):
        ui_server.App._zone_paint(app, {"name": "x", "events": []})


# ------------------------------------------------ quick.js logic (node, fake DOM, fake server)

_HARNESS = r"""
const fs = require('fs'), vm = require('vm');
class El {
  constructor(tag, id) { this.tagName = String(tag).toUpperCase(); this.id = id; this.children = [];
    this.dataset = {}; this.className = ''; this._text = ''; this._value = ''; this.checked = false;
    this.disabled = false; this.hidden = false; this._sel = undefined; this.src = ''; this.open = false;
    this.parent = null; this.on = {}; this.classes = {}; this.title = ''; this.attrs = {};
    this.classList = { toggle: (c, v) => { this.classes[c] = v === undefined ? !this.classes[c] : !!v; },
                       add() {}, remove() {} }; }
  set textContent(t) {
    // as a browser: an element that is removed loses the focus, which falls to the page
    const a = doc.activeElement;
    if (a && a !== this && this.contains(a)) doc.activeElement = doc.body;
    for (const c of this.children) c.parent = null;
    this._text = String(t); this.children = [];
  }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(''); }
  appendChild(c) { this.children.push(c); c.parent = this; return c; }
  contains(x) { for (let e = x; e; e = e.parent) if (e === this) return true; return false; }
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
  addEventListener(ev, fn) { this.on[ev] = fn; }
  focus() { if (this.disabled) return; focused.push(this.id); doc.activeElement = this; }
  scrollIntoView() { scrolled.push(this.id); }
  matches(sel) {
    // a tag name, or [attribute] (as the page's "[tabindex]")
    return sel.split(',').map((s) => s.trim()).some((s) => (/^\[[a-z]+\]$/.test(s)
      ? s.slice(1, -1) in this.attrs : this.tagName === s.toUpperCase()));
  }
  closest(sel) { for (let e = this; e; e = e.parent) if (e.matches(sel)) return e; return null; }
}
const focused = [];
const scrolled = [];
// the page's document: the element that has the keyboard focus (a browser's activeElement)
const doc = { addEventListener() {}, querySelector: (s) => el(s.slice(1)), querySelectorAll: () => [],
              createElement: (t) => new El(t), body: new El('body'), activeElement: null };
doc.activeElement = doc.body;
const TAGS = { qZoneSet: 'select', qZoneName: 'input', qZoneReason: 'input', qPaint: 'button',
  qZoneShow: 'button', qZoneDelete: 'button', qZoneAskYes: 'button', qZoneAskNo: 'button',
  qZoneImg: 'img', qMeasWhat: 'select', qMeasScan: 'select', qRigPreset: 'select',
  qQuickCheck: 'select', qQuickRecipe: 'select', qStation: 'select', qProfile: 'select',
  qStart: 'button', qMeasure: 'button', qConfirm: 'input', qMeasRoi: 'input', qQuickRoi: 'input',
  qMeasP2: 'input', qQuickThreshold: 'input' };
const byId = {};
const el = (id) => byId[id] || (byId[id] = new El(TAGS[id] || 'div', id));
const sent = [];
const srv = { rev: 0, rows: [], n: 0, jobs: {}, refuseSel: null, others: [],
  paint: null, show: null, del: { status: 'done', result: { trashed: 'trash/zones/g/x' } } };
function mkSel(extra) {
  // the server's zone selection (engine/zones/selection.status), from srv.rows
  const sets = srv.rows.map((r) => {
    const zones = (r.zones || []).map(([id, name, on]) => ({ id, name, on: on !== false }));
    const nOn = zones.filter((z) => z.on).length;
    return { set: r.set, on: !!r.on, ids: nOn === zones.length ? null : zones.filter((z) => z.on).map((z) => z.id),
      n_on: nOn, n_zones: zones.length, zones, found: r.found !== false,
      usable: r.usable !== false && r.found !== false, error: r.error || null,
      created: r.created || '2026-01-02T03:04:05', preset: 'bench', rig_changes: r.rig_changes || [] };
  });
  const on = sets.filter((s) => s.on);
  const problems = on.filter((s) => !s.found).map((s) => ({ set: s.set, zone: null, reason: 'missing',
    message: `zone set '${s.set}' is switched on but is not in the library any more (deleted or renamed)` }));
  const label = on.map((s) => (!s.found ? `${s.set} (NOT FOUND)`
    : `${s.set} (${s.n_on === s.n_zones ? s.n_on : `${s.n_on} of ${s.n_zones}`}${s.rig_changes.length ? ', CHECK' : ''})`)).join(', ') || 'whole field';
  srv.rev += 1;
  return { geometry_id: 'g', preset: 'bench', state: problems.length ? 'problem' : 'ok',
    ok: !problems.length, empty: !on.length, label, stamp: `s${srv.rev}`, problems, sets,
    text: on.length ? 'zoneset:' + on.map((s) => s.set + (s.ids ? ':' + s.ids.join(',') : '')).join('+') : '',
    n_sets_on: on.length, n_zones_on: on.reduce((a, s) => a + s.n_on, 0),
    check: on.filter((s) => s.rig_changes.length).map((s) => `${s.set}: ${s.rig_changes.join('; ')}`),
    time: '2026-01-02T09:14:02', by: 'operator page', ...(extra || {}) };
}
srv.bus = { state: 'READY', operating_mode: 'measure', station_config: 'own.json',
  station_configs: ['own.json', 'sel.json', 'rec.json', 'flow.json', 'cut.json'], unusable: {},
  signals: {}, jobs: {}, mode_jobs: [], active_preset: 'bench', run_preset: 'bench',
  rig: { ok: true, mount: 'fixed', message: '' }, painter: null, last_paint: null,
  zone_selection: mkSel() };
const stations = [
  { config: 'own.json', station_id: 'own', part_label: 'p', job_type: 'metric', preset: null, zone: null, zone_from_selection: false },
  { config: 'sel.json', station_id: 'sel', part_label: 'p', job_type: 'metric', preset: 'bench', zone: 'selection', zone_from_selection: true },
  { config: 'rec.json', station_id: 'rec', part_label: 'p', job_type: 'recipe', recipe: 'lid', zone: null, zones: ['a'], checks: ['presence'] },
  { config: 'flow.json', station_id: 'flow', part_label: 'p', job_type: 'flow', flow: [], profiles: [] },
  { config: 'cut.json', station_id: 'cut', part_label: 'p', job_type: 'metric', preset: null, zone: 'C4:D5', zone_from_selection: false }];
const setSel = (extra) => { srv.bus = { ...srv.bus, zone_selection: mkSel(extra) }; };
const sandbox = {
  console, setTimeout, clearTimeout, setInterval, clearInterval, encodeURIComponent,
  document: doc,
  fetch: async (path, opt) => {
    const body = opt && opt.body ? JSON.parse(opt.body) : undefined;
    sent.push({ path, body });
    let data = {}, status = 200;
    if (srv.down) throw new Error('network');
    if (srv.slow && path === srv.slow) await new Promise((r) => setTimeout(r, 40));
    if (path === '/api/cmd/state') data = { ...srv.bus, mode: 'real' };
    else if (path === '/api/stations') data = { mode: 'real', stations };
    else if (path === '/api/modes') data = { mode: srv.bus.operating_mode, loaded_preset: 'bench',
      presets: [{ name: 'bench', mount: 'fixed', calibrated: true }], recipes: ['lid'] };
    else if (path === '/api/zonesets') data = { geometry_id: 'g', preset: 'bench', others: srv.others,
      selection: srv.bus.zone_selection,
      sets: srv.rows.filter((r) => r.found !== false).map((r) => ({ name: r.set, n_zones: (r.zones || []).length,
        valid: r.usable !== false, error: r.error || null, created: r.created || '2026-01-02T03:04:05',
        zones: (r.zones || []).map(([id, name]) => ({ id, name })), rig_changes: r.rig_changes || [],
        used_by: r.used_by || { recipes: [], stations: [] },
        preview: `zones/g/${r.set}/preview.png`, camera: r.photo === false ? null : `zones/g/${r.set}/camera.png` })) };
    else if (path === '/api/zones/selection') {
      if (srv.refuseSel) { status = srv.refuseSel[0]; data = { accepted: false, message: srv.refuseSel[1],
        selection: srv.bus.zone_selection, bus: { ...srv.bus } }; }
      else {
        for (const e of body.sets) {
          const r = srv.rows.find((x) => x.set === e.set);
          if (!r) continue;
          r.on = e.on;
          r.zones = r.zones.map(([id, name]) => [id, name, e.ids === null || e.ids.includes(id)]);
        }
        srv.rows = srv.rows.filter((r) => r.found !== false || r.on);
        setSel();
        data = { accepted: true, message: '', selection: srv.bus.zone_selection, bus: { ...srv.bus } };
      }
    } else if (path === '/api/jobs') {
      srv.n += 1;
      if (srv.jobRefused) { status = 409; data = { error: srv.jobRefused }; }
      else if (body.verb === 'zone_paint') {
        data = { id: `j${srv.n}`, status: 'running', created: '2026-01-02T12:00:10' };
        srv.bus = { ...srv.bus, painter: { running: true, verb: 'zone_paint', name: body.args.name,
          state: 'running', since: '2026-01-02T12:00:10', job: `j${srv.n}` } };
      } else data = { id: `j${srv.n}`, ...(body.verb === 'zone_show' ? srv.show : srv.del) };
    } else if (path.startsWith('/api/jobs/')) data = srv.jobs[path.split('/').pop()] || { status: 'running' };
    else if (path === '/api/measure' || path === '/api/quick-check') { srv.n += 1; data = { id: `j${srv.n}`, status: 'queued', bus: { ...srv.bus } }; }
    else if (path.startsWith('/api/cmd/')) {
      const name = path.split('/').pop();
      if (name === 'mode') srv.bus = { ...srv.bus, operating_mode: body.mode };
      if (name === 'config') srv.bus = { ...srv.bus, station_config: body.config };
      data = { accepted: true, command: name.toUpperCase(), message: 'ok', bus: { ...srv.bus }, jobs: {} };
    }
    return { ok: status < 400, status, json: async () => data };
  },
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), sandbox, { filename: 'quick.js' });
const run = (code) => vm.runInContext(code, sandbox);
const jobs = () => sent.filter((s) => s.path === '/api/jobs').map((s) => s.body);
const posts = (p) => sent.filter((s) => s.path === p && s.body !== undefined).map((s) => s.body);
const opt = (id, v) => { const o = new El('option'); o.value = v; el(id).appendChild(o); };
const key = (k, target) => {
  const n = sent.length;
  let prevented = false;
  run('onKey')({ key: k, target: target || sandbox.document.body, ctrlKey: false, altKey: false,
                 metaKey: false, preventDefault() { prevented = true; } });
  return { prevented, n };
};
// a key pressed in a browser goes to the element that has the focus
const keyAt = (k) => key(k, doc.activeElement || doc.body);
// the page's structure where the focus rules read it (quick.html): the answers of the
// question are inside it; the Scan zones panel can take the focus (tabindex)
for (const id of ['qZoneAskText', 'qZoneReasonRow', 'qZoneAskYes', 'qZoneAskNo']) el('qZoneAsk').appendChild(el(id));
el('qZoneReasonRow').appendChild(el('qZoneReason'));
el('qScanZones').attrs.tabindex = '-1';
// the switch of a set (or of one zone of it) as it is drawn now
const sw = (set, zone) => {
  const row = el('qZoneRows').children.find((r) => r.dataset.set === set);
  if (!row) return null;
  return zone ? row.children[1].children.map((z) => z.children[0]).find((c) => c.dataset.zone === zone)
    : row.children[0].children[0];
};
// Space on a focused switch, as a browser does it: the box flips, then `change` is fired
const press = (c) => { c.checked = !c.checked; run('onZoneRowChange')({ target: c }); };
const active = () => { const a = doc.activeElement; return a === doc.body ? 'BODY' : (a.id || `${a.tagName}:${a.dataset.set || ''}/${a.dataset.zone || ''}`); };
const settle = () => new Promise((r) => setTimeout(r, 25));
const line = () => ({ line: el('qZoneLine').textContent, scope: el('qRunScope').textContent,
                      warn: !!el('qZoneLine').classes.qwarn });
const rowsNow = () => el('qZoneRows').children.map((r) => ({
  set: r.dataset.set, cls: r.className, text: r.textContent,
  on: r.children[0].children[0].checked, off: r.children[0].children[0].disabled,
  zones: r.children[1].children.map((z) => [z.children[0].dataset.zone, z.children[0].checked, z.children[0].disabled]) }));
const mode = async (m) => { await run(`selectMode('${m}')`); await settle(); };
// the real rig: the operator confirms the setup (another selection clears it again)
const confirm = () => { el('qConfirm').checked = true; run('refreshButtons()'); };
const poll = () => run('pollOnce()');
const out = {};
(async () => {
  for (const v of ['flatness', 'step', 'distance', 'dims', 'cylinder', 'endface', 'deviation']) opt('qMeasWhat', v);
  for (const v of ['metric', 'fast']) opt('qMeasScan', v);
  for (const v of ['flatness', 'golden', 'recipe']) opt('qQuickCheck', v);
  opt('qQuickRecipe', 'lid');
  el('qMeasRepeats').value = '1';
  el('qQuickThreshold').value = '0.5';
  el('qConfirm').checked = true;
  await run('loadStations()');
  await run('loadZoneSets()');

  // ---------- nothing painted yet
  out.empty = { geo: el('qZoneGeo').textContent, rows: rowsNow(), ...line(),
                options: el('qZoneSet').options.map((o) => o.textContent), img: el('qZoneImg').hidden };
  srv.others = [{ geometry_id: 'h', n_sets: 2, presets: ['cart'] }];
  await run('loadZoneSets()');
  out.emptyElsewhere = el('qZoneGeo').textContent;
  srv.others = [];
  // Paint zones without a name: asked for one, nothing starts
  await run('paintZones()');
  out.noName = { msg: el('qZoneResult').textContent, jobs: jobs().length, focus: focused.slice(-1)[0] };

  // ---------- a new name: ONE painter job; the page then follows the bus state
  el('qZoneName').value = 'cells';
  await run('paintZones()');
  out.paintNew = { job: jobs().pop(), msg: el('qZoneResult').textContent,
                   paint: el('qPaint').disabled, measure: el('qMeasure').disabled };
  // the painter holds the rig: the run buttons are off and every key but Esc is ignored
  run('refreshButtons()');
  out.painterOn = { measure: el('qMeasure').disabled, start: el('qStart').disabled, paint: el('qPaint').disabled,
                    show: el('qZoneShow').disabled, del: el('qZoneDelete').disabled, abort: el('qAbort').disabled };
  out.painterKeys = {};
  for (const k of ['m', ' ', 'Enter', 'h', 'p', 'n', '1', 't', 'a', 'k']) {
    const r = key(k); await settle();
    out.painterKeys[k] = sent.slice(r.n).map((s) => s.path);
  }
  const esc = key('Escape'); await settle();
  out.painterEsc = sent.slice(esc.n).map((s) => s.path);
  // contact is lost while painting: said as NOT KNOWN, never as saved or not saved
  srv.down = true;
  run('contactLost()');
  out.lost = el('qZoneResult').textContent;
  srv.down = false;
  // it ends SAVED: the set is listed, switched on, viewed, its photo shown - and what the run
  // button does with it is said in the same breath (Measure mode here)
  srv.rows = [{ set: 'cells', on: true, zones: [['Z1', 'part'], ['Z2', 'part']] }];
  setSel();
  const saved = { name: 'cells', outcome: 'saved', time: '2026-01-02T12:01:07', n_zones: 2, on: true,
    replaced: null, switches_reset: false, zones: [{ id: 'Z1', name: 'part' }, { id: 'Z2', name: 'part' }],
    photo: 'zones/g/cells/camera.png', preview: 'zones/g/cells/preview.png', selection_note: 'x',
    geometry_id: 'g', preset: 'bench', created: '2026-01-02T03:04:05', rig_error: null };
  srv.bus = { ...srv.bus, painter: null, last_paint: saved };
  const nScroll = scrolled.length;
  await poll(); await settle();
  out.saved = { msg: el('qZoneResult').textContent, msgWarn: !!el('qZoneResult').classes.qwarn,
                picked: el('qZoneSet').value, img: el('qZoneImg').src, imgHidden: el('qZoneImg').hidden,
                typed: el('qZoneName').value, rows: rowsNow(), ...line(),
                info: el('qZoneInfo').textContent, measure: el('qMeasure').disabled,
                revealed: scrolled.slice(nScroll) };
  // the same last_paint on a page that was just loaded: said again, not lost - as HISTORY:
  // with its date, and with what is true of the set NOW (never what was so when it was saved)
  const cells2 = (more) => [{ set: 'cells', on: true, zones: [['Z1', 'part'], ['Z2', 'part']], ...(more || {}) }];
  const reload = async (rows, extra) => {
    if (rows) { srv.rows = rows; setSel(extra); }
    run('state.paintKey = null; state.zoneMsg = null');
    await poll(); await settle();
    return el('qZoneResult').textContent;
  };
  const nScroll2 = scrolled.length;
  out.savedAfterReload = await reload();
  out.reloadRevealed = scrolled.slice(nScroll2);
  out.history = {
    oneZone: await reload(cells2({ zones: [['Z1', 'part'], ['Z2', 'part', false]] })),
    off: await reload(cells2({ on: false })),
    repainted: await reload(cells2({ created: '2026-01-03T10:00:00' })),
    deleted: await reload([]),
    elsewhere: await reload(cells2(), { geometry_id: 'h', preset: 'cart' }),
    scope: el('qRunScope').textContent,
  };
  await reload(cells2());
  // every other end, by its outcome; a saved set's switch is told from the stored selection
  out.ends = {};
  for (const [k, lp, rows] of [
    ['cancelled', { outcome: 'cancelled' }], ['aborted', { outcome: 'aborted' }],
    ['idle_timeout', { outcome: 'idle_timeout', idle_timeout_s: 600 }], ['script_ended', { outcome: 'script_ended' }],
    ['refused', { outcome: 'refused', reason: "ZoneError: zone set 'cells' exists at geometry g" }],
    ['dropped', { outcome: 'dropped' }],
    ['replaced', { ...saved, replaced: 'trash/zones/g/cells', switches_reset: true, n_zones: 3,
                   zones: [...saved.zones, { id: 'Z3', name: '' }] },
      cells2({ zones: [['Z1', 'part'], ['Z2', 'part'], ['Z3', '']] })],
    ['off', { ...saved, on: false }, cells2({ on: false })],
    ['offReplaced', { ...saved, on: false, replaced: 'trash/zones/g/cells' }, cells2({ on: false })],
    ['noPhoto', { ...saved, photo: null }],
    ['rigError', { ...saved, rig_error: 'RuntimeError: camera close failed' }],
    // changed by somebody since the save: the message says what is so now
    ['offSince', saved, cells2({ on: false })],
    ['zoneOffSince', saved, cells2({ zones: [['Z1', 'part'], ['Z2', 'part', false]] })]]) {
    srv.rows = rows || cells2();
    setSel();
    const mm = String(Object.keys(out.ends).length).padStart(2, '0');
    srv.bus = { ...srv.bus, last_paint: { name: 'cells', ...lp, time: `2026-01-02T12:${mm}:59` } };
    await poll();
    out.ends[k] = { msg: el('qZoneResult').textContent, warn: !!el('qZoneResult').classes.qwarn };
  }
  // a painter that is still queued behind another job is not announced as open
  srv.rows = cells2();
  setSel();
  srv.bus = { ...srv.bus, painter: { running: true, verb: 'zone_paint', name: 'late', state: 'queued',
    since: '2026-01-02T12:30:00', job: 'j99' } };
  await poll();
  out.queued = { msg: el('qZoneResult').textContent, measure: el('qMeasure').disabled };
  srv.bus = { ...srv.bus, painter: null };
  await poll();
  await settle();

  // ---------- painting over an existing set asks first; the job goes only after Yes
  srv.rows[0].used_by = { recipes: ['lid'], stations: ['sel.json'] };
  await run("loadZoneSets('cells')");
  const nJobs = jobs().length;
  el('qZoneName').value = '';
  await run('paintZones()');                               // nothing typed: the viewed set
  out.askPaint = { open: !el('qZoneAsk').hidden, text: el('qZoneAskText').textContent,
                   yes: el('qZoneAskYes').textContent, reason: !el('qZoneReasonRow').hidden,
                   jobs: jobs().length - nJobs, focus: focused.slice(-1)[0] };
  // while the question is open the keys start nothing; Esc closes it (and is still ABORT)
  out.askKeys = {};
  for (const k of ['m', ' ', 'Enter', 'h']) {
    const r = key(k); await settle();
    out.askKeys[k] = sent.slice(r.n).map((s) => s.path);
  }
  const askEsc = keyAt('Escape'); await settle();            // (the focus is on Cancel)
  out.askEsc = { open: !el('qZoneAsk').hidden, sent: sent.slice(askEsc.n).map((s) => s.path),
                 msg: el('qZoneResult').textContent, jobs: jobs().length - nJobs,
                 focus: active() };                          // back on the button that asked
  await run('paintZones()');
  await run('confirmAsk()');
  out.paintOver = { job: jobs().pop(), open: !el('qZoneAsk').hidden, focus: active() };
  srv.bus = { ...srv.bus, painter: null };
  await poll();
  // typed = an existing name: asked too (never replaced because of a typed name)
  el('qZoneName').value = 'cells';
  const nJobs2 = jobs().length;
  await run('paintZones()');
  out.askTyped = { open: !el('qZoneAsk').hidden, jobs: jobs().length - nJobs2, focus: active() };
  run("closeAsk(); zoneSay('Cancelled: nothing was changed.')");   // (the Cancel button)
  out.askCancel = { open: !el('qZoneAsk').hidden, focus: active() };
  const nCancel = sent.length;
  keyAt(' '); await settle();                                // Space is Paint zones' now: no run
  out.askCancel.sent = sent.slice(nCancel).map((s) => s.path).filter((p) => p !== '/api/cmd/state');
  el('qZoneName').value = '';

  // ---------- two sets: switches, the selection line, the row order
  srv.rows = [{ set: 'cells', on: true, zones: [['Z1', 'part'], ['Z2', 'part']] },
              { set: 'tube', on: false, zones: [['Z1', 'tube'], ['Z2', ''], ['Z3', '']] },
              { set: 'aaa', on: false, zones: [['Z1', '']] }];
  setSel();
  await run('loadZoneSets()');
  await poll();
  out.two = { rows: rowsNow(), ...line(), geo: el('qZoneGeo').textContent };
  await run("toggleSet('tube', true)");
  out.tubeOn = { post: posts('/api/zones/selection').pop(), rows: rowsNow().map((r) => [r.set, r.on]),
                 ...line(), msg: el('qZoneResult').textContent, confirm: el('qConfirm').checked };
  await run("toggleZone('tube', 'Z3', false)");
  out.zoneOff = { post: posts('/api/zones/selection').pop(), ...line(), msg: el('qZoneResult').textContent };
  // the switch listener: the checkbox that changed says which set / zone
  run('onZoneRowChange')({ target: { dataset: { set: 'cells' }, checked: false } });
  await settle();
  out.cellsOff = { post: posts('/api/zones/selection').pop(), ...line() };
  // switching off the last zone of a set that is on switches the set off
  await run("toggleZone('tube', 'Z1', false)");
  await run("toggleZone('tube', 'Z2', false)");
  out.lastZone = { post: posts('/api/zones/selection').pop(), ...line(), msg: el('qZoneResult').textContent };
  // a refused switch is redrawn from what the server stores, and the reason is said
  srv.refuseSel = [409, 'zone_select refused while the station is ARMED: the zones a station inspects are fixed'];
  await run("toggleSet('cells', true)");
  out.refused = { rows: rowsNow().map((r) => [r.set, r.on]), msg: el('qZoneResult').textContent,
                  msgWarn: !!el('qZoneResult').classes.qwarn, ...line() };
  srv.refuseSel = [400, "no zone set 'cells' at geometry g"];
  await run("toggleSet('cells', true)");
  out.refused400 = { rows: rowsNow().map((r) => [r.set, r.on]), msg: el('qZoneResult').textContent };
  srv.refuseSel = null;

  // ---------- the keyboard: a switch keeps the focus when the rows are redrawn (left on the
  // page, the next Space - meant for the switch - would be the run button's)
  srv.rows = [{ set: 'cells', on: true, zones: [['Z1', 'part'], ['Z2', 'part']] },
              { set: 'tube', on: true, zones: [['Z1', 'tube'], ['Z2', '']] }];
  setSel();
  await run('loadZoneSets()'); await poll();
  const kb = {};
  const first = sw('tube');
  first.focus();
  kb.start = active();
  const nMeas0 = posts('/api/measure').length;
  const nSel0 = posts('/api/zones/selection').length;
  srv.slow = '/api/zones/selection';
  press(first);                                            // Space on the focused switch
  // ... while the change is on its way: the page says so, a second Space / Enter starts
  // nothing, a second change is not taken
  const other = sw('cells', 'Z2');
  press(other);
  kb.busy = { msg: el('qZoneResult').textContent, other: other.checked, space: key(' ').prevented,
              enter: key('Enter').prevented, posts: posts('/api/zones/selection').length - nSel0,
              measure: posts('/api/measure').length - nMeas0 };
  await settle(); await settle(); await settle();
  srv.slow = null;
  kb.after = { active: active(), redrawn: doc.activeElement !== first, on: sw('tube').checked,
               msg: el('qZoneResult').textContent, line: el('qZoneLine').textContent };
  keyAt(' '); await settle();                              // the key goes to the switch ...
  press(doc.activeElement);                                // ... which flips again
  await settle(); await settle();
  kb.again = { active: active(), on: sw('tube').checked, measure: posts('/api/measure').length - nMeas0,
               posts: posts('/api/zones/selection').length - nSel0 };
  // the same for one zone's switch
  sw('tube', 'Z2').focus();
  press(doc.activeElement);
  await settle(); await settle();
  kb.zone = { active: active(), on: sw('tube', 'Z2').checked, line: el('qZoneLine').textContent };
  // a switch that cannot be pressed any more (the station was ARMED meanwhile): the focus
  // goes to the Scan zones panel, where Space / Enter start nothing
  srv.bus = { ...srv.bus, state: 'ARMED' };
  await poll();
  kb.armed = { active: active(), off: sw('tube', 'Z2').disabled };
  const nArmed = sent.length;
  keyAt(' '); keyAt('Enter'); await settle();
  kb.armed.sent = sent.slice(nArmed).map((s) => s.path).filter((p) => p !== '/api/cmd/state');
  srv.bus = { ...srv.bus, state: 'READY' };
  await poll();
  // a setup job of this page (Project grid, Signal check) holds the scanner: the switches,
  // Paint and Delete are off meanwhile (the server would refuse the change)
  run('state.setupBusy = true; refreshButtons(); renderZones()');
  kb.setup = { rows: rowsNow().map((r) => [r.off, r.zones.map((z) => z[2])]), paint: el('qPaint').disabled,
               del: el('qZoneDelete').disabled };
  run('state.setupBusy = false; refreshButtons(); renderZones()');
  kb.setupDone = rowsNow().map((r) => r.off);
  out.keyboard = kb;

  // ---------- what the run button does with the selection, by mode (one function)
  const scopes = {};
  const what = async (w) => { el('qMeasWhat').value = w; run('measFields()'); run('renderZones()'); run('refreshButtons()'); };
  srv.rows = [{ set: 'cells', on: true, zones: [['Z1', 'part'], ['Z2', 'part']] },
              { set: 'tube', on: true, zones: [['Z1', 'tube'], ['Z2', ''], ['Z3', '', false]] }];
  setSel();
  await run('loadZoneSets()'); await poll();
  const stamp = srv.bus.zone_selection.stamp;
  for (const w of ['flatness', 'cylinder', 'step', 'dims', 'deviation', 'endface', 'distance']) {
    await what(w);
    let args = null, err = null;
    try { args = run('measureArgs()'); } catch (e) { err = String(e.message); }
    scopes[`measure.${w}`] = { ...line(), zone: (args && args.zone) || null, err, off: el('qMeasure').disabled,
                              pane: el('qMeasZone').textContent };
  }
  el('qMeasP2').value = '10,20';
  await what('distance');
  scopes['measure.distance.p2'] = { ...line(), zone: run('measureArgs()').zone, off: el('qMeasure').disabled };
  el('qMeasP2').value = '';
  el('qMeasRoi').value = '10,20,300,400';
  await what('flatness');
  scopes['measure.flatness.region'] = line();
  el('qMeasRoi').value = '';
  await what('flatness');
  // one zone on
  srv.rows = [{ set: 'cells', on: true, zones: [['Z1', 'part'], ['Z2', 'part', false]] }];
  setSel(); await poll();
  for (const w of ['flatness', 'dims', 'step', 'distance']) {
    await what(w);
    scopes[`measure.one.${w}`] = { ...line(), off: el('qMeasure').disabled };
  }
  // nothing on
  srv.rows = [{ set: 'cells', on: false, zones: [['Z1', 'part'], ['Z2', 'part']] }];
  setSel(); await poll();
  await what('step');
  scopes['measure.none.step'] = { ...line(), off: el('qMeasure').disabled };
  await what('flatness');
  scopes['measure.none'] = { ...line(), zone: run('measureArgs()').zone, stamp: srv.bus.zone_selection.stamp,
                            off: el('qMeasure').disabled };
  // a set that is on needs a look (the rig changed since it was painted)
  srv.rows = [{ set: 'cells', on: true, zones: [['Z1', 'part'], ['Z2', 'part']],
                rig_changes: ['rig check REHOMED (2026-01-03T08:00:00)'] }];
  setSel(); await poll();
  scopes['measure.check'] = { ...line(), row: rowsNow()[0], off: el('qMeasure').disabled };
  // a set that is on and gone: the run would be refused, so MEASURE is off and says why
  srv.rows = [{ set: 'cells', on: true, zones: [['Z1', 'part'], ['Z2', 'part']] },
              { set: 'tube', on: true, found: false, zones: [], error: 'not in the library any more (deleted or renamed)' }];
  setSel(); await poll();
  let err = null;
  const nMeas = posts('/api/measure').length;
  try { run('measureArgs()'); } catch (e) { err = String(e.message); }
  await run('sendMeasure()');
  scopes['measure.problem'] = { ...line(), err, off: el('qMeasure').disabled, rows: rowsNow(),
                               sent: posts('/api/measure').length - nMeas };
  await what('endface');
  scopes['measure.problem.endface'] = { ...line(), off: el('qMeasure').disabled };
  await what('flatness');
  // switching the set that is gone off takes it out of the selection
  await run("toggleSet('tube', false)");
  scopes['measure.fixed'] = { ...line(), rows: rowsNow().map((r) => r.set), post: posts('/api/zones/selection').pop() };
  // the selection is not known: never read as "nothing on"; MEASURE stays off, nothing is sent
  const keepSel = srv.bus.zone_selection;
  srv.bus = { ...srv.bus }; delete srv.bus.zone_selection;
  await poll();
  const nM = posts('/api/measure').length;
  await run('sendMeasure()');
  const sp = key(' '); await settle();
  scopes['measure.unknown'] = { ...line(), off: el('qMeasure').disabled, geo: el('qZoneGeo').textContent,
                               sent: posts('/api/measure').length - nM };
  srv.bus = { ...srv.bus, zone_selection: { ...keepSel, state: 'unknown', ok: false, empty: false, label: 'NOT KNOWN',
    text: '', problems: [{ set: null, zone: null, reason: 'unreadable', message: 'the newest line of _selection.jsonl cannot be read: not UTF-8 JSON' }] } };
  await poll();
  scopes['measure.unreadable'] = { ...line(), off: el('qMeasure').disabled, geo: el('qZoneGeo').textContent };
  srv.bus = { ...srv.bus, zone_selection: null };
  await poll();
  scopes['measure.nopreset'] = { ...line(), off: el('qMeasure').disabled, geo: el('qZoneGeo').textContent };

  // Quick check
  srv.rows = [{ set: 'cells', on: true, zones: [['Z1', 'part'], ['Z2', 'part']] }];
  setSel();
  await mode('quick_check'); await poll();
  const qstamp = srv.bus.zone_selection.stamp;
  for (const c of ['flatness', 'golden', 'recipe']) {
    el('qQuickCheck').value = c; run('quickFields()'); run('renderZones()'); run('refreshButtons()');
    el('qQuickGolden').value = 'good1';
    confirm();
    scopes[`quick.${c}`] = { ...line(), zone: run('quickArgs()').zone || null, off: el('qStart').disabled };
  }
  el('qQuickCheck').value = 'flatness'; run('quickFields()');
  el('qQuickRoi').value = '1,2,30,40'; run('renderZones()');
  scopes['quick.region'] = line();
  el('qQuickRoi').value = '';
  srv.rows[0].on = false; setSel(); await poll();
  scopes['quick.none'] = { ...line(), zone: run('quickArgs()').zone, stamp: srv.bus.zone_selection.stamp };
  srv.rows = [{ set: 'cells', on: true, zones: [['Z1', 'part'], ['Z2', 'part']], rig_changes: ['a new rig session (2026-01-03T08:00:00)'] }];
  setSel(); await poll();
  confirm();
  scopes['quick.check'] = { ...line(), off: el('qStart').disabled };
  srv.rows[0].rig_changes = []; setSel(); await poll();
  const nQ = posts('/api/quick-check').length;
  confirm();
  await run('sendQuick()');
  out.quickPost = { body: posts('/api/quick-check').pop(), n: posts('/api/quick-check').length - nQ };
  out.quickStamp = srv.bus.zone_selection.stamp;

  // Station: the configuration decides; the selection only when its job says "zone": "selection"
  await mode('station');
  const cfg = async (name) => { await run(`selectConfig('${name}')`); await settle(); await poll(); };
  for (const name of ['own.json', 'cut.json', 'rec.json', 'flow.json', 'sel.json']) {
    await cfg(name);
    confirm();
    const shown = { ...line(), start: el('qStart').disabled, lights: el('qZone').textContent };
    const n = sent.length;
    await run("send('start')");
    scopes[`station.${name}`] = { ...shown,
                                 body: sent.slice(n).filter((s) => s.path === '/api/cmd/start').map((s) => s.body)[0] };
  }
  out.selStamp = srv.bus.zone_selection.stamp;
  srv.rows[0].on = false; setSel(); await poll();
  confirm();
  scopes['station.sel.empty'] = { ...line(), start: el('qStart').disabled };
  await cfg('own.json');
  scopes['station.own.none'] = line();
  await cfg('sel.json');
  srv.rows = [{ set: 'cells', on: true, zones: [['Z1', 'part'], ['Z2', 'part']] },
              { set: 'tube', on: true, found: false, zones: [], error: 'not in the library any more (deleted or renamed)' }];
  setSel(); await poll();
  scopes['station.sel.problem'] = line();
  await cfg('own.json');
  scopes['station.own.problem'] = line();
  await cfg('sel.json');
  srv.rows = [{ set: 'cells', on: true, zones: [['Z1', 'part'], ['Z2', 'part']] }];
  setSel();
  srv.bus = { ...srv.bus, active_preset: 'cart', run_preset: 'bench' };
  await poll();
  scopes['station.sel.preset'] = line();
  srv.bus = { ...srv.bus, active_preset: 'bench' };
  // the station is ARMED: the switches, Paint and Delete are off; the save message keeps up
  srv.bus = { ...srv.bus, state: 'ARMED' };
  await poll();
  scopes['station.armed'] = { rows: rowsNow().map((r) => [r.off, r.zones.map((z) => z[2])]),
                              paint: el('qPaint').disabled, del: el('qZoneDelete').disabled, show: el('qZoneShow').disabled };
  srv.bus = { ...srv.bus, state: 'READY' };
  // the save message follows the mode: the same set, saved while in Station mode
  srv.bus = { ...srv.bus, last_paint: { ...saved, time: '2026-01-02T13:00:00' } };
  await cfg('own.json');
  scopes['saved.station.own'] = el('qZoneResult').textContent;
  await cfg('sel.json');
  scopes['saved.station.sel'] = el('qZoneResult').textContent;
  await mode('quick_check'); await poll();
  scopes['saved.quick'] = el('qZoneResult').textContent;
  out.scopes = scopes;

  // ---------- Show selected zones
  await mode('measure'); await poll();
  srv.show = { status: 'done', result: { zones: [{ id: 'Z1', key: 'cells/Z1', seen: true }, { id: 'Z2', key: 'cells/Z2', seen: false }],
    warnings: ['zone Z2 does not land'], overlay_png: 'output/previews/x_overlay.png' } };
  await run('showZonesOnPart()');
  out.show = { job: jobs().pop(), msg: el('qZoneResult').textContent, img: el('qZoneImg').src,
               stamp: srv.bus.zone_selection.stamp };
  srv.show.result.zones[1].seen = true;
  await run('showZonesOnPart()');
  out.showAll = el('qZoneResult').textContent;
  srv.rows[0].on = false; setSel(); await poll();
  const nShow = jobs().length;
  await run('showZonesOnPart()');
  out.showNone = { msg: el('qZoneResult').textContent, jobs: jobs().length - nShow };

  // ---------- Delete: a set that is on is switched off first; then a question with a reason
  srv.rows = [{ set: 'cells', on: true, zones: [['Z1', 'part'], ['Z2', 'part']] },
              { set: 'tube', on: false, zones: [['Z1', 'tube']], used_by: { recipes: [], stations: ['cut.json'] } }];
  setSel();
  await run("loadZoneSets('cells')"); await poll();
  const nDel = jobs().length;
  run('deleteZoneSet()');
  out.delOn = { msg: el('qZoneResult').textContent, open: !el('qZoneAsk').hidden, jobs: jobs().length - nDel };
  el('qZoneSet').value = 'tube';
  run('deleteZoneSet()');
  out.delAsk = { open: !el('qZoneAsk').hidden, text: el('qZoneAskText').textContent, yes: el('qZoneAskYes').textContent,
                 reason: !el('qZoneReasonRow').hidden, focus: focused.slice(-1)[0], jobs: jobs().length - nDel };
  await run('confirmAsk()');                               // no reason typed
  out.delNoReason = { text: el('qZoneAskText').textContent, open: !el('qZoneAsk').hidden, jobs: jobs().length - nDel };
  el('qZoneReason').value = 'fixture changed';
  srv.rows = srv.rows.filter((r) => r.set !== 'tube');
  const nSel = sent.length;
  key('Enter', el('qZoneReason'));                         // Enter in the reason box presses Yes
  await settle(); await settle();
  out.delDone = { job: jobs().pop(), msg: el('qZoneResult').textContent, open: !el('qZoneAsk').hidden,
                  focus: active(),
                  measure: sent.slice(nSel).filter((s) => s.path === '/api/measure').length,
                  options: el('qZoneSet').options.map((o) => o.value) };
  srv.del = { status: 'error', error: "ZoneError: zone set 'cells' is used by recipe(s) lid: change them first, or force the delete" };
  srv.rows[0].on = false; setSel(); await run("loadZoneSets('cells')"); await poll();
  run('deleteZoneSet()');
  el('qZoneReason').value = 'x';
  await run('confirmAsk()');
  out.delRefused = el('qZoneResult').textContent;

  // ---------- which area a result was about: from the run's own record; cleared with it
  const show = (r) => { run('showInspected')(r); return { inspected: el('qInspected').textContent,
    counts: el('qZoneCounts').children.map((li) => [li.className, li.textContent]), hidden: el('qZoneCounts').hidden,
    warn: el('qZoneWarn').textContent }; };
  out.results = {
    zones: show({ run_id: 'r1', zone_selection: { text: 'zoneset:cells', source: 'selection',
      selected: { time: '2026-01-02T09:14:02', by: 'operator page' }, outside: 12, warnings: ['zone set cells was painted ...'],
      by_zone: [{ key: 'cells/Z1', name: 'part', verdict: 'PASS', max_mm: 0.0213, points: 21034, coverage_pct: 98.4 },
                { key: 'cells/Z2', name: 'part', verdict: 'FAIL', max_mm: 0.812, points: 20110, coverage_pct: 97 }] } }),
    nothingOn: show({ run_id: 'r2', zone_selection: { text: '', source: 'selection', by_zone: [] } }),
    whole: show({ run_id: 'r3', job: { type: 'metric', zone: null, config: 'quick_check.json' }, capture: { zone: null }, operating_mode: 'quick_check' }),
    station: show({ run_id: 'r4', job: { type: 'metric', zone: 'C4:D5', config: 'cut.json' }, capture: { zone: 'C4:D5' } }),
    recipe: show({ run_id: 'r5', job: { type: 'recipe', recipe: 'lid', config: 'rec.json' } }),
    measure: show({ run_id: 'r6', operating_mode: 'measure', spec: { what: 'flatness', zone: null, roi: [1, 2, 30, 40] } }),
    endface: show({ run_id: 'r7', operating_mode: 'measure', spec: { what: 'endface' } }),
  };
  show({ run_id: 'r1', zone_selection: { text: 'zoneset:cells', by_zone: [{ key: 'cells/Z1', points: 5 }] } });
  run('clearResult()');
  out.cleared = { inspected: el('qInspected').textContent, counts: el('qZoneCounts').children.length,
                  hidden: el('qZoneCounts').hidden, warn: el('qZoneWarn').textContent };
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed; quick.js logic not checked")
    h = tmp_path_factory.mktemp("zui_node") / "zones_harness.js"
    h.write_text(_HARNESS, encoding="utf-8")
    r = subprocess.run([node, str(h), str(STATIC / "quick.js")], capture_output=True, text=True,
                       encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


SAVED = "SAVED 2026-01-02 12:01: zone set cells, 2 zones: Z1 part, Z2 part. It is switched ON."
PER_ZONE = ("MEASURE lights only these zones and measures each one on its own (one plane per zone, "
            "named set/zone).")


def test_paint_without_a_name_asks_for_one_and_starts_nothing(page):
    e = page["empty"]
    assert e["geo"] == "No zone set painted at this rig placement yet. Name one and press Paint zones."
    assert e["rows"] == [] and e["options"] == ["(a new set: type its name)"] and e["img"] is True
    assert e["line"] == "Zones: whole field" and e["scope"] == "MEASURE uses the whole field."
    assert page["emptyElsewhere"] == (
        "No zone set at this rig placement (preset bench). 2 sets were painted with preset cart "
        "(another rig placement): load that preset to use them.")
    n = page["noName"]
    assert n["msg"] == "Name the new zone set first (New set name), or pick a set to edit."
    assert n["jobs"] == 0 and n["focus"] == "qZoneName"


def test_save_message_after_a_painter_session(page):
    p = page["paintNew"]
    assert p["job"] == {"verb": "zone_paint", "args": {"name": "cells", "select": True}}
    assert p["msg"].startswith("Painting cells: draw on the part with the mouse. SPACE on the "
                               "projector saves, ESC cancels. ABORT here ends it")
    assert page["lost"] == ("NOT KNOWN: lost contact with the server while painting cells. Look at "
                            "the list when it is back: cells is new only if its painted time is "
                            "after 12:00.")
    s = page["saved"]
    # saved as NAME with N zones, switched on - and what the run button does with it, now
    assert s["msg"] == f"{SAVED} {PER_ZONE}" and s["msgWarn"] is False and s["warn"] is False
    assert s["picked"] == "cells" and s["typed"] == "" and s["imgHidden"] is False
    # ... and the line is brought into view when it arrives (the operator scrolled down to the
    # Paint zones button); a page that is merely loaded later is not scrolled
    assert s["revealed"] == ["qZoneResult"] and page["reloadRevealed"] == []
    assert s["img"].startswith("/files/zones/g/cells/camera.png")
    assert s["line"] == "Zones: cells (2)" and s["scope"] == PER_ZONE and s["measure"] is False
    assert s["rows"] == [{"set": "cells", "cls": "qzrow on", "text": "cells2 zonesZ1 partZ2 part",
                          "on": True, "off": False, "zones": [["Z1", True, False], ["Z2", True, False]]}]
    assert s["info"].startswith("Viewing cells: Z1 part, Z2 part (painted 2026-01-02 03:04:05")
    e = page["ends"]
    assert e["cancelled"]["msg"] == ("NOT SAVED 2026-01-02 12:00: cells - the painter was cancelled "
                                     "(ESC). Nothing was written.")
    assert e["aborted"]["msg"] == ("NOT SAVED 2026-01-02 12:01: cells - ABORT ended the painter. "
                                   "Nothing was written.")
    assert e["idle_timeout"]["msg"] == ("NOT SAVED 2026-01-02 12:02: cells - no input for 600 s, so "
                                        "the rig was given back. Nothing was written.")
    assert e["script_ended"]["msg"] == ("NOT SAVED 2026-01-02 12:03: cells - the scripted session "
                                        "ended without SPACE.")
    assert e["refused"]["msg"] == ("NOT SAVED 2026-01-02 12:04: the painter did not start - "
                                   "ZoneError: zone set 'cells' exists at geometry g.")
    assert e["dropped"]["msg"] == "NOT SAVED 2026-01-02 12:05: ABORT came before the painter started."
    assert all(e[k]["warn"] for k in ("cancelled", "aborted", "idle_timeout", "script_ended",
                                      "refused", "dropped"))
    assert e["replaced"]["msg"] == (
        "SAVED 2026-01-02 12:06: zone set cells replaced, 3 zones: Z1 part, Z2 part, Z3. The "
        "earlier one is in the library trash. It stays switched ON. All 3 zones are on (the "
        f"per-zone switches were reset: the zones were painted again). {PER_ZONE}")
    assert e["off"]["msg"] == ("SAVED 2026-01-02 12:07: zone set cells, 2 zones: Z1 part, Z2 part. "
                               "It is NOT switched on: switch it on in the list to select it.")
    assert "It stays switched OFF, as before" in e["offReplaced"]["msg"]
    assert e["noPhoto"]["msg"].endswith("No photo of the zones could be taken (the drawing is shown instead).")
    assert not e["replaced"]["warn"] and not e["off"]["warn"]
    # saved, but the rig was not given back cleanly: SAVED, with the warning (never "not saved")
    assert e["rigError"]["msg"].startswith(SAVED.replace("12:01", "12:10")) and e["rigError"]["warn"]
    assert e["rigError"]["msg"].endswith(
        "The rig was NOT given back cleanly after the save (RuntimeError: camera close failed): "
        "check the camera and the projector before the next scan.")
    # the sentence after "switched ON" is the run button's own, whatever the mode
    sc = page["scopes"]
    assert sc["saved.station.own"] == (
        "SAVED 2026-01-02 13:00: zone set cells, 2 zones: Z1 part, Z2 part. It is switched ON. "
        "START runs own.json as written: the whole field. The selected zones (cells (2)) are NOT "
        "used. To inspect them, use Quick check or Measure, or a station configuration whose job "
        "says \"zone\": \"selection\".")
    assert sc["saved.station.sel"].startswith(
        "SAVED 2026-01-02 13:00: zone set cells, 2 zones: Z1 part, Z2 part. It is switched ON. "
        "START runs sel.json, which inspects the SELECTED zones: cells (2)")
    assert sc["saved.quick"].startswith(
        "SAVED 2026-01-02 13:00: zone set cells, 2 zones: Z1 part, Z2 part. It is switched ON. "
        "START lights only these zones and checks each one against the limit on its own")


def test_the_save_message_never_says_what_is_no_longer_so(page):
    """What was saved is said with its date; what is true of that set NOW comes from the stored
    selection every time the message is drawn - on a page that was reloaded, in a tab that
    stayed open, after the set was switched off, painted again, deleted, or another preset
    was loaded."""
    said = "SAVED 2026-01-02 12:01: zone set cells, 2 zones: Z1 part, Z2 part."
    old = f"Last painter session - {said}"
    # a page that was just loaded: the session is history - no "the run button does ..." of now
    assert page["savedAfterReload"] == f"{old} It is switched ON now."
    h = page["history"]
    assert h["oneZone"] == f"{old} It is switched ON now (1 of 2 zones)."
    assert h["off"] == f"{old} It is switched OFF now: switch it on in the list to select it."
    assert h["repainted"] == f"{old} It was painted again since."
    assert h["deleted"] == f"{old} It is not in the list any more: it was deleted since."
    assert h["elsewhere"] == (f"{old} That was at another rig placement (preset bench): it is not "
                              "one of the sets listed here.")
    for text in h.values():
        assert "MEASURE" not in text or text == h["scope"], text
    # a message this page saw arrive follows the stored selection too
    e = page["ends"]
    assert e["offSince"]["msg"] == (said.replace("12:01", "12:11") + " It is NOT switched on: "
                                    "switch it on in the list to select it.")
    assert e["zoneOffSince"]["msg"] == (said.replace("12:01", "12:12") + " It is switched ON. 1 of 2 "
                                        "zones are on. MEASURE lights and measures only this zone.")
    # a painter that waits behind another job is said as waiting, not as open
    q = page["queued"]
    assert q["msg"] == ("Waiting to paint late: another job still holds the scanner. The painter "
                        "opens on the projector when that job ends; nothing is drawn yet. ABORT "
                        "here drops it.")
    assert q["measure"] is True


def test_painting_over_a_set_asks_before_replacing(page):
    a = page["askPaint"]
    assert a["open"] and a["jobs"] == 0 and a["yes"] == "Replace cells" and a["reason"] is False
    assert a["focus"] == "qZoneAskNo"                           # the answer that changes nothing
    assert a["text"] == (
        "Zone set cells exists (2 zones, painted 2026-01-02 03:04:05). Paint over it? The painter "
        "opens with its zones. When you save with SPACE the set as it is now moves to the library "
        "trash (it can be brought back). ESC in the painter leaves it unchanged. It is used by "
        "recipe lid and station configuration sel.json: what they inspect changes with it.")
    # the keys start nothing while the question waits
    assert page["askKeys"] == {"m": [], " ": [], "Enter": [], "h": []}
    e = page["askEsc"]
    assert e["open"] is False and e["jobs"] == 0 and e["sent"] == ["/api/cmd/abort"]
    assert e["msg"] == "Cancelled (Esc): nothing was changed. ABORT was sent too."
    p = page["paintOver"]
    assert p["open"] is False and p["job"] == {"verb": "zone_paint", "args": {
        "name": "cells", "source": "cells", "replace": True, "select": True}}
    assert page["askTyped"] == {"open": True, "jobs": 0, "focus": "qZoneAskNo"}
    # the question closes: the focus goes back to the button that asked (hidden with the
    # question it would fall to the page, where Space / Enter is the run button). After Yes
    # that button is switched off: the Scan zones panel takes it
    assert e["focus"] == "qPaint" and p["focus"] == "qScanZones"
    assert page["askCancel"] == {"open": False, "focus": "qPaint", "sent": []}


def test_run_buttons_and_page_keys_are_off_while_the_painter_is_open(page):
    assert page["painterOn"] == {"measure": True, "start": True, "paint": True, "show": True,
                                 "del": True, "abort": False}
    # P polygon, N name, H legend, digits ...: the painter's keys - nothing here reacts
    assert all(v == [] for v in page["painterKeys"].values()) and len(page["painterKeys"]) == 10
    assert page["painterEsc"] == ["/api/cmd/abort"]             # Esc is ABORT, always
    armed = page["scopes"]["station.armed"]
    assert armed == {"rows": [[True, [True, True]]], "paint": True, "del": True, "show": True}


def test_several_sets_are_switched_on_and_off_and_single_zones_inside_them(page):
    t = page["two"]
    assert [r["set"] for r in t["rows"]] == ["cells", "aaa", "tube"]      # on first, then by name
    assert [r["cls"] for r in t["rows"]] == ["qzrow on", "qzrow", "qzrow"]
    assert t["rows"][2]["zones"] == [["Z1", True, False], ["Z2", True, False], ["Z3", True, False]]
    assert t["line"] == "Zones: cells (2)"
    assert t["geo"] == ("Zone sets at this rig placement (preset bench). Switch a set on to select "
                        "it; the line above the run button says what the next run does with the "
                        "selection. Last changed 2026-01-02 09:14:02 by operator page.")
    on = page["tubeOn"]
    assert on["post"]["sets"] == [{"set": "cells", "on": True, "ids": None},
                                  {"set": "tube", "on": True, "ids": None},
                                  {"set": "aaa", "on": False, "ids": None}]
    assert on["post"]["source"] == "quick" and re.fullmatch(r"s\d+", on["post"]["was"])
    assert on["rows"] == [["cells", True], ["tube", True], ["aaa", False]]
    assert on["line"] == "Zones: cells (2), tube (3)" and on["msg"] == "tube is switched ON."
    assert on["confirm"] is False                               # another selection: confirm again
    off = page["zoneOff"]
    assert {"set": "tube", "on": True, "ids": ["Z1", "Z2"]} in off["post"]["sets"]
    assert off["line"] == "Zones: cells (2), tube (2 of 3)" and off["msg"] == "tube/Z3 is switched off."
    c = page["cellsOff"]
    assert {"set": "cells", "on": False, "ids": None} in c["post"]["sets"]
    assert c["line"] == "Zones: tube (2 of 3)"
    last = page["lastZone"]
    assert {"set": "tube", "on": False, "ids": ["Z2"]} in last["post"]["sets"]
    assert last["line"] == "Zones: whole field"
    assert last["msg"] == "tube/Z2 was the last zone that was on: tube is switched OFF."
    r = page["refused"]
    assert r["rows"] == [["aaa", False], ["cells", False], ["tube", False]]   # as stored: not on
    assert r["msgWarn"] and r["line"] == "Zones: whole field"
    assert r["msg"] == ("NOT changed: zone_select refused while the station is ARMED: the zones a "
                        "station inspects are fixed")
    assert page["refused400"]["msg"] == "NOT changed: no zone set 'cells' at geometry g"
    assert all(not on_ for _, on_ in page["refused400"]["rows"])


def test_a_switch_keeps_the_keyboard_focus_and_space_never_becomes_a_run(page):
    """Every change redraws the rows. The switch that was pressed must have the focus again
    afterwards: on the page itself Space / Enter is the run button, so a second Space meant
    for the switch would start a measurement (Measure mode asks for no confirmation)."""
    kb = page["keyboard"]
    assert kb["start"] == "INPUT:tube/"
    # while the change is on its way to the server
    b = kb["busy"]
    assert b["msg"] == "Changing the zone selection..."
    assert b["space"] is True and b["enter"] is True and b["measure"] == 0     # no run key
    assert b["other"] is True and b["posts"] == 1                              # no second change
    # afterwards: the same switch, redrawn, has the focus - and the next Space is its own
    a = kb["after"]
    assert a["active"] == "INPUT:tube/" and a["redrawn"] is True and a["on"] is False
    assert a["msg"] == "tube is switched OFF." and a["line"] == "Zones: cells (2)"
    g = kb["again"]
    assert g == {"active": "INPUT:tube/", "on": True, "measure": 0, "posts": 2}
    assert kb["zone"] == {"active": "INPUT:tube/Z2", "on": False, "line": "Zones: cells (2), tube (1 of 2)"}
    # a switch that cannot be pressed any more: the panel takes the focus, the keys start nothing
    assert kb["armed"] == {"active": "qScanZones", "off": True, "sent": []}
    # a setup job of this page holds the scanner: no switch, no painter, no delete meanwhile
    assert kb["setup"] == {"rows": [[True, [True, True]], [True, [True, True]]], "paint": True, "del": True}
    assert kb["setupDone"] == [False, False]
    # the page's own structure the rule rests on
    html = (STATIC / "quick.html").read_text(encoding="utf-8")
    js = (STATIC / "quick.js").read_text(encoding="utf-8")
    assert '<div class="qpanel qzones" id="qScanZones" tabindex="-1">' in html
    assert "[tabindex]" in js[js.index("const ACTIVATABLE"):js.index("function onKey")]


def test_measure_and_quick_check_send_the_selection_and_say_so(page):
    sc = page["scopes"]
    stamp = sc["measure.flatness"]["zone"]
    assert re.fullmatch(r"selection@s\d+", stamp)
    f = sc["measure.flatness"]
    assert f["line"] == "Zones: cells (2), tube (2 of 3)" and f["scope"] == PER_ZONE == f["pane"]
    assert f["off"] is False and f["err"] is None and not f["warn"]
    assert sc["measure.cylinder"]["scope"].endswith("(one cylinder per zone, named set/zone).")
    # a measurement that has no meaning across several zones: MEASURE is off and says why
    for w, said in (("step", "Step height needs region A and region B"),
                    ("dims", "The bounding dimensions are of ONE part"),
                    ("deviation", "The deviation from the golden is ONE compare"),
                    ("distance", "Point 1 to the plane of the region needs ONE zone")):
        s = sc[f"measure.{w}"]
        assert s["off"] is True and s["zone"] is None and s["scope"].startswith(said), (w, s)
        assert s["err"] == s["scope"] and s["warn"]
    assert sc["measure.distance.p2"]["zone"] == stamp and sc["measure.distance.p2"]["off"] is False
    assert sc["measure.distance.p2"]["scope"] == "MEASURE lights only these zones: pick the points inside them."
    # the end face never uses the selection, and says so (no zone is sent)
    e = sc["measure.endface"]
    assert e["line"] == "End face: whole field" and e["zone"] is None and e["off"] is False
    assert e["scope"] == ("The end-face measurement uses the whole field (it finds the rim on the "
                          "fully lit part). The selected zones (cells (2), tube (2 of 3)) are NOT used.")
    assert sc["measure.flatness.region"]["scope"] == (
        f"{PER_ZONE} Only the part of these zones inside Region 10,20,300,400 is measured.")
    assert sc["measure.one.flatness"] == {"line": "Zones: cells (1 of 2)", "warn": False, "off": False,
                                          "scope": "MEASURE lights and measures only this zone."}
    assert sc["measure.one.dims"]["off"] is False and sc["measure.one.step"]["off"] is True
    assert sc["measure.one.distance"]["scope"].endswith("Point 1 is also measured to the plane of this zone.")
    n = sc["measure.none"]
    assert n["line"] == "Zones: whole field" and n["scope"] == "MEASURE uses the whole field."
    assert n["zone"] == f"selection@{n['stamp']}" and n["off"] is False       # "nothing on", as shown
    assert sc["measure.none.step"]["off"] is False
    # a rig change since a set was painted: marked on the row, the line and the sentence
    c = sc["measure.check"]
    assert c["line"] == "Zones: cells (2, CHECK)" and c["warn"] and c["off"] is False
    assert c["scope"].endswith("Check them first (Show selected zones): since they were painted, "
                               "cells: rig check REHOMED (2026-01-03T08:00:00).")
    assert c["row"]["cls"] == "qzrow on check" and "CHECK: since it was painted, rig check REHOMED" in c["row"]["text"]
    # a selection that cannot be used: refused on the page, nothing sent
    p = sc["measure.problem"]
    assert p["line"] == "Zones: cells (2), tube (NOT FOUND)" and p["warn"] and p["off"] is True
    assert p["scope"] == ("This run will be refused: zone set 'tube' is switched on but is not in "
                          "the library any more (deleted or renamed). Switch that set off or paint "
                          "it again.")
    assert p["err"] == p["scope"] and p["sent"] == 0
    gone = next(r for r in p["rows"] if r["set"] == "tube")
    assert gone["cls"] == "qzrow on bad" and gone["on"] and not gone["off"] and "NOT FOUND" in gone["text"]
    # ... but the refusal is promised only where the run uses the selection
    pe = sc["measure.problem.endface"]
    assert pe["off"] is False and pe["line"] == "End face: whole field"
    assert pe["scope"].endswith("The selection has a problem (zone set 'tube' is switched on but is "
                                "not in the library any more (deleted or renamed)); this run does "
                                "not use it.")
    assert sc["measure.fixed"]["line"] == "Zones: cells (2)" and sc["measure.fixed"]["rows"] == ["cells"]
    assert {"set": "tube", "on": False, "ids": None} in sc["measure.fixed"]["post"]["sets"]
    # not known is never "nothing on": the line says so and nothing can be sent
    u = sc["measure.unknown"]
    assert u["line"] == "Zones: NOT KNOWN" and u["warn"] and u["off"] is True and u["sent"] == 0
    assert u["scope"].startswith("This run will be refused: the server has not said which zone sets")
    assert "NOT KNOWN" in u["geo"]
    r = sc["measure.unreadable"]
    assert r["line"] == "Zones: NOT KNOWN" and r["off"] is True
    assert r["geo"].startswith("The stored zone selection cannot be read (the newest line of")
    np_ = sc["measure.nopreset"]
    assert np_["line"] == "Zones: no preset loaded" and np_["off"] is True
    assert np_["geo"] == "Load a preset first: zone sets belong to a rig placement."
    assert np_["scope"] == "Load a preset first: MEASURE is off."
    # Quick check
    q = sc["quick.flatness"]
    assert q["line"] == "Zones: cells (2)" and re.fullmatch(r"selection@s\d+", q["zone"]) and not q["off"]
    assert q["scope"] == ("START lights only these zones and checks each one against the limit on "
                          "its own; the result names a zone that fails. A zone the camera does not "
                          "see well enough PULLs the part.")
    g = sc["quick.golden"]
    assert g["zone"] == q["zone"] and g["scope"].startswith(
        "START lights only these zones and compares what they show with the golden, which must "
        "have been scanned through the same zones (else the part is PULLed).")
    rc = sc["quick.recipe"]
    assert rc["line"] == "Recipe lid: its own zones" and rc["zone"] is None
    assert rc["scope"] == "START runs recipe lid with its own zones. The selected zones (cells (2)) are NOT used."
    assert sc["quick.region"]["scope"].endswith("Only the part of these zones inside Region 1,2,30,40 is checked.")
    qn = sc["quick.none"]
    assert qn["scope"] == "START checks the whole field." and qn["zone"] == f"selection@{qn['stamp']}"
    qc = sc["quick.check"]
    assert qc["line"] == "Zones: cells (2, CHECK)" and qc["off"] is False
    assert "START will PULL until they were shown on the part (Show selected zones) or painted again" in qc["scope"]
    assert page["quickPost"]["n"] == 1
    assert page["quickPost"]["body"] == {"check": "flatness", "source": "quick", "operator_confirmed": True,
                                         "threshold_mm": 0.5, "preset": "bench",
                                         "zone": f"selection@{page['quickStamp']}"}


def test_station_mode_words_for_own_zone_selection_recipe_and_flow(page):
    sc = page["scopes"]
    used = ("To inspect them, use Quick check or Measure, or a station configuration whose job says "
            "\"zone\": \"selection\".")
    own = sc["station.own.json"]
    assert own["line"] == "Station zone: whole field (own.json)" and not own["warn"]
    assert own["scope"] == ("START runs own.json as written: the whole field. The selected zones "
                            f"(cells (2)) are NOT used. {used}")
    assert own["lights"] == "whole field" and own["start"] is False
    assert "zones" not in own["body"] and own["body"]["config"] == "own.json"   # no stamp: its own zone
    cut = sc["station.cut.json"]
    assert cut["line"] == "Station zone: C4:D5 (cut.json)"
    assert cut["scope"].startswith("START runs cut.json as written: its own zone C4:D5. The selected "
                                   "zones (cells (2)) are NOT used.")
    rec = sc["station.rec.json"]
    assert rec["line"] == "Station zones: recipe lid's own (rec.json)"
    assert rec["scope"].startswith("START runs rec.json as written: recipe lid with its own zones.")
    flow = sc["station.flow.json"]
    assert flow["line"] == "Station zones: each work state's recipe (flow.json)"
    assert flow["scope"].startswith("START runs flow.json as written: each work state is checked by "
                                    "its recipe's own zones. The selected zones (cells (2)) are NOT used.")
    s = sc["station.sel.json"]
    assert s["line"] == "Zones: cells (2)" and not s["warn"] and s["start"] is False
    assert s["scope"] == ("START runs sel.json, which inspects the SELECTED zones: cells (2), each one "
                          "against the limit on its own. Changing the selection changes what this "
                          "station inspects; it is fixed while the station is ARMED or in a cycle.")
    assert s["lights"] == "the selected zones (cells (2))"
    assert s["body"]["zones"] == page["selStamp"] and s["body"]["config"] == "sel.json"
    e = sc["station.sel.empty"]
    assert e["warn"] and e["start"] is False                    # START stays: the bus PULLs, and says so
    assert e["scope"] == ("sel.json inspects the selected zones and none is switched on: START will "
                          "be refused and the part PULLed. Switch a zone set on.")
    assert sc["station.own.none"]["scope"] == "START runs own.json as written: the whole field."
    p = sc["station.sel.problem"]
    assert p["line"] == "Zones: cells (2), tube (NOT FOUND)" and p["scope"].startswith(
        "sel.json inspects the selected zones and the selection cannot be used (zone set 'tube' is "
        "switched on but is not in the library any more (deleted or renamed)): START will be "
        "refused and the part PULLed.")
    op = sc["station.own.problem"]
    assert op["line"] == "Station zone: whole field (own.json)" and not op["warn"]
    assert "The selection has a problem" in op["scope"] and "this run does not use it" in op["scope"]
    assert "will be refused" not in op["scope"]
    pr = sc["station.sel.preset"]
    assert pr["scope"] == ("sel.json inspects the selected zones with preset bench, and preset cart "
                           "is loaded: START will be refused and the part PULLed. Select the station "
                           "configuration again (that loads bench).")


def test_show_selected_zones_and_delete_asks_needs_a_reason(page):
    s = page["show"]
    assert s["job"] == {"verb": "zone_show", "args": {"zones": f"selection@{s['stamp']}", "hold_s": 5}}
    assert s["msg"] == ("Shown: 1 of 2. NOT SEEN: cells/Z2 (outside the camera's view, in shadow, "
                        "or on nothing).")
    assert s["img"] == "/files/output/previews/x_overlay.png"
    assert page["showAll"].startswith("Shown: 2 of 2 zones land in the camera image.")
    assert page["showNone"] == {"msg": "No zone set is switched on: there is nothing to show.", "jobs": 0}
    # delete
    assert page["delOn"] == {"msg": "cells is switched on. Switch it off first, then delete it.",
                             "open": False, "jobs": 0}
    a = page["delAsk"]
    assert a["open"] and a["reason"] and a["jobs"] == 0 and a["yes"] == "Move tube to the trash"
    assert a["focus"] == "qZoneReason"
    assert a["text"] == ("Move zone set tube (1 zone) to the library trash? It can be brought back "
                         "from there. Give a reason (it is recorded). It is used by station "
                         "configuration cut.json: the delete is refused while they name it.")
    n = page["delNoReason"]
    assert n["open"] and n["jobs"] == 0 and n["text"].endswith("Type a reason first.")
    d = page["delDone"]
    assert d["job"] == {"verb": "zone_delete", "args": {"name": "tube", "reason": "fixture changed",
                                                        "confirm": True}}
    assert d["msg"] == "DELETED: tube is in the library trash." and d["open"] is False
    assert d["measure"] == 0 and d["options"] == ["", "cells"]   # Enter answered: it measured nothing
    assert d["focus"] == "qScanZones"                            # not left on the page
    assert page["delRefused"] == ("NOT deleted: ZoneError: zone set 'cells' is used by recipe(s) lid: "
                                  "change them first, or force the delete")
    # the buttons and the switches are wired (one listener on the rows: no inline handler)
    js = (STATIC / "quick.js").read_text(encoding="utf-8")
    for wired in ("$('#qZoneRows').addEventListener('change', onZoneRowChange)",
                  "$('#qZoneAskYes').addEventListener('click', confirmAsk)",
                  "$('#qZoneAskNo').addEventListener('click'",
                  "$('#qZoneDelete').addEventListener('click', deleteZoneSet)",
                  "$('#qZoneShow').addEventListener('click', showZonesOnPart)",
                  "$('#qPaint').addEventListener('click', paintZones)"):
        assert wired in js, wired


def test_every_result_says_which_area_it_is_about(page):
    r = page["results"]
    z = r["zones"]
    assert z["inspected"] == "Inspected: cells/Z1, cells/Z2 (the zone selection of 2026-01-02 09:14:02)"
    assert z["hidden"] is False and z["counts"] == [
        ["z-pass", "cells/Z1 part PASSmax 0.021 mm, 21034 points, 98 % of the zone seen"],
        ["z-fail", "cells/Z2 part FAILmax 0.812 mm, 20110 points, 97 % of the zone seen"]]
    assert z["warn"] == ("zone set cells was painted ... | 12 point(s) decoded outside every zone "
                         "(reported, not judged)")
    assert r["nothingOn"]["inspected"] == "Inspected: whole field (no zone set was switched on)"
    assert r["whole"]["inspected"] == "Inspected: whole field" and r["whole"]["hidden"] is True
    assert r["station"]["inspected"] == "Inspected: station configuration cut.json, its own zone C4:D5"
    assert r["recipe"]["inspected"] == "Inspected: recipe lid's own zones"
    assert r["measure"]["inspected"] == "Inspected: whole field, inside region 1,2,30,40"
    assert r["endface"]["inspected"] == "Inspected: whole field (the end-face measurement)"
    # nothing of the last result stays when it is cleared (another mode, another cycle)
    assert page["cleared"] == {"inspected": "", "counts": 0, "hidden": True, "warn": ""}
