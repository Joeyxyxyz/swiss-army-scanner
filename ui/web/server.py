"""
Local web UI for the Swiss Army Scanner engine.

    python -m ui.web.server [--mock] [--port 8765] [--library PATH]
                            [--presets PATH] [--preset NAME] [--open]

Standard library only (http.server + json). The page in static/ is plain
HTML/JS/CSS with no CDN or build step, because the bench PC may be offline.

Why one worker thread: the projector is a fullscreen OpenCV window on the
EVM's HDMI output. That window must be created and pumped on ONE thread,
and camera + projector are one physical rig, so hardware operations are
serial. Every engine call (verbs, the live view, the session's preset)
runs on a single worker thread fed by a job queue. HTTP handler threads
only enqueue work and read state: the worker's status snapshot, job
records, listings the worker produced, library files, presets on disk,
the output index and the session log.

Safety: binds 127.0.0.1 only. The mode (mock / real) is fixed at launch.
POSTs must be JSON from this page's own origin, and every response forbids
framing (X-Frame-Options + CSP frame-ancestors), so a foreign web page can
neither post to the rig nor clickjack this page. /files serves only
whitelisted types from inside the library root. Nothing here talks to
hardware: the engine enforces §9.

Operator page (/quick): the top selects the STATION CONFIGURATION (stations/*.json: the
station's parameters + its profile library set) and the active PROFILE (which part variant:
long tube / short tube); the middle shows the WORK STATES, the steps of the configuration's
flow. START begins a cycle; STEP verifies the current work state: the line's sensor claims
(POST /api/io/inputs, for every part: a claim belongs to one cycle; MOCK: the simulated
line's sensors report per cycle) are read, the DLP scans (`flow_verify`: one recipe run of
the active
profile, engine/flow/verify.py) and the work state is CONFIRMED (its permission, PERMIT_<NAME>,
is granted) or CORRECTED / NOT_MET / DEFECT / DOUBT (withheld; the fault is painted on the
part, a prompt waits for ACK, the part is kicked out ...). Each cycle writes a cycle folder
(result.json: station config, profile, steps, permissions, disposition). A single-job
station config is a one-step flow: START runs it through
the same code path as `scan run` (ui/cli.py cmd_run) as one job (a recipe job:
job.type recipe, engine/recipes/; its verdict is then painted on the part by a separate
`recipe_feedback` job that any next command - START, TRIGGER, HOME, ABORT, another
configuration - clears); ABORT sets
the job's cancel event, which the engine checks before every capture (the
rig is closed on the way out: projector idle, rig lock released) and the run
is recorded as ABORTED. The lamp shows the run's disposition (engine/disposition.py): PASS
(CONTINUE), FAIL (REJECT) or, amber, "PULL - quality inspection" with the pull reasons
(any doubt: INVALID, ABORTED, a guard band, an unreadable code), and never a PASS the bus
outputs do not give (an ABORT after the run's verdict, a START refused on an unusable
configuration: the bus PULLs, the lamp shows PULL; the job's result and run log say so too, and no
verdict is projected after an ABORT). ABORT in any state also
stops what runs outside a cycle: the projected verdict, the live view, station / recipe runs and rig checks submitted as jobs.
HOME (when idle, or queued behind an ABORT) runs the
rig check. Every command (page buttons, keys, POST /api/cmd/*, hardware
inputs) goes through one state machine: engine/control/commands.py. Every bus job
reports back to it when it ends, also when it is cancelled or dropped before it ran;
a busy state with no job left to report back is settled (never wedged).

Operating modes (engine/modes/): the page's mode switch sends SELECT_MODE to the bus (POST
/api/cmd/mode). Station is everything above. In MEASURE mode POST /api/measure runs a
`measure_run` job (a triangulation preset, what to measure, an roi or two picked camera px):
numbers with units and uncertainty, a neutral MEASURED / NOT MEASURABLE - never a lamp
verdict. In QUICK CHECK mode POST /api/quick-check runs a `quick_check` job (a preset + one
check): CONTINUE / REJECT / PULL on the page's lamp. Neither is a bus job: outside Station
mode the bus starts no cycle and keeps READY, PASS and every permission off, so nothing a
measurement or a quick check says reaches the line's outputs. Both verbs are refused in any
other mode. ABORT stops them; HOME is the rig check a MOBILE rig (preset rig.mount) needs
before any mm (engine/rigstate.py). The server's start is a new rig session: a marker is
recorded for every mobile preset it knows, and for one it first sees later the session is
still never older than the server's start (rigstate.open_process_session) - nothing that
verified a rig before the server started counts in it. The rig's state in the bus snapshot
is cached (the record's size and time, a few seconds): polling it does not read the record.

Live view: it holds the rig (camera + DMD + the machine-wide rig lock), so
it stops when a rig job is queued, when the operator presses Stop (even
while it is still queued), and when no viewer has polled it for
LIVE_IDLE_S. Rig-free jobs (measure, inspect, ...) run between live frames.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import re
import sys
import threading
import time
import traceback
import webbrowser
from collections import OrderedDict, deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np

from engine import api, config, modes, rigstate, rules
from engine import disposition as D
from engine import runs as R
from engine.control.commands import BUSY as BUS_BUSY
from engine.control.commands import IN_CYCLE as BUS_IN_CYCLE
from engine.control.commands import Reply
from engine.flow import schema as flow_schema
from engine.flow import verify as flow_verify
from engine.library import presets as preset_lib
from engine.library.store import OUTPUT_KINDS
from engine.reconstruct import pointcloud
from engine.results import TIER_DEPTH, TIER_LATERAL, TIER_RELIEF, ToleranceBand

BIND = "127.0.0.1"            # never 0.0.0.0: this server drives a physical rig
DEFAULT_PORT = 8765
DEFAULT_PRESET = "cell_wall_2p7ft_vstripe"   # fringes vertical in camera, projector right
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY = 5 * 1024 * 1024
MAX_JOBS_KEPT = 300
_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
STATIONS_DIR = Path(__file__).resolve().parents[2] / "stations"
RUN_FILES = {"result.json": "application/json", "run.log.jsonl": "text/plain; charset=utf-8",
             "deviation.png": "image/png", "overlay.png": "image/png",
             "feedback_dmd.png": "image/png", "feedback_photo.png": "image/png",
             "values.csv": "text/csv; charset=utf-8", "spec.json": "application/json",
             "scan.ply": "application/octet-stream", "quick_check.json": "application/json"}
# ^ what /runs/<run_id>/ serves of a run folder (recipe runs: overlay + the DMD feedback;
#   measure runs: values.csv, the overlay, the scan's PLY, the spec; quick checks: their config)
LIVE_IDLE_S = 30.0            # live view stops when nobody polled it for this long
MAX_LIVE_S = 1800.0           # longest live view one start may ask for (it holds the rig)
MAX_HOLD_S = 300.0            # pumped grid / zone holds can't be interrupted: keep them short
LOG_SCAN_BYTES = 64 * 1024 * 1024   # how far back /api/log looks for n matching records

# every response: this page must never be framed by another site (clickjacking)
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": ("default-src 'self'; img-src 'self' data: blob:; "
                                "style-src 'self' 'unsafe-inline'; script-src 'self'; "
                                "object-src 'none'; base-uri 'none'; form-action 'none'; "
                                "frame-ancestors 'none'"),
}

FILE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
              ".json": "application/json", ".csv": "text/csv; charset=utf-8",
              ".ply": "text/plain; charset=utf-8", ".xyz": "text/plain; charset=utf-8",
              ".txt": "text/plain; charset=utf-8",
              # meshes: always sent as downloads (never rendered inline)
              ".stl": "model/stl", ".obj": "model/obj"}
DOWNLOAD_ONLY = frozenset({".stl", ".obj"})
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml",
                ".png": "image/png", ".ico": "image/x-icon"}

# library kinds an entry can be resolved from / deleted in
DELETABLE = ("bases", "objects", "calibrations", "readings", "spectral", "banks", "maps",
             "materials", "sensors", "metric", "calib_poses", "dentsets", "faces", "facescans",
             "face_golden", "station_golden")
LIST_KINDS = ("bases", "objects", "calibrations", "readings", "outputs", "materials",
              "spectral", "banks", "maps", "sensors", "metric", "calib_poses", "dentsets",
              "faces", "facescans", "face_golden", "station_golden")
DENT_SUBSETS = ("all", "production", "production_nolines", "minimal")
# where a virtual-sensor spot / zone goes (project_spot, teach_presence)
AT_DOC = ("grid cell 'H8' (add '@<spacing_px>' to pin the grid you saw, e.g. 'H8@96'), "
          "{\"dmd\": [col, row]} DMD px, or {\"camera\": [x, y]} camera px (needs a projector "
          "map)")
LIVE_COLOURS = {"white": (255, 255, 255), "red": (0, 0, 255), "green": (0, 255, 0),
                "blue": (255, 0, 0)}                       # BGR, as the HDMI window draws
LIVE_PATTERNS = ("flood", "stripes", "checker")
# UI-only listings (not engine list_library kinds): light separations for
# bake_maps, Surfer atlases with their regions
UI_LIST_KINDS = ("lights", "atlases")
REGION_DOC = ("'atlas:<name>/<region>' (import_atlas first, e.g. atlas:part/part), a grid zone "
              "'P17:P22@48', or a DMD rect [x0, y0, x1, y1]")
ORIENTATIONS = ("v", "h", "d45", "d135")
FLASH_DOC = "docs/FLASH_PATTERN_MODE.md"
FASTSCAN_DOC = "docs/FLASH_FASTSCAN_4500.md"
SOFTSYNC_DOC = "docs/FLASH_SOFTSYNC.md"
FLASH_SUMMARY = (
    "Flash / hardware-trigger pattern mode (fast path): the pack holds BMPs + manifest.json + "
    "procedure.txt for the DLPC350 (flash images, or video-port bit-planes). A HUMAN switches "
    "the controller into pattern mode in the TI LightCrafter 4500 GUI; the engine never does. "
    "About 0.14 s per axis instead of ~15 s over HDMI, one exposure per pattern (no "
    "averaging). Nothing has run on the real rig yet: HDMI video mode stays the default.")

# ---------------------------------------------------------------- verb table
# The whitelist: every engine.api verb except configure / mock_bench, plus
# three UI verbs (tests/test_ui_server.py checks nothing is missing). The
# browser passes ids, never objects; the worker resolves them through the
# library listings. Doc strings double as help for AI agents (GET /api/verbs).
VERBS: dict = {
    "load_preset": ("Load a preset (validated; binds the latest calibration for its geometry).",
                    {"name": "preset name (GET /api/presets)"}),
    "preset_add": ("Add a USER preset (part recipe) to this library: a copy of `base` (built-in "
                   "or user) with fields overridden. Built-in presets are read-only; a new "
                   "geometry_id starts UNCALIBRATED.",
                   {"name": "new user preset name [A-Za-z0-9_.-]",
                    "base": "preset to copy (GET /api/presets)",
                    "overrides": "{\"camera.exposure_us\": 33333, \"projection.duty\": 0.8, ...}",
                    "geometry_id": "new geometry_id for a new rig profile (optional)",
                    "description": "text: what part / line it is for",
                    "replace": "bool: replace an existing USER preset (old one to trash/)"}),
    "preset_select": ("Select (and load) the library's active preset: the default when no preset "
                      "is named (this server loads it at start; a station run naming no preset "
                      "uses the LOADED one). A station config's own preset still wins. Refused "
                      "while a station run is busy.", {"name": "preset name (GET /api/presets)"}),
    "preset_delete": ("Move a USER preset to the library trash (recoverable). Refused for "
                      "built-ins, the active preset, the selected recipe's preset, and presets a "
                      "station config, a recipe or a recipe station config uses (unless force). "
                      "Needs confirm=true.",
                      {"name": "user preset name", "reason": "why (recorded)",
                       "confirm": "bool: must be true (the operator confirmed the delete)",
                       "force": "bool: delete even when a station config, a recipe (built-in "
                                "or user) or a recipe station config uses it (never the "
                                "selected recipe's preset)"}),
    "check_signal": ("Short live probe: gc_conf, valid %, focus ratio, coded-axis check.", {}),
    "capture_base": ("EMPTY scene -> base reference (the datum) for the active geometry. "
                     "zone: light only that grid zone (use the same zone as the zone scans).",
                     {"label": "text", "zone": "grid zone, e.g. 'H8:I11@96' (optional)"}),
    "capture_object": ("Object scene -> relief vs a base. Optional known dims and grid zone.",
                       {"label": "text", "base_id": "base id (default: latest usable base)",
                        "known_dims": "{H, W, L} in mm (any subset)",
                        "zone": "grid zone: 'C4', 'C4:D6', 'B2,E5' (see project_grid); add "
                                "'@<spacing_px>' (e.g. 'C4@96') to name cells of the grid "
                                "you saw, else the last projected grid's spacing is used"}),
    "reconstruct": ("Relief (+ calibration) -> .ply / .xyz in output/pointclouds.",
                    {"object_id": "scan id", "fmt": "ply | xyz"}),
    "measure": ("H/W/L: mm with a matching calibration, else px + warning (rule 6).",
                {"object_id": "scan id"}),
    "compare": ("Deviation vs a golden scan -> PASS / FAIL / INDETERMINATE (+ deviation PNG).",
                {"object_id": "scan id", "golden_id": "golden scan id",
                 "fail_mm": "mm band (default: preset tolerance.ch_fail_mm)",
                 "fail_px": "px band (used when the geometry is uncalibrated)",
                 "min_defect_px": "int", "smooth_px": "int"}),
    "calibrate": ("Fit Z (from known H) and XY (from known W/L); bound to the geometry.",
                  {"object_ids": "[scan ids with known dims]"}),
    "read_codes": ("Read QR / barcodes under bracketed DMD illumination.",
                   {"label": "text", "levels": "[DMD levels], default 255,170,110"}),
    "inspect": ("Defect blobs on a scan's relief vs the 0.7 mm trigger.",
                {"object_id": "scan id", "n_sigma": "float", "min_area_px": "int",
                 "bg_window": "odd int", "whole_field": "bool"}),
    "delete": ("Move a library entry to trash/ (recoverable).",
               {"kind": "|".join(DELETABLE), "id": "entry id", "reason": "text"}),
    "reject_base": ("Mark a base unusable (kept, skipped by latest_base).",
                    {"base_id": "base id", "reason": "text"}),
    "profile": ("Height along a line (p0 -> p1 camera px, or axis auto/x/y).",
                {"object_id": "scan id", "p0": "[x, y]", "p1": "[x, y]", "width_px": "int",
                 "axis": "auto | x | y"}),
    "bank_add": ("Add a known-good scan to a good memory bank.",
                 {"object_id": "scan id", "bank": "bank name"}),
    "bank_check": ("Flag anything in a scan not seen in the bank's good parts.",
                   {"object_id": "scan id", "bank": "bank name", "k_sigma": "float",
                    "min_area_px": "int"}),
    "bank_remove": ("Drop a scan from a good memory bank (appends a remove record; the scan "
                    "itself is kept).", {"object_id": "scan id", "bank": "bank name",
                                         "reason": "text"}),
    "tune": ("Scan assistant: sweep exposure (and duty); recommend settings. Never edits "
             "the preset.", {"exposures": "[us]", "duties": "[0..1]"}),
    "measure_gamma": ("Fit the projector's gray response (video-mode degamma).",
                      {"n_levels": "int"}),
    "capture_map": ("2-axis camera<->projector map of the current scene (grid zones need it).",
                    {"label": "text"}),
    "project_grid": ("Project a labelled grid (A1, B2, ...) and photograph it.",
                     {"spacing_px": "DMD px per cell", "label": "text",
                      "hold_s": f"float, 0..{MAX_HOLD_S:g} s (holds the rig; not interruptible)"}),
    "project_zones": ("Light custom zones with patterns (flood/outline/grid/stripes/checker).",
                      {"zones": "[{name, polygon [[x,y],...], space, pattern, level}]",
                       "label": "text",
                       "hold_s": f"float, 0..{MAX_HOLD_S:g} s (holds the rig; not interruptible)"}),
    # painted zone sets (engine/zones): the operator draws on the part with the mouse
    "zone_paint": ("The zone painter: the projector window takes the mouse (its cursor is "
                   "projected on the part); the operator drags rectangles / ellipses (E) / "
                   "polygons (P) over the areas, moves / resizes / rotates and names them; "
                   "Space saves the zone set (zones/<geometry>/<name>/: zones.json, masks.npz, "
                   "preview.png, camera.png), Esc cancels. The job holds the rig until then "
                   "(or until idle_timeout_s without input; cancelling the job ends it, nothing "
                   "saved). An existing name is refused unless replace. -> {saved, outcome "
                   "(saved | cancelled | aborted | idle_timeout | script_ended: how the session "
                   "ended), reason, zones, preview, camera, replaced, on, selection_note}. GET "
                   "/api/cmd/state carries the same for every page (painter, last_paint). "
                   "Refused (409) unless the station is READY or FAULT.",
                   {"name": "the zone set to save [A-Za-z0-9_.-]",
                    "source": "a saved set to start from (edit it)",
                    "replace": "bool: replace an existing set of this name (old one to trash/)",
                    "select": "bool: switch a NEW set on in the zone selection (a set painted "
                              "over keeps its on / off); either way its per-zone switches "
                              "start again at all zones",
                    "labels": "[names] the N key cycles through (optional)",
                    "polarity": "measure (default) | normal | inverted",
                    "note": "text stored with the set",
                    "idle_timeout_s": "give the rig back after this long without input "
                                      "(default 600)",
                    "events": "MOCK ONLY: the scripted session, a list of painter events "
                              "(default in MOCK: a built-in demo session)"}),
    "zone_show": ("Project saved zones on the part and report where each zone lands in "
                  "the camera image (camera px, centroid, box; seen or not), outlines + ids "
                  "photographed and held. Several sets: each zone carries key '<set>/<id>' and "
                  "zoneset. A show held 1 s or longer with every zone seen is recorded: it "
                  "answers the rig changes since the set was painted.",
                  {"zones": "zone set, optionally some zones: 'lid' or 'lid:Z1,Z3'; several "
                            "sets 'cells+tube:Z1'; or 'selection' (the sets switched on: GET "
                            "/api/zones/selection)",
                   "hold_s": f"float, 0..{MAX_HOLD_S:g} s (default 5; cancel the job to end it)",
                   "label": "text"}),
    "zone_delete": ("Move a zone set (at the loaded preset's geometry) to the library trash "
                    "(recoverable). Refused (unless force) while a recipe or a station "
                    "configuration names it, or it is switched on in the zone selection. Needs "
                    "confirm=true. Refused (409) unless the station is READY or FAULT.",
                    {"name": "zone set name (GET /api/zonesets)", "reason": "why (recorded)",
                     "confirm": "bool: must be true (the operator confirmed the delete)",
                     "force": "bool: delete even when it is named or switched on"}),
    "zone_select": ("Change the zone selection of the loaded preset's rig placement: which "
                    "painted zone sets (and zones) are switched on. Runs that ask for the "
                    "selection (the operator page's MEASURE / START in Quick check, `--zones "
                    "selection`, a metric station job with \"zone\": \"selection\") light and "
                    "judge exactly these. Stored append-only in the library (zones/<geometry>/"
                    "_selection.jsonl); a set that is switched on must resolve now. Refused "
                    "(409) unless the station is READY or FAULT and no measurement, quick "
                    "check, painter or zone show is queued or running. -> the selection (GET "
                    "/api/zones/selection). The page uses POST /api/zones/selection.",
                    {"sets": "the whole selection: [{set, on, ids}] (ids null = every zone)",
                     "zones": "or: exactly these are on, 'cells+tube:Z1' or a list of terms",
                     "on": "or: terms to switch on ('tube:Z1,Z2'; 'tube' = with the switches "
                           "it had)",
                     "off": "set names to switch off",
                     "clear": "bool: everything off (the whole field)",
                     "by": "who changes it (recorded with the selection)",
                     "expect": "the stamp of the selection the sender was shown: refused when "
                               "it changed meanwhile"}),
    "spectral_capture": ("R/G/B LED reflectance capture.", {"label": "text"}),
    "learn_material": ("Teach a material from a camera ROI of a spectral reading.",
                       {"reading_id": "spectral reading id", "name": "material name",
                        "roi": "[x0, y0, x1, y1] camera px"}),
    "classify": ("Per-pixel material map vs learned materials.",
                 {"reading_id": "spectral reading id", "materials": "[names] (default all)",
                  "max_distance": "float"}),
    "separate_light": ("Direct vs global light from shifted checkerboards.",
                       {"label": "text", "square_px": "int", "shifts": "int"}),
    # virtual sensors: the DMD draws the spot, the camera is the receiver
    "spot_size": ("Laser-spot-sensor-style spot sizing: a spot of D mm at the part -> DMD px, columns x rows "
                  "(the DLP4500 diamond array: a round spot spans 2x as many rows as columns). "
                  "NOMINAL from rig standoff / throw ratio unless a MEASURED scale is given.",
                  {"diameter_mm": "spot diameter at the part, mm",
                   "mm_per_dmd_px": "optional MEASURED mm per DMD column: project a spot, "
                                    "measure it on the part, measured_mm / diameter_dmd_px"}),
    "project_spot": ("Project one bright spot, photograph and measure it -> SpotReading "
                     "(centroid, diameter in camera px; mm only with a lateral calibration).",
                     {"at": AT_DOC, "diameter_mm": "mm at the part",
                      "mm_per_dmd_px": "optional MEASURED scale (see spot_size)",
                      "shape": "disk | square", "label": "text", "level": "DMD level 1-255",
                      "hold_s": f"float, 0..{MAX_HOLD_S:g} s (keeps the spot up to aim; "
                                "holds the rig; not interruptible)"}),
    "teach_presence": ("Teach a part-present sensor on the EMPTY scene. Clearing the field is "
                       "a human step: pass empty_confirmed=true only after the operator "
                       "confirmed the sensor's field is EMPTY.",
                       {"name": "sensor name", "at": AT_DOC + "; or a zone 'G7:H9' "
                                                          "(photoelectric only)",
                        "diameter_mm": "spot diameter, mm (spot sensors)",
                        "method": "photoelectric | bgs (background suppression)",
                        "empty_confirmed": "bool: the operator confirmed the field is EMPTY",
                        "mm_per_dmd_px": "optional MEASURED scale (see spot_size)",
                        "threshold": "photoelectric: relative change that means PRESENT "
                                     "(fraction, default 0.25)",
                        "min_height_mm": "bgs: height above the taught background, mm "
                                         "(default 3)",
                        "level": "DMD level 1-255"}),
    "check_presence": ("Fire a taught sensor -> PRESENT / ABSENT / INDETERMINATE.",
                       {"name": "sensor name (list_library kind 'sensors')",
                        "threshold": "override the taught fraction",
                        "min_height_mm": "override the taught bgs height, mm",
                        "min_relief_px": "bgs without any mm scale: decide on DMD px",
                        "label": "text"}),
    "teach_alignment": ("Store the reference pose (centroid, angle, extents) of the part in a "
                        "scan.", {"name": "sensor name", "object_id": "scan id"}),
    "check_alignment": ("dx / dy / dtheta of the part in a scan vs the taught pose -> ALIGNED / "
                        "NOT_ALIGNED / INDETERMINATE (mm only with a lateral calibration).",
                        {"name": "alignment sensor name", "object_id": "scan id",
                         "tol_mm": "position tolerance, mm (default 1)",
                         "tol_deg": "angle tolerance, deg (default 2)",
                         "tol_px": "position tolerance, camera px (uncalibrated)"}),
    # 3D meshes + cell unwrap (rig-free: work on a stored scan)
    "export_mesh": ("Scan -> STL / OBJ mesh + meta JSON in output/pointclouds (mm only with a "
                    "matching calibration, rule 6). kind: relief heightmap, the base surface, "
                    "or the measured open cylinder arc (never a closed, invented tube).",
                    {"object_id": "scan id", "fmt": "stl | obj", "kind": "relief | base | cylinder",
                     "stride": "int >= 1, camera px per vertex (default 2)",
                     "radius_mm": "cylinder: known cylinder radius, mm (optional)",
                     "roi": "[x0, y0, x1, y1] camera px (optional)"}),
    "cell_unwrap": ("Cylinder fit + (theta, z) unwrap of a standing cylinder -> arc degrees, end-face "
                    "height, dent heatmap PNG (open arc only).",
                    {"object_id": "scan id", "radius_mm": "known radius, mm (optional)",
                     "step": "grid step, units (optional)"}),
    "cell_dents": ("Dents on the unwrapped cylinder -> PASS / FAIL / INDETERMINATE (mm verdict only "
                   "with a depth calibration, rule 6) + table and heatmap.",
                   {"object_id": "scan id", "radius_mm": "known radius, mm (optional)",
                    "n_sigma": "detection threshold x noise (default 5)",
                    "min_area_bins": "int (default 6)",
                    "depth_trigger_mm": "FAIL depth, mm (default: preset ch_fail_mm)"}),
    "bake_maps": ("Aligned per-pixel map stack of a scan: height, cavity, normal, validity, "
                  "density (+ albedo / subsurface / dust with readings) -> PNG + npz.",
                  {"object_id": "scan id", "spectral_id": "spectral reading id -> albedo",
                   "light_id": "light separation id (GET /api/library/lights) -> subsurface",
                   "golden_id": "spectral reading of the golden part -> dust",
                   "sigma_px": "float (default 2)", "label": "text"}),
    "checker_scan": ("Project a checkerboard, find its corners (camera px <-> DMD mirrors), local "
                     "Jacobian -> stretch / contrast PNGs + summary.",
                     {"square_px": "DMD px (default 32)", "shifts": "int (default 1)",
                      "label": "text"}),
    # Surfer atlas + methods
    "import_atlas": ("Copy a Surfer atlas JSON into library/atlases/<geometry>/<name>/ (the source "
                     "is only read). Its regions then work as zones 'atlas:<name>/<region>'.",
                     {"path": "local .json file", "name": "atlas name (e.g. part)"}),
    "list_atlases": ("Imported atlases (+ their regions).", {"geometry_id": "filter"}),
    "flood_verify": ("Light only a region (black elsewhere), photograph, report where it lands.",
                     {"region": REGION_DOC, "level": "DMD level 1-255", "label": "text"}),
    "anti_glare_sweep": ("Pick the tilted-carrier pattern that least clips glare in a region "
                         "(pattern choice, not optical wavefront shaping).",
                         {"region": REGION_DOC, "carriers": "[[fx, fy(, z)], ...] (default set)",
                          "steps": "int", "cycles": "float", "level": "1-255", "label": "text"}),
    "multi_orientation_scan": ("H / V / diagonal fringes in a region fused by modulation -> "
                               "quality / modulation maps.",
                               {"region": REGION_DOC,
                                "orientations": "subset of " + ", ".join(ORIENTATIONS),
                                "steps": "int", "period_px": "float", "level": "1-255",
                                "label": "text"}),
    "make_flash_pack": ("Build a DLPC350 flash / video-port pattern pack (files only, never "
                        "touches the rig; see " + FLASH_DOC + ").",
                        {"axes": "x | y (one axis per pack)", "video_port": "bool",
                         "label": "text"}),
    "make_fastscan_pack": ("Build the 1-bit soft-sync fast-scan pack: a full-field and (with zone "
                           "sources) a zone-masked 24-bit flash image, 16 one-bit slots each, "
                           "-> output/flashpacks (files only, never touches the rig; see "
                           + FASTSCAN_DOC + ").",
                           {"zone_rings": "[local face_dmd_ring.npy, ...] (ellipse + margin)",
                            "zone_masks": "[local DMD mask .npy, ...]",
                            "band_cam": "local camera-space band mask .npy",
                            "band_field": "local both-axes field.npz for band_cam",
                            "margin_px": "int (default 6)",
                            "calibration": "stereo calibration folder (optional)",
                            "study": "bool: simulate the decode error per level",
                            "label": "text"}),
    "fast_scan_flash": ("Fast metric scan (~0.3-0.5 s of frames) from the DLPC350 flash loop: "
                        "a HUMAN loads the fast-scan pack with the 4500 MKII GUI's firmware link "
                        "and plays it (Pattern Sequence, Flash, Internal/External, Play Once "
                        "off); the "
                        "engine opens ONLY the camera, free-running, sorts the frames in software "
                        "and decodes -> a metric scan (mode flash_softsync). Refused when the loop "
                        "is not playing / incomplete / out of order. See " + SOFTSYNC_DOC + ".",
                        {"pack": "local fast-scan pack folder (output/flashpacks/...)",
                         "variant": "full | zone (the image the GUI plays; default: reported)",
                         "label": "text", "zone": "camera window from grid cells / atlas region",
                         "roi": "[x0, y0, x1, y1] camera px window",
                         "loops": "int >= 2 (default: auto)",
                         "exposure_us": "camera exposure (default: the pack's)",
                         "fps": "camera frame rate (default: the window's maximum)",
                         "gain_db": "camera gain (default: the preset's)",
                         "keep_recording": "bool (default true): raw frames to output/flashscans",
                         "dry_run": "bool: the timing plan only, nothing is opened",
                         "fine_prefilter_px": "camera px low-pass of the fine fringes, 0..3 "
                                              "(1 = the pack's recommendation at a sharp "
                                              "projector; default off)"}),
    "sort_flash_recording": ("Offline soft sync: re-sort a saved flash recording "
                             "(output/flashscans/*.npz) into per-pattern images (+ projector "
                             "columns with decode). Never touches the rig.",
                             {"recording": "local recording .npz",
                              "pack": "local pack folder (default: the layout in the recording)",
                              "variant": "full | zone", "decode": "bool", "label": "text",
                              "fine_prefilter_px": "with decode: fine-fringe low-pass, camera "
                                                   "px 0..3 (default off)"}),
    "list_library": ("Browse the library.", {"kind": "|".join(LIST_KINDS),
                                             "geometry_id": "filter (outputs: output kind)"}),
    "latest_base": ("Newest usable base for the active geometry.", {}),
    # metric calibration + measurement (mm only with a stereo calibration, rule 6)
    "make_charuco": ("Printable ChArUco board -> output/calibration (PNG + true-size PDF + spec "
                     "JSON). Print at 100 %, then measure a square with calipers.",
                     {"squares_x": "int (default 10)", "squares_y": "int (default 7)",
                      "square_mm": "square side, mm (default 25)",
                      "marker_mm": "marker side, mm (default 18)",
                      "dictionary": "cv2.aruco dictionary (default DICT_5X5_100)",
                      "dpi": "print resolution (default 600)", "label": "text"}),
    "row_parity": ("Diamond DMD row-parity check (bench step B3): which rows sit half a column "
                   "right. Stored with the geometry; later scans draw straight fringes.",
                   {"label": "text"}),
    "capture_calib_pose": ("One ChArUco board pose for calibrate_stereo: flood + both-axis "
                           "multifreq decode. The operator holds the board still (12-20 poses, "
                           "tilts +-30 deg).",
                           {"set_label": "pose set name (default stereo)",
                            "steps": "phase steps (default 12)",
                            "periods": "[T, ...] DMD px, fine -> coarse (default: the pose "
                                       "set's, else 8, 80, 1280 - captured again at 16,160,1280 "
                                       "/ 24,192,1280 when under 60 % of the corners get a "
                                       "projector position; given: used as they are)",
                            "diamond_parity": "auto | 0 | 1 | null (square grid)",
                            "fixture": "calibration fixture spec JSON path: one PLACEMENT of the "
                                       "fixture, one view per face (3 placements in all)"}),
    "calibrate_stereo": ("Camera + projector (inverse camera) + stereoCalibrate from a pose set "
                         "-> geometry-bound stereo calibration (per-pose and per-axis RMS, the "
                         "poses that gave projector points). An error, nothing saved, when the "
                         "fit is not a calibration (degenerate projector intrinsics, stereo RMS "
                         "> 2 px, too few poses with projector points).",
                         {"set_label": "pose set (default: the newest)",
                          "square_mm": "caliper-measured printed square, mm (single board)",
                          "projector_distortion": "bool: fit projector k1/k2 (default false)",
                          "mock_poses": "MOCK ONLY: N board poses placed by the simulator",
                          "steps": "phase steps (mock poses)", "periods": "[T, ...] DMD px",
                          "diamond_parity": "auto | 0 | 1 | null",
                          "fixture": "fixture set: the spec JSON path with the caliper-measured "
                                     "squares (default: the spec stored with the placements)",
                          "mock_fixture": "MOCK ONLY: N fixture placements by the simulator",
                          "rigid": "bool: rigid-fixture refinement (default true)"}),
    "scan_metric": ("Both-axis multifreq scan -> point cloud in mm (+ .ply, per-pixel "
                    "consistency error). Refused without a stereo calibration (rule 6).",
                    {"label": "text", "method": "auto | plane | rays",
                     "steps": "phase steps", "periods": "[T, ...] DMD px",
                     "max_consistency_mm": "drop pixels whose rays miss by more (optional)"}),
    "triangulate": ("Re-triangulate a stored scan with the current stereo calibration -> new "
                    "metric scan + .ply (mm).",
                    {"object_id": "scan id (3D scan)", "metric_id": "or a metric scan id",
                     "method": "auto | plane | rays", "label": "text"}),
    "cylinder_deviation": ("Cylinder fit on a metric scan -> radial deviation map (mm, dents "
                           "negative) + dent blobs with depth and area.",
                           {"metric_id": "metric scan id", "radius_mm": "fix the radius, mm",
                            "n_sigma": "detection threshold x noise (default 4)",
                            "min_depth_mm": "smallest reported dent (default 0.05)",
                            "roi": "[x0, y0, x1, y1] camera px (optional)"}),
    "compare_metric": ("Point-to-plane ICP of a metric scan onto a golden metric scan -> signed "
                       "deviation (mm) and below-surface blobs.",
                       {"metric_id": "scan", "golden_id": "golden metric scan",
                        "max_dist_mm": "ICP pairing distance (default 5)",
                        "n_sigma": "blob threshold x noise (default 4)"}),
    "accuracy_check": ("Known reference object over repeat metric scans -> mean / max error, "
                       "repeatability, GO / NO-GO (bench step B6).",
                       {"kind": "plane | step | cylinder | sphere",
                        "height_mm": "step: known height", "diameter_mm": "cylinder / sphere",
                        "metric_ids": "[ids] of existing metric scans (else new scans)",
                        "repeats": "new scans (default 3)", "tol_mm": "mean error (default 0.1)",
                        "repeat_tol_mm": "repeat sigma (default 0.03)", "label": "text"}),
    "accuracy_block": ("Scan the printed known-geometry check block: reference flatness, step "
                       "heights, bar / dome diameter and height vs the caliper values -> "
                       "CONTINUE / REJECT (recalibrate) / PULL (doubt).",
                       {"block": "check block spec JSON path", "measured": "filled-in measured "
                        "template JSON path (without it: PULL)",
                        "metric_ids": "[ids] of existing metric scans (else new scans)",
                        "repeats": "new scans (default 1)",
                        "tol_height_mm": "heights (default: the spec's)",
                        "tol_diameter_mm": "diameters (default: the spec's)",
                        "flatness_max_mm": "reference face RMS limit (default: the spec's)",
                        "label": "text"}),
    "predict_precision": ("Predicted depth noise sigma_z in mm before scanning (Lv & Kemao).",
                          {"z_mm": "working distance (default: rig standoff)",
                           "A_dn": "fringe offset, grey levels", "B_dn": "modulation, grey levels",
                           "period_px": "finest period (default 8)", "steps": "(default 12)",
                           "frames_averaged": "(default: preset)", "gain_db": "(default: preset)",
                           "metric_id": "read A / B / T / N from this metric scan",
                           "extra_noise_dn": "extra per-frame noise, grey levels"}),
    "export_metric_mesh": ("Metric scan -> STL / OBJ in mm, open surface only (depth jumps left "
                           "open, nothing invented) + meta JSON.",
                           {"metric_id": "metric scan id", "fmt": "stl | obj",
                            "stride": "camera px per vertex (default 1)",
                            "max_edge_mm": "longest triangle edge (default 4 x spacing)",
                            "foreground_only": "bool: drop the background plane"}),
    # dent detection on shiny cylindrical parts (mm only with a stereo calibration, rule 6)
    "capture_dent_set": ("One pose of a cylinder for the dent detector: long pass (the active preset, "
                         "normally the dark 66 667 us one: floods + sin0 T12/T36 + sin90 "
                         "T12/T36) and "
                         "short pass (16 667 us: floods + the same sinusoids + lines_v), one rig "
                         "session each, + ONE metric scan of the same zone -> dentsets/. Refused "
                         "before the rig without a stereo calibration (rule 6).",
                         {"zone": "grid zone, e.g. 'H9:I12' (optional)", "label": "text",
                          "short_preset": "preset name of the short pass (default: the active "
                                          "preset at short_exposure_us)",
                          "short_exposure_us": "short-pass exposure, us (default 16667)",
                          "metric_preset": "preset name of the metric scan (default: active)",
                          "lines_v": "bool (default true)", "metric": "bool (default true)",
                          "steps": "metric scan phase steps", "periods": "[T, ...] DMD px",
                          "pattern_set": "production | minimal (drops sin90 T36)"}),
    "calibrate_dent_null": ("Negative control for the dent detector: run the four finders on "
                            "dent set(s) of KNOWN-GOOD parts of one type in one fixture; their "
                            "maxima become a geometry-bound calibration (1.0 = the good part's "
                            "strongest response).",
                            {"dentset_ids": "[dent set ids] (library dent sets only)",
                             "cell_diameter_mm": "cylinder diameter, mm", "label": "text",
                             "metric_ids": "[metric scan ids] of the one good dent set (default: "
                                           "its own)", "fixture": "fixture name (default: zone; "
                                                                  "detect_dents then needs it)",
                             "subset": "|".join(DENT_SUBSETS)}),
    "detect_dents": ("Dents / scratches on a cylinder -> candidates (camera px, theta / s on the cylinder, "
                     "fused score x the good-part max, level detection (> 1.25) / watch / edge, "
                     "depth mm with reliability and significance flags, size, per-finder "
                     "evidence, kind dent / scratch / speck) + overlay PNG. FLAG when any "
                     "detection.",
                     {"dentset_id": "dent set id (library dent sets only)",
                      "cell_diameter_mm": "mm (default: that of the newest matching null "
                                          "calibration)",
                      "metric_ids": "[metric scan ids] (default: the dent set's own)",
                      "null_id": "dent null calibration id (default: newest matching)",
                      "fixture": "fixture the null was made for (default: the dent set's zone)",
                      "label": "text", "subset": "|".join(DENT_SUBSETS)}),
    # end faces: a cylinder lying with its circular END FACE toward the camera (mm: rule 6)
    "locate_face": ("Locate a circular end face in a metric scan (both axes; scan_metric with "
                    "method plane and a zone around the face): face plane, centre, fixed-radius "
                    "rim circle, tilt, camera face mask and the camera -> DMD affine on the "
                    "decoded face (residual DMD px; warns > 0.5) -> face target in faces/. "
                    "Known size: the face's flood outline vs face_diameter_mm; more than "
                    "size_tol_pct off = result.meta.size_check mismatch, disposition PULL, "
                    "pull_reasons [size_mismatch] (calibration or lens suspect; nothing is "
                    "rescaled). result.meta.centre_check: the outline's centre vs the located one.",
                    {"metric_id": "metric scan id (the locate scan)",
                     "face_diameter_mm": "diameter of the bright FLAT the flood shows, mm "
                                         "(default 21.05; not the cylinder's when the rim is "
                                         "rounded)",
                     "silhouette_diameter_mm": "the part's OUTER diameter when its rim returns "
                                               "depth points (default: found from the rim)",
                     "size_tol_pct": "known-size tolerance, % (default 4; 0 = off, warned)",
                     "label": "text",
                     "roi": "[x0, y0, x1, y1] camera px to search (optional)",
                     "near_band_mm": "depth face band (default 12)",
                     "far_gap_mm": "background gap behind the face (default 30)",
                     "near_gap_mm": "a neighbour this far behind the face plane marks the rim "
                                    "(default 2; the neighbour split)",
                     "flood_metric_id": "metric scan whose flood pair outlines the face when "
                                        "the depth rim fails (default: the scan's own, else its "
                                        "decode modulation)"}),
    "locate_scan": ("The scan locate_face wants, made now: both axes, ray-plane, lit through a "
                    "zone, its flood pair kept (the size check needs it) -> metric scan.",
                    {"zone": "grid cells 'H10:I12', atlas:<name>/<region> or zoneset:<set> "
                             "(default: full field)", "label": "text",
                     "steps": "phase steps", "periods": "[T, ...] DMD px",
                     "exposure_us": "camera exposure for this scan only (optional)"}),
    "locate_face_live": ("Scan and locate an end face in one go, at the exposure the face "
                         "needs: with auto exposure the zone is scanned again (at most 3 scans, "
                         "frame-locked exposures) when no face is found or the located face is "
                         "dark / clipped, and the face located on the scans merged per pixel -> "
                         "face target (result.meta.locate_scan: scans, exposures, notes); the "
                         "size and centre checks as locate_face.",
                         {"zone": "grid cells / atlas region / zone set to scan",
                          "face_diameter_mm": "diameter of the bright flat, mm (default 21.05)",
                          "silhouette_diameter_mm": "the part's outer diameter (optional)",
                          "size_tol_pct": "known-size tolerance, % (default 4; 0 = off)",
                          "label": "text", "roi": "[x0, y0, x1, y1] camera px (optional)",
                          "steps": "phase steps", "periods": "[T, ...] DMD px",
                          "auto_exposure": "bool (default: on when the preset's exposure is "
                                           "whole video frames)",
                          "exposure_us": "force one exposure (one scan)",
                          "exposure_range_us": "[lo, hi] us (default 16667..100000)",
                          "near_band_mm": "(default 12)", "far_gap_mm": "(default 30)",
                          "near_gap_mm": "(default 2)"}),
    "face_exposure": ("The frame-locked camera exposure for this seating of a located face: "
                      "flood shots through the face mask, p95 inside 0.8 R near target_dn with "
                      "< max_clipped_pct clipped -> {ok, exposure_us, p95_dn, clipped_pct, "
                      "shots, why}; ok false = disposition PULL (too dark / clipped / mirror). "
                      "The preset is not edited, the LED currents are never touched.",
                      {"face_id": "face target id (default: newest)",
                       "target_dn": "camera DN (default 170)",
                       "exposure_range_us": "[lo, hi] us (default 16667..100000)",
                       "max_clipped_pct": "% of the face (default 0.5)",
                       "duty": "projection.duty override", "label": "text"}),
    "aim_ring": ("Project a face target's ring + cross and photograph it: the operator's "
                 "placement guide (hold_s keeps it up, pumped hold).",
                 {"face_id": "face target id (default: the newest at the preset's geometry)",
                  "hold_s": f"float, 0..{MAX_HOLD_S:g} s", "label": "text",
                  "offset_dmd": "[cols, rows] shift, DMD px (optional)"}),
    "check_ring": ("Is the face still where it was located? A flood through the face's DMD "
                   "ellipse must light >= 90 % of the located face (<= 10 % outside) and the "
                   "photographed ring must land on the located rim -> ON_FACE | OFF | NOT_FOUND.",
                   {"face_id": "face target id (default: newest)", "hold_s": "float s",
                    "label": "text", "tol_px": "ring centre tolerance, camera px (default 1.5)",
                    "offset_dmd": "[cols, rows] DMD px (optional)"}),
    "auto_intensity": ("projection.duty for a face p99 of ~target_dn camera DN at the preset's "
                       "exposure, from flood shots through the face mask. Never edits the preset "
                       "and never touches the LED currents; pass the duty to targeted_face_scan.",
                       {"face_id": "face target id (default: newest)",
                        "target_dn": "camera DN (default 200)", "tol_dn": "DN (default 10)",
                        "max_shots": "flood shots (default 4)", "label": "text"}),
    "targeted_face_scan": ("Targeted end-face scan: only the face ellipse (+ margin) is lit; N "
                           "repeats of the multifreq depth set (ray-plane triangulation) -> metric "
                           "scans, + the polar set (rings, spokes, floods) -> facescans/. Refused "
                           "before the rig without a stereo calibration (rule 6).",
                           {"face_id": "face target id (default: newest)", "label": "text",
                            "repeats": "1..20 (default 3)",
                            "margin_dmd_px": "lit margin, DMD px (default 6)",
                            "polar": "bool: rings + spokes (default true; the flood pair is "
                                     "always captured)", "duty": "projection.duty override",
                            "steps": "depth-set phase steps", "periods": "[T, ...] DMD px",
                            "polar_steps": "steps per polar family (default 8)",
                            "auto_exposure": "bool (default: on when the preset's exposure is "
                                             "whole video frames): a frame-locked exposure per "
                                             "seating; none fits = NOT scanned (job error "
                                             "ExposureNotFound: PULL / re-seat). "
                                             "result.meta.exposure_us / exposure_source / "
                                             "auto_exposure say what was used",
                            "exposure_us": "force a camera exposure for this scan",
                            "exposure_target_dn": "face p95 target, DN (default 170)",
                            "exposure_range_us": "[lo, hi] us (default 16667..100000)",
                            "max_clipped_pct": "% of the face clipped at most (default 0.5)"}),
    "face_profile": ("Face height map (face frame, mm, 0.25 mm grid), radial profile, sections, "
                     "flatness, per-point repeatability; the rim outline in the flood pair "
                     "(result.stats.rim_outline) with the known-size check (stats.size_check) "
                     "and the centre the profile was computed about (stats.centre: the flood "
                     "outline's when it shows whole); PNG + CSV + JSON + NPZ + open STL.",
                     {"facescan_id": "face scan id", "metric_ids": "or [metric scan ids] of one "
                      "placement (then face_id)", "face_id": "face target id (default: the face "
                      "scan's)", "label": "text", "grid_mm": "(default 0.25)",
                      "r_flat_mm": "flatness radius (default 0.855 x the face radius; a value "
                                   "given is used as given)",
                      "size_tol_pct": "known-size tolerance, % (default 4; 0 = off, warned)",
                      "stl": "bool (default true)"}),
    "face_golden": ("Golden end face from GOOD placements (face scans), bound to geometry, face "
                    "diameter, grid and stereo calibration: MF-CFAR (+ the reflectance FIR from the "
                    "flood pair; default detector) and band models, leave-one-out false triggers "
                    "and a synthetic-dent set -> face_golden/ (append-only).",
                    {"facescan_ids": "[face scan ids] of GOOD placements (>= 2, 3+ for "
                                     "leave-one-out)", "label": "text",
                     "detector": "mf_fir | mf | band (default mf_fir)",
                     "z_thresh": "MF trigger z (default 5.5)",
                     "min_depth_mm": "MF: ignore shallower dents (default 0)",
                     "fir": "bool: reflectance FIR (default true)",
                     "k_sigma": "band (default 4)", "floor_mm": "band (default 0.10)",
                     "smooth_cells": "band sigma median window (default 5)",
                     "rim_r_mm": "rim zone r >, ONE radius for every detector (default "
                                 "0.855 x the face radius; used as given or refused)",
                     "inspect_r_mm": "inspected zone r <= (default 0.926 x the face radius)",
                     "plane_r_mm": "robust plane region (default: the rim radius)",
                     "rim_factor": "(default 2)",
                     "rim_min_mm": "(default 0.3)", "height_smooth_mm": "(default 0.25)",
                     "min_area_mm2": "band: smallest trigger (default 0.25)",
                     "rim_tol_mm": "rim outline: a notch / bulge beyond this (default 0.4) ...",
                     "rim_min_arc_mm": "... over at least this much rim (default 1.5)",
                     "size_tol_pct": "known size: a placement further off the face diameter is "
                                     "refused (default 4; 0 = off, warned)",
                     "synthetic": "bool: synthetic-dent set (default true)"}),
    "check_face": ("A placement (face scan) vs the golden face: dents (zone face / rim, r, theta, "
                   "depth_mm = depth_centre_mm, depth_min_mm, width_half_depth_mm, z (MF) or "
                   "dev_mm (band), area_mm2) and edge dents (zone rim_outline: a notch / bulge "
                   "of the flood outline, depth_mm + = in, arc_mm) -> PASS / FAIL / INCOMPLETE "
                   "+ overlay PNG. result.disposition CONTINUE / REJECT / PULL with "
                   "result.pull_reasons [{reason, message}] (size_mismatch, guard_band, "
                   "incomplete, invalid); result.stats.size_check / rim_outline / centre_check "
                   "/ radii say why.",
                   {"facescan_id": "face scan id", "golden_id": "golden face id (default: newest "
                    "matching)", "label": "text", "detector": "mf_fir | mf | band (default: the "
                    "golden's)", "z_thresh": "MF trigger z (default: the golden's)",
                    "min_depth_mm": "MF: ignore shallower dents (optional)",
                    "min_area_mm2": "smallest trigger (optional)",
                    "rim_tol_mm": "rim outline tolerance, mm (default: the golden's)",
                    "rim_min_arc_mm": "rim outline minimum arc, mm (default: the golden's)",
                    "size_tol_pct": "known-size tolerance, % (default: the golden's; 0 = off, "
                                    "warned)"}),
    # HDMI fast zone scan + station (cylinders with end faces toward the camera + ribbed tube)
    "scan_fast": ("HDMI fast zone scan: the column-axis multifreq set only (default 4 steps x "
                  "8 / 80 / 1280 DMD px + white / black), lit through a zone and / or located "
                  "faces, camera average + HDMI hold for this run only (average < 2 is raised "
                  "to 2: HDMI flicker) -> metric scan (ray-plane, mm) + its flood pair. Bench: "
                  "5.3 s for the whole station scene. Refused without a stereo calibration.",
                  {"zone": "grid cells 'F10:J12' or atlas:<name>/<region> (default: full field)",
                   "zones": "or a painted zone set 'lid' / 'lid:Z1,Z3' (GET /api/zonesets), "
                            "several 'cells+tube:Z1', or 'selection' (the sets switched on): "
                            "only those zones are lit; result.meta.zones has the points per "
                            "zone (exact with axis xy; keyed '<set>/<id>' for several sets)",
                   "face_ids": "[face target ids] also lit (their DMD ellipse + margin)",
                   "margin_dmd_px": "face margin, DMD px (default 6)", "label": "text",
                   "steps": "phase steps (default 4)", "periods": "[T, ...] DMD px",
                   "average": "camera frames per pattern (default 2, min 2)",
                   "min_hold_s": "HDMI hold per pattern, s (default 0.12)", "axis": "x | xy"}),
    "station_measure": ("Cylinders with their end faces toward the camera + a ribbed/serpentine tube "
                        "between them, from one metric scan + a flood pair: end-face heights along "
                        "the common face normal, Cj-Ci, spacing; tube edge band, trapezoid wave, "
                        "zones T1..Tn with the edge height vs the face plane -> PNG + JSON + CSV "
                        "(output/measurements).",
                        {"metric_id": "metric scan id",
                         "flood_metric_id": "metric scan whose flood pair to use (default: the "
                                            "scan's own, else its decode modulation)",
                         "label": "text", "n_cells": "int (default 2)",
                         "face_diameter_mm": "mm (default 21.05)",
                         "ct_band_mm": "[lo, hi] tube edge height band, mm (default auto)",
                         "zone_skip": "[left, right] zones dropped at the ends (default [0, 0])",
                         "centre_method": "rim_circle | ellipse_ray"}),
    "station_golden": ("Station golden from GOOD scans (measured here): nominal end-face height "
                       "differences, spacings and tube zone heights, the zone model + numbering, and "
                       "the limits -> station_golden/ (append-only, bound to geometry + stereo "
                       "calibration).",
                       {"metric_ids": "[metric scan ids] of GOOD placements", "label": "text",
                        "cell_height_tol_mm": "nominal +- (default 0.15)",
                        "ct_zone_tol_mm": "nominal +- (default 0.30)",
                        "spacing_tol_mm": "nominal +- (optional)",
                        "face_position_tol_mm": "each end face within this of its golden "
                                                "position, 3D mm (default the face diameter)",
                        "cell_height_band_mm": "[lo, hi] absolute (optional)",
                        "ct_zone_band_mm": "[lo, hi] absolute (optional)",
                        "bands": "{item: [lo, hi]}, e.g. {'C2-C1': [-0.4, 0.1], 'T3': [-5, -4]}",
                        "n_cells": "int (default 2)", "face_diameter_mm": "mm (default 21.05)",
                        "ct_band_mm": "[lo, hi] tube edge height band for the measurement (default "
                                      "auto)",
                        "zone_skip": "[left, right] (default [0, 0]): the golden's numbering",
                        "centre_method": "rim_circle | ellipse_ray"}),
    "station_check": ("A metric scan vs the station golden, re-measured with the golden's "
                      "parameters and zones -> PASS / FAIL / INCOMPLETE with the triggered items "
                      "(PNG + JSON + CSV in output/defects).",
                      {"metric_id": "metric scan id", "golden_id": "station golden id (default: "
                       "newest matching)", "flood_metric_id": "metric scan whose flood pair to use",
                       "label": "text"}),
    "recipe_add": ("Add a USER recipe (work-state check) to this library: a copy of `base` (a "
                   "built-in / user recipe, GET /api/recipes) or the mapping `recipe`, with fields "
                   "overridden; validated (every problem listed), its preset must exist. Built-in "
                   "recipes are read-only.",
                   {"name": "new user recipe name [A-Za-z0-9_.-]",
                    "base": "recipe to copy (GET /api/recipes)",
                    "recipe": "a whole recipe mapping instead of base (schema_version 1)",
                    "overrides": "{\"feedback.hold_s\": 10, \"zones.0.checks.0.params.tol_mm\": "
                                 "0.4}",
                    "description": "text", "replace": "bool: replace an existing USER recipe"}),
    "recipe_select": ("Select the library's recipe (the default of `scan recipe-run`; a station "
                      "config's job.recipe still wins).", {"name": "recipe name"}),
    "recipe_delete": ("Move a USER recipe to the library trash (recoverable). Refused for "
                      "built-ins, the selected recipe and recipes a station config uses (unless "
                      "force). Needs confirm=true.",
                      {"name": "user recipe name", "reason": "why (recorded)",
                       "confirm": "bool: must be true", "force": "bool"}),
    "recipe_run": ("Run a recipe: scan with its pinned preset, run every zone's checks, verdict "
                   "PASS / FAIL / INVALID and disposition CONTINUE / REJECT / PULL (quality "
                   "inspection: any doubt), a new run folder (result.json with zones + checks, "
                   "disposition + pull_reasons, run.log.jsonl, overlay.png, feedback_dmd.png), "
                   "then paint it on the part (green outline CONTINUE, red fill + cross at the "
                   "fault REJECT, amber frame + PULL) for hold_s (cancel the job to clear it). "
                   "The real rig needs operator_confirmed.",
                   {"recipe": "recipe name (GET /api/recipes)",
                    "operator_confirmed": "bool (required on the real rig)",
                    "scan_id": "check a stored metric scan instead of capturing (optional)",
                    "project_feedback": "bool (default true)",
                    "hold_s": "feedback hold, s (default the recipe's)",
                    "mock_scene": "MOCK ONLY: scene:<name>[:<variant>] placed first"}),
    "recipe_golden": ("Scan a GOOD part with a recipe's scan settings and record it as the "
                      "recipe's golden (labelled the recipe's golden.scan; recipe_run uses only "
                      "recorded goldens - newest wins, 'latest' = this recipe's newest - or an "
                      "exact scan id).",
                      {"recipe": "recipe name", "mock_scene": "MOCK ONLY: scene:<name>[:<variant>]"}),
    "recipe_feedback": ("Paint a recipe run's verdict on the part again (its feedback_dmd.png) for "
                        "hold_s; cancel the job (or any station command) to clear it.",
                        {"run_id": "a run of this server (GET /runs/<run_id>/result.json)",
                         "hold_s": "seconds (default the run's)"}),
    "rig_check": ("Rig check (HOME): a fresh flood + a random block pattern vs the stored "
                  "reference -> camera / projector motion over the frame (shift + roll + zoom: "
                  "tile phase correlation + a similarity fit; the largest displacement is "
                  "judged), px and ~mm at the working distance. OK only with the pattern "
                  "matched and its roll / zoom determined | MOVED (also: the pattern visible "
                  "but no longer matching) | UNKNOWN (no pattern or texture, roll / zoom not "
                  "determined, or the flood no longer matching) | REFERENCE_SET (first check). "
                  "Recorded on the rig-state record: for a mm result an OK counts only at the "
                  "default tolerance or tighter, against a reference of the calibration in use "
                  "stored while the rig could be trusted (`warnings` say when it does not).",
                  {"set_reference": "bool: store a new reference (after base / calibration)",
                   "tol_px": "MOVED above this image motion, px (default 0.5; finite, >= 0; an "
                             "OK at a looser one does not verify the rig)",
                   "mock_shift_px": "MOCK ONLY: [dx, dy] or [dx, dy, roll_deg] or [dx, dy, "
                                    "roll_deg, scale]: moves the fresh frames"}),
    "flow_verify": ("The DLP check of ONE work state of a station configuration (profiles + "
                    "flow) for a profile: scan with the profile's preset, run the recipe that "
                    "work state verifies with, a new run folder; nothing is projected and no "
                    "permission is granted (the command bus does that: POST /api/cmd/start, "
                    "/step). -> the run's result + dlp {result, disposition, observed: {signal: "
                    "true | false | null}}. The real rig needs operator_confirmed.",
                    {"config": "station config file name (GET /api/stations)",
                     "profile": "profile name (default: the configuration's default profile)",
                     "step": "work state id",
                     "operator_confirmed": "bool (required on the real rig)",
                     "scan_id": "check a stored metric scan instead of capturing (optional)",
                     "mock_scene": "MOCK ONLY: scene:<name>[:<variant>] or none, placed first",
                     "cycle_id": "the open cycle this check belongs to (the command bus)",
                     "attempt": "the attempt the command bus asked for (echoed in dlp with "
                                "step and cycle_id: the bus takes only the answer to the "
                                "verify it asked for)"}),
    "flow_run": ("One whole cycle of a station configuration's flow, in line and unattended: "
                 "every work state is verified in turn against `inputs` (the sensor claims), "
                 "its actions run, a new cycle folder is written (result.json: station config, "
                 "profile, steps, permissions; CONTINUE only when every required work state "
                 "was CONFIRMED). A prompt nobody acknowledges ends the cycle PULL unless "
                 "acknowledge is true. The real rig needs operator_confirmed. The operator "
                 "page and a PLC walk the same cycle step by step: POST /api/cmd/*.",
                 {"config": "station config file name (GET /api/stations)",
                  "profile": "profile name (default: the configuration's default profile)",
                  "inputs": "{signal: true | false}: what the line's sensors say",
                  "operator_confirmed": "bool (required on the real rig)",
                  "acknowledge": "bool: ACK every prompt (default false: fail closed)",
                  "project_faults": "bool (default true): paint the faults on the part",
                  "hold_s": "fault projection hold, s (default the recipe's)",
                  "mock_scene": "MOCK ONLY: a scene, or {step id: scene}, placed before a scan",
                  "mock_case": "MOCK ONLY: a named case of the configuration (scene + inputs)"}),
    # operating modes (engine/modes/): Measure / Quick check. Each runs only while its mode
    # is selected (POST /api/cmd/mode): outside Station mode the bus starts no cycle
    "measure_run": ("MEASURE mode: scan and report numbers - no tolerance, no pass / fail, no "
                    "permission. A new run folder: result.json (operating_mode measure, status "
                    "MEASURED | NOT_MEASURABLE + reason + how_to_fix, values [{name, label, "
                    "value, units, n_points, uncertainty, near_resolution_limit, notes}], "
                    "result null, disposition NONE), values.csv, overlay.png, scan.ply, "
                    "spec.json. NOT_MEASURABLE (no number, never a px as mm): uncalibrated, "
                    "stale_calibration, rig_unverified (a mobile rig: HOME first), rig_check "
                    "(a fixed rig's MOVED / UNKNOWN), too_few_points, incomplete (a cut part, "
                    "an uncompared patch), invalid (no cylinder, a pick on an edge) ... Refused "
                    "unless Measure mode is "
                    "selected (POST /api/cmd/mode {\"mode\": \"measure\"}).",
                    {"what": "flatness | step | distance | cylinder | endface | dims | deviation",
                     "preset": "triangulation preset (default: the loaded one; another one is "
                               "loaded first)",
                     "spec": "a measurement spec mapping (the other args override it)",
                     "roi": "[x0, y0, x1, y1] camera px: region A (x1, y1 exclusive)",
                     "roi2": "[x0, y0, x1, y1] camera px: region B (step)",
                     "zone": "region A as grid cells, or painted zone sets "
                             "'zoneset:<set>[:Z1,Z3]' / 'zoneset:<a>+<b>' (lit + the px that "
                             "decode into them), or 'selection' / 'selection@<stamp>' (the sets "
                             "switched on; with the stamp: refused when the selection is not "
                             "the one that was shown). Several zones: flatness / cylinder per "
                             "zone, values named '<set>/<id>:<name>'; not for endface",
                     "zone2": "region B as grid cells (or ONE painted zone)",
                     "p1": "[x, y] camera px: a picked point (distance)",
                     "p2": "[x, y] camera px: the second point",
                     "golden": "deviation: the golden metric scan (id / label)",
                     "radius_mm": "cylinder: fix the radius", "axis": "cylinder: auto | x | y | z",
                     "min_height_mm": "dims: the part stands this far out (default 2)",
                     "face_diameter_mm": "endface: the expected diameter",
                     "n_faces": "endface: end faces in view (default 1)",
                     "repeats": "scans: > 1 gives each value's repeatability (default 1)",
                     "metric_ids": "[metric scan ids] measured instead of capturing",
                     "scan_kind": "metric (default) | fast", "steps": "phase steps",
                     "periods": "[T, ...] DMD px", "average": "camera frames per pattern",
                     "label": "scan label", "name": "a name for the spec",
                     "mock_scene": "MOCK ONLY: scene:<name>[:<variant>] or none, placed first"}),
    "rig_session": ("Start a new rig session at the loaded preset's geometry: the rig was set "
                    "down / powered up / moved now. A MOBILE rig (preset rig.mount) then gives "
                    "no mm until a rig check OK, a re-home or a fresh calibration (this server's "
                    "start already is one). -> the marker + the rig's state.",
                    {"reason": "why (recorded)"}),
    # UI verbs (not engine.api verbs)
    "quick_check": ("QUICK CHECK mode: a triangulation preset + ONE check -> PASS / FAIL / "
                    "INVALID and CONTINUE / REJECT / PULL; no sensors, no permissions, no bus "
                    "outputs. Runs the generated one-job station config exactly like `scan "
                    "quick-check` (a new run folder: result.json with operating_mode "
                    "quick_check, run.log.jsonl, quick_check.json). A mobile rig not verified "
                    "in this session, or a rig whose newest rig check said MOVED / UNKNOWN, is a "
                    "PULL. The real rig needs operator_confirmed. Refused unless Quick check "
                    "mode is selected (POST /api/cmd/mode {\"mode\": \"quick_check\"}).",
                    {"check": "flatness | golden | recipe",
                     "preset": "triangulation preset (default: the loaded one; a recipe "
                               "check: the recipe's own)",
                     "threshold_mm": "flatness / golden: the limit, mm",
                     "roi": "flatness / golden: [x0, y0, x1, y1] camera px",
                     "zone": "flatness / golden: light only these grid cells, or painted zone "
                             "sets 'zoneset:<set>[:Z1,Z3]' / 'zoneset:<a>+<b>', or 'selection' "
                             "/ 'selection@<stamp>' (the sets switched on). Painted zones are "
                             "judged one by one: the limit applies to each zone, and a zone "
                             "that cannot be judged is a PULL",
                     "golden": "golden: the golden metric scan (id / label)",
                     "recipe": "recipe: a recipe name (GET /api/recipes)",
                     "guard_band_pct": "flatness / golden: a PASS this close to the limit is "
                                       "PULLed (default 0)",
                     "operator_confirmed": "bool (required on the real rig)",
                     "mock_part": "MOCK ONLY: what the simulated human places (as --mock-part)"}),
    "run_station": ("Run one station config (stations/<config>.json) exactly like `scan run`: "
                    "a new run folder with result.json + run.log.jsonl; PASS / FAIL / INVALID / "
                    "ABORTED, disposition CONTINUE / REJECT / PULL (quality inspection) "
                    "(cancel the job to abort between captures). A config naming no "
                    "preset (and not a recipe) runs the LOADED preset (refused when none is "
                    "loaded). A configuration with profiles + a flow runs one whole cycle in "
                    "line (as flow_run, its default profile, no sensor claims). The real rig "
                    "needs operator_confirmed.",
                    {"config": "station config file name (GET /api/stations)",
                     "operator_confirmed": "bool: a human confirmed the physical setup "
                                           "(required on the real rig)",
                     "zones_expect": "a config whose job says \"zone\": \"selection\": the "
                                     "stamp of the zone selection the sender was shown (GET "
                                     "/api/zones/selection); the run is INVALID -> PULL when "
                                     "the selection is another one when it scans",
                     "mock_part": "MOCK ONLY: what the simulated human places (as --mock-part)"}),
    "live_view": ("Keep the rig open and stream a flood / focus pattern until stopped "
                  "(POST /api/live/stop), until a rig job is queued, or when nobody polled "
                  f"GET /api/live/state or /api/live.jpg for {LIVE_IDLE_S:g} s.",
                  {"pattern": "|".join(LIVE_PATTERNS), "colour": "|".join(LIVE_COLOURS),
                   "level": "0-255", "period_px": "stripe/checker period",
                   "max_s": f"float, 0..{MAX_LIVE_S:g} s (default {MAX_LIVE_S:g})"}),
    "mock_place": ("MOCK ONLY: stand in for the human and place an object on the simulated "
                   "bench.", {"object": "validation | box | label | cylinder", "h_mm": "",
                              "w_mm": "", "radius_mm": "cylinder (default 24.05)",
                              "height_mm": "cylinder (default 90)",
                              "dents": "cylinder: [{depth_mm, size_mm, angle_deg, z_mm}]",
                              "l_mm": "", "x_mm": "", "y_mm": "", "angle_deg": "",
                              "albedo": "box reflectance 0..1 (base is 0.6; default 0.7)",
                              "data": "QR text (label)", "size_mm": "label side"}),
    "mock_clear": ("MOCK ONLY: clear the simulated bench.", {}),
}

# Verbs that never open the camera / projector: they don't stop a running
# live view and run between its frames. Everything else needs the rig (or
# changes the active preset) and stops the live view first.
RIG_FREE_VERBS = frozenset({
    "reconstruct", "measure", "compare", "calibrate", "inspect", "reject_base", "profile",
    "bank_add", "bank_check", "bank_remove", "learn_material", "classify", "list_library",
    "latest_base", "mock_place", "mock_clear", "spot_size", "teach_alignment",
    "check_alignment", "export_mesh", "cell_unwrap", "cell_dents", "bake_maps", "import_atlas",
    "list_atlases", "make_flash_pack", "make_fastscan_pack", "sort_flash_recording",
    "make_charuco", "triangulate",
    "cylinder_deviation",
    "compare_metric", "predict_precision", "export_metric_mesh", "calibrate_dent_null",
    "detect_dents", "locate_face", "face_profile", "face_golden", "check_face",
    "station_measure", "station_golden", "station_check", "preset_add", "preset_delete",
    "recipe_add", "recipe_select", "recipe_delete", "zone_delete", "zone_select"})

# verbs that change which preset is loaded / selected: refused while a station run is busy
PRESET_CHANGE_VERBS = frozenset({"load_preset", "preset_select", "preset_delete"})
# verbs that change what "the selected zones" are (the selection itself, a set painted over,
# a set deleted): refused unless the station is READY or FAULT - what a station inspects
# never changes while it is ARMED or in a cycle
SELECTION_CHANGE_VERBS = frozenset({"zone_select", "zone_paint", "zone_delete"})
# the painter and the zone show hold the rig, and the projector display belongs to them
PAINTER_VERBS = ("zone_paint", "zone_show")


# running jobs that may be aborted: the engine checks the job's cancel event before every
# capture (ScanSession.cancel), so the rig is never left mid-sequence
CANCELLABLE_VERBS = frozenset({"run_station", "rig_check", "recipe_run", "recipe_feedback",
                               "flow_verify", "flow_run", "measure_run", "quick_check",
                               "zone_paint", "zone_show"})
# verbs of an operating mode: accepted only while the bus is in that mode (engine/modes/), so
# a measurement or a quick check never runs beside a cycle of the line
MODE_VERBS = {"measure_run": "measure", "quick_check": "quick_check"}
CONFIRMED_CMDS = frozenset({"START", "TRIGGER", "ARM"})   # bus commands that read operator_confirmed
# POST /api/cmd/<name> -> the bus command; `workstate` is the deprecated name of `config`
# (it always selected a station configuration, never a work state)
CMD_NAMES = {"config": "SELECT_CONFIG", "profile": "SELECT_PROFILE", "mode": "SELECT_MODE",
             "workstate": "SELECT_WORKSTATE"}
BUS_TICK_S = 0.5              # how often a waiting work state's timeout is looked at
RIG_GATE_TTL_S = 5.0          # the rig's state in the snapshot is re-read at least this often
ZONE_STATUS_TTL_S = 5.0       # ... and so is the zone selection (its files unchanged)
ZONE_SELECT_WAIT_S = 10.0     # POST /api/zones/selection waits this long for its files-only job
ZONE_SELECT_GRACE_S = 3.0     # ... and this long for a job ahead of it to end (a preset load)


class ArgError(ValueError):
    """Bad job arguments from the browser / agent."""


class EntryNotFound(LookupError):
    """An id the browser passed no longer resolves (deleted, other geometry)."""


class HTTPError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code, self.msg = code, msg


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def error_hint(e: BaseException) -> Optional[str]:
    """One plain-language line on what to do about a failed job."""
    if isinstance(e, NotImplementedError):
        return ("This mode is still being built in the engine. Nothing was captured "
                "or changed.")
    if isinstance(e, rules.GeometryMismatch):
        return ("Data from different rig geometries can't be mixed: capture a new base "
                "(and recalibrate) at this geometry.")
    if isinstance(e, rules.SafetyViolation):
        return "A hardware safety rule (§9) refused this; nothing was sent to the rig."
    if isinstance(e, getattr(rules, "RigBusy", ())):
        return ("Another program holds the camera + DMD: usually a second web UI (its live "
                "view or a job), a CLI run or an agent. Stop it there or wait for it to "
                "finish; this UI queues its own jobs. Nothing was sent to the rig.")
    if isinstance(e, rules.OverwriteRefused):
        return "Library data is never overwritten; retry (a new entry name is allocated)."
    if isinstance(e, rules.RuleViolation):
        return "An engine rule (§9) refused this; the operation did not run."
    if isinstance(e, config.ConfigError):
        return "Preset validation failed: fix the field named in the message."
    if isinstance(e, EntryNotFound):
        return "The entry may have been deleted or belongs to another geometry: refresh."
    if isinstance(e, ArgError):
        return "Check the inputs for this action."
    if type(e).__name__ == "ExposureNotFound":
        return ("No camera exposure shows this seating of the face well enough: it was NOT "
                "scanned. PULL the part (quality inspection) or re-seat it and scan again; "
                "an exposure can be forced (exposure_us).")
    if type(e).__name__ == "SequenceError":
        return ("The camera did not see the pack's loop playing in order: in the 4500 MKII GUI "
                "check Pattern Sequence (Pattern Source Flash, Internal/External, Play Once "
                "off, the pack's 16 patterns in order) and press Play, then retry. Nothing was "
                "decoded.")
    return None


# ------------------------------------------------------------ JSON safety

_OMIT = object()


def rel_path(p, root: Path) -> str:
    """Library-relative posix path (what /files/ serves), else absolute."""
    try:
        return Path(p).resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        return str(p)


def jsonable(o, root: Path, _depth: int = 0):
    """Verb results -> JSON: dataclasses -> dicts, Paths -> library-relative
    strings, NaN/inf -> null, big numpy arrays (images, fields) omitted.
    Short 1-D numeric arrays (profile samples) become lists."""
    if _depth > 14:
        return repr(o)[:200]
    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, Path):
        return rel_path(o, root)
    if isinstance(o, np.ndarray):
        if o.ndim == 1 and o.size <= 4096 and o.dtype.kind in "biuf":
            return [jsonable(x, root, _depth + 1) for x in o.tolist()]
        return _OMIT
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        d = {"__type__": type(o).__name__}
        for f in dataclasses.fields(o):
            if f.name.startswith("_"):
                continue
            v = jsonable(getattr(o, f.name), root, _depth + 1)
            if v is not _OMIT:
                d[f.name] = v
        path = getattr(o, "path", None)
        if isinstance(path, Path) and (path / "preview.png").is_file():
            d["preview"] = rel_path(path / "preview.png", root)
        return d
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            v = jsonable(v, root, _depth + 1)
            if v is not _OMIT:
                out[str(k)] = v
        return out
    if isinstance(o, (list, tuple, set, frozenset, deque)):
        return [v for v in (jsonable(x, root, _depth + 1) for x in o) if v is not _OMIT]
    return repr(o)[:500]


def _log_record(line: bytes, exclude: set) -> Optional[dict]:
    """One log.jsonl line -> record, or None (blank, torn, or an excluded verb)."""
    line = line.strip()
    if not line:
        return None
    try:
        rec = json.loads(line.decode("utf-8", errors="replace"))
    except ValueError:
        return None
    if not isinstance(rec, dict) or rec.get("verb") in exclude:
        return None
    return rec


def _entry_id(item) -> Optional[str]:
    if isinstance(item, dict):
        return item.get("id") or item.get("name")
    return getattr(item, "id", None) or getattr(item, "name", None)


def _entry_path(item) -> Optional[Path]:
    p = item.get("path") if isinstance(item, dict) else getattr(item, "path", None)
    return Path(p) if p else None


# ------------------------------------------------------------ arg parsing

def _s(a: dict, k: str, default=None, required: bool = False):
    v = a.get(k)
    if v is None or (isinstance(v, str) and not v.strip()):
        if required:
            raise ArgError(f"'{k}' is required")
        return default
    if not isinstance(v, (str, int, float)):
        raise ArgError(f"'{k}' must be text")
    return str(v).strip()


def _num(a: dict, k: str, default=None, cast=float, required: bool = False):
    v = a.get(k)
    if v is None or (isinstance(v, str) and not v.strip()):
        if required:
            raise ArgError(f"'{k}' is required")
        return default
    if isinstance(v, bool):
        raise ArgError(f"'{k}' must be a number")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ArgError(f"'{k}' must be a number, got {v!r}") from None
    if not math.isfinite(f):
        raise ArgError(f"'{k}' must be finite")
    if cast is int:
        if f != int(f):
            raise ArgError(f"'{k}' must be a whole number, got {v!r}")
        return int(f)
    return f


def _fine_prefilter(a: dict):
    """fine_prefilter_px: the optional fine-fringe low-pass of the flash scan, 0..3 camera px."""
    v = _num(a, "fine_prefilter_px")
    if v is not None and not 0.0 <= v <= 3.0:
        raise ArgError("fine_prefilter_px must be 0..3 camera px")
    return v


def _bool(a: dict, k: str, default: bool = False) -> bool:
    v = a.get(k)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _confirmed(a: dict, k: str) -> bool:
    """A human's confirmation (operator_confirmed, empty_confirmed, confirm): only a JSON
    true confirms; absent / null / false do not, and anything else ("false", "yes", 1) is
    refused rather than guessed."""
    v = a.get(k)
    if v is None or v is False:
        return False
    if v is True:
        return True
    raise ArgError(f"'{k}' must be a JSON boolean (true / false), got {type(v).__name__} {v!r}")


def _num_list(a: dict, k: str, cast=float) -> Optional[list]:
    v = a.get(k)
    if v is None or v == "" or v == []:
        return None
    if isinstance(v, str):
        v = [x for x in re.split(r"[,\s;]+", v.strip()) if x]
    if not isinstance(v, (list, tuple)):
        raise ArgError(f"'{k}' must be a list of numbers")
    return [_num({"v": x}, "v", cast=cast, required=True) for x in v]


def _point(a: dict, k: str) -> Optional[list]:
    v = a.get(k)
    if v is None or v == []:
        return None
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise ArgError(f"'{k}' must be [x, y] in camera px")
    return [_num({"v": x}, "v", required=True) for x in v]


def _hold_s(a: dict) -> float:
    """Pumped holds keep the rig (and the machine-wide rig lock) and can't be
    cancelled mid-hold, so the UI caps them well below the engine's 3600 s."""
    h = _num(a, "hold_s", 0.0)
    if not 0.0 <= h <= MAX_HOLD_S:
        raise ArgError(f"hold_s must be 0..{MAX_HOLD_S:g} s (a hold can't be interrupted), "
                       f"got {h:g}")
    return h


def _at(a: dict):
    """Where a virtual-sensor spot / zone goes, checked before it is queued:
    a grid cell or zone text, {"dmd": [col, row]}, {"camera": [x, y]},
    {"cell": "H8"} or {"zone": "G7:H9"}. The engine resolves it (grid, map)."""
    v = a.get("at")
    if isinstance(v, str) and v.strip():
        return v.strip()
    if isinstance(v, dict) and len(v) == 1:
        (k, val), = v.items()
        if k in ("dmd", "camera"):
            unit = "DMD px [col, row]" if k == "dmd" else "camera px [x, y]"
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                raise ArgError(f"'at' {{'{k}': ...}} needs two numbers, {unit}")
            return {k: [_num({"v": x}, "v", required=True) for x in val]}
        if k in ("cell", "zone") and isinstance(val, str) and val.strip():
            return {k: val.strip()}
    raise ArgError(f"'at' is required: {AT_DOC}")


def _kw(**kw) -> dict:
    """Only the optional args the caller gave: the engine keeps its own defaults."""
    return {k: v for k, v in kw.items() if v is not None}


def known_dims(v) -> Optional[dict]:
    if not v:
        return None
    if not isinstance(v, dict):
        raise ArgError("known_dims must be an object like {\"H\": 106.95, \"W\": 56.5}")
    out = {}
    for k, x in v.items():
        if x is None or (isinstance(x, str) and not x.strip()):
            continue
        out[str(k)] = _num({"v": x}, "v", required=True)
    return out or None


# ------------------------------------------------------------- live view

def live_pattern(settings: dict, shape=(1140, 912)) -> np.ndarray:
    """BGR uint8 DMD image, sent raw (the DMD shows exactly what is sent),
    so the live view is lit whatever the preset's polarity chain says."""
    level = int(np.clip(settings.get("level", 255), 0, 255))
    colour = LIVE_COLOURS.get(settings.get("colour", "white"), LIVE_COLOURS["white"])
    bgr = np.array([c * level // 255 for c in colour], np.uint8)
    img = np.zeros((*shape, 3), np.uint8)
    period = max(2, int(settings.get("period_px", 8)))
    pattern = settings.get("pattern", "flood")
    if pattern == "stripes":
        on = (np.arange(shape[1]) // max(1, period // 2)) % 2 == 0
        img[:, on] = bgr
    elif pattern == "checker":
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        sq = max(1, period // 2)
        img[((yy // sq) + (xx // sq)) % 2 == 0] = bgr
    else:
        img[:] = bgr
    return img


def frame_metrics(frame: np.ndarray, max_w: int = 960, quality: int = 80):
    """-> (jpeg bytes, metrics). Focus = variance of the Laplacian over the
    lit area, eroded so the lit patch's own edge doesn't count as detail.
    Saturated pixels are painted red: saturation flattens fringes."""
    import cv2
    f = np.asarray(frame, np.float32)
    h, w = f.shape[:2]
    small = cv2.resize(f, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    blur = cv2.blur(small, (9, 9))
    lo, hi = np.percentile(blur, [2, 98])
    if hi - lo > 6:
        lit_small = blur > lo + 0.5 * (hi - lo)
    else:
        lit_small = np.full(blur.shape, float(blur.mean()) > 20)
    lit_small = cv2.erode(lit_small.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    lit = cv2.resize(lit_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    lap = cv2.Laplacian(f, cv2.CV_32F, ksize=3)
    n_lit = int(lit.sum())
    focus = float(lap[lit].var()) if n_lit > 500 else 0.0
    sat = f >= 250
    area = lit if n_lit else np.ones_like(lit)
    metrics = {"focus": round(focus, 2), "mean": round(float(f[area].mean()), 1),
               "saturated_pct": round(100.0 * float(sat[area].mean()), 2),
               "lit_pct": round(100.0 * n_lit / lit.size, 1), "units": "camera counts"}
    vis = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    vis[sat] = (0, 0, 255)
    if w > max_w:
        vis = cv2.resize(vis, (max_w, int(round(h * max_w / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return (buf.tobytes() if ok else None), metrics


class LiveState:
    """Shared between the worker (writes) and HTTP threads (read / settings).
    Stopping is per job (Job.cancel), so a Stop pressed while the live view
    is still queued can't be lost when it starts."""

    def __init__(self):
        self.lock = threading.Lock()
        self.seen = time.monotonic()        # last viewer poll (heartbeat)
        self.stopped_because: Optional[str] = None
        self.settings = {"pattern": "flood", "colour": "white", "level": 255, "period_px": 8}
        self.jpeg: Optional[bytes] = None
        self.running = False
        self.job_id: Optional[str] = None
        self.frame = 0
        self.metrics: dict = {}
        self.peak = 0.0
        self.fps = 0.0
        self.started: Optional[str] = None
        self.error: Optional[str] = None

    def update_settings(self, s: dict) -> dict:
        with self.lock:
            if "pattern" in s:
                if s["pattern"] not in LIVE_PATTERNS:
                    raise ArgError(f"pattern must be one of {LIVE_PATTERNS}")
                self.settings["pattern"] = s["pattern"]
            if "colour" in s:
                if s["colour"] not in LIVE_COLOURS:
                    raise ArgError(f"colour must be one of {tuple(LIVE_COLOURS)}")
                self.settings["colour"] = s["colour"]
            if s.get("level") is not None:
                self.settings["level"] = int(np.clip(_num(s, "level", cast=int), 0, 255))
            if s.get("period_px") is not None:
                self.settings["period_px"] = int(np.clip(_num(s, "period_px", cast=int), 2, 256))
            if s.get("reset_peak"):
                self.peak = 0.0
            return dict(self.settings)

    def touch(self) -> None:
        """A viewer polled: keep the live view alive (heartbeat)."""
        self.seen = time.monotonic()

    def public(self) -> dict:
        with self.lock:
            return {"running": self.running, "job_id": self.job_id, "frame": self.frame,
                    "metrics": dict(self.metrics), "focus_peak": round(self.peak, 2),
                    "fps": round(self.fps, 2), "settings": dict(self.settings),
                    "started": self.started, "error": self.error,
                    "stopped_because": self.stopped_because,
                    "idle_stop_s": LIVE_IDLE_S, "has_image": self.jpeg is not None}


# ------------------------------------------------------------------ jobs

@dataclasses.dataclass
class Job:
    id: str
    verb: str
    args: dict
    status: str = "queued"               # queued | running | done | error
    result: Any = None
    links: dict = dataclasses.field(default_factory=dict)
    error: Optional[str] = None
    error_type: Optional[str] = None
    hint: Optional[str] = None
    trace: Optional[str] = None
    note: Optional[str] = None
    created: str = dataclasses.field(default_factory=_now)
    started: Optional[str] = None
    finished: Optional[str] = None
    duration_s: Optional[float] = None
    done: threading.Event = dataclasses.field(default_factory=threading.Event, repr=False)
    cancel: threading.Event = dataclasses.field(default_factory=threading.Event, repr=False)

    def public(self, full: bool = True) -> dict:
        d = {k: getattr(self, k) for k in ("id", "verb", "status", "error", "error_type",
                                           "hint", "note", "created", "started", "finished",
                                           "duration_s")}
        d["args"] = self.args
        if full:
            d.update(result=self.result, links=self.links, trace=self.trace)
        return d


class _UILink(str):
    """A ready URL for a job link (not a library path)."""


class _Call:
    """A short read the worker runs between jobs (listings)."""

    def __init__(self, fn: Callable):
        self.fn, self.done = fn, threading.Event()
        self.value = self.exc = None

    def run(self):
        try:
            self.value = self.fn()
        except BaseException as e:      # noqa: BLE001 - handed back to the caller
            self.exc = e
        finally:
            self.done.set()


class App:
    """The engine side: one worker thread owns every engine call."""

    def __init__(self, *, mock: bool, library=None, presets_dir=None,
                 preset_name: Optional[str] = None, stations_dir=None):
        self.mock = bool(mock)
        self.stations_dir = Path(stations_dir) if stations_dir else STATIONS_DIR
        self._runs: dict = {}                   # run_id -> run folder (what /runs/ may serve)
        # the station command bus (START / ABORT / HOME / TRIGGER ...): every source (page
        # buttons, keys, POST /api/cmd/*, hardware inputs) goes through its state machine
        from engine.control.commands import CommandBus, Handlers
        self.bus = CommandBus(Handlers(start=self._bus_start, abort=self._bus_abort,
                                       home=self._bus_home, select=self._bus_select,
                                       pending=self._bus_pending, verify=self._bus_verify,
                                       project=self._bus_project,
                                       cycle_begin=self._bus_cycle_begin,
                                       cycle_end=self._bus_cycle_end, log=self._bus_log,
                                       gate=self._bus_gate),
                              require_confirmation=not self.mock)
        # painted zones: how the last painter session ended (every page shows the same save
        # message, also after a reload), the cached zone selection of the loaded preset
        # (GET /api/cmd/state is polled twice a second) and the selection stamp the bus
        # accepted for the cycle it is starting (handed to that cycle's run job)
        self.last_paint: Optional[dict] = None
        self._zone_cache: dict = {}
        self._zone_token: Optional[str] = None
        self._cycles: dict = {}                 # open cycle id -> its folder, log, config
        self.last_cycle: Optional[dict] = None  # the last cycle written (links for /quick)
        # MOCK only: the case of the simulated line (the scene the simulated human places
        # for the next scans + what the simulated sensors say); None = the profile's good part
        self.mock_case: Optional[str] = None
        self.mock_scene: Optional[str] = None   # an explicit scene instead of a named case
        # ... and what its simulated sensors say (signal -> level): reported to the bus when
        # they change and again for every cycle (a claim belongs to one cycle)
        self.mock_inputs: dict = {}
        self._mock_config: Optional[str] = None  # the configuration the mock line belongs to
        # job id -> "run" | "home": registered under _cv BEFORE the job is queued (so the
        # worker can never finish it unannounced), popped by _bus_notify. Lock order:
        # bus._lock, then _cv - never the other way round.
        self._bus_jobs: dict = {}
        # geometries of MOBILE rigs (preset rig.mount) this server has opened a rig session
        # for: its start is a new session, so each must be verified again before any mm
        self._mobile_sessions: set = set()
        # (preset, mount, strict) -> (record signature, when, the gate's answer): GET
        # /api/cmd/state is polled twice a second and must not read the rig-state record
        self._rig_cache: dict = {}
        self.library = Path(library) if library else None
        self.presets_dir = Path(presets_dir) if presets_dir else None
        self.preset_name = preset_name
        self.session = None
        self.root: Optional[Path] = None
        self.port: Optional[int] = None
        self.started = _now()
        self.startup_warnings: list = []
        self.live = LiveState()
        self._cv = threading.Condition()
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._queue: deque = deque()
        self._quick: deque = deque()
        self._current: Optional[Job] = None
        self._inline: Optional[Job] = None      # rig-free job running between live frames
        self._closing = False
        self._ids = itertools.count(1)
        self._snapshot: dict = {}
        self._list_cache: dict = {}
        self._ready = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name="engine-worker", daemon=True)
        self._ticker = threading.Thread(target=self._tick_loop, name="bus-tick", daemon=True)
        self._stop_tick = threading.Event()

    # ----------------------------------------------------------- lifecycle
    def start(self, timeout: float = 120.0) -> "App":
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("engine worker did not start")
        if self._startup_error is not None:
            raise self._startup_error
        self._ticker.start()
        return self

    def _tick_loop(self) -> None:
        """A work state that waits for STEP / ACK longer than its timeout_s ends its cycle
        PULL even when nobody polls the page (CommandBus.tick)."""
        while not self._stop_tick.wait(BUS_TICK_S):
            try:
                self.bus.tick()
            except Exception:           # noqa: BLE001 - the next tick tries again
                pass

    def close(self, timeout: float = 10.0) -> None:
        self._stop_tick.set()
        with self._cv:
            self._stop_live_locked("server shutting down")
            self._closing = True
            dropped = list(self._queue)
            for job in dropped:
                self._fail(job, "server shutting down", "Cancelled")
            self._queue.clear()
            self._cv.notify_all()
        for job in dropped:                     # never ran: the bus still hears it ended
            self._bus_notify(job)
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout)
        if getattr(self, "session", None) is not None:
            rigstate.close_process_session(self.session.store)   # its rig session ends with it

    @property
    def presets_path(self) -> Path:
        return self.presets_dir or config.PRESETS_DIR

    def _startup(self):
        """Mode is fixed here, once: api.configure(mock=...) on the worker."""
        self.session = api.configure(self.library, mock=self.mock, presets_dir=self.presets_dir)
        self.root = self.session.store.root.resolve()
        # this server owns the rig from now: no mobile rig's session is older than its start
        rigstate.open_process_session(self.session.store)
        names = self.preset_names()
        sel = api.preset_selection()
        gone = bool(sel.get("name")) and not sel.get("exists")
        if gone and not self.preset_name:
            # selected but gone (deleted, renamed): nothing is loaded in its place - a default
            # loaded here would be what every station run naming no preset scans with, even
            # once the selection resolves again
            self.startup_warnings.append(
                f"the selected preset {sel['name']!r} no longer exists in this library: no "
                "preset loaded - select one in Setup before measuring")
        elif gone:
            self.startup_warnings.append(
                f"the library's selected preset {sel['name']!r} no longer exists: station "
                f"configs that name no preset are refused (not run with {self.preset_name!r}) "
                "until one is selected in Setup")
        if gone and not self.preset_name:
            name = None
        else:
            name = self.preset_name or api.preset_active() or (
                DEFAULT_PRESET if DEFAULT_PRESET in names else (names[0] if names else None))
            if not name:
                self.startup_warnings.append(f"no presets in {self.presets_path}")
        if name:
            try:
                api.load_preset(name)
            except Exception as e:      # noqa: BLE001 - reported in the UI, not fatal
                self.startup_warnings.append(f"could not load preset {name!r}: "
                                             f"{type(e).__name__}: {e}")
        self._mobile_sessions_at_start()
        self._refresh_snapshot()
        self.refresh_configs()
        self._mock_line_default()

    def _mobile_session(self, preset) -> None:
        """A MOBILE rig (preset rig.mount) is assumed moved since its last use, and this
        server's start is a new session: the first time its geometry is seen here a session
        marker is recorded (engine/rigstate.py), so whatever verified it before no longer
        does - HOME (the rig check), a re-home or a calibration must, before any mm. Always
        BEFORE the preset's first job (a marker written after one would undo what that job
        verified); a preset that slips past this is still held by the server's own session
        (rigstate.open_process_session in _startup)."""
        if preset is None or preset.rig.mount != "mobile" or \
                preset.geometry_id in self._mobile_sessions:
            return
        self._mobile_sessions.add(preset.geometry_id)
        try:
            rigstate.new_session(self.session.store, preset,
                                 "the web UI server started (a new rig session)",
                                 by=self.session.holder or "web UI")
        except OSError as e:            # unrecorded: the rig stays unverified at most 12 h
            self.startup_warnings.append(f"rig session marker for {preset.name} not written: {e}")

    def _mobile_sessions_at_start(self) -> None:
        """Every mobile preset of this library (and the loaded one) starts a rig session."""
        for name in self.preset_names():
            try:
                p = config.load_preset(name, presets_dir=self.session.presets_dir,
                                       store=self.session.store, mock=self.mock)
            except Exception:           # noqa: BLE001 - an invalid preset is listed as such
                continue
            self._mobile_session(p)
        self._mobile_session(self.session.preset)

    def _run(self):
        try:
            self._startup()
        except BaseException as e:      # noqa: BLE001 - re-raised by start()
            self._startup_error = e
            self._ready.set()
            return
        self._ready.set()
        while True:
            with self._cv:
                while not (self._quick or self._queue or self._closing):
                    self._cv.wait()
                if self._quick:
                    item, job = self._quick.popleft(), None
                elif self._queue:
                    item, job = None, self._queue.popleft()
                    self._current = job
                    job.status, job.started = "running", _now()
                else:
                    break                                   # closing, nothing left
            if item is not None:
                item.run()
                continue
            self._execute(job)
            with self._cv:
                self._current = None

    def drain_quick(self) -> None:
        """Run pending short reads now (called by long jobs between frames)."""
        while True:
            with self._cv:
                if not self._quick:
                    return
                item = self._quick.popleft()
            item.run()

    def drain_rig_free(self) -> None:
        """Run queued rig-free jobs now, in queue order (the live view calls
        this between frames). Stops at the first job that needs the rig."""
        while True:
            with self._cv:
                if not self._queue or self._queue[0].verb not in RIG_FREE_VERBS:
                    return
                job = self._queue.popleft()
                job.status, job.started = "running", _now()
                self._inline = job
            try:
                self._execute(job)
            finally:
                with self._cv:
                    self._inline = None

    def call(self, fn: Callable, timeout: float = 60.0):
        """Run fn on the worker and return its value (TimeoutError if the
        worker is busy longer than `timeout`; the call still runs later)."""
        if threading.current_thread() is self._thread:
            return fn()
        c = _Call(fn)
        with self._cv:
            if self._closing:
                raise RuntimeError("server is shutting down")
            self._quick.append(c)
            self._cv.notify_all()
        if not c.done.wait(timeout):
            raise TimeoutError("engine worker busy")
        if c.exc is not None:
            raise c.exc
        return c.value

    # ---------------------------------------------------------------- jobs
    def submit(self, verb: str, args: Optional[dict] = None,
               bus_kind: Optional[str] = None, only_if_free: bool = False) -> Job:
        """Queue a job. bus_kind ('run' | 'home'): a job the command bus started; it is
        registered before the worker can see it, so its end always reaches the bus.
        only_if_free: refused (409) instead of queued behind another job (jobs_ahead) - for
        a change whose sender waits for the answer and must not see it applied later."""
        if verb not in VERBS:
            raise HTTPError(400, f"unknown or not exposed verb {verb!r}; see GET /api/verbs")
        args = dict(args or {})
        allowed = set(VERBS[verb][1])
        extra = set(args) - allowed
        if extra:
            raise HTTPError(400, f"{verb}: unknown args {sorted(extra)}; expected "
                                 f"{sorted(allowed) or 'none'}")
        if verb in PRESET_CHANGE_VERBS and self.bus.state in (*BUS_BUSY, *BUS_IN_CYCLE):
            raise HTTPError(409, f"{verb} refused while the station is {self.bus.state}: "
                                 "the running job / cycle uses the loaded preset")
        if verb in SELECTION_CHANGE_VERBS and self.bus.state not in ("READY", "FAULT"):
            raise HTTPError(409, f"{verb} refused while the station is {self.bus.state}: the "
                                 "zones a station inspects are fixed until it is idle (DISARM "
                                 "or wait for the cycle to end)")
        if verb == "zone_select":
            busy = self.mode_jobs() or ([self.painter_state()["job"]]
                                        if self.painter_state() else [])
            if busy:
                raise HTTPError(409, f"zone_select refused: {', '.join(busy)} is still queued "
                                     "or running (a measurement, a quick check, the painter or "
                                     "a zone show): wait for it, or ABORT")
        if verb in MODE_VERBS and self.painter_state():
            raise HTTPError(409, f"{verb} refused: {self._painter_holds()}")
        need = MODE_VERBS.get(verb)
        if need is not None and self.bus.mode != need:
            raise HTTPError(409, f"{verb} runs in {modes.TITLES[need]} mode and the station is "
                                 f"in {modes.TITLES[self.bus.mode]} mode: select it first (POST "
                                 f"/api/cmd/mode {{\"mode\": \"{need}\"}}) - outside Station "
                                 "mode the bus starts no cycle")
        with self._cv:
            if self._closing:
                raise HTTPError(503, "server is shutting down")
            if verb == "live_view":
                for j in [self._current, *self._queue]:
                    if j is not None and j.verb == "live_view" and not j.cancel.is_set():
                        return j
            ahead = self.jobs_ahead() if only_if_free else []
            if ahead:
                raise HTTPError(409, f"{verb} refused: another job holds the scanner ("
                                     + ", ".join(ahead[:3]) + (", ..." if len(ahead) > 3 else "")
                                     + "): wait for it to end, or ABORT, then do it again")
            job = Job(id=f"j{next(self._ids)}", verb=verb, args=args)
            if verb != "live_view" and verb not in RIG_FREE_VERBS:
                # the rig is one physical device: free it for the job (a live
                # view still waiting in the queue is dropped too)
                if self._stop_live_locked(f"stopped so {verb} ({job.id}) could use the rig"):
                    job.note = "live view stopped so this job could use the rig"
            if bus_kind is not None:
                self._bus_jobs[job.id] = bus_kind
            self._jobs[job.id] = job
            self._queue.append(job)
            self._prune()
            self._cv.notify_all()
            return job

    def _stop_live_locked(self, why: str) -> int:
        """Stop the running live view and drop queued ones (caller holds _cv).
        -> how many live jobs were stopped or dropped."""
        n = 0
        for j in [j for j in self._queue if j.verb == "live_view"]:
            self._queue.remove(j)
            self._fail(j, f"live view cancelled before it started: {why}", "Cancelled")
            n += 1
        cur = self._current
        if cur is not None and cur.verb == "live_view" and cur.status == "running":
            if not cur.cancel.is_set():
                self.live.stopped_because = why
            cur.cancel.set()
            n += 1
        return n

    def stop_live(self, why: str = "stopped by the operator") -> int:
        with self._cv:
            return self._stop_live_locked(why)

    def cancel(self, job_id: str) -> Job:
        dropped = None
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None:
                raise HTTPError(404, f"no job {job_id}")
            if job.status == "queued":
                self._queue.remove(job)
                self._fail(job, "cancelled by the operator before it started", "Cancelled")
                dropped = job
            elif job.status == "running" and job.verb == "live_view":
                self.live.stopped_because = "stopped by the operator"
                job.cancel.set()
            elif job.status == "running" and job.verb in CANCELLABLE_VERBS:
                job.cancel.set()                # checked before the next capture
                job.note = "abort requested: stops before the next capture"
            elif job.status == "running":
                raise HTTPError(409, "a running capture can't be interrupted mid-sequence")
        if dropped is not None:                 # never ran: the bus still hears it ended
            self._bus_notify(dropped)
            if dropped.verb == "zone_paint":    # ... and the page: nothing was painted
                self._paint_record(dropped.args.get("name"), "dropped",
                                   reason="the painter job was cancelled before it started")
        return job

    def _fail(self, job: Job, msg: str, etype: str):
        job.status, job.error, job.error_type = "error", msg, etype
        job.finished = _now()
        job.done.set()

    def _prune(self):
        while len(self._jobs) > MAX_JOBS_KEPT:
            for k, j in self._jobs.items():
                if j.status in ("done", "error"):
                    del self._jobs[k]
                    break
            else:
                return

    def job(self, job_id: str) -> dict:
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None:
                raise HTTPError(404, f"no job {job_id}")
            return job.public()

    def wait(self, job_id: str, timeout: float) -> dict:
        with self._cv:
            job = self._jobs.get(job_id)
        if job is None:
            raise HTTPError(404, f"no job {job_id}")
        job.done.wait(max(0.0, min(timeout, 300.0)))
        return self.job(job_id)

    def recent_jobs(self, n: int = 50) -> list:
        with self._cv:
            jobs = list(self._jobs.values())[-n:]
            return [j.public(full=False) for j in reversed(jobs)]

    def mode_jobs(self) -> list:
        """Ids of the measurement / quick-check jobs still queued or running."""
        with self._cv:
            return [j.id for j in [self._current, *self._queue]
                    if j is not None and j.verb in MODE_VERBS and j.status in ("queued", "running")]

    def jobs_ahead(self) -> list:
        """The jobs a new rig-free job would wait behind -> ["j14 calibrate_stereo", ...]: the
        one that is running and every queued one. A running live view is not one (it runs
        queued rig-free jobs between its frames)."""
        with self._cv:
            return [f"{j.id} {j.verb}" for j in [self._current, *self._queue]
                    if j is not None and j.status in ("queued", "running")
                    and not (j is self._current and j.verb == "live_view")]

    # ---------------------------------------------------- painted zones: painter, selection
    def painter_state(self) -> Optional[dict]:
        """The zone painter / zone show job that is queued or running (it holds the rig and
        the projector display belongs to it), or None: {running, verb, name, state, since,
        job}. In the bus snapshot, so every page - also one that was just reloaded - shows
        "Painting ..." and keeps its run buttons and keys off."""
        with self._cv:
            for j in [self._current, *self._queue]:
                if j is not None and j.verb in PAINTER_VERBS and j.status in ("queued", "running"):
                    return {"running": True, "verb": j.verb, "state": j.status, "job": j.id,
                            "name": j.args.get("name") or j.args.get("zones"),
                            "since": j.started or j.created}
        return None

    def _painter_holds(self) -> str:
        """Why nothing else starts while the painter / a zone show is queued or running."""
        p = self.painter_state() or {}
        if p.get("verb") == "zone_show":
            return ("the zones are being shown on the part (the rig is held): wait for it to "
                    "end, or ABORT")
        return ("the zone painter holds the rig: finish it on the projector (SPACE saves, ESC "
                "cancels) or ABORT")

    def _zone_signature(self, p) -> tuple:
        """What the zone selection's status rests on, by a few stat calls: the selection
        file, the record of zone shows, the geometry's zone folder (a set added, replaced or
        trashed renames an entry in it) and the rig-state record (rig changes since a set
        was painted)."""
        from engine.zones import selection as zone_sel
        from engine.zones import store as zstore
        out = [p.name]
        try:
            folder = zone_sel.file_of(self.session.store, p.geometry_id).parent
            for f in (folder / zone_sel.FILE, folder / zstore.SHOWN_FILE, folder):
                try:
                    st = f.stat()
                    out.append((st.st_size, st.st_mtime_ns))
                except OSError:
                    out.append(None)
        except ValueError:
            out.append(None)
        return (*out, *self._rig_signature(p))

    def zone_status(self, fresh: bool = False) -> Optional[dict]:
        """The zone selection of the loaded preset's rig placement (api.zone_selection: which
        sets and zones are on, its problems, its stamp), or None when no preset is loaded.
        Cached while its files are unchanged, for at most ZONE_STATUS_TTL_S (a set's
        zones.json edited by hand shows up then); fresh: read now (a gate never decides on a
        cached answer). A read that fails is "unknown": never "nothing is on"."""
        p = self.session.preset if self.session else None
        if p is None:
            return None
        sig, now = self._zone_signature(p), time.monotonic()
        hit = self._zone_cache.get(p.geometry_id)
        if not fresh and hit is not None and hit[0] == sig and now - hit[1] < ZONE_STATUS_TTL_S:
            return hit[2]
        try:
            st = jsonable(api.zone_selection(p), self.root)
        except Exception as e:          # noqa: BLE001 - unreadable: unknown, and it says why
            why = f"the zone selection could not be read: {type(e).__name__}: {e}"
            st = {"geometry_id": p.geometry_id, "preset": p.name, "state": "unknown",
                  "ok": False, "empty": False, "text": "", "label": "NOT KNOWN",
                  "stamp": "unknown", "sets": [], "check": [], "time": None, "by": None,
                  "n_sets_on": 0, "n_zones_on": 0,
                  "problems": [{"set": None, "zone": None, "reason": "unreadable",
                                "message": why}]}
        if len(self._zone_cache) > 16:
            self._zone_cache.clear()
        self._zone_cache[p.geometry_id] = (sig, time.monotonic(), st)
        return st

    def station_zone(self, config: Optional[str]) -> dict:
        """What the station configuration `config` says about its zone: {"uses_selection":
        its job says "zone": "selection" (then - and only then - the zone selection decides
        what its run lights and judges), "zone": its own zone text or None}. A configuration
        with profiles + a flow, a recipe job, or an unreadable file: its own zones."""
        from ui import run_job as RJ
        out = {"uses_selection": False, "zone": None}
        try:
            f = self._station_file(config)
            if not flow_schema.is_legacy(flow_schema.read_doc(f)):
                return out
            c = RJ.load_station_config(f)
        except Exception:               # noqa: BLE001 - the bus marks it unusable itself
            return out
        return {"uses_selection": RJ.uses_selection(c.zone), "zone": c.zone}

    def _bus_gate(self, cmd: str, arg, armed=None) -> Optional[dict]:
        """The bus asks before START / TRIGGER start a cycle and before ARM arms
        (Handlers.gate). Refused, nothing starts: the painter / a zone show holds the rig;
        the page's `zones` stamp is not the stored selection's (it changed under the page).
        Refused with a PULL - only for a configuration whose job says "zone": "selection":
        another preset is loaded than the run uses (the page shows the loaded one's zone
        sets), the selection is unknown, unusable or empty, a selected set has a rig change
        on record since it was painted and last shown, or the selection is not the one the
        station was ARMED with. -> {"token": the selection's stamp} when it may start."""
        if self.painter_state():
            return {"message": self._painter_holds(), "pull": False}
        self._zone_token = None
        info = self.station_zone(self.bus.config)
        if not info["uses_selection"]:
            return None
        cfg = self.bus.config
        p = self.session.preset
        run = self.bus.run_preset() or (p.name if p is not None else None)
        if p is None or run != p.name:
            return {"pull": True, "message": (
                f"{cfg} inspects the selected zones with preset {run or '(none)'}, and "
                f"{'preset ' + p.name if p is not None else 'no preset'} is loaded: the zone "
                "selection shown belongs to the loaded preset - select the station "
                "configuration again (that loads its preset)")}
        st = self.zone_status(fresh=True)
        if st["state"] == "unknown" or not st["ok"]:
            return {"pull": True, "message": (
                f"{cfg} inspects the selected zones and the zone selection cannot be used: "
                f"{st['problems'][0]['message']}")}
        if st["empty"]:
            return {"pull": True, "message": (
                f"{cfg} inspects the selected zones and none is switched on: switch a zone "
                "set on (Scan zones)")}
        if st.get("check"):
            return {"pull": True, "message": (
                f"{cfg} inspects the selected zones, and since they were painted the rig-state "
                f"record has - {' | '.join(st['check'])}: show them on the part first (Show "
                "selected zones), or paint the set again")}
        want = arg.get("zones") if isinstance(arg, dict) else None
        if want and str(want) != st["stamp"]:
            return {"pull": False, "message": (
                f"the zone selection changed (it is now: {st['label']}): look at the page "
                "and press it again")}
        if armed is not None and armed != st["stamp"]:
            return {"pull": True, "message": (
                f"the zones changed since ARM (they are now: {st['label']}): ARM again")}
        self._zone_token = st["stamp"]
        return {"token": st["stamp"]}

    # ------------------------------------------------------------ command bus
    def clear_feedback(self, why: str = "cleared by a new command") -> list:
        """Stop the projected verdict of the last recipe run (queued or holding): the next
        command needs the rig, and the operator acted on it. -> ids touched."""
        touched = []
        with self._cv:
            for j in [j for j in self._queue if j.verb == "recipe_feedback"]:
                self._queue.remove(j)
                self._fail(j, why, "Cancelled")
                touched.append(j.id)
            cur = self._current
            if cur is not None and cur.verb == "recipe_feedback" and cur.status == "running":
                cur.cancel.set()
                cur.note = why
                touched.append(cur.id)
        return touched

    def _bus_start(self, config: str, confirmed: bool) -> str:
        """START on a single-job station config: its job, as `scan run`. A configuration that
        inspects the selected zones: the run is handed the selection's stamp the bus just
        accepted (_bus_gate), and is a PULL when the selection is another one when it scans."""
        if self.painter_state():                # the gate refuses this first: never a run
            raise RuntimeError(self._painter_holds())       # queued behind the painter
        self.clear_feedback("cleared: the next cycle started")
        args = {"config": config, "operator_confirmed": confirmed is True}
        if self._zone_token:
            args["zones_expect"] = self._zone_token
        return self.submit("run_station", args, bus_kind="run").id

    def _bus_select(self, config: str, preset: Optional[str]) -> Optional[str]:
        """SELECT_CONFIG / SELECT_PROFILE accepted: load the preset the configuration's
        active profile pins."""
        self.clear_feedback("cleared: another station configuration / profile was selected")
        self._mock_line_default()
        if not preset or preset not in self.preset_names():
            return None
        return self.submit("load_preset", {"name": preset}).id

    def _mock_line_default(self) -> None:
        """MOCK only: a newly selected station configuration starts with ITS simulated line:
        its default mock case (the scene the simulated human places + what the simulated
        sensors say), so scene and sensor claims always belong together. The real rig's
        claims only ever come from the line (POST /api/io/inputs, an input backend)."""
        if not self.mock or self._mock_config == self.bus.config:
            return
        self._mock_config = self.bus.config
        self.mock_case = self.mock_scene = None
        self.mock_inputs = {}
        cfg = self.bus.flow_of(self.bus.config)
        try:
            c = cfg.mock_case(None, self.bus.profile) if cfg is not None else {"name": None}
        except flow_schema.StationConfigError:
            return
        if c.get("name"):
            self.mock_case = c["name"]
            self.mock_inputs = dict(c["inputs"])
            self._mock_line_report("mock case")

    def _mock_line_report(self, source: str = "mock line", missing_only: bool = False) -> None:
        """MOCK only: the simulated line reports what its simulated sensors say (as a line's
        gateway would for every part). missing_only: only the signals nobody reported for
        this cycle (a claim posted to /api/io/inputs wins for the cycle it was made for)."""
        if not self.mock:
            return
        held = self.bus.inputs
        known = set(self.bus.known_signals())
        say = {k: v for k, v in self.mock_inputs.items()
               if (not known or k in known) and not (missing_only and k in held)}
        if say:
            try:
                self.bus.set_inputs(say, source=source)
            except ValueError:          # a case of another configuration: nothing to report
                pass

    # the cycle of a flow: the bus walks the work states (engine/flow/cycle.py), the server
    # runs each DLP check as a `flow_verify` job and keeps the cycle folder
    def _bus_cycle_begin(self, info: dict) -> dict:
        """A cycle starts: a NEW cycle folder + its log (step run folders go under it)."""
        cfg = self.bus.flow_of(info["config"])
        opened = flow_verify.open_cycle(cfg, info["profile"], mock=self.mock,
                                        operator_confirmed=info.get("confirmed") is True,
                                        source=str(info.get("source") or ""))
        with self._cv:
            self._cycles[opened["cycle_id"]] = {**opened, "cfg": cfg,
                                                "confirmed": info.get("confirmed") is True}
            self._runs[opened["cycle_id"]] = opened["cycle_dir"]
        self.clear_feedback("cleared: the next cycle started")
        self._mock_line_report(missing_only=True)       # MOCK: the simulated sensors, per part
        return {"cycle_id": opened["cycle_id"]}

    def _bus_verify(self, req: dict) -> str:
        """The current work state is verified: one DLP check, as a job that reports back."""
        self.clear_feedback("cleared: the next work state is verified")
        return self.submit("flow_verify", {"config": req["config"], "profile": req["profile"],
                                           "step": req["step"], "cycle_id": req.get("cycle_id"),
                                           "attempt": req.get("attempt"),
                                           "operator_confirmed": req.get("confirmed") is True},
                           bus_kind="run").id

    def _bus_project(self, req: dict) -> Optional[str]:
        """project_fault: paint that work state's fault on the part (its own job: the next
        verify, HOME or ABORT clears it)."""
        if not req.get("run_id") or req["run_id"] not in self._runs:
            return None
        return self.submit("recipe_feedback", {"run_id": req["run_id"]}).id

    def _bus_log(self, meta: dict, event: dict) -> None:
        with self._cv:
            c = self._cycles.get((meta or {}).get("cycle_id"))
        if c is not None:
            flow_verify.log_event(c["log"], event)

    def _bus_cycle_end(self, record: dict) -> None:
        """The cycle ended: its result.json (written once) and run_end in its log."""
        with self._cv:
            c = self._cycles.pop(record.get("cycle_id"), None)
        if c is None:
            return
        doc = flow_verify.write_cycle(c["cfg"], record, c, mock=self.mock,
                                      operator_confirmed=c["confirmed"])
        self.last_cycle = {"cycle_id": c["cycle_id"], "result": doc["result"],
                           "disposition": doc["disposition"],
                           "links": {"result_json": f"/runs/{c['cycle_id']}/result.json",
                                     "log": f"/runs/{c['cycle_id']}/run.log.jsonl"}}

    def _bus_abort(self) -> list:
        return self.abort_runs("ABORT")

    def _bus_home(self) -> str:
        """HOME accepted (now, or after the abort it was queued behind): the rig check with
        that HOME request's options (bus.home_args: set_reference, tol_px, mock_shift_px)."""
        self.clear_feedback("cleared: HOME")
        args = dict(self.bus.home_args or {})
        return self.submit("rig_check", args, bus_kind="home").id

    def _bus_pending(self) -> bool:
        """Is a job the bus started still going to report back? (Handlers.pending)"""
        with self._cv:
            return bool(self._bus_jobs)

    def bus_jobs(self) -> dict:
        with self._cv:
            return dict(self._bus_jobs)

    def _bus_notify(self, job: Job) -> None:
        """A bus job ended - ran, failed, or was cancelled / dropped before it ran: tell the
        bus. The pop and job_finished happen under the bus lock, so bus_reconcile never sees
        the gap between them as a lost job."""
        with self.bus._lock:
            with self._cv:
                kind = self._bus_jobs.pop(job.id, None)
            if kind is None:
                return
            r = job.result if isinstance(job.result, dict) else {}
            error = job.error if job.status == "error" else None
            if kind == "run":
                result, disp = r.get("result"), r.get("disposition")
                if error and job.error_type == "Cancelled":    # dropped before it ran: an
                    result, error, disp = "ABORTED", None, None     # abort, not a crash
                aborting = self.bus.state == "ABORTING"
                # a work state's DLP check (flow_verify) carries its answer: what the DLP saw
                self.bus.job_finished("run", result, error, disposition=disp,
                                      step=r.get("dlp") if job.verb == "flow_verify" else None)
                if (aborting and self.bus.last_result == "ABORTED" and isinstance(job.result, dict)
                        and r.get("result") not in (None, "ABORTED")):
                    job.result = self._aborted_after_verdict(job.result)
            else:
                self.bus.job_finished("home", r.get("status"), error)

    def _aborted_after_verdict(self, res: dict) -> dict:
        """ABORT came after the run wrote its verdict (result.json is written once, never
        rewritten) but before the job ended: the bus counts the cycle ABORTED and PULLs the
        part, so the job's result says the same (disposition PULL, reason `aborted`) and the
        run's log records it - the page, the API and the outputs never disagree."""
        why = (f"ABORT was pressed after this run wrote its verdict ({res.get('result')} / "
               f"{res.get('disposition')} in result.json): the cycle is ABORTED on the line "
               "and the part is PULLed")
        out = {**res, "disposition": "PULL", "line_result": "ABORTED",
               "pull_reasons": [*(res.get("pull_reasons") or []), D.reason("aborted", why)],
               "warnings": [*(res.get("warnings") or []), why]}
        run_dir = self._runs.get(res.get("run_id"))
        if run_dir is not None and (Path(run_dir) / "run.log.jsonl").is_file():
            try:
                R.append_event(Path(run_dir) / "run.log.jsonl", res.get("run_id"),
                               "aborted_after_verdict", level="warning", result="ABORTED",
                               disposition="PULL", message=why)
            except OSError as e:
                out["warnings"].append(f"run log not updated: {e}")
        return out

    def bus_reconcile(self) -> None:
        """Recovery: a busy bus state (RUNNING / ABORTING / HOMING) with no job left that
        will report back is settled (job_lost) instead of wedging the station."""
        with self.bus._lock:
            if self.bus.state in BUS_BUSY and not self._bus_pending():
                self.bus.job_lost("no station job is running or queued any more")

    def cmd_state(self) -> dict:
        """The bus snapshot for /quick and a PLC gateway (station configuration, profile, the
        cycle's work states, permissions, sensor inputs, signals, registers), plus the bus
        jobs, the loaded preset (active_preset), the preset START would run (run_preset: the
        configuration's / profile's own preset, else the loaded one), the last cycle's
        links and, in MOCK, the simulated line's case."""
        self.bus_reconcile()
        self.bus.tick()
        snap = self.bus.snapshot()
        snap["jobs"] = self.bus_jobs()
        p = self.session.preset
        snap["active_preset"] = p.name if p is not None else None
        snap["run_preset"] = snap.get("config_preset") or snap["active_preset"]
        snap["warnings"] = list(self.startup_warnings)     # e.g. the selected preset is gone
        snap["mode"] = "mock" if self.mock else "real"
        snap["mode_jobs"] = self.mode_jobs()
        # Measure / Quick check ask strictly; Station mode asks as the station does (a fixed
        # rig is not gated there, and nothing is read for it)
        snap["rig"] = self.rig_gate(p, strict=self.bus.mode != modes.DEFAULT_MODE)
        # painted zones: the selection of the loaded preset's rig placement (None: no preset
        # loaded), the painter / zone show that holds the rig now, how the last painter
        # session ended - what every page draws its Scan zones panel from
        snap["zone_selection"] = self.zone_status()
        snap["painter"] = self.painter_state()
        snap["last_paint"] = dict(self.last_paint) if self.last_paint else None
        snap["last_cycle"] = dict(self.last_cycle) if self.last_cycle else None
        cyc = snap.get("cycle") or {}
        if cyc.get("cycle_id") and cyc.get("active"):
            snap["cycle_links"] = {"log": f"/runs/{cyc['cycle_id']}/run.log.jsonl"}
        if self.mock:
            cfg = self.bus.flow_of(self.bus.config)
            snap["mock"] = {"case": self.mock_case, "scene": self.mock_scene,
                            "inputs": dict(self.mock_inputs),
                            "cases": cfg.public()["mock_cases"] if cfg is not None else []}
        return snap

    def _rig_signature(self, p) -> tuple:
        """What the rig's state rests on, by two stat calls: the rig-state record (it only
        grows) and the geometry's calibration folder (an entry added or trashed)."""
        out = []
        try:
            folder = rigstate.log_path(self.session.store, p.geometry_id).parent
            for f in (folder / rigstate.LOG_FILE, folder):
                try:
                    st = f.stat()
                    out.append((st.st_size, st.st_mtime_ns))
                except OSError:
                    out.append(None)
        except ValueError:
            out.append(None)
        return tuple(out)

    def rig_gate(self, preset=None, strict: bool = True) -> Optional[dict]:
        """May the loaded preset's rig give mm now (engine/rigstate.gate; strict: what Measure
        and Quick check ask)? {ok, code, message, how_to_fix, mount}; None: no preset. Cached
        while the rig-state record and the calibration folder are unchanged, for at most
        RIG_GATE_TTL_S (a session also ends by the clock)."""
        p = preset if preset is not None else (self.session.preset if self.session else None)
        if p is None:
            return None
        key, sig = (p.name, p.rig.mount, bool(strict)), self._rig_signature(p)
        now = time.monotonic()
        hit = self._rig_cache.get(key)
        if hit is not None and hit[0] == sig and now - hit[1] < RIG_GATE_TTL_S:
            return dict(hit[2])
        try:
            g = rigstate.gate(self.session.store, p, strict=bool(strict),
                              presets_dir=self.session.presets_dir)
        except Exception as e:          # noqa: BLE001 - unreadable: no mm, and it says why
            return {"ok": False, "code": "rig_unverified" if p.rig.mount == "mobile" else
                    "rig_check", "mount": p.rig.mount, "how_to_fix": "",
                    "message": f"the rig's state could not be read: {type(e).__name__}: {e}"}
        out = {k: g[k] for k in ("ok", "code", "message", "how_to_fix", "mount")}
        if len(self._rig_cache) > 64:
            self._rig_cache.clear()
        self._rig_cache[key] = (sig, time.monotonic(), out)     # (timed when it was known)
        return dict(out)

    def modes_info(self) -> dict:
        """What the operator page's mode switch and pickers show: the modes, the selected
        one, the triangulation presets (mount, stereo calibration), the recipes."""
        from engine.modes import measure as measure_mode
        from engine.modes import quick as quick_mode
        p = self.session.preset
        presets = [{k: i.get(k) for k in ("name", "tier", "mount", "calibrated", "geometry_id",
                                          "valid", "description")}
                   for i in preset_lib.listing(self.session.store, self.presets_path, self.mock)]
        # a geometry_id names one rig: a preset at a geometry another preset calls mobile is
        # gated as mobile (engine/rigstate.mount_of), and listed so
        mobile = {i["geometry_id"]: i["name"] for i in presets
                  if i.get("mount") == "mobile" and i.get("geometry_id")}
        for i in presets:
            by = mobile.get(i.get("geometry_id"))
            if by and i.get("mount") != "mobile":
                i.update(mount="mobile", mount_declared_by=by)
        try:
            recipes = [i["name"] for i in api.recipe_list() if i.get("valid")]
        except Exception:               # noqa: BLE001 - listed empty rather than a 500
            recipes = []
        return {"mode": self.bus.mode,
                "modes": [{"name": m, "title": modes.TITLES[m], "needs": modes.NEEDS[m]}
                          for m in modes.MODES],
                "measure": {"whats": list(measure_mode.WHATS)},
                "quick_check": {"checks": list(quick_mode.CHECKS)},
                "presets": presets, "recipes": recipes,
                "loaded_preset": p.name if p is not None else None,
                "rig": self.rig_gate(p)}

    def refresh_configs(self) -> list:
        """Re-read the station configurations and hand them to the bus: every config in
        file-name order (an invalid one keeps its number and is marked unusable, so a PLC's
        configuration number never shifts because another config broke), with its profiles
        and work states. The bus never switches the selected configuration or profile on its
        own, and keeps the list as it is while ARMED or in a cycle."""
        flows: dict = {}
        items = self.stations(flows)
        ok = [i for i in items if not i.get("error")]
        self.bus.update_configs(
            [i["config"] for i in items],
            unusable={i["config"]: i["error"] for i in items if i.get("error")},
            presets={i["config"]: i.get("preset") for i in ok},
            info={i["config"]: {"recipe": i["recipe"], "result_codes": i.get("result_codes")}
                  for i in ok if i.get("recipe")},
            flows=flows)
        return items

    def set_inputs(self, values: dict, source: str = "http") -> dict:
        """Sensor claims from the line (POST /api/io/inputs): {signal: true | false}. MOCK:
        a signal the line itself reports is the line's from then on - the simulated sensor
        of that name stops reporting it (until a mock case is chosen again), so a gateway
        tested against the mock server must report for every part, as on the real line."""
        try:
            held = self.bus.set_inputs(values, source=source)
        except ValueError as e:
            raise HTTPError(400, str(e)) from None
        for k in values:
            self.mock_inputs.pop(k, None)
        return held

    def set_mock_case(self, case=None, scene=None, inputs=None) -> dict:
        """MOCK only: the simulated line. A named case of the selected configuration (the
        scene the simulated human places for the next scans + what the simulated sensors
        say), or an explicit scene / inputs ({signal: true | false}; null: that simulated
        sensor stops reporting). {} goes back to the configuration's default case. The
        simulated sensors are reported to the bus now and again for every cycle, as a line's
        gateway reports its sensors for every part."""
        if not self.mock:
            raise HTTPError(400, "the simulated line is MOCK ONLY: on the real rig the part "
                                 "is placed by a human and the sensors are the line's own")
        cfg = self.bus.flow_of(self.bus.config)
        if case is not None:
            if cfg is None or not isinstance(case, str):
                raise HTTPError(400, "case must be the name of a mock case")
            try:
                c = cfg.mock_case(case, self.bus.profile)
            except flow_schema.StationConfigError as e:
                raise HTTPError(400, str(e)) from None
            self.mock_case, self.mock_scene = c["name"], None
            self.mock_inputs = dict(c["inputs"])
            self._mock_line_report("mock case")
        elif scene is not None or inputs is not None:
            if scene is not None:
                if not isinstance(scene, str):
                    raise HTTPError(400, "scene must be text: scene:<name>[:<variant>] or none")
                try:
                    flow_verify.mock_placer(self.session, scene)
                except ValueError as e:
                    raise HTTPError(400, str(e)) from None
                self.mock_case, self.mock_scene = None, scene
            if inputs is not None:
                try:                    # strict: JSON booleans / null, declared signals only
                    self.bus.set_inputs(inputs, source="mock")
                except ValueError as e:
                    raise HTTPError(400, str(e)) from None
                self.mock_inputs = {k: v for k, v in {**self.mock_inputs, **inputs}.items()
                                    if v is not None}
        else:                           # the configuration's default case again
            self.mock_case = self.mock_scene = None
            self.mock_inputs = {}
            try:
                if cfg is not None:
                    self.mock_inputs = dict(cfg.mock_case(None, self.bus.profile)["inputs"])
            except flow_schema.StationConfigError:
                pass
            self._mock_line_report("mock case")
        return {"case": self.mock_case, "scene": self.mock_scene,
                "inputs": dict(self.bus.inputs), "simulated": dict(self.mock_inputs)}

    def _mock_scene_now(self, cfg, profile: str) -> Optional[str]:
        """MOCK: what the simulated human places before a work state's scan: the explicit
        scene, else the selected case's, else the configuration's default case / the
        profile's good part."""
        if not self.mock:
            return None
        if self.mock_scene is not None:
            return self.mock_scene
        try:
            return cfg.mock_case(self.mock_case, profile)["scene"]
        except flow_schema.StationConfigError:      # a case of another configuration
            return cfg.mock_case(None, profile)["scene"]

    def abort_runs(self, why: str = "aborted by the operator") -> list:
        """ABORT (in any bus state): drop queued station / recipe runs, rig checks and
        projected verdicts, abort the running one, stop the live view.
        -> ids of the jobs touched."""
        dropped = []
        with self._cv:
            touched = []
            for j in [j for j in self._queue if j.verb in CANCELLABLE_VERBS]:
                self._queue.remove(j)
                self._fail(j, f"cancelled before it started: {why}", "Cancelled")
                touched.append(j.id)
                dropped.append(j)
            cur = self._current
            if cur is not None and cur.verb in CANCELLABLE_VERBS and cur.status == "running":
                cur.cancel.set()
                cur.note = "abort requested: stops before the next capture"
                touched.append(cur.id)
            if self._stop_live_locked(why):
                touched.append("live view")
        for j in dropped:                       # never ran: the bus still hears it ended
            self._bus_notify(j)
            if j.verb == "zone_paint":          # ... and the page: nothing was painted
                self._paint_record(j.args.get("name"), "dropped",
                                   reason="ABORT came before the painter started")
        return touched

    def _execute(self, job: Job):
        # The final status is published only AFTER the status snapshot is
        # refreshed: a client that polls the job and then reads /api/status
        # must see the job's effects (e.g. a new calibration), never a stale
        # snapshot (race seen in the full test run).
        t0 = time.perf_counter()
        final = "error"
        try:
            if job.verb in CANCELLABLE_VERBS:
                out = self._cancellable(job)
            else:
                out = getattr(self, "_v_" + job.verb)(dict(job.args))
            result, links = out if isinstance(out, tuple) else (out, {})
            job.result = jsonable(result, self.root)
            job.links = {k: jsonable(v, self.root) for k, v in (links or {}).items()
                         if v is not None}
            final = "done"
        except Exception as e:          # noqa: BLE001 - a verb must never kill the worker
            job.error = f"{type(e).__name__}: {e}"
            job.error_type = type(e).__name__
            job.hint = error_hint(e)
            job.trace = "".join(traceback.format_exception(e)[-3:])[-2000:]
        finally:
            job.finished = _now()
            job.duration_s = round(time.perf_counter() - t0, 3)
            try:
                self._refresh_snapshot()
            except Exception as e:      # noqa: BLE001
                self.startup_warnings.append(f"status refresh failed: {e}")
            job.status = final
            job.done.set()
            self._bus_notify(job)

    # ------------------------------------------------------------ helpers
    def _preset(self):
        p = self.session.preset
        if p is None:
            raise ArgError("no preset loaded: pick one in Setup and press Load")
        return p

    def _list(self, kind: str, geometry_id: Optional[str] = None) -> list:
        """api.list_library for the UI's own browsing / id lookups, logged as
        ONE compact record (kind + count), not the full listing: full listings
        would crowd the real actions out of the session log (the Log tab's
        window). Agents calling the list_library verb still get it in full."""
        if kind == "lights":
            return self._lights(geometry_id)
        if kind == "atlases":
            return self._atlases(geometry_id)
        raw = getattr(api.list_library, "__wrapped__", None)
        if raw is None:                         # engine changed: fall back, fully logged
            return api.list_library(kind, geometry_id)
        out: dict = {}

        def call():
            out["items"] = raw(kind, geometry_id)
            return {"kind": kind, "geometry_id": geometry_id, "n": len(out["items"])}

        self.session.run("list_library", {"kind": kind, "geometry_id": geometry_id,
                                          "logged": "count only (web UI browsing)"}, call)
        return out["items"]

    def _entry(self, kind: str, entry_id, items=None):
        if not entry_id:
            raise ArgError(f"no {kind[:-1]} selected")
        items = self._list(kind) if items is None else items
        for it in items:
            if _entry_id(it) == entry_id:
                return it
        raise EntryNotFound(f"no {kind[:-1]} with id {entry_id!r} in the library")

    def _object(self, a: dict, key: str = "object_id", items=None):
        return self._entry("objects", _s(a, key, required=True), items)

    def _base(self, base_id, p):
        if not base_id:
            b = api.latest_base(p)
            if b is None:
                raise EntryNotFound(f"no usable base for geometry {p.geometry_id}: capture one "
                                    "(Base library) with the scene EMPTY")
            return b
        b = self._entry("bases", base_id)
        if b.meta.get("rejected"):
            raise ArgError(f"base {base_id} was rejected ({b.meta['rejected']}); pick another")
        return b

    def _bench_state(self) -> Optional[dict]:
        if not self.mock:
            return None
        b = api.mock_bench()
        return {"objects": [dataclasses.asdict(o) for o in b.objects],
                "labels": [{"data": lb.data, "center_mm": list(lb.center_mm),
                            "size_mm": lb.size_mm} for lb in b.labels]}

    def _refresh_snapshot(self):
        s = self.session
        p = s.preset
        snap: dict = {"mode": "mock" if self.mock else "real", "mock": self.mock,
                      "preset": None, "geometry_id": None, "calibration": None,
                      "latest_base": None, "output_kinds": list(OUTPUT_KINDS),
                      "paths": {k: str(Path(v).resolve()) for k, v in api.paths().items()},
                      "log_path": str(s.log_path), "bench": self._bench_state(),
                      "presets_dir": str(self.presets_path.resolve()),
                      "warnings": list(self.startup_warnings)}
        if p is not None:
            rig = p.rig
            snap["preset"] = {
                "name": p.name, "geometry_id": p.geometry_id,
                "rig_geometry_id": p.rig_geometry_id, "warnings": list(p.warnings),
                "source": str(p.source) if p.source else None,
                "camera": dataclasses.asdict(p.camera),
                "projection": dataclasses.asdict(p.projection),
                "decode": dataclasses.asdict(p.decode),
                "tolerance": dataclasses.asdict(p.tolerance), "bases": p.bases,
                "rig": {**dataclasses.asdict(rig), "coded_axis": rig.coded_axis,
                        "nominal_mm_per_relief_px": rig.nominal_mm_per_relief_px,
                        "triangulation_angle_deg": rig.triangulation_angle_deg,
                        "note": "nominal = pre-calibration estimate, never a measurement"}}
            snap["geometry_id"] = p.geometry_id
            # the engine's own rule-6 lookup, never a re-implementation of it
            snap["calibration"] = api.calibration_for(p.geometry_id,
                                                      config.geometry_fingerprint(p))
            snap["latest_base"] = api.latest_base(p)
            try:
                snap["grid"] = self._latest_grid(p.geometry_id)
            except Exception as e:      # noqa: BLE001 - a bad grid file must not break status
                snap["grid"] = None
                snap["warnings"].append(f"could not read the latest grid shot: {e}")
        self._snapshot = jsonable(snap, self.root)

    def _latest_grid(self, geometry_id: str) -> Optional[dict]:
        """The grid shot the engine resolves zone names against (the newest
        grid JSON for this geometry, whoever projected it: this page, another
        browser, the CLI or an agent), in the shape project_grid returns, so
        every client can show the grid the engine will use."""
        from engine.capture import zones as zones_mod
        store = self.session.store
        for row in reversed(store.outputs("previews")):
            if row.get("geometry_id") != geometry_id or \
                    not row.get("file", "").endswith(zones_mod.GRID_JSON_EXT):
                continue
            f = store.root / row["file"]
            try:
                doc = json.loads(f.read_text(encoding="utf-8"))
                spacing = int(doc["spacing_px"])
            except (OSError, ValueError, KeyError, TypeError):
                continue

            def rel(p):
                return rel_path(p, self.root) if p else None
            return {"__type__": "ZoneShot", "kind": "grid", "source": "engine",
                    "json_path": rel(f), "created": doc.get("created"),
                    "label": doc.get("label"), "preset": doc.get("preset"),
                    "grid": {"__type__": "GridSpec", "spacing_px": spacing,
                             **{k: doc.get(k) for k in ("n_cols", "n_rows", "col_labels",
                                                        "row_labels")}},
                    "camera_png": rel(doc.get("camera_png")),
                    "overlay_png": rel(doc.get("overlay_png")),
                    "dmd_png": rel(doc.get("dmd_png")),
                    "cells_visible": doc.get("cells_visible") or [],
                    "cell_centers": doc.get("cell_centers") or {},
                    "map_id": doc.get("map_id"), "warnings": doc.get("warnings") or []}
        return None

    # --------------------------------------------------------- read side
    def status(self) -> dict:
        with self._cv:
            running = self._inline or self._current    # an inline job runs inside the live view
            cur = running.public(full=False) if running else None
            queued = [j.public(full=False) for j in self._queue]
        return {**self._snapshot, "job": cur, "queued": queued, "live": self.live.public(),
                "server": {"started": self.started, "pid": os.getpid(), "port": self.port,
                           "bind": BIND}}

    def _busy(self) -> Optional[str]:
        """Verb of a running job that blocks the worker (live view drains reads)."""
        cur = self._inline or self._current
        return cur.verb if cur is not None and cur.verb != "live_view" else None

    def listing(self, kind: str, geometry_id: Optional[str] = None):
        """-> (payload, raw items). Runs the listing on the worker; while a
        capture holds the worker, returns the last listing marked stale."""
        if kind not in LIST_KINDS + UI_LIST_KINDS:
            raise HTTPError(400, f"kind must be one of {LIST_KINDS + UI_LIST_KINDS}")
        key = (kind, geometry_id or "")

        def fn():
            items = self._list(kind, geometry_id)
            data = jsonable(items, self.root)
            self._list_cache[key] = (data, items)
            return data, items

        busy = self._busy()
        try:
            data, items = self.call(fn, timeout=1.5 if busy else 20.0)
            return {"kind": kind, "items": data, "stale": False}, items
        except TimeoutError:
            data, items = self._list_cache.get(key, ([], []))
            return {"kind": kind, "items": data, "stale": True,
                    "note": f"worker busy ({busy or 'job'}): showing the last listing"}, items

    def _find_scan(self, object_id: str):
        """-> (ObjectScan or None, stale). A just-captured scan must be found
        even while a job holds the worker: the cached listings are keyed by
        geometry and may predate it, so fall back to the other cached
        listings, then to a read-only look at the library folder."""
        payload, items = self.listing("objects")
        scan = next((o for o in items if o.id == object_id), None)
        if scan is not None or not payload.get("stale"):
            return scan, False
        for (kind, _geo), (_data, cached) in list(self._list_cache.items()):
            if kind == "objects":
                scan = next((o for o in cached if o.id == object_id), None)
                if scan is not None:
                    return scan, True
        try:                                    # read-only: listing folders + meta.json
            from engine.library import objects as object_lib
            scan = next((o for o in object_lib.list_objects(self.session.store)
                         if o.id == object_id), None)
        except Exception:                       # noqa: BLE001 - e.g. a scan being written
            scan = None
        return scan, True

    def pointcloud(self, object_id: str, max_points: int = 20000) -> dict:
        """Subsampled points for the browser viewer, in the same units as
        reconstruct() would write (mm only with a matching calibration)."""
        scan, stale = self._find_scan(object_id)
        if scan is None:
            if stale:
                raise HTTPError(503, f"no scan with id {object_id!r} in the last listing and "
                                     f"the engine worker is busy ({self._busy() or 'job'}): "
                                     "try again when the job finishes")
            raise HTTPError(404, f"no scan with id {object_id!r}")
        scan = dataclasses.replace(scan, _field=None, _relief=None)  # don't pin arrays in the cache
        # the calibration reconstruct() uses for this scan: its geometry AND its settings
        cal = api.calibration_for(scan.geometry_id, (scan.meta or {}).get("geometry_fp"))
        pts, xy_u, z_u = pointcloud.points(scan, cal)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        n_total = int(len(pts))
        max_points = int(np.clip(max_points, 100, 300000))
        if n_total > max_points:
            idx = np.sort(np.random.default_rng(0).choice(n_total, max_points, replace=False))
            pts = pts[idx]
        tier = (TIER_LATERAL if xy_u == "mm" and z_u == "mm" else
                TIER_DEPTH if z_u == "mm" else TIER_RELIEF)
        lo = pts.min(axis=0).tolist() if len(pts) else [0, 0, 0]
        hi = pts.max(axis=0).tolist() if len(pts) else [0, 0, 0]
        return {"object_id": scan.id, "label": scan.label, "geometry_id": scan.geometry_id,
                "n_total": n_total, "n": int(len(pts)), "units": {"xy": xy_u, "z": z_u},
                "tier": tier, "calibration_id": cal.id if cal is not None and (
                    xy_u == "mm" or z_u == "mm") else None,
                "bounds": {"min": [round(v, 3) for v in lo], "max": [round(v, 3) for v in hi]},
                "note": "xy = camera px from the image centre unless calibrated; "
                        "z = relief (projector px) unless calibrated",
                "points": np.round(pts, 3).tolist()}

    def log_tail(self, n: int = 200, exclude=()) -> dict:
        """The last n records, oldest first, skipping verbs in `exclude`.
        The filter runs BEFORE the cut and the file is read backwards in
        chunks until n records match (or LOG_SCAN_BYTES were read), so
        browsing records can't crowd the real actions out of the window."""
        path = Path(self._snapshot.get("log_path") or "")
        if not path.is_file():
            return {"path": str(path), "records": []}
        exclude = set(exclude or ())
        recs: list = []
        scanned = 0
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            tail = b""                          # partial first line of the chunk after
            while pos > 0 and len(recs) < n and scanned < LOG_SCAN_BYTES:
                step = min(1 << 20, pos)
                pos -= step
                f.seek(pos)
                buf = f.read(step) + tail
                scanned += step
                lines = buf.split(b"\n")
                tail = lines.pop(0) if pos > 0 else b""   # may be cut: finish it next round
                for line in reversed(lines):
                    rec = _log_record(line, exclude)
                    if rec is not None:
                        recs.append(rec)
                        if len(recs) >= n:
                            break
        recs.reverse()
        return {"path": str(path), "records": jsonable(recs, self.root),
                "excluded": sorted(exclude), "truncated": pos > 0 and len(recs) < n}

    # -------------------------------------------------------------- presets
    def preset_names(self) -> list:
        """Built-in presets, then this library's user presets."""
        users = preset_lib.user_names(self.session.store) if self.session is not None else []
        return config.list_presets(self.presets_path) + users

    def preset_tiers(self) -> list:
        """Both tiers with flags (tier, active = selected, geometry_id, calibrated, description)."""
        return jsonable(preset_lib.listing(self.session.store, self.presets_path, self.mock),
                        self.root)

    def check_preset_name(self, name: str) -> str:
        """Only names listed in the presets folder: config.resolve_preset_path
        would also accept a path, and the browser must not pick arbitrary files."""
        if not _NAME_RE.match(name or "") or name not in self.preset_names():
            raise HTTPError(404, f"no preset named {name!r} in {self.presets_path}")
        return name

    def preset_detail(self, name: str) -> dict:
        """JSON form for the editor. Parsed as REAL so the preset's own
        calibration block survives an edit + save (mock mode ignores it)."""
        name = self.check_preset_name(name)
        try:
            p = config.load_preset(name, presets_dir=self.presets_path, mock=False,
                                   user_dir=config.user_presets_dir(self.root))
        except Exception as e:          # noqa: BLE001 - shown verbatim in the editor
            raise HTTPError(400, f"{type(e).__name__}: {e}") from None
        rig = p.rig
        active = self.session.preset.name if self.session.preset is not None else None
        tier = preset_lib.tier_of(self.session.store, self.presets_path, name)
        return {"name": name, "active": name == active, "path": str(p.source), "tier": tier,
                "selected": name == api.preset_active(),
                "preset": config.preset_to_dict(p), "warnings": list(p.warnings),
                "geometry_id": (config.MOCK_PREFIX if self.mock else "") + p.rig_geometry_id,
                "derived": {"coded_axis": rig.coded_axis,
                            "nominal_mm_per_relief_px": rig.nominal_mm_per_relief_px,
                            "triangulation_angle_deg": rig.triangulation_angle_deg}}

    def save_preset(self, body: dict) -> dict:
        """The Setup editor's Save: a USER preset in this library (<library>/presets/<name>/,
        runtime data), never a file in the built-in presets folder (read-only, tracked in
        the repo). Same path as the preset_add verb: a built-in name is refused, and
        overwrite replaces an existing USER preset (the old one goes to the library trash).
        body: {"preset": {...}, "base": the preset the editor opened, "overwrite": false,
        "description": text}."""
        usage = ("body must be {\"preset\": {...}, \"base\": \"<the preset it was edited from>\", "
                 "\"overwrite\": false}")
        d = body.get("preset")
        if not isinstance(d, dict):
            raise HTTPError(400, usage)
        name = d.get("name")
        if not isinstance(name, str) or not name.strip():
            raise HTTPError(400, "preset.name is required")
        overwrite = body.get("overwrite", False)
        if not isinstance(overwrite, bool):
            raise HTTPError(400, "overwrite must be a JSON boolean (true / false)")
        base = body.get("base")
        if base is None and name in self.preset_names():
            base = name                         # re-saving an existing preset under its name
        if not isinstance(base, str) or not base:
            raise HTTPError(400, "'base' (the preset this one was edited from) is required: "
                                 + usage)
        try:
            self.check_preset_name(base)
        except HTTPError as e:
            raise HTTPError(400, f"base: {e.msg}") from None
        geo = d.get("geometry_id")
        overrides = {k: v for k, v in d.items() if k not in ("name", "geometry_id")}
        try:
            r = preset_lib.add(self.session.store, self.presets_path, name, base, overrides,
                               geo, str(body.get("description") or ""), replace=overwrite)
        except Exception as e:          # noqa: BLE001 - validation errors go back verbatim
            raise HTTPError(400, f"{type(e).__name__}: {e}") from None
        return {"saved": r["name"], "tier": "user", "path": str(r["path"]),
                "overwrote": bool(r.get("replaced")),
                "replaced": str(r["replaced"]) if r.get("replaced") else None,
                "warnings": list(r.get("warnings") or []), "note": r.get("note")}

    # ================================================================ verbs
    # Each _v_<verb> runs ON THE WORKER. Returns result or (result, links).

    def _v_load_preset(self, a):
        name = _s(a, "name", required=True)
        try:
            self.check_preset_name(name)
        except HTTPError as e:
            raise ArgError(e.msg) from None
        p = api.load_preset(name)
        self._mobile_session(p)
        return p

    def _v_preset_add(self, a):
        ov = a.get("overrides")
        if ov is not None and not isinstance(ov, dict):
            raise ArgError("'overrides' must be an object {\"camera.exposure_us\": 33333, ...}")
        base = self.check_preset_name_arg(_s(a, "base", required=True))
        return api.preset_add(_s(a, "name", required=True), base, ov, _s(a, "geometry_id"),
                              _s(a, "description", "") or "", _bool(a, "replace"))

    def _v_preset_select(self, a):
        name = self.check_preset_name_arg(_s(a, "name", required=True))
        rec = api.preset_select(name)
        p = api.load_preset(name)
        self._mobile_session(p)
        return {"selected": rec, "loaded": p.name, "geometry_id": p.geometry_id,
                "warnings": list(p.warnings)}

    def _v_preset_delete(self, a):
        if not _confirmed(a, "confirm"):
            raise ArgError("preset_delete needs confirm=true (the operator confirmed the delete)")
        name = self.check_preset_name_arg(_s(a, "name", required=True))
        p = self.session.preset
        if p is not None and p.name == name:
            raise ArgError(f"{name!r} is the loaded preset: load or select another one first")
        return api.preset_delete(name, _s(a, "reason", required=True), _bool(a, "force"),
                                 stations_dir=self.stations_dir)

    def check_preset_name_arg(self, name: str) -> str:
        try:
            return self.check_preset_name(name)
        except HTTPError as e:
            raise ArgError(e.msg) from None

    def _v_check_signal(self, a):
        return api.check_signal(self._preset())

    def _v_capture_base(self, a):
        return api.capture_base(self._preset(), _s(a, "label", "base"), zone=_s(a, "zone"))

    def _v_capture_object(self, a):
        p = self._preset()
        base = self._base(_s(a, "base_id"), p)
        return api.capture_object(p, base, _s(a, "label", required=True),
                                  known_dims=known_dims(a.get("known_dims")),
                                  zone=_s(a, "zone"))

    def _v_reconstruct(self, a):
        fmt = _s(a, "fmt", "ply")
        if fmt not in ("ply", "xyz"):
            raise ArgError("fmt must be 'ply' or 'xyz'")
        out = api.reconstruct(self._object(a), fmt=fmt)
        return out, {"file": out}

    def _v_measure(self, a):
        return api.measure(self._object(a))

    def _v_compare(self, a):
        objs = self._list("objects")
        scan, golden = self._object(a, items=objs), self._object(a, "golden_id", items=objs)
        tol = None
        if any(a.get(k) not in (None, "") for k in ("fail_mm", "fail_px", "min_defect_px",
                                                       "smooth_px")):
            # like the CLI: a px band alone keeps the preset's mm band, so a
            # calibrated geometry can still judge (it judges in mm)
            p = self._preset()
            tol = ToleranceBand(fail_mm=_num(a, "fail_mm", p.tolerance.ch_fail_mm),
                                fail_px=_num(a, "fail_px"),
                                min_defect_px=_num(a, "min_defect_px",
                                                   p.tolerance.min_defect_px, int),
                                smooth_px=_num(a, "smooth_px", 5, int))
        store = self.session.store
        n0 = len(store.outputs("comparisons"))
        v = api.compare(scan, golden, tol)
        links = {}
        for row in store.outputs("comparisons")[n0:]:     # what this compare just wrote
            f = store.root / row.get("file", "")
            key = {".png": "png", ".json": "json"}.get(f.suffix.lower())
            if key and f.is_file():
                links[key] = f
        return v, links

    def _v_calibrate(self, a):
        ids = a.get("object_ids") or []
        if not isinstance(ids, list) or not ids:
            raise ArgError("pick at least one scan with known dims")
        objs = self._list("objects")
        return api.calibrate(self._preset(), [self._entry("objects", i, objs) for i in ids])

    def _v_read_codes(self, a):
        levels = _num_list(a, "levels", int)
        kw = {"levels": tuple(levels)} if levels else {}
        r = api.read_codes(self._preset(), _s(a, "label", "codes"), **kw)
        links = {}
        if r.path is not None and (Path(r.path) / "full_on.png").is_file():
            links["frame_png"] = Path(r.path) / "full_on.png"
        return r, links

    def _v_inspect(self, a):
        insp = api.inspect(self._object(a), n_sigma=_num(a, "n_sigma", 6.0),
                           min_area_px=_num(a, "min_area_px", 50, int),
                           bg_window=_num(a, "bg_window", 121, int),
                           whole_field=_bool(a, "whole_field"))
        links = {}
        if insp.path is not None and Path(insp.path).with_suffix(".png").is_file():
            links["annotated_png"] = Path(insp.path).with_suffix(".png")
        return insp, links

    def _v_delete(self, a):
        kind = _s(a, "kind", required=True)
        if kind not in DELETABLE:
            raise ArgError(f"kind must be one of {DELETABLE}")
        eid = _s(a, "id", required=True)
        # banks and sensors are keyed by name, so only look in the active geometry's
        items = (self._list(kind, self._preset().geometry_id)
                 if kind in ("banks", "sensors") else None)
        entry = self._entry(kind, eid, items)
        if kind == "calibrations":            # listed as JSON; the id is <geometry>/<ts>
            target = self.session.store.root / "calibrations" / eid
        else:
            target = _entry_path(entry)
        if target is None or not target.is_dir():
            raise EntryNotFound(f"{kind} {eid}: entry folder not found")
        dest = api.delete(target, reason=_s(a, "reason", "deleted from the web UI"))
        note = None
        p = self.session.preset
        if kind == "calibrations" and p is not None:
            api.load_preset(p.name)            # rebind: the trashed fit must not stay active
            note = f"reloaded preset {p.name} so the active calibration is current"
        return {"kind": kind, "id": eid, "trashed_to": dest, "note": note}

    def _v_reject_base(self, a):
        reason = _s(a, "reason", required=True)
        return api.reject_base(self._entry("bases", _s(a, "base_id", required=True)), reason)

    def _v_profile(self, a):
        axis = _s(a, "axis", "auto")
        if axis not in ("auto", "x", "y"):
            raise ArgError("axis must be auto, x or y")
        pr = api.profile(self._object(a), _point(a, "p0"), _point(a, "p1"),
                         _num(a, "width_px", 5, int), axis)
        return pr, {"csv": getattr(pr, "csv_path", None), "png": getattr(pr, "png_path", None)}

    def _v_bank_add(self, a):
        return api.bank_add(self._object(a), _s(a, "bank", "default"))

    def _v_bank_check(self, a):
        v = api.bank_check(self._object(a), _s(a, "bank", "default"),
                           _num(a, "k_sigma", 4.0), _num(a, "min_area_px", 25, int))
        return v, {"png": getattr(v, "png_path", None), "json": getattr(v, "json_path", None)}

    def _v_bank_remove(self, a):
        return api.bank_remove(self._object(a), _s(a, "bank", "default"),
                               _s(a, "reason", "removed from the web UI"))

    def _v_tune(self, a):
        return api.tune(self._preset(), _num_list(a, "exposures", int),
                        _num_list(a, "duties", float))

    def _v_measure_gamma(self, a):
        return api.measure_gamma(self._preset(), _num(a, "n_levels", 12, int))

    def _v_capture_map(self, a):
        return api.capture_map(self._preset(), _s(a, "label", "map"))

    def _v_project_grid(self, a):
        return api.project_grid(self._preset(), _num(a, "spacing_px", 96, int),
                                _s(a, "label", "grid"), _hold_s(a))

    def _v_project_zones(self, a):
        from engine.capture.zones import Zone
        raw = a.get("zones")
        if not isinstance(raw, list) or not raw:
            raise ArgError("zones must be a non-empty list of {name, polygon, pattern, ...}")
        names = {f.name for f in dataclasses.fields(Zone)}
        zones = []
        for z in raw:
            if not isinstance(z, dict) or set(z) - names:
                raise ArgError(f"each zone is an object with keys {sorted(names)}")
            poly = z.get("polygon")
            if not isinstance(poly, list) or len(poly) < 2:
                raise ArgError(f"zone {z.get('name')!r}: polygon needs >= 3 [x, y] points "
                               "(or 2 opposite corners of a rectangle)")
            zones.append(Zone(**{**z, "polygon": [_point({"p": pt}, "p") for pt in poly]}))
        return api.project_zones(zones, self._preset(), _s(a, "label", "zones"), _hold_s(a))

    # ---- painted zone sets (engine/zones)
    def zone_sets(self) -> dict:
        """The painted zone sets at the loaded preset's geometry (files only: answered while
        a job holds the worker, e.g. the painter itself). Each set also says whether it is
        switched on (`on`, `ids_on`: null = every zone; each zone its own `on`) and who names
        it (`used_by`: recipes, station configurations); `selection` is the zone selection
        (zone_status); `others` lists the rig placements that have sets of their own."""
        p = self.session.preset
        geo = p.geometry_id if p is not None else None
        items = api.zone_sets(all_geometries=True)
        sel = self.zone_status(fresh=True)
        rows = {r["set"]: r for r in (sel or {}).get("sets") or []}
        mine = []
        for i in items:
            if i["geometry_id"] != geo:
                continue
            r = rows.get(i["name"]) or {}
            on_ids = {z["id"] for z in r.get("zones") or [] if z["on"]}
            try:
                used = api.zone_users(i["name"], self.stations_dir)
            except Exception:           # noqa: BLE001 - a listing never fails on one file
                used = {"recipes": [], "stations": []}
            mine.append({**i, "on": bool(r.get("on")), "ids_on": r.get("ids"),
                         "usable": bool(r.get("usable")) if r else bool(i.get("valid")),
                         "zones": [{**z, "on": z["id"] in on_ids} for z in i.get("zones") or []],
                         "used_by": used})
        others: dict = {}
        for i in items:
            if i["geometry_id"] != geo:
                o = others.setdefault(i["geometry_id"], {"geometry_id": i["geometry_id"],
                                                         "n_sets": 0, "presets": []})
                o["n_sets"] += 1
                if i.get("preset") and i["preset"] not in o["presets"]:
                    o["presets"].append(i["preset"])
        return jsonable({"geometry_id": geo, "preset": p.name if p is not None else None,
                         "sets": mine, "selection": sel,
                         "others": [others[g] for g in sorted(others)],
                         "other_geometries": sorted({i["geometry_id"] for i in items} - {geo})},
                        self.root)

    def _paint_record(self, name, outcome: str, **more) -> None:
        """How a painter session ended, for every page (the bus snapshot's last_paint). It
        says where (preset, geometry_id) and - saved - which painted time the set got: a page
        tells what is true of that set NOW from the selection, never from this record."""
        p = self.session.preset if self.session else None
        self.last_paint = {"name": name, "outcome": outcome, "time": _now(), "reason": None,
                           "n_zones": None, "zones": [], "replaced": None, "on": None,
                           "selection_note": None, "photo": None, "preview": None,
                           "geometry_id": p.geometry_id if p is not None else None,
                           "preset": p.name if p is not None else None, "created": None,
                           "mock": self.mock, **more}

    def _v_zone_paint(self, a):
        """The painter job. Whatever its end - saved, cancelled (ESC), aborted, the idle
        timeout, refused before it opened - is kept in last_paint: the save step is told by
        the bus snapshot, not by the one browser tab that started the job."""
        name = a.get("name")
        try:
            out, links = self._zone_paint(a)
        except Exception as e:          # noqa: BLE001 - refused / crashed: nothing was saved
            self._paint_record(name, "refused", reason=f"{type(e).__name__}: {e}")
            raise
        j = jsonable(out, self.root)
        self._paint_record(
            out["name"], out["outcome"], reason=out.get("reason"), n_zones=out.get("n_zones"),
            zones=[{"id": z["id"], "name": z.get("name") or ""} for z in j.get("zones") or []]
            if out["saved"] else [], replaced=j.get("replaced"), on=out.get("on"),
            switches_reset=bool(out.get("switches_reset")),
            selection_note=out.get("selection_note"), photo=j.get("camera"),
            preview=j.get("preview"), note=out.get("note"),
            idle_timeout_s=out.get("idle_timeout_s"), rig_error=out.get("rig_error"),
            geometry_id=out.get("geometry_id"), preset=out.get("preset"),
            created=out.get("created") if out["saved"] else None,
            warnings=list(out.get("warnings") or []))
        return out, links

    def _zone_paint(self, a):
        from engine.zones import painter
        events, note = a.get("events"), None
        if self.mock:
            if events is None:          # MOCK: the simulated human draws the demo session
                events = list(painter.DEMO_EVENTS)
                note = "MOCK: no events given - the built-in demo session was replayed"
            elif not isinstance(events, list):
                raise ArgError("events must be a list of painter events")
        elif events is not None:
            raise ArgError("events is MOCK ONLY: on the real rig the operator draws on the part")
        labels = a.get("labels")
        if labels is not None and (not isinstance(labels, list)
                                   or not all(isinstance(x, str) for x in labels)):
            raise ArgError("labels must be a list of names")
        try:
            out = api.zone_paint(_s(a, "name", required=True), self._preset(),
                                 source=_s(a, "source"), events=events,
                                 replace=_bool(a, "replace"), labels=labels,
                                 polarity=_s(a, "polarity", "measure"),
                                 note=_s(a, "note", "") or "",
                                 idle_timeout_s=_num(a, "idle_timeout_s", 600.0),
                                 cancel=self.session.cancel, select=_bool(a, "select"))
        except ValueError as e:         # a bad name / an existing set / a malformed event
            raise ArgError(str(e)) from None
        if note:
            out = {**out, "note": note}
        return out, ({"preview": out.get("camera") or out.get("preview")} if out["saved"] else {})

    def _v_zone_show(self, a):
        h = _num(a, "hold_s", 5.0)
        if not 0.0 <= h <= MAX_HOLD_S:
            raise ArgError(f"hold_s must be 0..{MAX_HOLD_S:g} s, got {h:g}")
        zones = a.get("zones")
        if isinstance(zones, list) and zones and all(isinstance(z, str) for z in zones):
            pass                        # a list of terms
        else:
            zones = _s(a, "zones", required=True)
        try:
            out = api.zone_show(zones, self._preset(), h, _s(a, "label"),
                                cancel=self.session.cancel)
        except ValueError as e:
            raise ArgError(str(e)) from None
        return out, {"preview": out.get("overlay_png") or out.get("camera_png")}

    def _v_zone_delete(self, a):
        if not _confirmed(a, "confirm"):
            raise ArgError("zone_delete needs confirm=true (the operator confirmed the delete)")
        try:
            return api.zone_delete(_s(a, "name", required=True), _s(a, "reason", required=True),
                                   self._preset(), force=_bool(a, "force"),
                                   stations_dir=self.stations_dir)
        except ValueError as e:
            raise ArgError(str(e)) from None

    def _v_zone_select(self, a):
        """Change the zone selection of the loaded preset's rig placement (files only)."""
        sets = a.get("sets")
        if sets is not None and not isinstance(sets, list):
            raise ArgError("sets must be a list of {set, on, ids}")
        terms = {}
        for k in ("zones", "on", "off"):
            v = a.get(k)
            if v is not None and not (isinstance(v, str) or (
                    isinstance(v, list) and all(isinstance(x, str) for x in v))):
                raise ArgError(f"{k} must be text or a list of texts")
            terms[k] = v
        expect = a.get("expect")
        if expect is not None and not isinstance(expect, str):
            raise ArgError("expect must be the stamp of the selection that was shown (text)")
        from engine.zones.selection import SelectionChanged
        try:
            return api.zone_select(zones=terms["zones"], on=terms["on"], off=terms["off"],
                                   clear=_bool(a, "clear"), sets=sets, preset=self._preset(),
                                   by=_s(a, "by") or "", expect=expect)
        except SelectionChanged:        # the sender was shown another selection: its own code
            raise
        except (ValueError, rules.GeometryMismatch) as e:
            raise ArgError(str(e)) from None

    def _v_spectral_capture(self, a):
        return api.spectral_capture(self._preset(), _s(a, "label", "spectral"))

    def _v_learn_material(self, a):
        roi = _num_list(a, "roi", float)
        if not roi or len(roi) != 4:
            raise ArgError("roi must be [x0, y0, x1, y1] in camera px (draw a rectangle)")
        x0, y0, x1, y1 = (int(round(v)) for v in roi)
        roi_t = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        reading = self._entry("spectral", _s(a, "reading_id", required=True))
        return api.learn_material(reading, _s(a, "name", required=True), roi_t)

    def _v_classify(self, a):
        reading = self._entry("spectral", _s(a, "reading_id", required=True))
        names = a.get("materials") or None      # names: the engine resolves its own library
        if names is not None and (not isinstance(names, list)
                                  or not all(isinstance(n, str) for n in names)):
            raise ArgError("materials must be a list of material names")
        return api.classify(reading, names, _num(a, "max_distance", 3.0))

    def _v_separate_light(self, a):
        return api.separate_light(self._preset(), _s(a, "label", "light"),
                                  _num(a, "square_px", 16, int), _num(a, "shifts", 8, int))

    # ------------------------------------------------------ virtual sensors
    def _v_spot_size(self, a):
        return api.spot_size(_num(a, "diameter_mm", required=True), _num(a, "mm_per_dmd_px"),
                             self._preset())

    def _v_project_spot(self, a):
        return api.project_spot(_at(a), _num(a, "diameter_mm", required=True), self._preset(),
                                **_kw(mm_per_dmd_px=_num(a, "mm_per_dmd_px"),
                                      shape=_s(a, "shape"), hold_s=_hold_s(a),
                                      label=_s(a, "label"), level=_num(a, "level", cast=int)))

    def _v_teach_presence(self, a):
        # empty_confirmed is the operator's answer to "is the field EMPTY?" (the page
        # asks before it sends it); the engine refuses to teach without it
        return api.teach_presence(
            _s(a, "name", required=True), _at(a), _num(a, "diameter_mm"),
            _s(a, "method", "photoelectric"), self._preset(),
            empty_confirmed=_confirmed(a, "empty_confirmed"),
            **_kw(mm_per_dmd_px=_num(a, "mm_per_dmd_px"), threshold=_num(a, "threshold"),
                  min_height_mm=_num(a, "min_height_mm"), level=_num(a, "level", cast=int)))

    def _v_check_presence(self, a):
        return api.check_presence(
            _s(a, "name", required=True), self._preset(),
            **_kw(threshold=_num(a, "threshold"), min_height_mm=_num(a, "min_height_mm"),
                  min_relief_px=_num(a, "min_relief_px"), label=_s(a, "label")))

    def _v_teach_alignment(self, a):
        return api.teach_alignment(_s(a, "name", required=True), self._object(a),
                                   self._preset())

    def _v_check_alignment(self, a):
        return api.check_alignment(
            _s(a, "name", required=True), self._object(a),
            **_kw(tol_mm=_num(a, "tol_mm"), tol_deg=_num(a, "tol_deg"),
                  tol_px=_num(a, "tol_px")), preset=self._preset())

    def _v_list_library(self, a):
        return api.list_library(_s(a, "kind", "bases"), _s(a, "geometry_id"))

    def _v_latest_base(self, a):
        return api.latest_base(self._preset())

    # ------------------------------------------- 3D meshes, cell, maps, checker
    def _n_outputs(self) -> int:
        return len(self.session.store.outputs())

    def _new_outputs(self, n0: int, exts=(".png",)) -> dict:
        """Output-bin files written since n0 -> {name without timestamp: path}."""
        store = self.session.store
        out = {}
        for row in store.outputs()[n0:]:
            f = store.root / row.get("file", "")
            if f.suffix.lower() in exts and f.is_file():
                out[re.sub(r"^\d{8}T\d{6}_\d+_", "", f.name)] = f
        return out

    def _light(self, light_id):
        for d in self.session.store.entries("spectral", "light"):
            if d.name == light_id and (d / "separation.npz").is_file():
                return d
        raise EntryNotFound(f"no light separation with id {light_id!r} in the library")

    def _lights(self, geometry_id=None) -> list:
        rows = []
        for d in self.session.store.entries("spectral", "light"):
            try:
                m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if geometry_id and m.get("geometry_id") != geometry_id:
                continue
            rows.append({"id": d.name, "label": m.get("label"), "created": m.get("created"),
                         "geometry_id": m.get("geometry_id"),
                         "global_fraction_median": m.get("global_fraction_median"), "path": d})
        return rows

    def _atlases(self, geometry_id=None) -> list:
        from engine.capture import atlas as atlas_mod
        rows = api.list_atlases(geometry_id)
        out = []
        for r in rows:
            r = dict(r)
            try:
                at = atlas_mod.load_atlas(r["latest"], r["name"])
                r["step_px"] = at.step_px
                r["regions"] = [{"name": g.name, "rect": list(g.rect), "label": g.label,
                                 "zone": g.zone_spec, "spec": f"atlas:{r['name']}/{g.name}"}
                                for g in at.regions.values()]
                r["warnings"] = list(at.warnings)
            except Exception as e:          # noqa: BLE001 - one bad file must not hide the rest
                r["regions"], r["error"] = [], f"{type(e).__name__}: {e}"
            out.append(r)
        return out

    # ------------------------------------------- metric calibration + measurement
    def _metric(self, a: dict, key: str = "metric_id"):
        return self._entry("metric", _s(a, key, required=True))

    def _v_make_charuco(self, a):
        return api.make_charuco(_num(a, "squares_x", 10, int), _num(a, "squares_y", 7, int),
                                _num(a, "square_mm", 25.0), _num(a, "marker_mm", 18.0),
                                _s(a, "dictionary", "DICT_5X5_100"), _num(a, "dpi", 600.0),
                                _s(a, "label", "charuco"))

    def _v_row_parity(self, a):
        return api.row_parity(self._preset(), _s(a, "label", "parity"))

    @staticmethod
    def _parity_arg(a):
        v = a.get("diamond_parity", "auto")
        if v in ("auto", None, 0, 1):
            return v
        if isinstance(v, str) and v.strip().lower() in ("null", "none", ""):
            return None
        if isinstance(v, str) and v.strip() in ("0", "1"):
            return int(v)
        raise ArgError("diamond_parity must be auto, 0, 1 or null")

    def _v_capture_calib_pose(self, a):
        return api.capture_calib_pose(self._preset(), _s(a, "set_label", "stereo"),
                                      steps=_num(a, "steps", None, int),
                                      periods=_num_list(a, "periods"),
                                      diamond_parity=self._parity_arg(a),
                                      fixture=_s(a, "fixture"))

    def _v_calibrate_stereo(self, a):
        n = _num(a, "mock_poses", 0, int)
        nf = _num(a, "mock_fixture", 0, int)
        if (n or nf) and not self.mock:
            raise ArgError("mock_poses / mock_fixture are mock-only")
        return api.calibrate_stereo(self._preset(), _s(a, "set_label"), _num(a, "square_mm"),
                                    projector_distortion=_bool(a, "projector_distortion"),
                                    mock_poses=n, steps=_num(a, "steps", None, int),
                                    periods=_num_list(a, "periods"),
                                    diamond_parity=self._parity_arg(a),
                                    fixture=_s(a, "fixture"), mock_fixture=nf,
                                    rigid=_bool(a, "rigid", True))

    def _v_accuracy_block(self, a):
        ids = a.get("metric_ids") or None
        if ids is not None and not isinstance(ids, list):
            raise ArgError("metric_ids must be a list of metric scan ids")
        scans = [self._entry("metric", i) for i in ids] if ids else None
        return api.accuracy_block(_s(a, "block", required=True), _s(a, "measured"), scans,
                                  _num(a, "repeats", 1, int), self._preset(),
                                  _num(a, "tol_height_mm"), _num(a, "tol_diameter_mm"),
                                  _num(a, "flatness_max_mm"), _s(a, "label", "block"))

    def _v_scan_metric(self, a):
        method = _s(a, "method", "auto")
        if method not in ("auto", "plane", "rays"):
            raise ArgError("method must be auto, plane or rays")
        ms = api.scan_metric(self._preset(), _s(a, "label", "metric"), method,
                             _num(a, "steps", None, int), _num_list(a, "periods"),
                             _num(a, "max_consistency_mm"))
        return ms, ({"preview": Path(ms.path) / "preview.png"}
                    if (Path(ms.path) / "preview.png").is_file() else {})

    def _v_triangulate(self, a):
        scan = self._metric(a) if a.get("metric_id") else self._object(a)
        method = _s(a, "method", "auto")
        if method not in ("auto", "plane", "rays"):
            raise ArgError("method must be auto, plane or rays")
        return api.triangulate(scan, method, _s(a, "label"))

    def _v_cylinder_deviation(self, a):
        roi = _num_list(a, "roi", int)
        if roi is not None and len(roi) != 4:
            raise ArgError("roi must be [x0, y0, x1, y1] camera px")
        dev = api.cylinder_deviation(self._metric(a), _num(a, "radius_mm"),
                                     _num(a, "n_sigma", 4.0), _num(a, "min_depth_mm", 0.05),
                                     tuple(roi) if roi else None)
        links = {k: v for k, v in (dev.files or {}).items() if v is not None and k != "npz"}
        return dev, links

    def _v_compare_metric(self, a):
        return api.compare_metric(self._metric(a), self._metric(a, "golden_id"),
                                  _num(a, "max_dist_mm", 5.0), _num(a, "n_sigma", 4.0))

    def _v_accuracy_check(self, a):
        kind = _s(a, "kind", required=True)
        known = {}
        for k in ("height_mm", "diameter_mm"):
            if _num(a, k) is not None:
                known[k] = _num(a, k)
        ids = a.get("metric_ids") or None
        if ids is not None and not isinstance(ids, list):
            raise ArgError("metric_ids must be a list of metric scan ids")
        scans = [self._entry("metric", i) for i in ids] if ids else None
        return api.accuracy_check(kind, known, scans, _num(a, "repeats", 3, int),
                                  self._preset(), _num(a, "tol_mm", 0.1),
                                  _num(a, "repeat_tol_mm", 0.03), _s(a, "label", "accuracy"))

    def _v_predict_precision(self, a):
        return api.predict_precision(
            self._preset(), _num(a, "z_mm"), _num(a, "A_dn"), _num(a, "B_dn"),
            _num(a, "period_px"), _num(a, "steps", None, int),
            _num(a, "frames_averaged", None, int), _num(a, "gain_db"),
            self._metric(a) if a.get("metric_id") else None, _num(a, "extra_noise_dn", 0.0))

    def _v_export_metric_mesh(self, a):
        fmt = _s(a, "fmt", "stl")
        if fmt not in ("stl", "obj"):
            raise ArgError("fmt must be 'stl' or 'obj'")
        stride = _num(a, "stride", 1, int)
        if stride < 1 or stride > 64:
            raise ArgError("stride must be 1..64 camera px")
        p = Path(api.export_metric_mesh(self._metric(a), fmt, stride, _num(a, "max_edge_mm"),
                                        _bool(a, "foreground_only")))
        mp = p.with_name(p.name + ".json")
        return {"file": p, "meta_file": mp}, {"file": p, "meta": mp}

    # ------------------------------------------------ dent detection
    def _named_preset(self, a: dict, key: str):
        """A second preset by name for one job (validated; the active preset is unchanged)."""
        name = _s(a, key)
        if not name:
            return None
        if not _NAME_RE.match(name):
            raise ArgError(f"'{key}' must be a preset name")
        return config.load_preset(name, presets_dir=self.session.presets_dir,
                                  store=self.session.store, mock=self.mock)

    def _dentset_arg(self, a: dict, key: str = "dentset_id"):
        """A library dent set by id. Like every other handler, never a server-side path
        (legacy offline folders: CLI / engine API only)."""
        return self._entry("dentsets", _s(a, key, required=True))

    def _dent_null(self, a: dict):
        """A dent null calibration by its library id (or its folder name) -> its folder."""
        v = _s(a, "null_id")
        if v is None:
            return None
        for c in self._list("calibrations"):
            if c.get("kind") == "dent_null" and (c.get("id") == v or Path(c["path"]).name == v):
                return str(c["path"])
        raise EntryNotFound(f"no dent null calibration with id {v!r} in the library")

    @staticmethod
    def _pattern_set(a: dict) -> str:
        v = _s(a, "pattern_set", "production")
        if v not in ("production", "minimal"):
            raise ArgError("pattern_set must be production or minimal")
        return v

    def _dent_subset(self, a: dict) -> str:
        sub = _s(a, "subset", "all")
        if sub not in DENT_SUBSETS:
            raise ArgError(f"subset must be one of {DENT_SUBSETS}")
        return sub

    def _metric_list(self, a: dict, key: str = "metric_ids"):
        ids = a.get(key) or None
        if ids is None:
            return None
        if not isinstance(ids, list):
            raise ArgError(f"{key} must be a list of metric scan ids")
        return [self._entry("metric", str(i)) for i in ids]

    def _v_capture_dent_set(self, a):
        ds = api.capture_dent_set(_s(a, "zone"), _s(a, "label", "dentset"), self._preset(),
                                  self._named_preset(a, "short_preset"),
                                  _num(a, "short_exposure_us", None, int),
                                  _bool(a, "lines_v", True),
                                  _bool(a, "metric", True), self._named_preset(a, "metric_preset"),
                                  _num(a, "steps", None, int), _num_list(a, "periods"),
                                  self._pattern_set(a))
        return ds, ({"preview": Path(ds.path) / "preview.png"}
                    if (Path(ds.path) / "preview.png").is_file() else {})

    def _v_calibrate_dent_null(self, a):
        ids = a.get("dentset_ids")
        if isinstance(ids, str):
            ids = [ids]
        if not ids or not isinstance(ids, list):
            raise ArgError("dentset_ids must list the dent sets of KNOWN-GOOD parts")
        sets = [self._dentset_arg({"dentset_id": i}) for i in ids]
        return api.calibrate_dent_null(sets, _num(a, "cell_diameter_mm", required=True),
                                       _s(a, "label", "null"), self._metric_list(a),
                                       _s(a, "fixture"), self._dent_subset(a))

    def _v_detect_dents(self, a):
        rep = api.detect_dents(self._dentset_arg(a), self._metric_list(a),
                               _num(a, "cell_diameter_mm"), self._dent_null(a), _s(a, "label"),
                               self._dent_subset(a), _s(a, "fixture"))
        links = {k: v for k, v in (rep.files or {}).items() if k in ("png", "json")}
        return rep, links

    # ------------------------------------------------ end faces
    def _face_arg(self, a: dict, key: str = "face_id"):
        """A face target by library id (None = the engine's newest at the preset's geometry)."""
        v = _s(a, key)
        return None if v is None else self._entry("faces", v)

    def _facescan_arg(self, a: dict, key: str = "facescan_id"):
        return self._entry("facescans", _s(a, key, required=True))

    @staticmethod
    def _face_detector(a: dict):
        d = _s(a, "detector")
        if d is not None and d not in ("mf_fir", "mf", "band"):
            raise ArgError("detector must be mf_fir, mf or band")
        return d

    @staticmethod
    def _offset(a: dict):
        off = _num_list(a, "offset_dmd")
        if off is not None and len(off) != 2:
            raise ArgError("offset_dmd must be [cols, rows] DMD px")
        return off

    @staticmethod
    def _locate_args(a: dict) -> tuple:
        """(roi, LocateParams overrides) of a locate verb."""
        roi = _num_list(a, "roi", int)
        if roi is not None and len(roi) != 4:
            raise ArgError("roi must be [x0, y0, x1, y1] camera px")
        params = _kw(near_band_mm=_num(a, "near_band_mm"), far_gap_mm=_num(a, "far_gap_mm"),
                     near_gap_mm=_num(a, "near_gap_mm"))
        return roi, params or None

    @staticmethod
    def _exposure_range(a: dict):
        rng = _num_list(a, "exposure_range_us")
        if rng is not None and len(rng) != 2:
            raise ArgError("exposure_range_us must be [lo, hi] us")
        return rng

    @staticmethod
    def _face_result(t):
        """A face target as a job result: its preview, and the disposition a size mismatch
        gave it at the top (a PULL is never only inside meta)."""
        links = ({"preview": Path(t.path) / "preview.png"}
                 if (Path(t.path) / "preview.png").is_file() else {})
        return t, links

    def _v_locate_face(self, a):
        roi, params = self._locate_args(a)
        t = api.locate_face(self._metric(a), _num(a, "face_diameter_mm", 21.05),
                            _s(a, "label", "face"), roi, params, self._flood_metric(a),
                            size_tol_pct=_num(a, "size_tol_pct"),
                            silhouette_diameter_mm=_num(a, "silhouette_diameter_mm"))
        return self._face_result(t)

    def _v_locate_scan(self, a):
        ms = api.locate_scan(_s(a, "zone"), self._preset(), _s(a, "label", "face_locate"),
                             _num(a, "steps", None, int), _num_list(a, "periods"),
                             _num(a, "exposure_us"))
        return ms, ({"preview": Path(ms.path) / "preview.png"}
                    if (Path(ms.path) / "preview.png").is_file() else {})

    def _v_locate_face_live(self, a):
        roi, params = self._locate_args(a)
        ae = a.get("auto_exposure")
        t = api.locate_face_live(_s(a, "zone"), self._preset(),
                                 _num(a, "face_diameter_mm", 21.05), _s(a, "label", "face"),
                                 _num(a, "steps", None, int), _num_list(a, "periods"), roi,
                                 params, size_tol_pct=_num(a, "size_tol_pct"),
                                 auto_exposure=None if ae is None else _bool(a, "auto_exposure"),
                                 exposure_us=_num(a, "exposure_us"),
                                 exposure_range_us=self._exposure_range(a),
                                 silhouette_diameter_mm=_num(a, "silhouette_diameter_mm"))
        return self._face_result(t)

    def _v_face_exposure(self, a):
        return api.face_exposure(self._face_arg(a), self._preset(), _num(a, "target_dn"),
                                 self._exposure_range(a), _num(a, "max_clipped_pct"),
                                 label=_s(a, "label", "auto_exposure"), duty=_num(a, "duty"))

    def _v_aim_ring(self, a):
        r = api.aim_ring(self._face_arg(a), self._preset(), _hold_s(a), _s(a, "label", "aim"),
                         self._offset(a))
        return r, {"photo": r.photo, "overlay": r.overlay}

    def _v_check_ring(self, a):
        r = api.check_ring(self._face_arg(a), self._preset(), _hold_s(a),
                           _s(a, "label", "ring_check"), _num(a, "tol_px", 1.5), self._offset(a))
        return r, {"photo": r.photo, "overlay": r.overlay}

    def _v_auto_intensity(self, a):
        return api.auto_intensity(self._face_arg(a), self._preset(), _num(a, "target_dn", 200.0),
                                  _num(a, "tol_dn", 10.0), _num(a, "max_shots", 4, int),
                                  _s(a, "label", "auto_intensity"))

    def _v_targeted_face_scan(self, a):
        fs = api.targeted_face_scan(self._face_arg(a), self._preset(), _s(a, "label", "face"),
                                    _num(a, "repeats", 3, int), _num(a, "margin_dmd_px", 6.0),
                                    _bool(a, "polar", True), _num(a, "duty"),
                                    _num(a, "steps", None, int), _num_list(a, "periods"),
                                    _num(a, "polar_steps", 8, int),
                                    auto_exposure=(None if a.get("auto_exposure") is None
                                                   else _bool(a, "auto_exposure")),
                                    exposure_us=_num(a, "exposure_us"),
                                    exposure_target_dn=_num(a, "exposure_target_dn"),
                                    exposure_range_us=self._exposure_range(a),
                                    max_clipped_pct=_num(a, "max_clipped_pct"))
        return fs, ({"preview": Path(fs.path) / "preview.png"}
                    if (Path(fs.path) / "preview.png").is_file() else {})

    def _v_face_profile(self, a):
        if a.get("facescan_id"):
            scans = self._facescan_arg(a)
        else:
            scans = self._metric_list(a)
            if not scans:
                raise ArgError("face_profile needs facescan_id or metric_ids")
        pr = api.face_profile(scans, self._face_arg(a), _s(a, "label"), _num(a, "grid_mm", 0.25),
                              _num(a, "r_flat_mm"), _bool(a, "stl", True),
                              size_tol_pct=_num(a, "size_tol_pct"))
        return pr, {k: v for k, v in pr.files.items() if k in ("png", "json", "csv_radial", "stl")}

    def _v_face_golden(self, a):
        ids = a.get("facescan_ids")
        if isinstance(ids, str):
            ids = [ids]
        if not ids or not isinstance(ids, list):
            raise ArgError("facescan_ids must list the face scans of GOOD placements")
        places = [self._facescan_arg({"facescan_id": i}) for i in ids]
        g = api.face_golden(places, _s(a, "label", "golden"),
                            **_kw(k_sigma=_num(a, "k_sigma"), floor_mm=_num(a, "floor_mm"),
                                  smooth_cells=_num(a, "smooth_cells", None, int),
                                  rim_r_mm=_num(a, "rim_r_mm"), rim_factor=_num(a, "rim_factor"),
                                  rim_min_mm=_num(a, "rim_min_mm"),
                                  height_smooth_mm=_num(a, "height_smooth_mm"),
                                  min_area_mm2=_num(a, "min_area_mm2"),
                                  detector=self._face_detector(a), z_thresh=_num(a, "z_thresh"),
                                  min_depth_mm=_num(a, "min_depth_mm"),
                                  radii=_kw(inspect_r_mm=_num(a, "inspect_r_mm"),
                                            plane_r_mm=_num(a, "plane_r_mm")) or None,
                                  rim=_kw(tol_mm=_num(a, "rim_tol_mm"),
                                          min_arc_mm=_num(a, "rim_min_arc_mm")) or None,
                                  size_tol_pct=_num(a, "size_tol_pct")),
                            synthetic=_bool(a, "synthetic", True), fir=_bool(a, "fir", True))
        return g, {"preview": g.files.get("png")}

    def _v_check_face(self, a):
        gid = _s(a, "golden_id")
        golden = None if gid is None else self._entry("face_golden", gid)
        r = api.check_face(self._facescan_arg(a), golden, _s(a, "label"), _num(a, "min_area_mm2"),
                           self._face_detector(a), _num(a, "z_thresh"), _num(a, "min_depth_mm"),
                           rim_tol_mm=_num(a, "rim_tol_mm"),
                           rim_min_arc_mm=_num(a, "rim_min_arc_mm"),
                           size_tol_pct=_num(a, "size_tol_pct"))
        return r, {k: v for k, v in r.files.items() if k in ("png", "json")}

    # ------------------------------------------- HDMI fast zone scan + station
    def _v_scan_fast(self, a):
        axis = _s(a, "axis", "x")
        if axis not in ("x", "xy"):
            raise ArgError("axis must be x or xy")
        ids = a.get("face_ids") or []
        if isinstance(ids, str):
            ids = [ids]
        if not isinstance(ids, list):
            raise ArgError("face_ids must be a list of face target ids")
        faces = [self._entry("faces", str(i)) for i in ids] or None
        zones = _s(a, "zones")
        ms = api.scan_fast(_s(a, "zone"), self._preset(), _s(a, "label", "fast"),
                           _num(a, "steps", 4, int), _num_list(a, "periods"),
                           _num(a, "average", 2, int), _num(a, "min_hold_s", 0.12), axis, faces,
                           _num(a, "margin_dmd_px", 6.0), **({"zones": zones} if zones else {}))
        return ms, ({"preview": Path(ms.path) / "preview.png"}
                    if (Path(ms.path) / "preview.png").is_file() else {})

    @staticmethod
    def _station_prm(a) -> dict:
        prm = {}
        band = _num_list(a, "ct_band_mm")
        if band is not None:
            if len(band) != 2:
                raise ArgError("ct_band_mm must be [lo, hi] mm")
            prm["ct_band_mm"] = band
        skip = _num_list(a, "zone_skip", int)
        if skip is not None:
            if len(skip) != 2:
                raise ArgError("zone_skip must be [left, right]")
            prm["zone_skip"] = skip
        cm = _s(a, "centre_method")
        if cm is not None:
            if cm not in ("rim_circle", "ellipse_ray"):
                raise ArgError("centre_method must be rim_circle or ellipse_ray")
            prm["centre_method"] = cm
        prm["n_cells"] = _num(a, "n_cells", 2, int)
        prm["face_diameter_mm"] = _num(a, "face_diameter_mm", 21.05)
        return prm

    def _flood_metric(self, a):
        v = _s(a, "flood_metric_id")
        return None if v is None else self._entry("metric", v)

    def _v_station_measure(self, a):
        prm = self._station_prm(a)
        m = api.station_measure(self._metric(a), self._flood_metric(a), _s(a, "label"),
                                prm.pop("n_cells"), prm.pop("face_diameter_mm"), prm or None)
        return m, {k: v for k, v in m.files.items() if k in ("png", "json", "csv") and v}

    def _v_station_golden(self, a):
        scans = self._metric_list(a)
        if not scans:
            raise ArgError("metric_ids must list the metric scans of GOOD placements")
        prm = self._station_prm(a)
        bands = a.get("bands") or None
        if bands is not None and not isinstance(bands, dict):
            raise ArgError("bands must be {item: [lo, hi]}")

        def band(k):
            v = _num_list(a, k)
            if v is not None and len(v) != 2:
                raise ArgError(f"{k} must be [lo, hi] mm")
            return v
        g = api.station_golden(scans, _s(a, "label", "golden"),
                               **_kw(cell_height_tol_mm=_num(a, "cell_height_tol_mm"),
                                     ct_zone_tol_mm=_num(a, "ct_zone_tol_mm"),
                                     spacing_tol_mm=_num(a, "spacing_tol_mm"),
                                     face_position_tol_mm=_num(a, "face_position_tol_mm"),
                                     cell_height_band_mm=band("cell_height_band_mm"),
                                     ct_zone_band_mm=band("ct_zone_band_mm"), bands=bands),
                               params=prm)
        return g, ({"preview": g.files.get("png")} if g.files.get("png")
                   and Path(g.files["png"]).is_file() else {})

    def _v_station_check(self, a):
        gid = _s(a, "golden_id")
        golden = None if gid is None else self._entry("station_golden", gid)
        r = api.station_check(self._metric(a), golden, self._flood_metric(a), _s(a, "label"))
        return r, {k: v for k, v in r.files.items() if k in ("png", "json", "csv") and v}

    def _v_export_mesh(self, a):
        fmt, kind = _s(a, "fmt", "stl"), _s(a, "kind", "relief")
        if fmt not in ("stl", "obj"):
            raise ArgError("fmt must be 'stl' or 'obj'")
        if kind not in ("relief", "base", "cylinder"):
            raise ArgError("kind must be 'relief', 'base' or 'cylinder'")
        stride = _num(a, "stride", 2, int)
        if stride < 1 or stride > 64:
            raise ArgError("stride must be 1..64 camera px")
        roi = _num_list(a, "roi", int)
        if roi is not None and len(roi) != 4:
            raise ArgError("roi must be [x0, y0, x1, y1] camera px")
        scan = self._object(a)
        p = Path(api.export_mesh(scan, fmt, stride, kind, _num(a, "radius_mm"),
                                 tuple(roi) if roi else None))
        mp = p.with_name(p.name + ".json")
        meta = None
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            mp = None
        return ({"scan_id": scan.id, "kind": kind, "fmt": fmt, "stride": stride, "file": p,
                 "meta_file": mp, "meta": meta},
                {"file": p, "meta": mp})

    def _v_cell_unwrap(self, a):
        scan = self._object(a)
        n0 = self._n_outputs()
        u = api.cell_unwrap(scan, _num(a, "radius_mm"), _num(a, "step"))
        th = [float(t) for t in np.asarray(u.theta_deg).ravel() if np.isfinite(t)]
        res = {"scan_id": scan.id, "label": u.label, "arc_deg": u.arc_deg,
               "cell_height": u.cell_height, "radius": u.radius, "step": u.step,
               "units": u.units, "calibration_id": u.calibration_id, "zone": u.zone,
               "n_theta": int(np.asarray(u.theta_deg).size), "n_z": int(np.asarray(u.z).size),
               "theta_range_deg": [min(th), max(th)] if th else None,
               "fit": u.fit, "stats": u.stats, "warnings": u.warnings}
        links = self._new_outputs(n0, (".png", ".npz"))
        return res, {("png" if k.endswith(".png") else "npz"): v for k, v in links.items()}

    def _v_cell_dents(self, a):
        scan = self._object(a)
        n0 = self._n_outputs()
        rep = api.cell_dents(scan, _num(a, "radius_mm"), _num(a, "n_sigma", 5.0),
                             _num(a, "min_area_bins", 6, int), _num(a, "depth_trigger_mm"))
        links = self._new_outputs(n0, (".png", ".json"))
        res = {"scan_id": scan.id, **{f.name: getattr(rep, f.name)
                                      for f in dataclasses.fields(rep)}}
        return res, {("png" if k.endswith(".png") else "json"): v for k, v in links.items()}

    def _v_bake_maps(self, a):
        scan = self._object(a)
        spec_items = None

        def spectral(key):
            nonlocal spec_items
            i = _s(a, key)
            if not i:
                return None
            spec_items = self._list("spectral") if spec_items is None else spec_items
            return self._entry("spectral", i, spec_items)

        sp, gold = spectral("spectral_id"), spectral("golden_id")
        light = self._light(_s(a, "light_id")) if _s(a, "light_id") else None
        ms = api.bake_maps(scan, spectral=sp, light=light, golden=gold,
                           sigma_px=_num(a, "sigma_px", 2.0), label=_s(a, "label"))
        return ms, dict(ms.png_paths or {})

    def _v_checker_scan(self, a):
        sq, sh = _num(a, "square_px", 32, int), _num(a, "shifts", 1, int)
        if not 4 <= sq <= 256 or not 1 <= sh <= 16:
            raise ArgError("square_px must be 4..256 and shifts 1..16")
        c = api.checker_scan(self._preset(), sq, sh, _s(a, "label", "checker"))
        jac = c.jacobian

        def med(name):
            v = np.asarray(getattr(jac, name, []), float)
            v = v[np.isfinite(v)]
            return float(np.median(v)) if v.size else None

        res = {"label": c.label, "geometry_id": c.geometry_id, "n_corners": len(c.corners),
               "n_cells": len(jac), "square_px": c.corners.square_px,
               "cam_square_px": c.corners.cam_square_px, "stretch_ref": jac.stretch_ref,
               "median": {k: med(k) for k in jac.VALUES},
               "png_paths": c.png_paths, "json_path": c.json_path, "npz_path": c.npz_path,
               "units": c.units, "warnings": list(c.warnings) + list(c.corners.warnings)}
        return res, dict(c.png_paths or {})

    # ------------------------------------------------- Surfer atlas + methods
    def _v_import_atlas(self, a):
        raw = _s(a, "path", required=True)
        src = Path(raw).expanduser()
        if src.suffix.lower() != ".json":
            raise ArgError("path must be an atlas .json file")
        if not src.is_file():
            raise ArgError(f"no file {raw}")
        name = _s(a, "name", required=True)
        if not _NAME_RE.match(name):
            raise ArgError("name must be letters, digits, _ . - only")
        out = api.import_atlas(str(src), name, self._preset())
        rows = [r for r in self._atlases(self._preset().geometry_id) if r["name"] == name]
        return {"file": out, "name": name, "source": str(src),
                "atlas": rows[-1] if rows else None}, {"json": out}

    def _v_list_atlases(self, a):
        return self._atlases(_s(a, "geometry_id"))

    @staticmethod
    def _region(a):
        r = a.get("region")
        if isinstance(r, list):
            if len(r) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                      for v in r):
                raise ArgError("a rect region is [x0, y0, x1, y1] DMD px")
            return tuple(int(v) for v in r)
        r = _s(a, "region", required=True).strip()
        if r.lower().startswith("atlas:"):
            return "atlas:" + r[6:].strip()
        return r.upper()

    def _v_flood_verify(self, a):
        region = self._region(a)
        r = api.flood_verify(region, self._preset(), _num(a, "level", 255, int),
                             _s(a, "label", "flood_verify"))
        return r, {"camera_png": r.camera_png, "diff_png": r.diff_png, "dmd_png": r.dmd_png}

    def _v_anti_glare_sweep(self, a):
        region = self._region(a)
        carriers = a.get("carriers")
        if carriers in (None, "", []):
            carriers = None
        elif not isinstance(carriers, list) or not all(
                isinstance(c, list) and len(c) in (2, 3)
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in c)
                for c in carriers):
            raise ArgError("carriers must be a list of [fx, fy] or [fx, fy, z]")
        n0 = self._n_outputs()
        r = api.anti_glare_sweep(region, carriers, self._preset(), _num(a, "steps", 4, int),
                                 _num(a, "cycles", 6.0), _num(a, "level", 255, int),
                                 _s(a, "label", "anti_glare"))
        return r, self._new_outputs(n0)

    def _v_multi_orientation_scan(self, a):
        region = self._region(a)
        ors = a.get("orientations") or list(ORIENTATIONS)
        if not isinstance(ors, list) or not ors or not set(ors) <= set(ORIENTATIONS):
            raise ArgError(f"orientations must be a non-empty subset of {list(ORIENTATIONS)}")
        label = _s(a, "label", "multi_orient")
        n0 = self._n_outputs()
        r = api.multi_orientation_scan(region, self._preset(), tuple(ors),
                                       _num(a, "steps", 4, int), _num(a, "period_px", 24.0),
                                       _num(a, "level", 255, int), label)
        links = self._new_outputs(n0)
        q = self._quality_png(r.quality, label)
        if q is not None:
            links["quality.png"] = q
        return r, links

    def _quality_png(self, quality, label):
        """The engine saves only the fused modulation PNG: the orientation
        agreement (0..1) is drawn here, into the output bin (never over data)."""
        try:
            import cv2
            qa = np.asarray(quality, float)
            if qa.ndim != 2:
                return None
            u8 = (np.clip(np.nan_to_num(qa, nan=0.0), 0, 1) * 255).astype(np.uint8)
            img = cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)
            img[~np.isfinite(qa)] = 0
            store = self.session.store
            out = store.output_path("previews", f"{label}_quality", ".png")
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                return None
            with open(out, "xb") as f:
                f.write(buf.tobytes())
            store.record_output("previews", out, label=label,
                                geometry_id=self._preset().geometry_id)
            return out
        except Exception:                   # noqa: BLE001 - a preview is optional
            return None

    def _v_make_flash_pack(self, a):
        axes = a.get("axes") or ["x"]
        if isinstance(axes, str):
            axes = [axes]
        if not isinstance(axes, list) or len(axes) != 1 or axes[0] not in ("x", "y"):
            raise ArgError("axes must be 'x' or 'y' (one axis per pack)")
        label = _s(a, "label")
        if label is not None and not _NAME_RE.match(label):
            raise ArgError("label must be letters, digits, _ . - only")
        p = Path(api.make_flash_pack(self._preset(), tuple(axes), _bool(a, "video_port"),
                                     label))
        manifest = None
        try:
            m = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
            manifest = {k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))}
        except (OSError, ValueError):
            pass
        doc = Path(api.__file__).resolve().parents[1] / FLASH_DOC
        return {"folder": str(p), "manifest": manifest,
                "files": sorted(f.name for f in p.iterdir())[:200] if p.is_dir() else [],
                "doc": str(doc), "doc_summary": FLASH_SUMMARY}

    @staticmethod
    def _local_files(a, k: str, suffix: str) -> list:
        v = a.get(k) or []
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
            raise ArgError(f"{k} must be a list of local {suffix} paths")
        out = []
        for raw in v:
            f = Path(raw).expanduser()
            if f.suffix.lower() != suffix:
                raise ArgError(f"{k}: {raw} is not a {suffix} file")
            if not f.is_file():
                raise ArgError(f"{k}: no file {raw}")
            out.append(str(f))
        return out

    def _v_make_fastscan_pack(self, a):
        rings = self._local_files(a, "zone_rings", ".npy")
        masks = self._local_files(a, "zone_masks", ".npy")
        band_cam = self._local_files(a, "band_cam", ".npy")
        band_field = self._local_files(a, "band_field", ".npz")
        if len(band_cam) > 1 or len(band_field) > 1 or len(band_cam) != len(band_field):
            raise ArgError("a tube band needs exactly one band_cam .npy and one band_field .npz")
        cal = _s(a, "calibration")
        if cal is not None and not Path(cal).expanduser().exists():
            raise ArgError(f"no calibration {cal}")
        label = _s(a, "label")
        if label is not None and not _NAME_RE.match(label):
            raise ArgError("label must be letters, digits, _ . - only")
        margin = _num(a, "margin_px", 6, cast=int)
        if not 0 <= margin <= 60:
            raise ArgError("margin_px must be 0..60 DMD px")
        p = Path(api.make_fastscan_pack(tuple(rings), tuple(masks),
                                        band_cam[0] if band_cam else None,
                                        band_field[0] if band_field else None, margin,
                                        cal, _bool(a, "study"), label))
        m = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
        doc = Path(api.__file__).resolve().parents[1] / FASTSCAN_DOC
        return {"folder": str(p), "variants": list(m["variants"]),
                "timing": m["timing"], "led": m["led"],
                "images": [im["file"] for im in m["images"]],
                "files": sorted(f.name for f in p.iterdir())[:200],
                "doc": str(doc)}, {"json": p / "manifest.json"}

    def _v_fast_scan_flash(self, a):
        pack = _s(a, "pack")
        if pack is not None and not (Path(pack).expanduser() / "manifest.json").is_file():
            raise ArgError(f"pack: no manifest.json in {pack}")
        variant = _s(a, "variant")
        if variant is not None and variant not in ("full", "zone"):
            raise ArgError("variant must be full or zone")
        label = _s(a, "label", "flash")
        if not _NAME_RE.match(label):
            raise ArgError("label must be letters, digits, _ . - only")
        roi = _num_list(a, "roi", int)
        if roi is not None and len(roi) != 4:
            raise ArgError("roi must be [x0, y0, x1, y1] camera px")
        loops = _num(a, "loops", None, int)
        if loops is not None and not 2 <= loops <= 50:
            raise ArgError("loops must be 2..50")
        fpx = _fine_prefilter(a)
        r = api.fast_scan_flash(str(Path(pack).expanduser()) if pack else None, variant, label,
                                self._preset(), _s(a, "zone"), roi, loops,
                                _num(a, "exposure_us"), _num(a, "fps"), _num(a, "gain_db"),
                                "plane", _bool(a, "keep_recording", True), _bool(a, "dry_run"),
                                fpx)
        links = {}
        ms = r.get("scan")
        if ms is not None and (Path(ms.path) / "preview.png").is_file():
            links["preview"] = Path(ms.path) / "preview.png"
        return r, links

    def _v_sort_flash_recording(self, a):
        rec = _s(a, "recording", required=True)
        f = Path(rec).expanduser()
        if f.suffix.lower() != ".npz" or not f.is_file():
            raise ArgError(f"recording: no .npz file {rec}")
        pack = _s(a, "pack")
        if pack is not None and not (Path(pack).expanduser() / "manifest.json").is_file():
            raise ArgError(f"pack: no manifest.json in {pack}")
        variant = _s(a, "variant")
        if variant is not None and variant not in ("full", "zone"):
            raise ArgError("variant must be full or zone")
        label = _s(a, "label")
        if label is not None and not _NAME_RE.match(label):
            raise ArgError("label must be letters, digits, _ . - only")
        return api.sort_flash_recording(str(f), str(Path(pack).expanduser()) if pack else None,
                                        variant, _bool(a, "decode"), label, _fine_prefilter(a))

    # ------------------------------------------------ operator page (/quick)
    def _cancellable(self, job: Job):
        """Run a job with its cancel event wired into the engine (checked before every
        capture): an abort ends the capture sequence and closes the rig cleanly."""
        self.session.cancel = job.cancel
        try:
            return getattr(self, "_v_" + job.verb)(dict(job.args))
        finally:
            self.session.cancel = None

    def stations(self, flows: Optional[dict] = None) -> list:
        """The station configurations: stations/*.json. One with profiles + a flow lists its
        profile library set and its work states; an old single-job config is one implicit
        profile ("default") and a one-step flow (its job). flows (optional): filled with
        config -> StationConfiguration for every usable one (what the bus walks)."""
        from ui import run_job as RJ
        out = []
        flows = {} if flows is None else flows
        d = self.stations_dir
        for f in sorted(d.glob("*.json")) if d.is_dir() else []:
            if f.name.startswith("io_"):        # I/O mappings live here too, not stations
                continue
            item = {"config": f.name}
            try:
                cfg = flow_schema.load(f)
                pub = cfg.public()
                item.update(kind="single job" if cfg.legacy else "flow",
                            profiles=pub["profiles"], default_profile=pub["default_profile"],
                            flow=pub["flow"], permissions=pub["permissions"],
                            signals=pub["signals"], mock_cases=pub["mock_cases"])
                if not cfg.legacy:
                    item.update(station_id=cfg.station_id, part_label=pub["default_profile"],
                                job_type="flow", preset=cfg.preset_for(None),
                                description=cfg.description, threshold_mm=None, zone=None,
                                measure=None, flow_source=cfg.flow_source)
                    probs = flow_verify.check_config(cfg)
                    if probs:
                        raise flow_schema.StationConfigError("; ".join(probs), probs)
                    flows[f.name] = cfg
                    out.append(item)
                    continue
                c = RJ.load_station_config(f)
                flows[f.name] = cfg
                item.update(station_id=c.station_id, part_label=c.part_label,
                            job_type=c.job_type, preset=c.preset, threshold_mm=c.threshold_mm,
                            zone=c.zone,
                            # its job says "zone": "selection": the zone sets switched on at
                            # the operator page decide what its run lights and judges
                            zone_from_selection=RJ.uses_selection(c.zone),
                            measure=c.measure if c.job_type == "metric" else None)
                if c.job_type == "recipe":          # the one work state is verified by a recipe
                    r = api.recipe_load(c.recipe)
                    item.update(recipe=r.name, preset=c.preset or r.preset,
                                recipe_preset=r.preset, description=r.description,
                                zones=[z.id for z in r.zones],
                                checks=sorted({ch.type for z in r.zones for ch in z.checks}),
                                zone=r.scan["zone"], feedback=r.feedback["enabled"],
                                result_codes=dict(r.outputs["result_codes"]))
            except Exception as e:      # noqa: BLE001 - shown as an unusable entry
                item["error"] = f"{type(e).__name__}: {e}"
                flows.pop(f.name, None)
            out.append(item)
        return out

    @staticmethod
    def _config_names_preset(f: Path) -> bool:
        """Does the station config pin its own preset (directly, or through its recipe)?
        An unreadable config counts as yes: the run then refuses it with its own error."""
        from ui import run_job as RJ
        try:
            if not flow_schema.is_legacy(flow_schema.read_doc(f)):
                return True             # profiles + flow: its profiles / recipes pin presets
            c = RJ.load_station_config(f)
        except Exception:               # noqa: BLE001 - cmd_run reports the config's error
            return True
        return bool(c.preset) or c.job_type == "recipe"

    def _station_file(self, name) -> Path:
        if not isinstance(name, str) or not _NAME_RE.match(name) or not name.endswith(".json"):
            raise ArgError("config must be a station config file name from GET /api/stations")
        f = (self.stations_dir / name).resolve()
        if not inside(self.stations_dir.resolve(), f) or not f.is_file():
            raise ArgError(f"no station config {name!r} in the stations folder")
        return f

    def _v_run_station(self, a):
        """`scan run <config>` in-process, on the server's own session and library."""
        from ui import cli
        f = self._station_file(a.get("config"))
        confirmed = _confirmed(a, "operator_confirmed")
        mock_part = _s(a, "mock_part")
        if mock_part is not None and not self.mock:
            raise ArgError("mock_part is MOCK ONLY")
        if not self.mock and not confirmed:
            raise rules.RuleViolation("the real rig needs the operator's confirmation of the "
                                      "physical setup (part placed, rig untouched)")
        argv = ["--json", "--library", str(self.root)] + (["--mock"] if self.mock else [])
        argv += (["--mock-part", mock_part] if mock_part else [])
        argv += ["run", str(f), "--no-feedback"] + (["--operator-confirmed"] if confirmed else [])
        args = cli.build_parser().parse_args(argv)
        args.preset_given = False               # the config's (or its recipe's) preset wins
        args.mock_part_given = mock_part is not None    # else the config's own mock_part
        # "zone": "selection": the selection's stamp the bus accepted for this cycle (what
        # the page showed / what the station was ARMED with); another one by now is a PULL
        args.zone_expect = _s(a, "zones_expect")
        loaded = self.session.preset
        if loaded is not None:
            # a config naming no preset runs the LOADED one: what /quick shows (run_preset)
            # and what the operator set up - never the CLI's default preset
            args.preset = loaded.name
        elif not self._config_names_preset(f):
            raise ArgError("this station config names no preset and none is loaded: load one "
                           "in Setup (or name one in the config)")
        return self._cli_job(cli.cmd_run, args)

    def _cli_job(self, cmd, args):
        """A station job through the CLI's own code path (cmd_run / cmd_quick_check) on the
        server's session -> (result + exit_code, links); a recipe job's verdict is then
        painted on the part by its own `recipe_feedback` job."""
        from ui import cli
        ctx = cli.Ctx(args)
        ctx.session = self.session
        ctx._outputs_before = len(self.session.store.outputs())
        try:
            code, data = cmd(ctx)
            res, run_dir = data["result"], Path(data["run_dir"])
        except Exception as e:          # noqa: BLE001 - a refused / crashed run is INVALID
            res = getattr(e, "run_result", None)
            if res is None:
                raise
            run_dir = Path(e.run_dir)
            # the code run.log.jsonl's run_end records: 2 refused, 4 crashed, ...
            code = getattr(e, "run_exit_code", cli.EXIT_REFUSED)
            res = {**res, "error": f"{type(e).__name__}: {e}"}
        self._runs[res["run_id"]] = run_dir
        fb = res.get("feedback") or {}
        if fb.get("rendered") and res.get("result") in ("PASS", "FAIL", "INVALID"):
            # the verdict painted on the part: its own job, so the bus already shows PASS /
            # FAIL and any next command (START, TRIGGER, HOME, ABORT) clears it. Checked and
            # queued under the job lock: an ABORT either came first (this cycle is ABORTED on
            # the bus: nothing is painted) or drops the queued feedback job after it
            with self._cv:
                cancel = self.session.cancel
                if cancel is not None and cancel.is_set():
                    res = {**res, "warnings": [*(res.get("warnings") or []),
                                               "feedback not projected: ABORT was pressed"]}
                else:
                    res = {**res, "feedback_job": self.submit("recipe_feedback",
                                                              {"run_id": res["run_id"]}).id}
        return {"exit_code": code, **res}, self._run_links(res, run_dir)

    # ------------------------------------------------ operating modes: Measure / Quick check
    def _mode_preset(self, a: dict):
        """The triangulation preset a measurement / quick check runs with: the loaded one,
        or `preset` (loaded first, so HOME - the rig check - is about the same rig)."""
        name = _s(a, "preset")
        cur = self.session.preset
        if name and (cur is None or cur.name != name):
            api.load_preset(self.check_preset_name_arg(name))
        p = self._preset()
        self._mobile_session(p)
        return p

    @staticmethod
    def _roi_arg(a: dict, k: str) -> Optional[list]:
        v = _num_list(a, k, int)
        if v is not None and len(v) != 4:
            raise ArgError(f"{k} must be [x0, y0, x1, y1] camera px")
        return v

    def _v_measure_run(self, a):
        """MEASURE mode: api.measure_run on the server's session (numbers, never a verdict)."""
        spec = a.get("spec")
        if spec is not None and not isinstance(spec, dict):
            raise ArgError("'spec' must be a measurement spec mapping")
        p = self._mode_preset(a)
        ids = a.get("metric_ids") or None
        if ids is not None and not isinstance(ids, list):
            raise ArgError("metric_ids must be a list of metric scan ids")
        scans = [self._entry("metric", i) for i in ids] if ids else None
        scene = self._mock_scene(a)
        if scene is None and self.mock and scans is None:
            scene = "none"              # MOCK: the simulated (stereo) bench, nothing placed
        try:
            out = api.measure_run(
                _s(a, "what"), p, spec, roi=self._roi_arg(a, "roi"),
                roi2=self._roi_arg(a, "roi2"), zone=_s(a, "zone"), zone2=_s(a, "zone2"),
                p1=_point(a, "p1"), p2=_point(a, "p2"), golden=_s(a, "golden"),
                radius_mm=_num(a, "radius_mm"), axis=_s(a, "axis"),
                min_height_mm=_num(a, "min_height_mm"),
                face_diameter_mm=_num(a, "face_diameter_mm"),
                n_faces=_num(a, "n_faces", None, int), repeats=_num(a, "repeats", None, int),
                scans=scans, scan_kind=_s(a, "scan_kind"), steps=_num(a, "steps", None, int),
                periods=_num_list(a, "periods"), average=_num(a, "average", None, int),
                label=_s(a, "label"), name=_s(a, "name"), mock_scene=scene)
        except ValueError as e:         # a malformed spec: nothing ran, no run folder
            raise ArgError(str(e)) from None
        doc, run_dir = out["result"], Path(out["run_dir"])
        rid = doc["run_id"]
        self._runs[rid] = run_dir
        links = {"result_json": f"/runs/{rid}/result.json", "log": f"/runs/{rid}/run.log.jsonl",
                 "values_csv": f"/runs/{rid}/values.csv", "spec": f"/runs/{rid}/spec.json"}
        if doc["outputs"].get("overlay"):
            links["preview"] = f"/runs/{rid}/overlay.png"
        if doc["outputs"].get("ply"):
            links["ply"] = f"/runs/{rid}/scan.ply"
        return ({"exit_code": out["exit_code"], **doc},
                {k: _UILink(v) for k, v in links.items()})

    def _v_rig_session(self, a):
        return api.rig_session(self._preset(), _s(a, "reason", "") or "")

    def _v_quick_check(self, a):
        """QUICK CHECK mode: `scan quick-check` in-process, on the server's own session."""
        from ui import cli
        check = _s(a, "check", required=True)
        confirmed = _confirmed(a, "operator_confirmed")
        mock_part = _s(a, "mock_part")
        if mock_part is not None and not self.mock:
            raise ArgError("mock_part is MOCK ONLY")
        if not self.mock and not confirmed:
            raise rules.RuleViolation("the real rig needs the operator's confirmation of the "
                                      "physical setup (part placed, rig untouched)")
        recipe = _s(a, "recipe")
        if recipe is not None:
            self.check_recipe_name(recipe)      # a name of this library, never a file path
        preset = _s(a, "preset")
        if preset is None and check != "recipe":
            preset = self._preset().name        # the loaded one: what the page shows
        if preset is not None:
            self.check_preset_name_arg(preset)
        argv = ["--json", "--library", str(self.root)] + (["--mock"] if self.mock else [])
        argv += (["--mock-part", mock_part] if mock_part else [])
        argv += ["quick-check"] + ([preset] if preset else []) + ["--check", check, "--no-feedback"]
        for opt, v in (("--threshold", _num(a, "threshold_mm")),
                       ("--guard-band", _num(a, "guard_band_pct")), ("--zone", _s(a, "zone")),
                       ("--golden", _s(a, "golden")), ("--recipe", recipe)):
            if v is not None:
                argv += [opt, str(v)]
        roi = self._roi_arg(a, "roi")
        if roi is not None:
            argv += ["--roi", ",".join(str(v) for v in roi)]
        argv += ["--operator-confirmed"] if confirmed else []
        try:
            args = cli.build_parser().parse_args(argv)
        except SystemExit:              # argparse refused a value (its message went to stderr)
            raise ArgError("quick_check: the arguments were refused (check, threshold_mm, "
                           "guard_band_pct ...)") from None
        args.preset_given = False               # cmd_quick_check takes the preset argument
        args.mock_part_given = mock_part is not None
        if preset is not None:                  # its rig session opens BEFORE the check runs
            try:
                self._mobile_session(config.load_preset(
                    preset, presets_dir=self.session.presets_dir, store=self.session.store,
                    mock=self.mock))
            except config.ConfigError:
                pass                            # the run refuses the preset with its own error
        return self._cli_job(cli.cmd_quick_check, args)

    def _run_links(self, res: dict, run_dir: Path) -> dict:
        links = {"result_json": f"/runs/{res['run_id']}/result.json",
                 "log": f"/runs/{res['run_id']}/run.log.jsonl"}
        if (run_dir / "feedback_dmd.png").is_file():
            links["feedback"] = f"/runs/{res['run_id']}/feedback_dmd.png"
        dm = (res.get("outputs") or {}).get("deviation_map")
        if dm:
            p = Path(dm)
            if p.parent.resolve() == run_dir.resolve() and p.name in RUN_FILES:
                links["preview"] = f"/runs/{res['run_id']}/{p.name}"
            elif self.root is not None and inside(self.root, p.resolve()):
                links["preview"] = "/files/" + rel_path(p, self.root)
        return {k: _UILink(v) for k, v in links.items()}

    def run_file(self, run_id: str, name: str) -> Path:
        d = self._runs.get(run_id)
        if d is None or name not in RUN_FILES:
            raise HTTPError(404, f"no run file {run_id}/{name} (only this server's runs)")
        f = Path(d) / name
        if not f.is_file():
            raise HTTPError(404, f"no file {name} in run {run_id}")
        return f

    # ------------------------------------------------ work states (flows)
    def _flow_cfg(self, a: dict, cycle: Optional[dict] = None):
        """The station configuration a flow verb runs: the open cycle's own (what the bus
        walks), else the file named `config` (must have profiles + a flow, and pass the
        checks the schema can't see)."""
        if cycle is not None:
            return cycle["cfg"]
        cfg = flow_schema.load(self._station_file(a.get("config")))
        if cfg.legacy:
            raise ArgError(f"{a.get('config')} is a single-job station config (no profiles, no "
                           "flow): run it with run_station")
        probs = flow_verify.check_config(cfg)
        if probs:
            raise ArgError("; ".join(probs))
        return cfg

    def _v_flow_verify(self, a):
        """The DLP check of one work state (the job behind STEP; also on its own)."""
        cid = _s(a, "cycle_id")
        with self._cv:
            cycle = self._cycles.get(cid) if cid else None
        if cid and cycle is None:
            raise ArgError(f"no open cycle {cid!r} of this server")
        cfg = self._flow_cfg(a, cycle)
        profile = _s(a, "profile") or cfg.default_profile
        step = _s(a, "step", required=True)
        confirmed = _confirmed(a, "operator_confirmed")
        scan = _s(a, "scan_id")
        if not self.mock and not confirmed and scan is None:
            raise rules.RuleViolation("the real rig needs the operator's confirmation of the "
                                      "physical setup (part placed, rig untouched)")
        if scan is not None:
            self._entry("metric", scan)
        scene = self._mock_scene(a)
        if scene is None and scan is None:
            scene = self._mock_scene_now(cfg, profile)
        runs_dir = (Path(cycle["cycle_dir"]) / flow_verify.STEPS_DIR) if cycle is not None \
            else None
        out = api.flow_verify(cfg, profile, step, runs_dir=runs_dir,
                              operator_confirmed=confirmed, mock_scene=scene, scan=scan)
        res, run_dir = out["result"], Path(out["run_dir"])
        self._runs[res["run_id"]] = run_dir
        # the answer says which verify it answers (work state, attempt, cycle): the command
        # bus takes no other
        dlp = {**out["dlp"], "step": step, "attempt": a.get("attempt"), "cycle_id": cid}
        return ({"exit_code": out["exit_code"], **res, "dlp": dlp},
                self._run_links(res, run_dir))

    def _v_flow_run(self, a):
        """One whole cycle in line (unattended): api.flow_run."""
        cfg = self._flow_cfg(a)
        profile = _s(a, "profile") or cfg.default_profile
        confirmed = _confirmed(a, "operator_confirmed")
        if not self.mock and not confirmed:
            raise rules.RuleViolation("the real rig needs the operator's confirmation of the "
                                      "physical setup (part placed, rig untouched)")
        inputs = a.get("inputs")
        if inputs is not None and (not isinstance(inputs, dict) or not all(
                isinstance(v, bool) for v in inputs.values())):
            raise ArgError("'inputs' must be an object {signal: true | false}")
        scene, case = a.get("mock_scene"), _s(a, "mock_case")
        if (scene is not None or case is not None) and not self.mock:
            raise ArgError("mock_scene / mock_case are MOCK ONLY")
        if scene is not None and not isinstance(scene, (str, dict)):
            raise ArgError("'mock_scene' must be a scene or an object {step id: scene}")
        if self.mock and scene is None:
            try:
                c = cfg.mock_case(case, profile)
            except flow_schema.StationConfigError as e:
                raise ArgError(str(e)) from None
            scene = c["scene"]
            if inputs is None:
                inputs = c["inputs"]
        out = api.flow_run(cfg, profile, inputs=inputs, operator_confirmed=confirmed,
                           mock_scene=scene, acknowledge=_bool(a, "acknowledge"),
                           project_faults=_bool(a, "project_faults", True),
                           hold_s=_num(a, "hold_s"))
        res, cycle_dir = out["result"], Path(out["cycle_dir"])
        self._runs[res["run_id"]] = cycle_dir
        return {"exit_code": out["exit_code"], **res}, self._run_links(res, cycle_dir)

    # ------------------------------------------------ recipes (work-state checks)
    def recipe_names(self) -> list:
        return [i["name"] for i in api.recipe_list()]

    def check_recipe_name(self, name) -> str:
        if not isinstance(name, str) or not _NAME_RE.match(name) or name not in self.recipe_names():
            raise ArgError(f"no recipe named {name!r} (GET /api/recipes)")
        return name

    def _mock_scene(self, a):
        sc = _s(a, "mock_scene")
        if sc is not None and not self.mock:
            raise ArgError("mock_scene is MOCK ONLY")
        return sc

    def _v_recipe_add(self, a):
        ov = a.get("overrides")
        if ov is not None and not isinstance(ov, dict):
            raise ArgError("'overrides' must be an object {\"feedback.hold_s\": 10, ...}")
        rec = a.get("recipe")
        if rec is not None and not isinstance(rec, dict):
            raise ArgError("'recipe' must be a recipe mapping")
        base = _s(a, "base")
        if base is not None:
            self.check_recipe_name(base)
        return api.recipe_add(_s(a, "name", required=True), base, ov,
                              _s(a, "description", "") or "", _bool(a, "replace"), recipe=rec)

    def _v_recipe_select(self, a):
        return api.recipe_select(self.check_recipe_name(_s(a, "name", required=True)))

    def _v_recipe_delete(self, a):
        if not _confirmed(a, "confirm"):
            raise ArgError("recipe_delete needs confirm=true (the operator confirmed the delete)")
        return api.recipe_delete(self.check_recipe_name(_s(a, "name", required=True)),
                                 _s(a, "reason", required=True), _bool(a, "force"),
                                 stations_dir=self.stations_dir)

    def _v_recipe_run(self, a):
        name = self.check_recipe_name(_s(a, "recipe", required=True))
        confirmed = _confirmed(a, "operator_confirmed")
        scan = _s(a, "scan_id")
        fb = _bool(a, "project_feedback", True)
        if not self.mock and not confirmed and not (scan is not None and not fb):
            raise rules.RuleViolation("the real rig needs the operator's confirmation of the "
                                      "physical setup (part placed, rig untouched); a stored "
                                      "scan_id re-checked with project_feedback false needs none")
        if scan is not None:
            self._entry("metric", scan)
        out = api.recipe_run(name, operator_confirmed=confirmed,
                             project_feedback=fb, scan=scan,
                             hold_s=_num(a, "hold_s"), mock_scene=self._mock_scene(a))
        res, run_dir = out["result"], Path(out["run_dir"])
        self._runs[res["run_id"]] = run_dir
        return {"exit_code": out["exit_code"], **res}, self._run_links(res, run_dir)

    def _v_recipe_golden(self, a):
        name = self.check_recipe_name(_s(a, "recipe", required=True))
        return api.recipe_golden(name, mock_scene=self._mock_scene(a))

    def _v_recipe_feedback(self, a):
        rid = _s(a, "run_id", required=True)
        d = self._runs.get(rid)
        if d is None:
            raise ArgError(f"no run {rid!r} of this server")
        me = self._current
        return api.recipe_feedback(d, self._preset(), _num(a, "hold_s"),
                                   cancel=me.cancel if me is not None else None)

    def _v_rig_check(self, a):
        shift = _num_list(a, "mock_shift_px")
        if shift is not None:
            if not self.mock:
                raise ArgError("mock_shift_px is MOCK ONLY")
            if not 2 <= len(shift) <= 4:
                raise ArgError("mock_shift_px must be [dx, dy] camera px (+ roll_deg, + scale)")
        r = api.rig_check(self._preset(), _bool(a, "set_reference"), _num(a, "tol_px", 0.5),
                          shift)
        return r, ({"flood": Path(r["path"]) / "flood.png"}
                   if r.get("path") and (Path(r["path"]) / "flood.png").is_file() else {})

    def _v_mock_place(self, a):
        from engine.capture.mock_bench import VALIDATION_OBJECT, Box, Label
        bench = api.mock_bench()               # raises in real mode: physical steps are human
        what = _s(a, "object", "validation")
        c = (_num(a, "x_mm", 0.0), _num(a, "y_mm", 0.0))
        if what == "validation":
            bench.place(dataclasses.replace(VALIDATION_OBJECT))
        elif what == "box":
            albedo = _num(a, "albedo")
            if albedo is not None and not 0.0 <= albedo <= 1.0:
                raise ArgError(f"albedo must be 0..1 (reflectance), got {albedo:g}")
            bench.place(Box(h_mm=_num(a, "h_mm", required=True), w_mm=_num(a, "w_mm", 40.0),
                            l_mm=_num(a, "l_mm", 30.0), center_mm=c,
                            angle_deg=_num(a, "angle_deg", 0.0),
                            **_kw(albedo=albedo)))
        elif what == "cylinder":
            from engine.capture.mock_bench import Cylinder
            dents = a.get("dents")
            if dents is None:
                dents = [{"depth_mm": 1.5, "size_mm": 12.0, "angle_deg": 10.0, "z_mm": 45.0}]
            if not isinstance(dents, list) or not all(
                    isinstance(d, dict) and {"depth_mm", "size_mm"} <= set(d)
                    and set(d) <= {"depth_mm", "size_mm", "angle_deg", "z_mm"} for d in dents):
                raise ArgError("dents must be a list of {depth_mm, size_mm, angle_deg, z_mm}")
            r, hh = _num(a, "radius_mm", 24.05), _num(a, "height_mm", 90.0)
            if r <= 0 or hh <= 0:
                raise ArgError("radius_mm and height_mm must be > 0")
            try:
                dl = [{k: float(v) for k, v in d.items()} for d in dents]
            except (TypeError, ValueError):
                raise ArgError("dent values must be numbers") from None
            bench.place(Cylinder(radius_mm=r, height_mm=hh, center_mm=c, dents=dl))
        elif what == "label":
            bench.place(Label(data=_s(a, "data", required=True), center_mm=c,
                              size_mm=_num(a, "size_mm", 45.0)))
        else:
            raise ArgError("object must be validation, box, label or cylinder")
        return self._bench_state()

    def _v_mock_clear(self, a):
        api.mock_bench().clear()
        return self._bench_state()

    def _v_live_view(self, a):
        """Rig stays open; shoot a raw flood / focus pattern until stopped.
        Between frames the worker serves pending listings (drain_quick) and
        runs queued rig-free jobs (drain_rig_free). Stops on its own job's
        cancel event (Stop, a queued rig job, shutdown), after max_s, or when
        no viewer polled for LIVE_IDLE_S (an abandoned browser must not keep
        the rig, the flood and the machine-wide rig lock)."""
        p = self._preset()
        live = self.live
        me = self._current
        stop = me.cancel if me is not None else threading.Event()
        settings = {k: a[k] for k in ("pattern", "colour", "level", "period_px") if k in a}
        if "level" not in settings:
            settings["level"] = int(round(255 * p.projection.duty))
        max_s = _num(a, "max_s", MAX_LIVE_S)
        if not 0.0 < max_s <= MAX_LIVE_S:
            raise ArgError(f"max_s must be > 0 and <= {MAX_LIVE_S:g} s (the live view holds "
                           f"the rig and the flood), got {max_s:g}")
        # rule 3 over time: a live view is a sustained projection of up to max_s
        rules.check_led_duration(p.projection.led, max_s, "live view")
        live.update_settings({**settings, "reset_peak": True})
        live.stopped_because = None
        live.touch()                            # the viewer gets LIVE_IDLE_S to start polling

        def loop():
            n, t0 = 0, time.monotonic()
            with live.lock:
                live.running, live.error, live.started = True, None, _now()
                live.frame, live.fps, live.metrics = 0, 0.0, {}
            try:
                with self.session.rig(p) as sync:
                    sync.average = 1                 # live preview: no frame averaging
                    shown = None
                    while not stop.is_set() and not self._closing:
                        if time.monotonic() - t0 >= max_s:
                            live.stopped_because = f"max_s {max_s:g} s reached"
                            break
                        if time.monotonic() - live.seen > LIVE_IDLE_S:
                            live.stopped_because = (f"no viewer for {LIVE_IDLE_S:g} s "
                                                    "(nobody polled the live view)")
                            break
                        with live.lock:
                            s = dict(live.settings)
                        t1 = time.monotonic()
                        if s != shown:              # new pattern: project, settle, grab
                            frame = sync.shoot(live_pattern(s, (p.projection.size[1],
                                                                 p.projection.size[0])), raw=True)
                            shown = s
                        else:                        # same pattern: keep the window pumped
                            sync.projector.window.hold(0.03)   # (§9.2) and grab a fresh frame
                            frame = sync.camera.grab().astype(np.float32)
                        jpg, m = frame_metrics(frame)
                        n += 1
                        with live.lock:
                            live.jpeg, live.metrics, live.frame = jpg, m, n
                            live.peak = max(live.peak, m["focus"])
                            dt = time.monotonic() - t1
                            live.fps = 0.8 * live.fps + 0.2 * (1.0 / dt) if dt > 0 else live.fps
                        self.drain_quick()
                        self.drain_rig_free()
                        if self.mock:                       # no real frame pacing in mock
                            stop.wait(0.12)
            except Exception as e:
                with live.lock:
                    live.error = f"{type(e).__name__}: {e}"
                raise
            finally:
                with live.lock:
                    live.running = False
            with live.lock:
                return {"frames": n, "duration_s": round(time.monotonic() - t0, 2),
                        "focus_peak": round(live.peak, 2), "last": dict(live.metrics),
                        "settings": dict(live.settings),
                        "stopped_because": live.stopped_because or "stopped"}

        with live.lock:
            live.job_id = me.id if me is not None else None
        return self.session.run("ui_live_view", {**settings, "preset": p.name}, loop)


# ================================================================== HTTP

class UIServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False     # on Windows, reuse lets a 2nd server share the port
    app: App
    verbose = False


ROUTES = [
    ("GET", r"/(?:index\.html)?", "index"),
    ("GET", r"/static/(?P<name>[^/]+)", "static"),
    ("GET", r"/files/(?P<rel>.+)", "files"),
    ("GET", r"/api/status", "status"),
    ("GET", r"/api/verbs", "verbs"),
    ("GET", r"/api/jobs", "jobs"),
    ("POST", r"/api/jobs", "submit"),
    ("GET", r"/api/jobs/(?P<jid>[A-Za-z0-9]+)", "job"),
    ("POST", r"/api/jobs/(?P<jid>[A-Za-z0-9]+)/cancel", "cancel"),
    ("GET", r"/api/live\.jpg", "live_jpg"),
    ("GET", r"/api/live/state", "live_state"),
    ("POST", r"/api/live/start", "live_start"),
    ("POST", r"/api/live/settings", "live_settings"),
    ("POST", r"/api/live/stop", "live_stop"),
    ("GET", r"/api/presets", "presets"),
    ("POST", r"/api/presets", "preset_save"),
    ("GET", r"/api/presets/(?P<name>[^/]+)", "preset_get"),
    ("GET", r"/api/zonesets", "zonesets"),
    ("GET", r"/api/zones/selection", "zone_selection"),
    ("POST", r"/api/zones/selection", "zone_selection_set"),
    ("GET", r"/api/recipes", "recipes"),
    ("GET", r"/api/recipes/(?P<name>[^/]+)", "recipe_get"),
    ("POST", r"/api/presets/(?P<name>[^/]+)/load", "preset_load"),
    ("GET", r"/api/library/(?P<kind>[a-z]+)", "library"),
    ("GET", r"/api/outputs", "outputs"),
    ("GET", r"/api/pointcloud/(?P<oid>[^/]+)", "pointcloud"),
    ("GET", r"/api/log", "log"),
    ("GET", r"/quick(?:\.html)?", "quick"),
    ("GET", r"/api/stations", "stations"),
    ("GET", r"/api/cmd/state", "cmd_state"),
    ("POST", r"/api/cmd/(?P<name>start|abort|home|trigger|step|ack|retry|skip|arm|disarm|reset"
             r"|config|profile|mode|workstate)", "cmd"),
    ("GET", r"/api/modes", "modes"),
    ("POST", r"/api/measure", "measure"),
    ("POST", r"/api/quick-check", "quick_check"),
    ("GET", r"/api/io/inputs", "io_inputs"),
    ("POST", r"/api/io/inputs", "io_inputs_set"),
    ("POST", r"/api/mock/case", "mock_case"),
    ("GET", r"/runs/(?P<run_id>[A-Za-z0-9_.\-]+)/(?P<name>[A-Za-z0-9_.\-]+)", "run_file"),
]
_ROUTES = [(m, re.compile(p), name) for m, p, name in ROUTES]


def inside(root: Path, target: Path) -> bool:
    r = os.path.normcase(str(root))
    t = os.path.normcase(str(target))
    return t.startswith(r.rstrip("\\/") + os.sep)


class Handler(BaseHTTPRequestHandler):
    server: UIServer
    server_version = "SwissArmyScannerUI/0.1"

    # ------------------------------------------------------------ plumbing
    @property
    def app(self) -> App:
        return self.server.app

    def log_message(self, fmt, *args):
        if self.server.verbose:
            super().log_message(fmt, *args)

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def _origins(self) -> set:
        port = self.server.server_address[1]
        return {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}

    def _dispatch(self, method: str):
        try:
            host = self.headers.get("Host")
            if host is not None and f"http://{host}" not in self._origins():
                raise HTTPError(403, "unexpected Host header (this UI answers on 127.0.0.1 only)")
            url = urlsplit(self.path)
            body = None
            if method == "POST":
                origin = self.headers.get("Origin")
                if origin is not None and origin not in self._origins():
                    raise HTTPError(403, f"cross-origin request from {origin} refused")
                ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                if ctype != "application/json":
                    raise HTTPError(415, "POST bodies must be application/json")
                body = self._read_json()
            for m, rx, name in _ROUTES:
                mt = rx.fullmatch(url.path)
                if mt and m == method:
                    q = {k: v[-1] for k, v in parse_qs(url.query).items()}
                    groups = {k: unquote(v) for k, v in mt.groupdict().items()}
                    return getattr(self, "r_" + name)(q=q, body=body, **groups)
            raise HTTPError(404, f"no route {method} {url.path}")
        except HTTPError as e:
            self._json({"error": e.msg}, e.code)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:          # noqa: BLE001 - a handler bug must not kill the server
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise HTTPError(413, "request body too large")
        raw = self.rfile.read(n) if n else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, ValueError) as e:
            raise HTTPError(400, f"invalid JSON body: {e}") from None
        if not isinstance(data, dict):
            raise HTTPError(400, "JSON body must be an object")
        return data

    def _send(self, code: int, data: bytes, ctype: str, extra: Optional[dict] = None):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for k, v in {**SECURITY_HEADERS, **(extra or {})}.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _json(self, obj, code: int = 200):
        try:
            data = json.dumps(obj, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError):
            data = json.dumps(jsonable(obj, self.app.root or Path(".")), allow_nan=False)
        self._send(code, data.encode("utf-8"), "application/json")

    def _file(self, path: Path, ctype: str, download: bool = False):
        extra = {}
        if download:
            extra["Content-Disposition"] = f'attachment; filename="{path.name}"'
        try:
            size = path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-cache")
            for k, v in {**SECURITY_HEADERS, **extra}.items():
                self.send_header(k, v)
            self.end_headers()
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(256 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    # -------------------------------------------------------------- static
    def r_index(self, q, body):
        self._file(STATIC_DIR / "index.html", STATIC_TYPES[".html"])

    def r_quick(self, q, body):
        self._file(STATIC_DIR / "quick.html", STATIC_TYPES[".html"])

    def r_stations(self, q, body):
        self._json({"stations": self.app.refresh_configs(),
                    "mode": "mock" if self.app.mock else "real",
                    "dir": str(self.app.stations_dir)})

    def r_io_inputs(self, q, body):
        """What the line's sensors say now, and what the scanner says back: the sensor
        claims held, the signals the selected configuration expects, the permissions."""
        st = self.app.cmd_state()
        self._json({"inputs": st["inputs"], "signals": st["signal_names"],
                    "permissions": st["permissions"], "outputs": st["signals"],
                    "registers": st["registers"]})

    def r_io_inputs_set(self, q, body):
        """Sensor claims from the line (a PLC / HMI gateway, a test): {signal: true | false,
        ...} (null forgets one), or {"inputs": {...}, "source": "plc"} - one form or the
        other, never both. Only JSON booleans count, and only signals a station configuration
        declares: anything else is a 400, never guessed, nothing applied. A claim belongs to
        one cycle (forgotten when it ends): report the sensors for every part. The DLP
        confirms or corrects them when a work state is verified."""
        values, src = body, "http"
        if "inputs" in body:
            extra = sorted(set(body) - {"inputs", "source"})
            if not isinstance(body["inputs"], dict) or extra:
                raise HTTPError(400, "send either {\"<signal>\": true | false, ...} or "
                                     "{\"inputs\": {...}, \"source\": \"...\"}, not both"
                                     + (f" (also got {', '.join(map(str, extra))})" if extra
                                        else ": 'inputs' must be an object"))
            values, src = body["inputs"], str(body.get("source") or "http")[:40]
        held = self.app.set_inputs(values, source=src)
        self._json({"inputs": held, "bus": self.app.cmd_state()})

    def r_mock_case(self, q, body):
        """MOCK only: the simulated line for the next scans: {"case": "<name>"} (a case of
        the selected configuration: scene + sensor claims), {"scene": ..., "inputs": {...}},
        or {} to clear it."""
        out = self.app.set_mock_case(body.get("case"), body.get("scene"), body.get("inputs"))
        self._json({**out, "bus": self.app.cmd_state()})

    def r_cmd_state(self, q, body):
        self._json(self.app.cmd_state())

    def r_cmd(self, q, body, name):
        """One command through the bus (page buttons, keys, a PLC / HMI gateway). 200 when
        accepted, 409 when the state machine refused it (the reply says why), 400 for a
        malformed body. body: operator_confirmed (start / trigger / arm on the real rig: a
        JSON boolean, only true confirms); config (config: the station configuration's file
        name or 1-based number to select; start / trigger / arm: the one the sender expects -
        refused when another is selected); profile (profile: the profile name or 1-based
        number to select; start / trigger / arm: the one the sender expects); step (step: the
        work state id / number the sender expects to verify); home: set_reference, tol_px,
        mock_shift_px. Deprecated: /api/cmd/workstate and the `workstate` key are the old
        names of /api/cmd/config and `config`."""
        app = self.app
        cmd = CMD_NAMES.get(name, name.upper())
        confirmed = body.get("operator_confirmed")
        # only the commands that start something read the confirmation: a malformed one is
        # refused there; ABORT / DISARM / HOME / RESET ignore the field (never refused for it)
        if cmd in CONFIRMED_CMDS and confirmed is not None and not isinstance(confirmed, bool):
            raise HTTPError(400, "operator_confirmed must be a JSON boolean (true / false), got "
                                 f"{type(confirmed).__name__} {confirmed!r}")
        if cmd not in CONFIRMED_CMDS:
            confirmed = None
        if cmd == "HOME":           # the options travel with the request the bus accepts
            arg = {k: body[k] for k in ("set_reference", "tol_px", "mock_shift_px")
                   if body.get(k) is not None}
        elif cmd in ("SELECT_CONFIG", "SELECT_WORKSTATE"):
            arg = body.get("config", body.get("workstate"))
        elif cmd == "SELECT_PROFILE":
            arg = body.get("profile")
        elif cmd == "SELECT_MODE":
            arg = body.get("mode")
        elif cmd == "STEP":
            arg = body.get("step")
        elif cmd in CONFIRMED_CMDS:     # the configuration (+ profile) the sender expects,
            # and the zone selection it was shown (its stamp: checked for a configuration
            # whose job says "zone": "selection")
            arg = {"config": body.get("config", body.get("workstate")),
                   "profile": body.get("profile"), "zones": body.get("zones")}
        else:
            arg = None
        src = str(body.get("source") or "http")[:40]
        if cmd != "ABORT":          # ABORT settles a lost job itself (and wins over a queued HOME)
            app.bus_reconcile()
        busy = app.mode_jobs() if cmd == "SELECT_MODE" else []
        if busy:                    # its measurement / quick check is still on the rig
            r = Reply(False, cmd, app.bus.state,
                      f"refused: {', '.join(busy)} of {modes.TITLES[app.bus.mode]} mode is still "
                      "queued or running: wait for it, or ABORT")
        else:
            r = app.bus.command(cmd, source=src, arg=arg, confirmed=confirmed is True)
        if "workstate" in body and name != "workstate" and "deprecated" not in r.message:
            # the old body key (it named a station configuration, never a work state)
            r.message = (r.message + " " if r.message else "") + \
                "[the `workstate` key is deprecated: use `config`]"
        st = app.cmd_state()
        out = {**r.to_dict(), "bus": st, "jobs": st["jobs"],
               "active_preset": st["active_preset"]}
        self._json(out, 200 if r.accepted else 409)

    def r_modes(self, q, body):
        """The operating modes, the selected one and what their pickers offer."""
        self._json(jsonable(self.app.modes_info(), self.app.root))

    def _mode_job(self, verb: str, body: dict):
        args = {k: v for k, v in body.items() if k != "source" and v is not None}
        self._zone_as_shown(args)
        job = self.app.submit(verb, args)
        self._json({**self.app.job(job.id), "bus": self.app.cmd_state()})

    def _zone_as_shown(self, args: dict) -> None:
        """zone "selection@<stamp>" = "the zone selection exactly as my page showed it": a
        stamp that is not the stored selection's is refused at once (409, nothing queued; the
        job checks again when it runs). Only for the loaded preset: a job that names another
        one loads it first, and its own check decides."""
        from engine.zones import selection as zone_sel
        zone = args.get("zone")
        stamp = zone_sel.word_stamp(zone) if isinstance(zone, str) else None
        p = self.app.session.preset
        if not stamp or p is None or args.get("preset") not in (None, p.name):
            return
        st = self.app.zone_status(fresh=True)
        if st["stamp"] != stamp:
            raise HTTPError(409, f"the zone selection changed (it is now: {st['label']}): "
                                 "look at the page, then press it again")

    def r_measure(self, q, body):
        """MEASURE mode: one measurement (the measure_run verb's arguments) as a job ->
        {id, status, ...}; poll GET /api/jobs/<id>. 409 unless Measure mode is selected, while
        the zone painter holds the rig, and when zone "selection@<stamp>" names a selection
        that is not the stored one any more."""
        self._mode_job("measure_run", body)

    def r_quick_check(self, q, body):
        """QUICK CHECK mode: one quick check (the quick_check verb's arguments) as a job.
        409 unless Quick check mode is selected (and as for /api/measure)."""
        self._mode_job("quick_check", body)

    def r_zone_selection(self, q, body):
        """The zone selection of the loaded preset's rig placement: which painted zone sets
        (and zones) are switched on -> {selection: {state ok | problem | unknown, ok, empty,
        text, label, stamp, problems, sets [{set, on, ids, n_on, n_zones, zones, ...}],
        time, by} | null (no preset loaded)}. Files only."""
        self._json({"selection": self.app.zone_status(fresh=True)})

    def r_zone_selection_set(self, q, body):
        """Change the zone selection: {sets: [{set, on, ids}]} (the whole selection; ids
        null = every zone of the set), or {zones | on | off | clear} as the zone_select verb
        takes them; was: the stamp of the selection the sender was shown; source: who.
        Always answers {accepted, message, selection, bus} with the STORED selection, so a
        page redraws from it and never keeps a switch the server refused. 200 changed; 400
        a set / zone that does not resolve (nothing written); 409 the station is not idle, a
        job is queued or running, or `was` is not the stored selection any more.

        The answer is final: "not changed" is never followed by the change. A job ahead of
        it that ends within ZONE_SELECT_GRACE_S (the preset load of a configuration just
        selected) is waited for; one that holds the worker longer refuses the change (it
        would wait in the queue and be applied later, with nobody looking); one that still
        got stuck behind a job is taken out of the queue before the answer goes out."""
        app = self.app
        args = {k: body[k] for k in ("sets", "zones", "on", "off", "clear")
                if body.get(k) is not None}
        if not args:
            raise HTTPError(400, "send sets (the whole selection: [{set, on, ids}]) or zones / "
                                 "on / off / clear")
        if body.get("was") is not None:
            args["expect"] = str(body["was"])
        src = str(body.get("source") or "http")[:40]
        args["by"] = "operator page" if src == "quick" else src
        code, msg = 200, ""
        until = time.monotonic() + ZONE_SELECT_GRACE_S
        while app.jobs_ahead() and time.monotonic() < until:
            time.sleep(0.05)
        try:
            job = app.submit("zone_select", args, only_if_free=True)
        except HTTPError as e:
            if e.code != 409:
                raise
            code, msg = 409, e.msg
        else:
            j = app.wait(job.id, ZONE_SELECT_WAIT_S)
            if j["status"] == "queued":         # a job got in ahead of it: not later, not at all
                try:
                    app.cancel(job.id)
                except HTTPError:               # it started this instant: it ends in a moment
                    pass
                j = app.wait(job.id, ZONE_SELECT_WAIT_S)
            elif j["status"] == "running":      # files only: it ends in a moment
                j = app.wait(job.id, ZONE_SELECT_WAIT_S)
            if j["status"] == "error":
                kind = str(j.get("error_type") or "")
                code = 409 if kind in ("SelectionChanged", "Cancelled") else 400
                msg = str(j.get("error") or "refused")
                if kind == "Cancelled":
                    msg = ("zone_select refused: another job holds the scanner: wait for it to "
                           "end, or ABORT, then do it again")
                elif kind and msg.startswith(kind + ": "):  # the reason, in plain words
                    msg = msg[len(kind) + 2:]
            elif j["status"] != "done":         # still running after all that: said as not known
                code, msg = 409, ("the server is still writing the change: it is NOT KNOWN yet "
                                  "whether it was stored - look at the switches again")
        self._json({"accepted": code == 200, "message": msg,
                    "selection": app.zone_status(fresh=True), "bus": app.cmd_state()}, code)

    def r_run_file(self, q, body, run_id, name):
        self._file(self.app.run_file(run_id, name), RUN_FILES[name],
                   download=name.endswith(".ply"))

    def r_static(self, q, body, name):
        target = (STATIC_DIR / name).resolve()
        if not inside(STATIC_DIR, target) or target.suffix not in STATIC_TYPES:
            raise HTTPError(403, "forbidden")
        if not target.is_file():
            raise HTTPError(404, f"no static file {name}")
        self._file(target, STATIC_TYPES[target.suffix])

    def r_files(self, q, body, rel):
        """Library files only: resolve, require inside the root, whitelist types."""
        root = self.app.root
        try:
            target = (root / rel).resolve()
        except (OSError, ValueError):
            raise HTTPError(400, "bad path") from None
        if not inside(root, target):
            raise HTTPError(403, "path is outside the library")
        ctype = FILE_TYPES.get(target.suffix.lower())
        if ctype is None:
            raise HTTPError(403, f"file type {target.suffix or '(none)'} is not served")
        if not target.is_file():
            raise HTTPError(404, f"no file {rel}")
        self._file(target, ctype, download=q.get("download") in ("1", "true")
                   or target.suffix.lower() in DOWNLOAD_ONLY)

    # ------------------------------------------------------------- status
    def r_status(self, q, body):
        self._json(self.app.status())

    def r_verbs(self, q, body):
        self._json({"how": "POST /api/jobs {\"verb\": ..., \"args\": {...}} -> {id}; "
                           "poll GET /api/jobs/<id>. Pass ids from GET /api/library/<kind>.",
                    "mode": "mock" if self.app.mock else "real",
                    "verbs": [{"verb": v, "doc": d, "args": a} for v, (d, a) in VERBS.items()
                              if self.app.mock or not v.startswith("mock_")]})

    # --------------------------------------------------------------- jobs
    def r_jobs(self, q, body):
        try:
            n = max(1, min(int(q.get("n", 50)), MAX_JOBS_KEPT))
        except ValueError:
            raise HTTPError(400, "n must be an integer") from None
        self._json({"jobs": self.app.recent_jobs(n)})

    def r_job(self, q, body, jid):
        self._json(self.app.job(jid))

    def r_submit(self, q, body):
        verb = body.get("verb")
        args = body.get("args") or {}
        if not isinstance(verb, str) or not isinstance(args, dict):
            raise HTTPError(400, "body must be {\"verb\": str, \"args\": {...}}")
        job = self.app.submit(verb, args)
        wait = body.get("wait")
        if isinstance(wait, (int, float)) and wait > 0:
            return self._json(self.app.wait(job.id, float(wait)))
        self._json(self.app.job(job.id))

    def r_cancel(self, q, body, jid):
        self._json(self.app.cancel(jid).public(full=False))

    # --------------------------------------------------------------- live
    def r_live_jpg(self, q, body):
        self.app.live.touch()
        jpg = self.app.live.jpeg
        if jpg is None:
            raise HTTPError(404, "no live frame yet: start the live view")
        self._send(200, jpg, "image/jpeg")

    def r_live_state(self, q, body):
        self.app.live.touch()
        self._json(self.app.live.public())

    def r_live_start(self, q, body):
        args = {k: body[k] for k in ("pattern", "colour", "level", "period_px", "max_s")
                if body.get(k) is not None}
        try:
            self.app.live.update_settings({k: v for k, v in args.items() if k != "max_s"})
        except ArgError as e:
            raise HTTPError(400, str(e)) from None
        self.app.live.touch()
        self._json(self.app.job(self.app.submit("live_view", args).id))

    def r_live_settings(self, q, body):
        self.app.live.touch()
        try:
            self._json({"settings": self.app.live.update_settings(body)})
        except ArgError as e:
            raise HTTPError(400, str(e)) from None

    def r_live_stop(self, q, body):
        """Stops the running live view AND drops one still waiting in the queue."""
        n = self.app.stop_live("stopped by the operator")
        self._json({**self.app.live.public(), "stopped": n})

    # ------------------------------------------------------------ presets
    def r_presets(self, q, body):
        p = self.app.session.preset
        self._json({"presets": self.app.preset_names(), "active": p.name if p else None,
                    "selected": api.preset_active(), "tiers": self.app.preset_tiers(),
                    "dir": str(self.app.presets_path.resolve()),
                    "user_dir": str(config.user_presets_dir(self.app.root))})

    def r_preset_get(self, q, body, name):
        self._json(self.app.preset_detail(name))

    def r_preset_save(self, q, body):
        self._json(self.app.save_preset(body))

    def r_preset_load(self, q, body, name):
        self.app.check_preset_name(name)
        job = self.app.submit("load_preset", {"name": name})
        self._json(self.app.wait(job.id, float(body.get("wait", 30))))

    def r_zonesets(self, q, body):
        """The painted zone sets (engine/zones) at the loaded preset's geometry: {geometry_id,
        preset, sets [{name, zones [{id, name, on}], created, preview, camera, valid, on,
        ids_on, usable, rig_changes, used_by {recipes, stations}}], selection (as GET
        /api/zones/selection), others [{geometry_id, n_sets, presets}], other_geometries}."""
        self._json(self.app.zone_sets())

    def r_recipes(self, q, body):
        self._json(jsonable({"recipes": api.recipe_list(), "selected": api.recipe_active(),
                             "checks": api.recipe_checks(),
                             "dir": str(api.paths()["recipes"]),
                             "user_dir": str(api.paths()["user_recipes"])}, self.app.root))

    def r_recipe_get(self, q, body, name):
        if not _NAME_RE.match(name or ""):
            raise HTTPError(404, f"no recipe named {name!r}")
        try:
            self._json(jsonable(api.recipe_show(name), self.app.root))
        except Exception as e:          # noqa: BLE001 - unknown / invalid: say why
            raise HTTPError(404, f"{type(e).__name__}: {e}") from None

    # ------------------------------------------------------------ library
    def r_library(self, q, body, kind):
        try:
            payload, _ = self.app.listing(kind, q.get("geometry_id") or None)
        except HTTPError:
            raise
        except Exception as e:          # noqa: BLE001 - e.g. a kind still being built
            payload = {"kind": kind, "items": [], "error": f"{type(e).__name__}: {e}",
                       "error_type": type(e).__name__, "hint": error_hint(e)}
        self._json(payload)

    def r_outputs(self, q, body):
        kind = q.get("kind") or None
        if kind is not None and kind not in OUTPUT_KINDS:
            raise HTTPError(400, f"kind must be one of {OUTPUT_KINDS}")
        payload, _ = self.app.listing("outputs", kind)
        payload["paths"] = {k: self.app._snapshot.get("paths", {}).get(k)
                            for k in ("library", "output", "output_index")}
        self._json(payload)

    def r_pointcloud(self, q, body, oid):
        try:
            n = int(q.get("max", 20000))
        except ValueError:
            raise HTTPError(400, "max must be an integer") from None
        self._json(self.app.pointcloud(oid, n))

    def r_log(self, q, body):
        """?n=200&exclude=list_library,ui_live_view: filtered BEFORE the cut."""
        try:
            n = max(1, min(int(q.get("n", 200)), 5000))
        except ValueError:
            raise HTTPError(400, "n must be an integer") from None
        exclude = [v for v in re.split(r"[,\s]+", q.get("exclude") or "") if v]
        self._json(self.app.log_tail(n, exclude))


# ================================================================== main

def make_server(*, mock: bool, library=None, presets_dir=None, preset: Optional[str] = None,
                port: int = DEFAULT_PORT, verbose: bool = False, stations_dir=None):
    """-> (httpd, app). The worker is started (engine configured) before the
    socket accepts anything, so no request can see an unconfigured engine."""
    app = App(mock=mock, library=library, presets_dir=presets_dir, preset_name=preset,
              stations_dir=stations_dir).start()
    try:
        httpd = bind(app, port, verbose)
    except OSError:
        app.close()
        raise
    return httpd, app


def bind(app: App, port: int, verbose: bool = False) -> UIServer:
    """Open the socket for a started App (OSError: the port is taken / not allowed)."""
    httpd = UIServer((BIND, port), Handler)
    httpd.app, httpd.verbose = app, verbose
    app.port = httpd.server_address[1]
    # a RigBusy message elsewhere (CLI, another UI) names this server as the holder
    app.session.holder = f"web UI (port {app.port})"
    return httpd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ui.web.server",
                                 description="Swiss Army Scanner local web UI (127.0.0.1 only)")
    ap.add_argument("--mock", action="store_true",
                    help="simulated bench, no hardware (default: REAL rig)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--library", help="library root (default: engine default / "
                                      "SWISS_SCANNER_LIBRARY)")
    ap.add_argument("--presets", help="presets folder (default: <repo>/presets)")
    ap.add_argument("--preset", help=f"preset to load at start (default: {DEFAULT_PRESET})")
    ap.add_argument("--stations", help="station configs folder for the operator page /quick "
                                       "(default: <repo>/stations)")
    ap.add_argument("--open", action="store_true", help="open the browser")
    ap.add_argument("--verbose", action="store_true", help="log every HTTP request")
    args = ap.parse_args(argv)

    # an engine start-up failure (library folder, preset...) is reported as itself;
    # only the socket bind means "is another UI already running?"
    app = App(mock=args.mock, library=args.library, presets_dir=args.presets,
              preset_name=args.preset, stations_dir=args.stations).start()
    try:
        httpd = bind(app, args.port, args.verbose)
    except OSError as e:
        app.close()
        print(f"could not listen on {BIND}:{args.port}: {e} (is another UI already running?)")
        return 2
    url = f"http://{BIND}:{app.port}/"
    print("Swiss Army Scanner web UI")
    print(f"  mode    : {'MOCK (simulated bench)' if app.mock else 'REAL HARDWARE'}")
    print(f"  library : {app.root}")
    print(f"  url     : {url}   (Ctrl+C to stop)")
    print(f"  operator: {url}quick   (START / ABORT / HOME)")
    if not app.mock:
        print("  REAL rig: close the LightCrafter GUI first. Nothing touches the rig until you")
        print("            run an action in the page. Don't move camera/projector between base")
        print("            and object captures.")
    for w in app.startup_warnings:
        print(f"  warning : {w}")
    sys.stdout.flush()
    if args.open:
        threading.Timer(0.6, webbrowser.open, (url,)).start()
    try:
        httpd.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("stopping...")
    finally:
        httpd.server_close()
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
