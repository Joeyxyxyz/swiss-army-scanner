"""
The painter's window (a thin shell around painter.PainterModel) and the rig side of zone sets.

    paint()        the rig opens (real: the machine-wide rig lock FIRST - RigBusy and nothing
                   is shown when another program has the rig), the painter is drawn on the DMD,
                   window events -> PainterModel -> render -> window, until Space (save) or
                   Esc (cancel). MOCK: headless, from a scripted event list.
    run_painter()  that loop alone, on any window with show() / poll(); it refuses a real
                   window that is not inside the rig lock
    show()         project a saved set: each zone lit on its own (where does it land in the
                   camera image?), then the outlines + ids held for the operator, photographed

Polarity: like the grid and the zone photos (engine/capture/zones.py) the painter is sent RAW
after a raw black vs raw white measurement, inverted when the HDMI path lands inverted, so a
bright outline is PHYSICALLY bright on the part. That one measurement shows a lit field for
one camera hold before the drawing starts (polarity="normal" / "inverted" skips it when the
operator knows); the drawing itself is thin lines on black. A measurement that cannot tell
(raw white and raw black look the same to the camera) REFUSES the session: guessing wrong
would hold a lit field on the part for the whole session.

Keys: a real window's show() pumps the window too, so a key pressed while the picture is being
redrawn (the mouse is moving) is kept by the window and handed out by its next poll()
(engine/capture/projector.py HDMIWindow) - never dropped.

The painter holds the rig (and its lock) while it is open: it gives up after idle_timeout_s
without input, saving nothing, so a forgotten session never blocks the line.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .. import config, rules
from ..library.store import json_safe, safe_label
from . import painter as P
from . import store as zstore
from . import use
from .model import DMD_SHAPE, ZoneError, ZoneSet, valid_set_name, valid_zone_name

POLL_MS = 15                    # the window pump between two looks at the events
IDLE_TIMEOUT_S = 600.0          # no input for this long: the session ends, nothing saved
MAX_IDLE_TIMEOUT_S = 3600.0
POLARITIES = ("measure", "normal", "inverted")
MIN_LANDING_PX = 16             # camera px a zone must light to count as seen
LANDING_FRACTION = 0.5          # ... each at least this much of its full-flood brightness
SHOW_EXT = ".zoneshow.json"
UNITS = {"dmd_px": "DMD mirrors of the zone", "camera_px": "camera px the zone lights",
         "centroid_px": "camera px [x, y]",
         "bbox_px": "camera px [x0, y0, x1, y1] (x1, y1 exclusive)",
         "zone coordinates": "projector px [col, row]"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sent(img: np.ndarray, inverted: bool) -> np.ndarray:
    return (255 - img) if inverted else img


def run_painter(model: P.PainterModel, window, *, mock: bool, rig_locked: bool = False,
                inverted: bool = False, twice: bool = False, cancel=None,
                idle_timeout_s: float = IDLE_TIMEOUT_S, clock=time.monotonic) -> str:
    """Pump the painter until it is done -> "save" | "cancel" (model.state.message says why).

    window: anything with show(img) and poll(ms) -> events (HDMIWindow / MockWindow); a poll
    that returns None means the scripted session has ended (cancelled: no Space, no save).
    Real mode (mock False) is refused unless the caller holds the rig lock (rig_locked: paint()
    runs this inside ScanSession.rig, which takes the lock before anything is opened)."""
    if not mock and not rig_locked:
        raise rules.RuleViolation(
            "the zone painter draws on the real DMD: it only runs while this process holds "
            "the rig lock (engine/capture/riglock.py). Start it through `scan zone-paint` / "
            "api.zone_paint, which take the lock first")
    if not callable(getattr(window, "poll", None)) or not callable(getattr(window, "show", None)):
        raise rules.RuleViolation("the painter needs a projector window with show() and poll()")

    def show() -> None:
        rules.check_scan_path_usb_free()
        sent = _sent(P.render(model.state), inverted)
        window.show(sent)
        if twice:                              # the DMD shows HDMI one frame behind
            window.show(sent)
        model.dirty = False

    show()
    last = clock()
    while model.state.done is None:
        if cancel is not None and cancel.is_set():
            model.cancel("aborted: nothing saved")
            break
        events = window.poll(POLL_MS)
        if events is None:
            model.cancel("the scripted session ended without Space (confirm): nothing saved")
            break
        for ev in events:
            model.handle(ev)
        now = clock()
        if events:
            last = now
        elif now - last > idle_timeout_s:
            model.cancel(f"no input for {idle_timeout_s:g} s: the rig was released, nothing saved")
            break
        if model.dirty:
            show()
    show()
    return model.state.done


def _polarity(sync, mode: str, warnings: list) -> bool:
    """Is the raw HDMI path inverted? Measured (raw black vs raw white, as the grid does),
    or as the operator says. A measurement that cannot tell is a ZoneError: the painter
    never starts on a guess (inverted, its whole picture would be a lit field)."""
    if mode != "measure":
        return mode == "inverted"
    from ..capture import zones as gz
    f0 = sync.shoot(np.zeros(DMD_SHAPE, np.uint8), raw=True)
    f1 = sync.shoot(np.full(DMD_SHAPE, 255, np.uint8), raw=True)
    diff = float(np.mean(f1)) - float(np.mean(f0))
    if abs(diff) < gz.POLARITY_MARGIN:
        raise ZoneError(f"the painter did not start: raw white and raw black look the same to "
                        f"the camera ({diff:+.1f} counts), so the polarity of the HDMI path "
                        "could not be measured - is the projector on and aimed at the scene? "
                        "Inverted, the painter would hold a lit field on the part; when you "
                        "know which way it lands, say so (polarity normal | inverted)")
    if diff < 0:
        warnings.append("raw HDMI path measured INVERTED; the painter was sent inverted so "
                        "it is physically bright-on-dim")
    return diff < 0


def _idle(v) -> float:
    try:
        t = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"idle timeout must be seconds (a number), got {v!r}") from None
    if not 1.0 <= t <= MAX_IDLE_TIMEOUT_S:
        raise ValueError(f"idle timeout must be 1..{MAX_IDLE_TIMEOUT_S:g} s (the painter holds "
                         f"the rig while it waits), got {v!r}")
    return t


def paint(s, preset, name: str, *, source: Optional[str] = None, events=None,
          replace: bool = False, labels=None, polarity: str = "measure", note: str = "",
          idle_timeout_s: float = IDLE_TIMEOUT_S, cancel=None) -> dict:
    """One painter session -> {"saved": bool, ...}. Saved: the new library entry
    (zones/<geometry_id>/<name>/: zones.json, masks.npz, preview.png, camera.png). Not saved
    (Esc, an abort, the idle timeout, a script without Space): "reason" says why and nothing
    was written.

    source: a saved set at this geometry to start from (--from). events: MOCK ONLY, the
    scripted session (painter.check_events); on the real rig the operator draws. Everything
    that can be refused is refused BEFORE the window opens: the name, an existing set
    without replace, a set of another geometry, the LED duration rule, and (real) a busy rig."""
    from ..decode import projmap
    if not valid_set_name(name):
        raise ZoneError(f"zone set name must be letters, digits, _ . - (max 64), got {name!r}")
    if polarity not in POLARITIES:
        raise ValueError(f"polarity must be one of {', '.join(POLARITIES)}, got {polarity!r}")
    idle = _idle(idle_timeout_s)
    projmap.dmd_shape_of(preset)
    if cancel is not None and not callable(getattr(cancel, "is_set", None)):
        raise ValueError("cancel must be a threading.Event (or anything with is_set())")
    if zstore.exists(s.store, preset.geometry_id, name) and not replace:
        raise ZoneError(f"zone set {name!r} exists at geometry {preset.geometry_id}: pick "
                        "another name, or replace it (--replace: the old one goes to trash/); "
                        f"--from {name} starts from its zones")
    start = use.for_preset(s.store, preset, source).zones if source else []
    if s.mock:
        if events is None:
            raise ValueError("MOCK: the painter runs headless from a scripted event list "
                             "(events / `--events FILE.json`; `--events demo` is a built-in one)")
        events = P.check_events(list(events))
    elif events is not None:
        raise ValueError("a scripted event list is MOCK ONLY: on the real rig the operator draws")
    labels = tuple(labels) if labels else P.LABELS
    for lab in labels:                          # each one is a zone's name: checked here, not
        if not valid_zone_name(lab):            # when Space refuses to save an hour later
            raise ValueError(f"labels must be zone names: letters, digits, space, _ . - (max "
                             f"32 characters), got {lab!r}")
    rules.check_led_duration(preset.projection.led, None, "the zone painter")
    # the grid underlay (G) is the grid `--zone C4` means at this geometry: the spacing of
    # the grid last projected for it, as engine/capture/zones.resolve reads a cell name
    from ..capture import zones as gz
    spacing = gz.last_grid_spacing(s, preset.geometry_id) or gz.DEFAULT_SPACING
    model = P.PainterModel(start, name=name, labels=labels, grid_spacing=spacing)
    warnings: list = []
    photo = zs = None
    t0 = time.perf_counter()
    with s.rig(preset) as sync:                 # real: the machine-wide rig lock first
        window = sync.projector.window
        inverted = _polarity(sync, polarity, warnings)
        window.enable_mouse()
        if s.mock:
            window.feed(events)
        done = run_painter(model, window, mock=s.mock, rig_locked=True, inverted=inverted,
                           twice=bool(preset.projection.push_twice), cancel=cancel or s.cancel,
                           idle_timeout_s=idle)
        if done == "save":
            zs = ZoneSet(name=name, geometry_id=preset.geometry_id, zones=model.zones(),
                         fingerprint=config.geometry_fingerprint(preset), created=_now(),
                         preset=preset.name, note=str(note or ""), mock=bool(s.mock),
                         polarity_inverted=inverted, source=source).check()
            from ..capture.zones import _photo_u8
            try:                                # the zones on the part, as the camera sees them
                photo = _photo_u8(sync.shoot(_sent(zstore.outline_image(zs), inverted), raw=True))
            except Exception as e:              # noqa: BLE001 - the operator's work comes first
                warnings.append(f"no photo of the zones on the part ({type(e).__name__}: {e}); "
                                "the set was saved without one")
            # saved while the rig is still held: a problem closing it never loses the drawing
            zs = zstore.save(s.store, zs, replace=replace, photo=photo)
    out = {"name": name, "geometry_id": preset.geometry_id, "preset": preset.name,
           "mock": bool(s.mock), "polarity_inverted": inverted, "source": source,
           "grid_spacing_px": int(spacing),
           "session_s": round(time.perf_counter() - t0, 2), "warnings": warnings}
    if zs is None:
        return {**out, "saved": False, "reason": model.state.message,
                "n_zones": len(model.state.zones)}
    return {**out, "saved": True, "reason": None, **zs.summary(),
            "zones_json": zs.path / zstore.ZONES_FILE, "masks": zs.path / zstore.MASKS_FILE,
            "preview": zs.path / zstore.PREVIEW_FILE,
            "camera": (zs.path / zstore.PHOTO_FILE) if photo is not None else None,
            "spec": use.spec_text(zs.name)}


# ------------------------------------------------------------------ zone-show

def _landing(frame: np.ndarray, dark: np.ndarray, contrast: np.ndarray, floor: float) -> dict:
    """Where one lit zone lands: the camera px that got at least LANDING_FRACTION of what a
    full flood gives them (and that a flood lights at all)."""
    lit = (((np.asarray(frame, np.float32) - dark) > LANDING_FRACTION * contrast)
           & (contrast > floor))
    n = int(lit.sum())
    if n < MIN_LANDING_PX:
        return {"camera_px": n, "seen": False, "centroid_px": None, "bbox_px": None,
                "at_image_edge": False, "mask": lit}
    ys, xs = np.nonzero(lit)
    H, W = lit.shape
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    return {"camera_px": n, "seen": True,
            "centroid_px": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
            "bbox_px": box, "mask": lit,
            "at_image_edge": bool(box[0] <= 0 or box[1] <= 0 or box[2] >= W or box[3] >= H)}


def show(s, preset, zs: ZoneSet, ids=None, hold_s: float = 5.0, label: Optional[str] = None,
         cancel=None) -> dict:
    """Project a saved zone set and report where each zone lands in the camera image.

    Raw black + raw white (polarity, and what a flood gives each camera px), each selected
    zone lit on its own (its camera px, centroid, box; seen / not seen), then the outlines +
    ids, photographed and held for hold_s (pumped, in slices; `cancel` ends it early).
    -> {"zones": [...], "all_seen", camera / overlay / DMD PNGs + a JSON record in
    output/previews, warnings}."""
    import cv2
    from ..capture import zones as gz
    from ..decode import projmap
    label = safe_label(label or f"zones_{zs.name}")
    hold = gz._hold(hold_s, preset, "zone-show")
    gz._check_cancel(cancel)
    projmap.dmd_shape_of(preset)
    rules.require_same_geometry(zs.geometry_id, preset.geometry_id, f"zone set {zs.name}")
    rules.require_same_fingerprint(zs.fingerprint, config.geometry_fingerprint(preset),
                                   f"zone set {zs.name}")
    chosen = zs.select(ids)
    masks = {z.id: z.mask(DMD_SHAPE) for z in chosen}
    outline = zstore.outline_image(zs, ids)
    warnings: list = []
    stale = zstore.stale_warning(s.store, zs)
    if stale:
        warnings.append(stale.split(": check them")[0] + ": look at where they land now")
    lands: dict = {}
    held = 0.0
    with s.rig(preset) as sync:
        f0 = sync.shoot(np.zeros(DMD_SHAPE, np.uint8), raw=True)
        f1 = sync.shoot(np.full(DMD_SHAPE, 255, np.uint8), raw=True)
        diff = float(f1.mean()) - float(f0.mean())
        inverted = diff < -gz.POLARITY_MARGIN
        if abs(diff) < gz.POLARITY_MARGIN:
            warnings.append(f"raw white and raw black look the same to the camera ({diff:+.1f} "
                            "counts): is the projector on and aimed at the scene?")
        elif inverted:
            warnings.append("raw HDMI path measured INVERTED; the zones were sent inverted so "
                            "they are physically bright-on-dim")
        bright, dark = (f0, f1) if inverted else (f1, f0)
        contrast = np.asarray(bright, np.float32) - np.asarray(dark, np.float32)
        floor = max(2.0 * gz.POLARITY_MARGIN, 0.1 * float(np.percentile(contrast, 99.5)))
        for z in chosen:
            phys = np.where(masks[z.id], np.uint8(255), np.uint8(0))
            lands[z.id] = _landing(sync.shoot(_sent(phys, inverted), raw=True),
                                   np.asarray(dark, np.float32), contrast, floor)
        photo = sync.shoot(_sent(outline, inverted), raw=True)
        if hold > 0:
            held = gz._pumped_hold(sync.projector.window, hold, cancel)
            if held < hold - 1e-9:
                warnings.append(f"hold cancelled after {held:.2f} of {hold:g} s")

    store = s.store
    photo8 = gz._photo_u8(photo)
    over = np.repeat((gz._photo_u8(bright) * gz.OVERLAY_TONE).astype(np.uint8)[..., None], 3,
                     axis=2)
    zones = []
    for z in chosen:
        ld = lands[z.id]
        m = ld.pop("mask")
        if ld["seen"]:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, cs, -1, gz.LINE_BGR, 2)
            gz._put_label(over, z.id + (f" {z.name}" if z.name else ""), ld["centroid_px"][0],
                          ld["centroid_px"][1], 0.8, 2)
            if ld["at_image_edge"]:
                warnings.append(f"zone {z.id} reaches the edge of the camera image: part of it "
                                "may be outside the camera's view")
        else:
            warnings.append(f"zone {z.id} does not land in the camera image ({ld['camera_px']} "
                            "px lit): outside the camera's view, in shadow, or on nothing")
        zones.append({"id": z.id, "name": z.name, "shape": z.shape,
                      "dmd_px": int(masks[z.id].sum()), **ld})
    camera_png = gz._save(store, preset, label, "camera", photo8)
    overlay_png = gz._save(store, preset, label, "overlay", over, source=zs.path)
    dmd_png = gz._save(store, preset, label, "dmd", outline)
    doc = {"kind": "zoneshow", "label": label, "zoneset": zs.name, "geometry_id": zs.geometry_id,
           "preset": preset.name, "mock": bool(s.mock), "created": _now(),
           "zoneset_created": zs.created, "zones": zones,
           "all_seen": all(z["seen"] for z in zones), "polarity_inverted": inverted,
           "hold_s": hold, "held_s": round(held, 3), "hold_cancelled": held < hold - 1e-9,
           "camera_png": camera_png, "overlay_png": overlay_png, "dmd_png": dmd_png,
           "units": UNITS, "warnings": warnings}
    out = store.output_path("previews", label, SHOW_EXT)
    store.write_json(out, json_safe(doc))
    store.record_output("previews", out, label=label, geometry_id=preset.geometry_id,
                        source=zs.path)
    return {**doc, "json_path": out}
