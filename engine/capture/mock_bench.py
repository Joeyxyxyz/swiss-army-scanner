"""
MockBench — a simulated rig with real parallax, for mock mode.

The swiss_army_sensor MockCameraDriver multiplies the DMD pattern by a
flat scene, so it has no parallax and relief is ~0 (first_scan_v10's own
note). This bench adds the missing geometry so mock scans produce real
relief, shadows and dimensions:

  * camera px -> base plane: orthographic, `cam_px_per_mm` px per mm
  * base plane -> projector: the lit patch maps linearly onto the DMD
  * a surface at height h shifts the projector column by
    `relief_px_per_mm * h` (triangulation parallax along x)
  * each box casts a shadow of length `shadow_ratio * h` mm on the base
    along +x (projector-side occlusion) -> invalid pixels, like the bench

Defaults put the §11 validation object (H 106.95 mm) at ~48 px of relief,
matching the real bench scan (relief range ≈ ±48 px).

Colour (spectral mode). The LightCrafter has separate R/G/B LEDs and, in
HDMI video mode, shows each colour channel of the image in its own LED
sub-frame; the mono camera integrates all three. So a (H, W, 3) BGR
pattern renders as

    signal = gain * sum_c  w_c * reflectance_c * pattern_c / 255
    w = led_power / sum(led_power)          (R, G, B; default 1.0, 1.0, 0.8)

`gain` stays "counts for a full-on WHITE frame at albedo 1": white is the
sum of the three sub-frames, so the weights sum to 1, a pure-red flood
gives ~1/3 of the white signal (the red LED is only on for its
sub-frame), R + G + B floods add up to the white flood, and a 3-channel
pattern with equal channels renders exactly like the 2-D gray pattern.
2-D (gray) patterns take the original code path unchanged.

Per-band reflectance is optional: Box.rgb, Label.rgb / Label.ink_rgb,
MockBench.base_rgb, each (r, g, b) in [0, 1]. None = neutral (the gray
albedo in every band). When rgb is given, the surface's gray albedo (what
2-D patterns see) is the LED-weighted mean w . rgb, so white = R + G + B
stays true; Box.albedo is then ignored.

`global_light` (default 0 = off) adds a crude global component for the
direct/global separation tests: every lit-patch pixel receives
global_light x its own reflectance x the mean DMD drive level (uniform
scatter / inter-reflection). It is low-frequency by construction, which
is the assumption Nayar et al. 2006 rely on.

Stereo mode (metric calibration layer). Placing a stereo shape
(capture/mock_stereo: Plane, Sphere, MetricCylinder, Block, CharucoTarget, EndFace)
switches the bench to an explicit pinhole camera (K, distortion) + pinhole
diamond-DMD projector (native col/row, f_v ~ 2 f_u, half-column row
parity) with the rig's R, t (projector 279.4 mm right, wall at 823 mm,
~18.8 deg), ray-cast in mm. `stereo_truth()` is what calibration must
recover. Placing an orthographic object (Box, Cylinder, Label) switches
back; every existing test scene is unaffected.

Hooks for the end-face tests (stereo mode): a DIM face is an EndFace with a low albedo, a
SPECULAR one (bare metal that mirrors the projector into the camera) an EndFace with
specular > 0 (mock_stereo: a lobe about the projector / camera bisector), an edge dent an
EndFace with rim_notches; zoom_lens(factor) changes the camera's focal length AFTER a
calibration was stored - what a lens whose zoom ring was turned looks like to the engine
(every lateral size reads x factor, nothing else says so).

In mock mode the bench stands in for the human's physical steps: tests
and agents call `place()` / `clear()` instead of touching real objects.
Change led_power / base_* before placing objects (the scene is cached
until the next place()/clear()).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..decode.deformation import DeformationField


@dataclass
class Box:
    h_mm: float
    w_mm: float
    l_mm: float
    center_mm: tuple = (0.0, 0.0)   # on the base plane, relative to field centre
    angle_deg: float = 0.0          # rotation of the W axis from camera +x
    albedo: float = 0.7
    rgb: Optional[tuple] = None     # per-band (r, g, b) reflectance; None = neutral albedo


@dataclass
class Cylinder:
    """A standing cell seen side-on by the frontal (wall-facing) rig: its axis
    runs along camera y, the wall behind it is the base plane. Height toward
    the camera is axis_standoff + sqrt(r^2 - dx^2) minus any dents.

    radius_mm, height_mm (extent along the axis = camera y), center_mm (axis
    x, cell mid-height y on the base plane). axis_standoff_mm: axis distance
    from the wall (None = radius: the can's back touches the wall).
    max_view_deg: surface steeper than this (|theta| from the camera-facing
    line) returns no data, as grazing faces do on the rig -> an open arc of
    2*max_view_deg, never 180+.
    dents: list of dicts {depth_mm, size_mm, angle_deg, z_mm}; angle from the
    camera-facing line, + toward camera +x; z_mm from the cell bottom (image
    +y end) along the axis. Each is a cosine-bowl radial dent of diameter size_mm.
    """
    radius_mm: float
    height_mm: float
    center_mm: tuple = (0.0, 0.0)
    axis_standoff_mm: Optional[float] = None
    max_view_deg: float = 60.0
    dents: list = field(default_factory=list)
    albedo: float = 0.7
    rgb: Optional[tuple] = None

    def surface(self, x_mm: np.ndarray, y_mm: np.ndarray):
        """-> (height_mm, footprint, seen) on the base-plane grid (mm, field-centred)."""
        R = float(self.radius_mm)
        dx = x_mm - self.center_mm[0]
        dy = y_mm - self.center_mm[1]
        fp = (np.abs(dx) <= R) & (np.abs(dy) <= self.height_mm / 2)
        s = np.clip(dx / R, -1.0, 1.0)
        theta = np.arcsin(s)
        dr = np.zeros_like(x_mm, dtype=np.float64)
        z = self.height_mm / 2 - dy                       # from the cell bottom
        for d in self.dents:
            rad = float(d["size_mm"]) / 2
            arc = R * (theta - math.radians(float(d.get("angle_deg", 0.0))))
            q = np.hypot(arc, z - float(d.get("z_mm", self.height_mm / 2))) / rad
            dr = np.maximum(dr, np.where(q < 1, float(d["depth_mm"]) * 0.5 * (1 + np.cos(np.pi * q)), 0))
        a = R if self.axis_standoff_mm is None else float(self.axis_standoff_mm)
        h = np.where(fp, a + (R - dr) * np.cos(theta), 0.0)
        seen = fp & (np.abs(theta) <= math.radians(self.max_view_deg))
        return h.astype(np.float32), fp, seen


@dataclass
class Label:
    """A printed QR label (reflectance only, no height), for code reading."""
    data: str
    center_mm: tuple = (0.0, 0.0)
    size_mm: float = 30.0            # side of the symbol incl. quiet zone
    dark: float = 0.08
    light: float = 0.85
    rgb: Optional[tuple] = None      # paper (light modules) per-band reflectance
    ink_rgb: Optional[tuple] = None  # ink (dark modules) per-band reflectance

    @property
    def coloured(self) -> bool:
        return self.rgb is not None or self.ink_rgb is not None

    def _light_modules(self) -> np.ndarray:
        import cv2
        q = cv2.QRCodeEncoder.create().encode(self.data)       # 0/255 modules
        q = np.pad(q, 4, constant_values=255)                   # quiet zone
        return q > 127

    def texture(self) -> np.ndarray:
        return np.where(self._light_modules(), self.light, self.dark).astype(np.float32)

    def texture_rgb(self) -> np.ndarray:
        """(n, n, 3) per-band reflectance, R, G, B order."""
        paper = np.asarray(self.rgb if self.rgb is not None else (self.light,) * 3, np.float32)
        ink = np.asarray(self.ink_rgb if self.ink_rgb is not None else (self.dark,) * 3,
                         np.float32)
        return np.where(self._light_modules()[..., None], paper, ink).astype(np.float32)


# §11: the known object on the bench
VALIDATION_OBJECT = Box(h_mm=106.95, w_mm=56.50, l_mm=21.90, center_mm=(-20.0, 0.0))


@dataclass
class MockBench:
    cam_shape: tuple = (1080, 1440)     # (H, W) — Blackfly S IMX273
    dmd_shape: tuple = (1140, 912)      # (H, W) — DLP4500
    cam_px_per_mm: float = 3.0
    relief_px_per_mm: float = 0.45
    shadow_ratio: float = 0.18
    lit_fraction: float = 0.8           # fraction of the camera frame the patch covers
    base_albedo: float = 0.6
    gain: float = 380.0                 # camera counts for full-on DMD at albedo 1
    ambient: float = 12.0
    corr_noise_px: float = 0.25
    seed: int = 0
    objects: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    base_rgb: Optional[tuple] = None    # per-band base reflectance; None = neutral base_albedo
    led_power: tuple = (1.0, 1.0, 0.8)  # relative R, G, B LED output seen by the mono camera
    global_light: float = 0.0           # uniform global (scatter) light; 0 = none
    stereo_rig: Optional[object] = None  # mock_stereo.StereoRig; None = the bench default
    stereo_shapes: list = field(default_factory=list)
    stereo_mode: bool = False
    stereo_wall: bool = True            # the wall plane (base) behind stereo shapes

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        h, w = self.cam_shape
        self._v, self._u = np.mgrid[0:h, 0:w].astype(np.float32)
        self._cache = None
        self._refl = None               # (H, W, 3) R, G, B reflectance, colour scenes only
        self._lit = None
        self._stereo = None

    def led_weights(self) -> np.ndarray:
        """R, G, B share of the white frame (sums to 1)."""
        p = np.asarray(self.led_power, np.float64)
        if p.shape != (3,) or np.any(p < 0) or p.sum() <= 0:
            raise ValueError(f"led_power must be three non-negative values, got {self.led_power}")
        return (p / p.sum()).astype(np.float32)

    def gray_albedo(self, rgb) -> float:
        """What a 2-D (all-LED) pattern sees of a surface with per-band reflectance rgb."""
        return float(np.dot(self.led_weights(), np.asarray(rgb, np.float32)))

    def _colour_scene(self) -> bool:
        return (self.base_rgb is not None or any(o.rgb is not None for o in self.objects)
                or any(lb.coloured for lb in self.labels))

    # ------------------------------------------------ the human's steps
    def place(self, obj) -> None:
        from .mock_stereo import STEREO_SHAPES
        if isinstance(obj, STEREO_SHAPES):
            self.stereo_shapes.append(obj)
            self.stereo_mode = True
            self._stereo = None
            return
        self.stereo_mode = False
        (self.labels if isinstance(obj, Label) else self.objects).append(obj)
        self._cache = None

    def clear(self) -> None:
        """Remove every object; a stereo bench stays in stereo mode (empty wall)."""
        self.objects.clear()
        self.labels.clear()
        self.stereo_shapes.clear()
        self._stereo = None
        self._cache = None

    # --------------------------------------------------- stereo (metric)
    def use_stereo(self, on: bool = True) -> None:
        """Switch the pinhole stereo model on (an empty wall until shapes are placed)."""
        self.stereo_mode = bool(on)
        self._stereo = None

    def rig_model(self):
        from .mock_stereo import StereoRig
        if self.stereo_rig is None:
            self.stereo_rig = StereoRig.bench()
        return self.stereo_rig

    def stereo_scene(self):
        from .mock_stereo import Plane, StereoScene
        if getattr(self, "_stereo", None) is None:
            wall = Plane(albedo=self.base_albedo) if self.stereo_wall else None
            self._stereo = StereoScene(self.rig_model(), list(self.stereo_shapes), wall=wall,
                                       gain=self.gain, ambient=self.ambient)
        return self._stereo

    def stereo_truth(self) -> dict:
        """Ground-truth K / distortion / R / t / parity the calibration must recover."""
        return self.rig_model().truth()

    def zoom_lens(self, factor: float) -> None:
        """Turn the camera's zoom ring: its focal length x factor from now on (the scene is
        re-rendered; a stereo calibration stored BEFORE still holds the old lens, which is
        the point: the engine must notice that sizes no longer come out right)."""
        from .mock_stereo import PinholeCamera, StereoRig
        factor = float(factor)
        if not factor > 0:
            raise ValueError(f"zoom factor must be > 0, got {factor!r}")
        rig = self.rig_model()
        K = np.array(rig.camera.K, np.float64, copy=True)
        K[0, 0] *= factor
        K[1, 1] *= factor
        self.stereo_rig = StereoRig(PinholeCamera(K, np.array(rig.camera.dist, copy=True),
                                                  tuple(rig.camera.size)),
                                    rig.projector, rig.R, rig.t)
        self._stereo = None

    @property
    def truth(self) -> dict:
        """Ground truth a calibration should recover."""
        return {"mm_per_relief_px": 1.0 / self.relief_px_per_mm,
                "cam_px_per_mm": self.cam_px_per_mm}

    # ------------------------------------------------------- geometry
    def _scene(self):
        if self._cache is not None:
            return self._cache
        H, W = self.cam_shape
        cu, cv = W / 2.0, H / 2.0
        k = self.cam_px_per_mm
        height = np.zeros(self.cam_shape, np.float32)
        base_gray = self.base_albedo if self.base_rgb is None else self.gray_albedo(self.base_rgb)
        albedo = np.full(self.cam_shape, base_gray, np.float32)
        # per-band reflectance only for colour scenes, so gray scenes cost nothing extra
        refl = None
        if self._colour_scene():
            refl = np.empty(self.cam_shape + (3,), np.float32)
            refl[...] = self.base_rgb if self.base_rgb is not None else (self.base_albedo,) * 3
        shadow = np.zeros(self.cam_shape, bool)
        cols = np.arange(W, dtype=np.float64)
        unseen = np.zeros(self.cam_shape, bool)
        for b in self.objects:
            if isinstance(b, Cylinder):
                ch, fp, seen = b.surface((self._u - cu) / k, (self._v - cv) / k)
                height = np.maximum(height, ch)
                albedo = np.where(fp, b.albedo if b.rgb is None else self.gray_albedo(b.rgb), albedo)
                if refl is not None:
                    refl[fp] = b.rgb if b.rgb is not None else (b.albedo,) * 3
                unseen |= fp & ~seen
                last = np.maximum.accumulate(np.where(fp, cols, -np.inf), axis=1)
                dist = cols - last
                hmax = float(ch.max())
                shadow |= (~fp) & (dist > 0) & (dist <= self.shadow_ratio * hmax * k)
                continue
            dx = (self._u - cu) / k - b.center_mm[0]
            dy = (self._v - cv) / k - b.center_mm[1]
            t = math.radians(b.angle_deg)
            a = dx * math.cos(t) + dy * math.sin(t)
            c = -dx * math.sin(t) + dy * math.cos(t)
            fp = (np.abs(a) <= b.w_mm / 2) & (np.abs(c) <= b.l_mm / 2)
            height = np.where(fp, np.maximum(height, b.h_mm), height)
            albedo = np.where(fp, b.albedo if b.rgb is None else self.gray_albedo(b.rgb), albedo)
            if refl is not None:
                refl[fp] = b.rgb if b.rgb is not None else (b.albedo,) * 3
            # distance (px) to the nearest footprint pixel on the left, per row
            last = np.maximum.accumulate(np.where(fp, cols, -np.inf), axis=1)
            dist = cols - last
            shadow |= (~fp) & (dist > 0) & (dist <= self.shadow_ratio * b.h_mm * k)
        shadow &= height == 0
        for lb in self.labels:
            import cv2
            side = int(round(lb.size_mm * k))
            x0 = int(round(cu + lb.center_mm[0] * k - side / 2))
            y0 = int(round(cv + lb.center_mm[1] * k - side / 2))
            ys, xs = slice(max(y0, 0), min(y0 + side, H)), slice(max(x0, 0), min(x0 + side, W))
            crop = (slice(ys.start - y0, ys.stop - y0), slice(xs.start - x0, xs.stop - x0))
            if lb.coloured:
                tex3 = cv2.resize(lb.texture_rgb(), (side, side), interpolation=cv2.INTER_NEAREST)
                refl[ys, xs] = tex3[crop]
                albedo[ys, xs] = (tex3 @ self.led_weights())[crop]
                continue
            tex = cv2.resize(lb.texture(), (side, side), interpolation=cv2.INTER_NEAREST)
            albedo[ys, xs] = tex[crop]
            if refl is not None:
                refl[ys, xs] = tex[crop][..., None]
        pw, ph = self.lit_fraction * W, self.lit_fraction * H
        lit = (np.abs(self._u - cu) <= pw / 2) & (np.abs(self._v - cv) <= ph / 2)
        dh, dw = self.dmd_shape
        corr_x = (self._u - (cu - pw / 2)) * (dw / pw) + self.relief_px_per_mm * height
        corr_y = (self._v - (cv - ph / 2)) * (dh / ph)
        valid = lit & ~shadow & ~unseen & (corr_x >= 0) & (corr_x <= dw - 1) & (corr_y <= dh - 1)
        self._refl, self._lit = refl, lit
        self._cache = (height, albedo, valid, corr_x, corr_y)
        return self._cache

    def reflectance(self) -> np.ndarray:
        """Ground-truth (H, W, 3) R, G, B reflectance (neutral scenes: albedo x 3)."""
        albedo = self._scene()[1]
        return self._refl if self._refl is not None else np.repeat(albedo[..., None], 3, axis=2)

    def height_map(self) -> np.ndarray:
        return self._scene()[0]

    # ------------------------------------------------------ observation
    def render(self, dmd_pattern: Optional[np.ndarray]) -> np.ndarray:
        """Camera frame (float, pre-noise) for what the DMD mirrors show."""
        if self.stereo_mode:
            return self.stereo_scene().render(dmd_pattern, self.led_weights())
        _, albedo, valid, cx, cy = self._scene()
        frame = np.full(self.cam_shape, self.ambient, np.float32)
        if dmd_pattern is None:
            return frame
        pat = np.asarray(dmd_pattern)
        dh, dw = self.dmd_shape
        # linear interpolation along the coded axis: a camera pixel sees a
        # blend of neighbouring mirrors, so relief is not quantized to 1 px
        x = np.clip(cx, 0, dw - 1)
        ix0 = np.floor(x).astype(np.intp)
        ix1 = np.minimum(ix0 + 1, dw - 1)
        fx = (x - ix0).astype(np.float32)
        iy = np.clip(np.rint(cy), 0, dh - 1).astype(np.intp)
        if pat.ndim == 3:
            return frame + self._render_colour(pat, albedo, valid, iy, ix0, ix1, fx)
        p = pat.astype(np.float32)
        val = ((1 - fx) * p[iy, ix0] + fx * p[iy, ix1]) / 255.0
        out = frame + np.where(valid, self.gain * albedo * val, 0.0).astype(np.float32)
        if self.global_light > 0:
            out += self._global(albedo, np.full(3, p.mean() / 255.0, np.float32))
        return out

    def _render_colour(self, pat, albedo, valid, iy, ix0, ix1, fx) -> np.ndarray:
        """A BGR image as sent over HDMI: each channel drives its own LED
        sub-frame; the mono camera sums them weighted by LED power."""
        if pat.shape[2] < 3:
            raise ValueError(f"colour pattern needs 3 (BGR) channels, got shape {pat.shape}")
        p = pat[..., 2::-1].astype(np.float32)                  # BGR(A) -> R, G, B
        w = self.led_weights()

        def drive(img):                                         # DMD image -> camera, 0..1
            return ((1 - fx) * img[iy, ix0] + fx * img[iy, ix1]) / 255.0

        if self._refl is None:                                  # neutral scene: one gray image
            sig = albedo * drive(np.ascontiguousarray(p @ w))
        else:
            sig = np.zeros(self.cam_shape, np.float32)
            for c in range(3):
                pc = p[..., c]
                lo, hi = float(pc.min()), float(pc.max())
                if hi <= 0 or w[c] == 0:
                    continue                                    # LED sub-frame dark
                d = lo / 255.0 if lo == hi else drive(np.ascontiguousarray(pc))
                sig += w[c] * self._refl[..., c] * d
        out = np.where(valid, self.gain * sig, 0.0).astype(np.float32)
        if self.global_light > 0:
            out += self._global(albedo, p.mean(axis=(0, 1)) / 255.0)
        return out

    def _global(self, albedo, level_rgb) -> np.ndarray:
        """Uniform scatter: reflectance x mean DMD drive, on the whole lit patch."""
        wl = self.led_weights() * np.asarray(level_rgb, np.float32)
        r = albedo * wl.sum() if self._refl is None else self._refl @ wl
        return np.where(self._lit, self.gain * self.global_light * r, 0.0).astype(np.float32)

    def oracle_field(self, geometry_id: str, meta: dict | None = None) -> DeformationField:
        """Ground-truth correspondence + noise, standing in for decode until
        the graycode/phaseshift bases land (build step 2)."""
        if self.stereo_mode:
            cx, cy, m = self.stereo_scene().oracle()
            n = self.corr_noise_px
            if n:
                cx = cx + self._rng.normal(0, n, cx.shape)
                cy = cy + self._rng.normal(0, n, cy.shape)
            g = self.stereo_scene().geometry()
            return DeformationField(corr_x=cx.astype(np.float32), corr_y=cy.astype(np.float32),
                                    corr_diag=None,
                                    conf=np.where(m, 0.5 * self.gain * g["albedo"], 0).astype(np.float32),
                                    mask=m, geometry_id=geometry_id, basis_names=["mock_oracle"],
                                    meta={**(meta or {}), "mock": True, "stereo": True})
        _, albedo, valid, cx, cy = self._scene()
        n = self.corr_noise_px
        corr_x = np.where(valid, cx + self._rng.normal(0, n, cx.shape), np.nan).astype(np.float32)
        corr_y = np.where(valid, cy + self._rng.normal(0, n, cy.shape), np.nan).astype(np.float32)
        conf = np.where(valid, 0.5 * self.gain * albedo, 0.0).astype(np.float32)
        return DeformationField(corr_x=corr_x, corr_y=corr_y, corr_diag=None,
                                conf=conf, mask=valid.copy(), geometry_id=geometry_id,
                                basis_names=["mock_oracle"],
                                meta={**(meta or {}), "mock": True,
                                      "objects": [vars(o) for o in self.objects]})
