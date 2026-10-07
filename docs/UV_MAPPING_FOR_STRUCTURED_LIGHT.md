# UV mapping ideas applied to structured light

Sources: a game-development text on UV unwrapping (UV coordinates, seams, planar/cylindrical/spherical mapping, texel density, checker tests, UDIM/Ptex) and a 3D-graphics text's unwrapping chapter ( texture map types: cavity, density, dust, subsurface).

This is an exploration only. Nothing here is built yet, and nothing touched the rig.

## The key observation: the scanner already produces a UV map

In games, a UV map says, for each point on a 3D surface, which texture pixel (u, v) paints it. Our decoder produces the mirror image of that:
- **Every camera pixel gets a projector coordinate (u, v).** `corr_x` / `corr_y` in `engine/decode/projmap.py` are the column and row of the DMD pixel that lit it.
- That is a UV map with **the DMD as the texture**, measured rather than authored.
- `ProjectorMap.to_camera` is the inverse: texture → surface.

So most game-texturing concepts have a direct structured-light meaning:

| Game concept | Structured-light equivalent | Use for us |
|---|---|---|
| UV coordinates | decoded (corr_x, corr_y) | already have it |
| Texture applied to a model | any pattern drawn in DMD space lands on the part | projection mapping: grids, zones, spot sensors (built) |
| **Checker texture to test distortion** | project a checker, measure how cells stretch in the camera: the local Jacobian of the UV map | local magnification, surface slope/normal, focus, and a lens-distortion check, all from one image |
| Texel density | mm per DMD pixel on the surface (varies with distance and tilt) | the true sampling resolution per region; tells you where sub-mm is achievable |
| **Seams** | where the map breaks: shadows, occlusion, depth edges, the ±24 px unwrap jumps | put "cuts" where the camera can't see anyway; explicit seam masks instead of noisy edges |
| **Planar / cylindrical / spherical mapping** | fit a primitive to the part, and parameterize its surface | **unwrap a cylindrical can into a flat (θ, z) image**, see below |
| UV islands, no overlap | per-face or per-zone patches | box faces, cylinder side vs top cap, grid zones |
| UDIM / multi-tile | several tiles per object | several zones or views of one part, each its own tile, stitched |
| Ptex (per-face, no unwrap) | per-zone data with no global parameterization | today's zone scans are effectively Ptex |
| Baked maps (cavity, density, dust, subsurface) | derived per-pixel maps | a per-part "material stack" for inspection, see below |

## 1. Cylindrical unwrap for cylinders (serves end-face height and dent)

A cylindrical part is exactly the "cylindrical mapping (arms, legs)" case:
1. Fit a cylinder (axis, radius) to the part's relief / point cloud.
2. Give each point coordinates **θ** (angle around the axis) and **z** (along the axis), and store the **radial deviation** r − R.
3. The result is a flat **dent map**: an image in (θ, z) where the healthy can is flat zero, and a dent is a dark blob whose depth is the pixel value.

**Why it helps:**
- **Placement-invariant.** A cylinder rotated or shifted in the fixture unwraps to the same image, only shifted in θ. That makes the good-memory bank and golden compare work on cylinders without a jig to the pixel.
- **Constant texel density.** Each unwrapped pixel is a fixed arc length × height, so dent area and depth mean the same thing everywhere. The raw camera view foreshortens toward the cylinder's silhouette.
- **End-face height falls out.** It's the z-extent of the valid band, or the step between the top cap and the tray in the same fit, measured along the true axis rather than the camera's.
- **Seam placement.** Put the θ seam on the side the camera can't see (the limb/shadow side), just as games hide seams under arms.
- **Dents become "cavity map" blobs**, so the existing `inspect` blob logic runs unchanged on the unwrapped image, with a much simpler background (a plane instead of a curved can).

Limits to be honest about:
- One view covers only about 120–140° of a can, and the sampling degrades near the silhouette. Full coverage needs a rotate/flip and a second scan, stitched as two UDIM-style tiles.
- The 25 µm dent spec stays a noise problem. Unwrapping doesn't add signal, but it makes area-averaging clean because the dent is a compact blob in a flat image.

## 2. Planar unwrap: "unfold the cardboard box"

The flipped case earlier was corner-on: two faces meeting at an edge, and a single background fit failed there. Fitting a plane per face and flattening each into its own (u, v) island gives two flat rectangles. The side-panel bulge we found by hand (~0.7–1 px) then appears directly as a blob, and a box's six faces become a standard cross-shaped layout. This generalizes the per-face fit I did by hand into a reusable step.

## 3. Checker texture to measure distortion

Games project a checker to spot stretching; we can do it physically, and get more than a pass/fail:
- Project a checker (the `checkerboard` basis from the spec, not yet built). Per cell, the camera sees a stretched quad, and **its size and skew are the local Jacobian** of the projector→camera map.
- **Stretch along the baseline** ≈ surface tilt; stretch orthogonal to it reveals curvature. That gives a surface-normal estimate without full triangulation.
- The same image gives corner points for **full stereo calibration** (the spec's future `calibrate/stereo.py`, TI's projector-as-inverse-camera method).
- A checker whose squares blur shows **focus per region**. On the bench, 16-mirror squares kept only 16% contrast, which was a focus/blur finding.

## 4. Baked maps: a per-part inspection stack

Game engines bake several maps onto one UV layout. We can bake a stack per scan, all in the same (unwrapped) coordinates:

| Map | From | Inspection meaning |
|---|---|---|
| Height / displacement | relief (calibrated) | geometry, end-face height |
| **Cavity** | curvature (Laplacian) of height | dents, pits and scratches pop out, regardless of overall shape |
| Normal | height gradient | orientation, bulges, the checker Jacobian |
| Albedo / colour | `spectral_capture` | labels, coatings, contamination |
| **Subsurface** | `separate_light` global component | translucent / plastic vs metal, wet or oily films |
| Density / valid | decode validity + modulation | trust map: where the measurement is reliable |
| **Dust** | albedo minus the golden albedo, in UV space | debris / residue on the part |

Aligned in UV space, these can be compared layer by layer against the good-memory bank. "Not in the good bank" then covers shape *and* appearance, which is the "flag anything not in the good memory bank" goal.

## 5. Texture going the other way: projection mapping

Because the map is invertible, anything computed in UV/part space can be **painted back onto the real part** through the DMD:
- defect outlines
- the dent map as a heatmap on the can
- pass/fail colouring per zone
- guidance arrows for an operator

This extends the grid/zone machinery that already works on the rig. It turns the scanner into an in-place AR display for the operator, or for Grok to "point" at things.

## Suggested order (fits the stated priority: cylinder dent and end-face height)

1. **Cylinder fit + (θ, z) unwrap + dent map + height from the unwrap**, prototyped on the mock bench (add a cylinder object to `MockBench`), then on a real cylinder.
2. **Cavity map + existing `inspect` on the unwrapped image**, then the good-memory bank in UV space (placement-invariant).
3. **Planar per-face unwrap**, reusing the case scans already in the library as test data.
4. **Checkerboard basis**: local Jacobian/normal map, focus map, then stereo calibration.
5. **Baked-map stack** and projecting results back onto the part.

Items 1–3 need no rig time to develop: the mock bench plus stored scans are enough. They only need a real cylinder scan at the end.
