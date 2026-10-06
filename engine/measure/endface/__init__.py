"""
End faces: the flat circular END FACE of a cylindrical cell lying with its axis toward the
camera + projector (bench 2026-09-28: a 21.05 mm face ~795 mm away, tilted 7-23 deg,
~55 camera px across, ~0.38 mm/px). api verbs: locate_face, aim_ring, check_ring,
auto_intensity, targeted_face_scan, face_profile, face_golden, check_face; the session
side is verbs.py.

    locate.py    face by DEPTH (the nearest depth-continuous surface of a metric scan that
                 is face-shaped), rim = contour points whose outward neighbourhood is far
                 background (the true silhouette, not an undecoded band), fixed-radius circle
                 fit in the face plane, the circle projected into the camera (the camera face
                 mask), camera -> DMD AFFINE fitted on the decoded face pixels (the stereo
                 calibration's projector ROW axis is not trusted for aiming: bench row-vs-column
                 triangulation disagreed by 3.6-4.6 mm; the column axis is fine)
    patterns.py  DMD images: aim ring + cross, the face ellipse mask (+ margin), the polar set
                 (concentric rings, spokes, floods in face-normalised polar coordinates), and
                 the ring-check measurement on the photo
    profile.py   face frame (robust plane, centre = the camera ray through the face ellipse
                 centre), height map on a 0.25 mm grid, radial profile, sections, flatness,
                 repeatability across repeats, PNG / CSV
    detectors.py the detector hook: 'mf_fir' (default) | 'mf' | 'band' (or a band trigger
                 function); every detector's golden model, leave-one-out false triggers, the
                 synthetic-dent battery (dents injected per camera pixel), overlay PNGs
    mfcfar.py    MF-CFAR: per-repeat planes, deviation from the golden RADIAL profile, a
                 dent-shaped matched filter bank (1-3 mm), z against the face's own robust
                 noise; the reflectance FIR (7x7 camera px on the flood pair) removes the
                 flood-predictable part of the height texture first
    golden.py    the band detector: median map + tolerance band (the bench prototype)

Face frame (every map, profile and trigger): n = the face normal toward the camera,
e1 = n x y_cam (~ camera +x, right), e2 = e1 x n (~ camera +y, down); u = (X - C).e1,
v = (X - C).e2, h = (X - C).n (mm, + toward the camera: a dent is NEGATIVE); theta =
atan2(v, u) in degrees (0 = camera right, 90 = camera down: the map reads like the photo).

mm only with the stereo calibration the metric scans were triangulated with, still current
for the geometry (rule 6); face targets, face scans and goldens are bound to the geometry and
its fingerprint (rule 5); the library is append-only (rule 7). Only aim_ring / check_ring /
auto_intensity / targeted_face_scan drive the rig, through the normal rig path (HDMI only,
pumped holds, rule 1 / 2).
"""
