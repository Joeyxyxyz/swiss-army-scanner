"""
Dent detection on shiny cylindrical cells (api verbs: capture_dent_set,
calibrate_dent_null, detect_dents; the session side is verbs.py).

    dentset.py       production patterns (DMD X = col, Y = row / 2), subsets, loading of
                     library entries and legacy dataset.json folders, flood polarity, HDR
    geometry.py      fixed-radius cylinder from the metric scans -> (theta, s), cosv, face,
                     projector shift per mm along the camera ray, metric radial residual
    finder_warp.py   (a) pattern warp: phasors vs the metric carrier, along-axis IRLS
                     residual, 2-D warp / depth, two-exposure AND rule
    finder_photo.py  (b) photometric: floods, sinusoid offset / modulation, direct / global,
                     shift-tolerant axial residual, shrunk Mahalanobis, censored glints
    finder_phase.py  (c) HDR phase-residual depth z (sin0 / sin90, ring carrier, per-column
                     noise)
    finder_line.py   (d) reflected room-line kink in the projector-dark flood
    fusion.py        null calibration (good-cell maxima), fusion, peaks, depth in mm,
                     scratch / speck / dent classification
    overlay.py       the PNG in the output bin

Every map is in crop pixels of the dent set; candidates are reported in full
camera pixels and in cell coordinates (theta_deg, s_mm). Offline: the finders
read saved frames and library files only; only capture_dent_set drives the rig
(through calibrate.flow.capture_patterns, HDMI path, rule 1).
"""
