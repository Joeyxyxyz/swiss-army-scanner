"""
Station measurement: cells lying with their circular END FACES toward the camera, held in a
ribbed tube (CT: a thin metal ribbon bent into a trapezoid wave, seen edge-on between the
cells a few mm behind the faces). From one metric scan (column-axis ray-plane cloud, mm) and a
flood pair (the scan's own from scan_fast, another scan's, a file, or - as a stand-in - the
decode's fringe modulation): api verbs station_measure, station_golden, station_check.

    geometry.py  cells on the flood (Hough, brightest non-overlapping discs that are flat,
                 proud faces in the scan), rim ellipse, face planes (robust), face centres
                 (sub-pixel flood rim edges back-projected onto the face plane, circle fitted
                 there; or the camera ray through the rim ellipse centre), the common face frame
                 (n = sum of the face normals, origin = mean face centre), end-face heights along n
                 and pairwise differences, spacing; the CT front edge (thin runs a few mm behind
                 the face plane, outside every cell body; height band given or found), the
                 trapezoid-wave fit of its centreline and the part-fixed zones T1..Tn with the
                 edge height per zone
    verbs.py     rules 5 / 6, the output bin (JSON + CSV + PNG), the golden (append-only, bound
                 to geometry + fingerprint + stereo calibration: nominal values, zone model +
                 numbering, user-set limits), the check (PASS / FAIL / INCOMPLETE + triggers)
    figure.py    the station figure (cv2)

Heights: along the common face normal, + = toward the camera (proud), - = behind the mean
cell-face plane. Zones: one zone = one plateau of the wave + the two half-ramps beside it
(half a pitch), indexed k from the cylinder-pair centre (part-fixed); a golden fixes the pitch +
phase and the numbering, so every check measures the same zones.
"""
