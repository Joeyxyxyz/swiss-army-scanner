"""Quick sanity tests for all modes."""
import sys; sys.path.insert(0, ".")
import numpy as np
from swiss_army_sensor import SensorRig, SensorMode


def test_all_modes_run_without_error():
    rig = SensorRig.create(mock=True, scene="step")
    try:
        # Geometric
        rig.set_mode(SensorMode.PHOTOELECTRIC, threshold=50)
        assert rig.read().intensity_mean >= 0

        rig.set_mode(SensorMode.DISTANCE)
        assert rig.read().spot_x >= 0

        rig.set_mode(SensorMode.PROFILOMETRY_3D, n_steps=4, period_px=24)
        assert rig.read().phase.shape == (1080, 1440)

        rig.set_mode(SensorMode.EDGE)
        assert rig.read().edge_strength > 0

        rig.set_mode(SensorMode.COLOR)
        rig.read()

        rig.set_mode(SensorMode.SINGLE_PIXEL, order=4, fov_px=128)
        assert rig.read().image.shape == (16, 16)

        rig.set_mode(SensorMode.VIBRATION, modulation_hz=50, n_frames=128, fps=1000)
        assert rig.read().dominant_hz > 0

        rig.set_mode(SensorMode.DEFECT, period_px=6)
        assert rig.read().specular.shape == (1080, 1440)

        rig.set_mode(SensorMode.MARK)
        rig.read()

        # Spectral
        rig.set_mode(SensorMode.SPOT_EMULATION, spot_diameter_mm=0.85)
        assert rig.read().spot_pixels_radius >= 1

        rig.set_mode(SensorMode.SPECTRAL_3BAND)
        rig.read()

        rig.set_mode(SensorMode.THIN_FILM, n_film=1.5)
        rig.read()

        rig.set_mode(SensorMode.MATERIAL_ID)
        rig.read()

        rig.set_mode(SensorMode.SPECKLE_ROUGHNESS, wavelength_nm=638)
        rig.read()

        rig.set_mode(SensorMode.NIR_SPECTROSCOPY,
                     bands={"nir_940": 0.45, "swir_1310": 0.65, "swir_1550": 0.55})
        rig.read()

        # Coherent
        rig.set_mode(SensorMode.HOLOGRAPHY, propagation_distance_mm=200)
        r = rig.read()
        assert 0 <= r.coherence_quality <= 1

        rig.set_mode(SensorMode.INTERFEROMETRY, n_steps=4, period_px=24)
        rig.read()

        # Barcode (4 methods)
        for method in ("flood_passive", "raster_single_pixel",
                       "hadamard_single_pixel", "line_scan"):
            kwargs = {}
            if method == "raster_single_pixel":
                kwargs["bucket_size"] = (32, 32)
            if method == "hadamard_single_pixel":
                kwargs["order"] = 4
            rig.set_mode(SensorMode.BARCODE, method=method, **kwargs)
            rig.read()

        print("[OK] all 18 modes + 4 barcode methods ran")
    finally:
        rig.close()


if __name__ == "__main__":
    test_all_modes_run_without_error()
