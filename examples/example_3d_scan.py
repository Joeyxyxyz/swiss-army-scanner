"""Run a 4-step phase-shifting 3D scan against a synthetic sphere."""

import numpy as np
from swiss_army_sensor import SensorRig, SensorMode


def main():
    with SensorRig.create(mock=True, scene="sphere") as rig:
        rig.set_mode(SensorMode.PROFILOMETRY_3D, n_steps=4, period_px=24)
        r = rig.read()
    print(f"Phase map shape : {r.phase.shape}")
    print(f"Modulation mean : {r.modulation.mean():.2f}")


if __name__ == "__main__":
    main()
