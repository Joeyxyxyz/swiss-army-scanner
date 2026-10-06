"""MNF nullification self-test."""
import sys; sys.path.insert(0, ".")
import numpy as np
from swiss_army_sensor.nullification import (
    DifferentialPair, PhaseShiftNu, HadamardNu, sweep_nu, run_self_test,
    synthetic_low_snr_hadamard_test, hadamard_mse,
)


def main():
    print("=" * 64)
    print("MNF NULLIFICATION SELF-TEST")
    print("=" * 64)

    results = run_self_test()

    print("\n--- ν=1 limit verifications ---")
    for v in results["verifications"]:
        status = "OK" if v.get("ok") else "FAIL"
        print(f"  [{status}] {v['name']}")

    print("\n--- Hadamard ν* vs SNR, 30-seed average ---")
    print(f'{"SNR (dB)":>9}  {"mean ν*":>9}  {"std ν*":>8}  {"% improve":>10}')
    for snr in (-15, -12, -10, -8, -6, -4, -2, 0, 3, 6, 10):
        nu_stars = []
        pcts = []
        for seed in range(30):
            inputs = synthetic_low_snr_hadamard_test(snr_db=snr, seed=seed, n_coeffs=128)
            sw = sweep_nu(HadamardNu(), inputs, hadamard_mse,
                          nu_values=np.linspace(0, 1, 101))
            nu_stars.append(sw.nu_star)
            pcts.append((sw.metric_at_nu_1 - sw.metric_at_nu_star)
                        / max(sw.metric_at_nu_1, 1e-12) * 100)
        print(f"{snr:>+9d}  {np.mean(nu_stars):>9.3f}  {np.std(nu_stars):>8.3f}  "
              f"{np.mean(pcts):>+9.3f}%")

    print("\n[OK] nullification self-test complete")


if __name__ == "__main__":
    main()
