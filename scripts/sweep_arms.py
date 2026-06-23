"""
Sweep each polarization controller arm (paddle) from position 0 to 154,
measure power at every step, save the raw data, and plot power vs position
with a fitted sine curve overlaid.
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pyvisa as visa
from datetime import datetime

sys.path.append(
    r"C:\Users\FTNK-LocalAdm\Desktop\mikael_speciale"
    r"\thjalfe_instrument_control\InstrumentControl\LabInstruments"
)
from pol_cons import ThorlabsMPC320

VISA_ADDR = "USB0::0x1313::0x8078::P0028333::INSTR"
POS_START = 0
POS_END = 154
SETTLE_TIME = 0.15  # seconds to wait after moving before measuring


def get_power_dbm(pm) -> float:
    watts = float(pm.query("READ?"))
    return 10.0 * np.log10(watts / 1e-3)


def sine_model(x, A, T, phi, C):
    return A * np.sin(2 * np.pi * x / T + phi) + C


def sweep_single_arm(pol_ctrl, pm, arm_idx, hold_positions):
    """Sweep arm `arm_idx` (0-based) through every integer position.

    The other two arms stay at `hold_positions`.
    Returns arrays of (positions, power_dbm).
    """
    positions = []
    powers = []

    # Set all arms to their hold positions first
    for i in range(3):
        pol_ctrl.set_position(i + 1, int(hold_positions[i]))

    for pos in range(POS_START, POS_END + 1):
        pol_ctrl.set_position(arm_idx + 1, int(pos))
        time.sleep(SETTLE_TIME)
        p = get_power_dbm(pm)
        positions.append(pos)
        powers.append(p)

    return np.array(positions), np.array(powers)


def fit_sine(positions, powers):
    """Fit a sine curve to the sweep data. Returns (params, fitted_curve)."""
    A0 = (np.max(powers) - np.min(powers)) / 2
    C0 = np.mean(powers)
    T0 = (POS_END - POS_START) / 2  # initial guess: ~2 cycles across range
    phi0 = 0.0

    try:
        popt, _ = curve_fit(
            sine_model,
            positions,
            powers,
            p0=[A0, T0, phi0, C0],
            maxfev=10000,
        )
        fitted = sine_model(positions, *popt)
        return popt, fitted
    except RuntimeError:
        return None, None


def main():
    print("Connecting to hardware...")
    pol_ctrl = ThorlabsMPC320()
    rm = visa.ResourceManager()
    pm = rm.open_resource(VISA_ADDR)

    arm_labels = ["Arm 1 (QWP)", "Arm 2 (HWP)", "Arm 3 (QWP)"]
    hold = [77, 77, 77]  # midpoint — neutral starting position for the other arms

    all_positions = []
    all_powers = []

    for arm_idx in range(3):
        print(f"Sweeping {arm_labels[arm_idx]} ({POS_START}→{POS_END})...")
        pos, pwr = sweep_single_arm(pol_ctrl, pm, arm_idx, hold)
        all_positions.append(pos)
        all_powers.append(pwr)

        # Update hold position for subsequent sweeps to keep the best so far
        best_pos = int(pos[np.argmin(pwr)])
        hold[arm_idx] = best_pos
        print(f"  Best position: {best_pos}  ({pwr.min():.2f} dBm)")

    # --- Save raw data ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"data/sweep_arms_{timestamp}.csv"
    with open(csv_path, "w") as f:
        f.write("arm,position,power_dbm\n")
        for arm_idx in range(3):
            for pos, pwr in zip(all_positions[arm_idx], all_powers[arm_idx]):
                f.write(f"{arm_idx + 1},{pos},{pwr:.4f}\n")
    print(f"Raw data saved to {csv_path}")

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle("Power vs. arm position (single-arm sweeps)", fontsize=14)

    for arm_idx, ax in enumerate(axes):
        pos = all_positions[arm_idx]
        pwr = all_powers[arm_idx]

        ax.scatter(pos, pwr, s=8, alpha=0.6, label="Measured", zorder=2)

        popt, fitted = fit_sine(pos, pwr)
        if fitted is not None:
            A, T, phi, C = popt
            ax.plot(pos, fitted, "r-", linewidth=1.5,
                    label=f"Sine fit  (A={A:.1f}, T={T:.1f}, C={C:.1f})",
                    zorder=3)

        ax.set_ylabel("Power [dBm]")
        ax.set_title(arm_labels[arm_idx])
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Paddle position (0–154)")
    plt.tight_layout()
    fig_path = f"output/figures/sweep_arms_{timestamp}.png"
    plt.savefig(fig_path, dpi=150)
    plt.show()
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
