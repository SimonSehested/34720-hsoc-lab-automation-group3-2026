# Automation of the High-Speed Optical Communications (HSOC) lab
# Course: 34720 | Group: 3 | Year: 2026
#
# This script automates polarization optimization of lasers using a
# Thorlabs MPC320 polarization controller and a power meter.
#
# --- Old variable names (for reference) ---
# pol_con1                                    -> pol_ctrl
# pow, db                                     -> power_dbm
# pos1, pos2, pos3                            -> positions[0], positions[1], positions[2]
# pow_arm1_list, pow_arm2_list, pow_arm3_list -> power_scan
# new_pos                                     -> removed (was unused)
# i (in optimize)                             -> step
# arm                                         -> arm_idx

import pyvisa as visa
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

# NOTE: Update this path to match your local installation of thjalfe_instrument_control
sys.path.append(r"C:\Users\FTNK-LocalAdm\Desktop\mikael_speciale\thjalfe_instrument_control\InstrumentControl\LabInstruments")
from pol_cons import ThorlabsMPC320

# Connect to polarization controller (serial: 444)
try:
    pol_ctrl = ThorlabsMPC320()
except:
    None

# Connect to power meter via USB
power_meter = visa.ResourceManager().open_resource('USB0::0x1313::0x8078::P0028333::INSTR')


def get_power():
    """Read current optical power from the power meter and return it in linear units (W)."""
    power_lin = float(power_meter.query('READ?'))
    return power_lin


def set_arms(arm1, arm2, arm3):
    """Set the position of all three polarization controller arms (range: 0-154)."""
    pol_ctrl.set_position(1, arm1)
    pol_ctrl.set_position(2, arm2)
    pol_ctrl.set_position(3, arm3)


def fit_sinusoid(xs, ys, n_positions=155):
    """Fit P(x) = a*sin(2π*x/T) + b*cos(2π*x/T) + C via linear least squares.

    Tries several candidate periods and picks the one with the best R².
    Returns (fitted_curve over n_positions, r_squared, best_period).
    """
    T_candidates = [70, 100, 130, 160, 200, 240, 280, 320]
    ss_tot = np.sum((ys - ys.mean()) ** 2) + 1e-30

    best_r2 = -np.inf
    best_fitted = None
    best_T = T_candidates[0]

    for T in T_candidates:
        M = np.column_stack([
            np.sin(2 * np.pi * xs / T),
            np.cos(2 * np.pi * xs / T),
            np.ones_like(xs, dtype=float),
        ])
        coeffs, _, _, _ = np.linalg.lstsq(M, ys, rcond=None)
        ssr = np.sum((M @ coeffs - ys) ** 2)
        r2 = 1.0 - ssr / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_T = T
            all_xs_full = np.arange(n_positions, dtype=float)
            M_full = np.column_stack([
                np.sin(2 * np.pi * all_xs_full / T),
                np.cos(2 * np.pi * all_xs_full / T),
                np.ones(n_positions),
            ])
            best_fitted = M_full @ coeffs

    return best_fitted, best_r2, best_T


def fit_sin2(xs, ys, n_positions=155):
    """Fit P(x) = A * sin(2π*x/T + phi)^2 + C via linear least squares.

    Uses the identity sin²(u) = (1 - cos(2u))/2 to rewrite the model as
    a*cos(4π*x/T) + b*sin(4π*x/T) + C', which is linear in (a, b, C').
    Tries several candidate periods; picks the best R².
    Returns (fitted_curve over n_positions, r_squared, best_T).
    """
    T_candidates = [70, 100, 130, 160, 200, 240, 280, 320]
    ss_tot = np.sum((ys - ys.mean()) ** 2) + 1e-30

    best_r2 = -np.inf
    best_fitted = None
    best_T = T_candidates[0]

    for T in T_candidates:
        M = np.column_stack([
            np.cos(4 * np.pi * xs / T),
            np.sin(4 * np.pi * xs / T),
            np.ones_like(xs, dtype=float),
        ])
        coeffs, _, _, _ = np.linalg.lstsq(M, ys, rcond=None)
        ssr = np.sum((M @ coeffs - ys) ** 2)
        r2 = 1.0 - ssr / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_T = T
            a, b, C_prime = coeffs
            A = 2 * np.sqrt(a**2 + b**2)
            phi = np.arctan2(b, -a) / 2
            C = C_prime - A / 2
            all_xs_full = np.arange(n_positions, dtype=float)
            best_fitted = A * np.sin(2 * np.pi * all_xs_full / T + phi)**2 + C

    return best_fitted, best_r2, best_T


def optimize():
    """
    Scan all 155 positions for each arm and move to the minimum.
    Fits a Jacobi elliptic curve to the scan data and plots the result.
    """
    positions = [0, 77, 77]
    set_arms(*positions)

    all_xs = np.arange(155, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for arm_idx in range(3):
        power_scan = []
        for step in range(155):
            scan_positions = positions.copy()
            scan_positions[arm_idx] = step
            set_arms(*scan_positions)
            power_scan.append(get_power())

        positions[arm_idx] = power_scan.index(min(power_scan))
        set_arms(*positions)

        scan_ys = np.array(power_scan)
        fitted, r2, period = fit_sin2(all_xs, scan_ys)

        ax = axes[arm_idx]
        ax.plot(all_xs, scan_ys, color='steelblue', alpha=0.7, label='Scan')
        ax.plot(all_xs, fitted, color='orange', linewidth=2,
                label=f'sin² fit (T={period}, R²={r2:.2f})')
        ax.axvline(positions[arm_idx], color='green', linestyle='--',
                   label=f'Min @ {positions[arm_idx]}')
        ax.set_title(f'Arm {arm_idx + 1}')
        ax.set_xlabel('Position')
        ax.set_ylabel('Power')
        ax.legend()

    plt.tight_layout()
    plt.show()


optimize()


'''
def test_drift():
    import random
    positions = [random.randint(0, 154), random.randint(0, 154), random.randint(0, 154)]
    set_arms(*positions)
    power_scan = []
    time.sleep(0.1)

    for i in range(300):
        power_scan.append(get_power())
        time.sleep(0.1)

    plt.plot(power_scan)
    plt.show()

test_drift()
'''

'''
def fit_fourier(xs, ys, n_positions=155):
    """Fit P(theta) = a0 + a1*cos(2t) + b1*sin(2t) + a2*cos(4t) + b2*sin(4t)
    where t = 2*pi*x / n_positions, via linear least squares.

    Returns (fitted_curve over n_positions, r_squared, fundamental_period_in_positions).
    """
    theta = 2 * np.pi * xs / n_positions
    M = np.column_stack([
        np.ones_like(xs, dtype=float),
        np.cos(2 * theta),
        np.sin(2 * theta),
        np.cos(4 * theta),
        np.sin(4 * theta),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(M, ys, rcond=None)

    ssr = np.sum((M @ coeffs - ys) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2) + 1e-30
    r2 = 1.0 - ssr / ss_tot

    all_xs = np.arange(n_positions, dtype=float)
    all_theta = 2 * np.pi * all_xs / n_positions
    M_full = np.column_stack([
        np.ones(n_positions),
        np.cos(2 * all_theta),
        np.sin(2 * all_theta),
        np.cos(4 * all_theta),
        np.sin(4 * all_theta),
    ])
    fitted = M_full @ coeffs
    period = n_positions / 2
    return fitted, r2, period


def get_stable_power(settle=1.0, n=10):
    """Settle for settle seconds, take n readings, return the median."""
    time.sleep(settle)
    return float(np.median([get_power() for _ in range(n)]))


def optimize_fast(start_positions=None):
    """Optimize polarization using sinusoidal curve fitting (8-sample version).

    For each arm:
      1. Measure power at 8 evenly-spaced positions using get_stable_power().
      2. Fit a sinusoid via linear least squares (numpy only, no scipy).
      3. Move the arm to the minimum of the fitted curve.

    start_positions: [arm1, arm2, arm3] to begin from (default [0, 77, 77]).
    """
    N_POSITIONS = 155
    N_SAMPLES = 8

    positions = list(start_positions) if start_positions is not None else [0, 77, 77]
    set_arms(*positions)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for arm_idx in range(3):
        sample_xs = np.linspace(0, N_POSITIONS - 1, N_SAMPLES, dtype=int)
        sample_ys = []

        for pos in sample_xs:
            scan_positions = positions.copy()
            scan_positions[arm_idx] = int(pos)
            set_arms(*scan_positions)
            sample_ys.append(get_stable_power())

        sample_xs = sample_xs.astype(float)
        sample_ys = np.array(sample_ys)
        all_xs = np.arange(N_POSITIONS, dtype=float)

        fitted, r2, period = fit_sinusoid(sample_xs, sample_ys, N_POSITIONS)
        best_pos = int(np.argmin(fitted))
        print(f"Arm {arm_idx + 1}: R²={r2:.2f}, T={period}  -> position {best_pos}")

        positions[arm_idx] = best_pos
        set_arms(*positions)

        ax = axes[arm_idx]
        ax.plot(all_xs, fitted, label=f'Fit (T={period}, R²={r2:.2f})')
        ax.scatter(sample_xs, sample_ys, color='red', zorder=5, label='Measured')
        ax.axvline(best_pos, color='green', linestyle='--', label=f'Min @ {best_pos}')
        ax.set_title(f'Arm {arm_idx + 1}')
        ax.set_xlabel('Position')
        ax.set_ylabel('Power')
        ax.legend()

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)

    print(f"Final positions: {positions}")
    return positions


TARGET_POWER = 0.01
EDGE_MARGIN = 10

start = [0, 77, 77]
iteration = 1
while True:
    print(f"\n--- Iteration {iteration} ---")
    positions = optimize_fast(start)
    power = get_stable_power()
    print(f"Power after iteration {iteration}: {power:.4f}")

    if power <= TARGET_POWER:
        print(f"Target of {TARGET_POWER} reached.")
        break

    edge_arms = [i + 1 for i, p in enumerate(positions)
                 if p < EDGE_MARGIN or p > 154 - EDGE_MARGIN]
    if edge_arms:
        print(f"Arm(s) {edge_arms} too close to edge — randomizing and restarting.")
    else:
        print("Target not reached — randomizing and retrying.")

    start = [int(np.random.randint(0, 155)) for _ in range(3)]
    print(f"New start positions: {start}")
    iteration += 1

plt.show()
'''
