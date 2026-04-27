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
import threading
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# NOTE: Update this path to match your local installation of thjalfe_instrument_control
sys.path.append(r"C:\Users\FTNK-LocalAdm\Desktop\mikael_speciale\thjalfe_instrument_control\InstrumentControl\LabInstruments")
from pol_cons import ThorlabsMPC320

# Connect to polarization controller (serial: 444)
pol_ctrl = None
try:
    pol_ctrl = ThorlabsMPC320()
except Exception as e:
    print(f"WARNING: MPC320 not connected ({e})")

# Connect to power meter via USB
power_meter = visa.ResourceManager().open_resource('USB0::0x1313::0x8078::P0024464::INSTR')


def get_power():
    """Read current optical power from the power meter and return it in linear units (W)."""
    power_lin = float(power_meter.query('READ?'))
    return power_lin


def set_arms(arm1, arm2, arm3):
    """Set the position of all three polarization controller arms (range: 0-154)."""
    if pol_ctrl is None:
        raise RuntimeError("MPC320 not connected — check USB and Kinesis drivers.")
    pol_ctrl.set_position(1, arm1)
    pol_ctrl.set_position(2, arm2)
    pol_ctrl.set_position(3, arm3)


def set_arm(arm_idx, position):
    """Set a single polarization controller arm."""
    if pol_ctrl is None:
        raise RuntimeError("MPC320 not connected â€” check USB and Kinesis drivers.")
    pol_ctrl.set_position(arm_idx + 1, int(position))


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


def fit_P(alpha, A0, A2c, A2s, A4c, A4s):
    """Fourier model from poli2: A0 + harmonics at 2α and 4α (α in radians)."""
    return (A0
            + A2c * np.cos(2 * alpha)
            + A2s * np.sin(2 * alpha)
            + A4c * np.cos(4 * alpha)
            + A4s * np.sin(4 * alpha))


def get_stable_power(settle_time=0.2, n_readings=3):
    """Let the optics settle, then return the median of a few power readings."""
    time.sleep(settle_time)
    readings = [get_power() for _ in range(n_readings)]
    return float(np.median(readings))


def measure_power_over_time(duration_minutes=1, interval_seconds=10, plot=True):
    """Measure optical power repeatedly over a fixed time window."""
    total_duration = duration_minutes * 60
    n_samples = int(total_duration // interval_seconds) + 1
    elapsed_times = []
    power_readings = []

    start_time = time.time()
    for sample_idx in range(n_samples):
        elapsed = time.time() - start_time
        power_w = get_power()

        elapsed_times.append(elapsed)
        power_readings.append(power_w)

        print(
            f"Sample {sample_idx + 1}/{n_samples}: "
            f"t = {elapsed:.0f} s, "
            f"P = {10 * np.log10(max(power_w * 1e3, 1e-30)):.2f} dBm"
        )

        if sample_idx < n_samples - 1:
            next_sample_time = start_time + (sample_idx + 1) * interval_seconds
            time.sleep(max(0, next_sample_time - time.time()))

    elapsed_times = np.asarray(elapsed_times, dtype=float)
    power_readings = np.asarray(power_readings, dtype=float)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(elapsed_times / 60.0, 10 * np.log10(np.maximum(power_readings * 1e3, 1e-30)))
        plt.xlabel("Time (minutes)")
        plt.ylabel("Power (dBm)")
        plt.title("Power over time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return elapsed_times, power_readings


def optimize_fourier(iterations=1, start_positions=None, n_samples=10,
                     settle_time=0.2, n_readings=3, plot=True):
    """Optimize polarization on the real setup using the Fourier fit from poli2.

    The theory from poli2 is used only as the fit model. The measured data
    comes from the actual polarization controller and power meter.
    """
    positions = list(start_positions) if start_positions is not None else [0, 77, 77]
    if len(positions) != 3:
        raise ValueError("start_positions must contain [arm1, arm2, arm3].")

    set_arms(*positions)

    sample_pos = np.linspace(0, 154, n_samples, dtype=int)
    sample_alpha = np.deg2rad(sample_pos.astype(float))
    fit_pos = np.arange(155, dtype=int)
    fit_alpha = np.deg2rad(fit_pos.astype(float))

    axes = None
    if plot:
        fig, axes = plt.subplots(iterations, 3, figsize=(14, 4 * iterations), squeeze=False)

    for iteration_idx in range(iterations):
        for arm_idx in range(3):
            power_scan = []

            for pos in sample_pos:
                scan_positions = positions.copy()
                scan_positions[arm_idx] = int(pos)
                set_arms(*scan_positions)
                power_scan.append(get_stable_power(settle_time=settle_time, n_readings=n_readings))

            power_scan = np.asarray(power_scan, dtype=float)
            best_pos = int(sample_pos[np.argmin(power_scan)])
            fitted = None

            try:
                params, _ = curve_fit(fit_P, sample_alpha, power_scan, maxfev=10000)
                fitted = fit_P(fit_alpha, *params)
                best_pos = int(fit_pos[np.argmin(fitted)])
            except RuntimeError:
                pass

            positions[arm_idx] = best_pos
            set_arms(*positions)

            if axes is not None:
                ax = axes[iteration_idx, arm_idx]
                ax.scatter(sample_pos, power_scan, color='red', zorder=5, label='Measured')
                if fitted is not None:
                    ax.plot(fit_pos, fitted, ':', color='green', label='Fourier fit')
                ax.axvline(best_pos, color='blue', linestyle='--', label=f'Min @ {best_pos}')
                ax.set_title(f'Iter {iteration_idx + 1}, arm {arm_idx + 1}')
                ax.set_xlabel('Position')
                ax.set_ylabel('Power (W)')
                ax.grid(True)
                ax.legend()

    if axes is not None:
        plt.tight_layout()
        plt.show()

    return positions


def optimize(plot=True):
    """
    Scan all 155 positions for each arm and move to the minimum.
    Fits a Jacobi elliptic curve to the scan data and plots the result.
    """
    positions = [0, 77, 77]
    set_arms(*positions)

    all_xs = np.arange(155, dtype=float)
    axes = None
    if plot:
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

        if axes is not None:
            ax = axes[arm_idx]
            ax.plot(all_xs, scan_ys, color='steelblue', alpha=0.7, label='Scan')
            ax.plot(all_xs, fitted, color='orange', linewidth=2,
                    label=f'sin2 fit (T={period}, R2={r2:.2f})')
            ax.axvline(positions[arm_idx], color='green', linestyle='--',
                       label=f'Min @ {positions[arm_idx]}')
            ax.set_title(f'Arm {arm_idx + 1}')
            ax.set_xlabel('Position')
            ax.set_ylabel('Power')
            ax.legend()

    if axes is not None:
        plt.tight_layout()
        plt.show()

    return positions


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

#test_drift()


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


def optimize_fast(iterations=1, start_positions=None, plot=True):
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

    axes = None
    if plot:
        fig, axes = plt.subplots(iterations, 3, figsize=(14, 4 * iterations), squeeze=False)

    for iteration_idx in range(iterations):
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
            positions[arm_idx] = best_pos
            set_arms(*positions)

            if axes is not None:
                ax = axes[iteration_idx, arm_idx]
                ax.plot(all_xs, fitted, label=f'Fit (T={period}, R2={r2:.2f})')
                ax.scatter(sample_xs, sample_ys, color='red', zorder=5, label='Measured')
                ax.axvline(best_pos, color='green', linestyle='--', label=f'Min @ {best_pos}')
                ax.set_title(f'Iter {iteration_idx + 1}, arm {arm_idx + 1}')
                ax.set_xlabel('Position')
                ax.set_ylabel('Power')
                ax.legend()

    if axes is not None:
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    return positions


'''
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



#optimize()
#optimize_fourier()
#measure_power_over_time()

def fast_sweep(coarse_iterations=1, fine_iterations=1, start_positions=None,
               sweep_start=0, sweep_end=155, sample_interval=0.01,
               fine_window=5, fine_settle_time=0.1, plot=True):
    def measure_during_sweep(arm_idx, start_pos, end_pos, positions, sample_interval):
        sweep = []
        timestamps = []
        stop_event = threading.Event()

        def sample_power(start_time):
            while not stop_event.is_set():
                timestamps.append(time.time() - start_time)
                sweep.append(get_power())
                time.sleep(sample_interval)

        positions[arm_idx] = int(start_pos)
        set_arms(*positions)

        start_time = time.time()
        sampler = threading.Thread(target=sample_power, args=(start_time,), daemon=True)
        sampler.start()
        try:
            set_arm(arm_idx, end_pos)
        finally:
            stop_event.set()
            sampler.join()

        positions[arm_idx] = int(end_pos)
        timestamps = np.asarray(timestamps, dtype=float)
        sweep = np.asarray(sweep, dtype=float)
        if timestamps.size == 0:
            raise RuntimeError("No power samples were collected during the sweep.")

        if timestamps[-1] > 0:
            position_trace = start_pos + (end_pos - start_pos) * (timestamps / timestamps[-1])
        else:
            position_trace = np.full_like(timestamps, start_pos, dtype=float)

        best_idx = int(np.argmin(sweep))
        best_pos = int(round(position_trace[best_idx]))
        return sweep, timestamps, position_trace, best_pos

    def refine_position(arm_idx, center_pos, positions, window=5, settle_time=0.1):
        measured = {}
        low = max(0, center_pos - window)
        high = min(154, center_pos + window)

        def measure_range(start_pos, end_pos):
            for pos in range(start_pos, end_pos + 1):
                if pos in measured:
                    continue
                scan_positions = positions.copy()
                scan_positions[arm_idx] = int(pos)
                set_arms(*scan_positions)
                time.sleep(settle_time)
                measured[pos] = get_power()

        measure_range(low, high)

        while True:
            local_positions = np.array(sorted(measured.keys()), dtype=int)
            local_powers = np.array([measured[pos] for pos in local_positions], dtype=float)
            best_local_idx = int(np.argmin(local_powers))
            best_local_pos = int(local_positions[best_local_idx])

            expanded = False
            if best_local_idx == 0 and low > 0:
                new_low = max(0, low - window)
                measure_range(new_low, low - 1)
                low = new_low
                expanded = True
            elif best_local_idx == len(local_positions) - 1 and high < 154:
                new_high = min(154, high + window)
                measure_range(high + 1, new_high)
                high = new_high
                expanded = True

            if not expanded:
                return local_positions, local_powers, best_local_pos

    def needs_restart(current_positions):
        return any(pos < 5 or pos > 150 for pos in current_positions)

    positions = list(start_positions) if start_positions is not None else [0, 77, 77]
    if len(positions) != 3:
        raise ValueError("start_positions must contain [arm1, arm2, arm3].")

    while True:
        last_sweeps = [None, None, None]
        last_refinements = [None, None, None]
        restart_required = False

        set_arms(*positions)

        for _ in range(coarse_iterations):
            for arm_idx in range(3):
                sweep, timestamps, position_trace, estimated_best_pos = measure_during_sweep(
                    arm_idx=arm_idx,
                    start_pos=sweep_start,
                    end_pos=sweep_end,
                    positions=positions,
                    sample_interval=sample_interval,
                )
                positions[arm_idx] = estimated_best_pos
                set_arms(*positions)
                last_sweeps[arm_idx] = (sweep, timestamps, position_trace, estimated_best_pos)

                if needs_restart(positions):
                    positions = [int(np.random.randint(0, 155)) for _ in range(3)]
                    restart_required = True
                    break
            if restart_required:
                break

        if restart_required:
            continue

        for _ in range(fine_iterations):
            for arm_idx in range(3):
                local_positions, local_powers, best_pos = refine_position(
                    arm_idx=arm_idx,
                    center_pos=positions[arm_idx],
                    positions=positions,
                    window=fine_window,
                    settle_time=fine_settle_time,
                )
                positions[arm_idx] = best_pos
                set_arms(*positions)
                last_refinements[arm_idx] = (local_positions, local_powers, best_pos)

                if needs_restart(positions):
                    positions = [int(np.random.randint(0, 155)) for _ in range(3)]
                    restart_required = True
                    break
            if restart_required:
                break

        if not restart_required:
            break

    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(9, 10), squeeze=False)
        for arm_idx in range(3):
            sweep, _, position_trace, estimated_best_pos = last_sweeps[arm_idx]
            local_positions, local_powers, best_pos = last_refinements[arm_idx]

            sweep_dbm = 10 * np.log10(np.maximum(sweep * 1e3, 1e-30))
            local_powers_dbm = 10 * np.log10(np.maximum(local_powers * 1e3, 1e-30))

            ax = axes[arm_idx, 0]
            ax.plot(position_trace, sweep_dbm, label="Measured during sweep")
            ax.scatter(local_positions, local_powers_dbm, color='orange', zorder=5, label='Fine check')
            ax.axvline(estimated_best_pos, color='blue', linestyle=':', label=f'Coarse pos {estimated_best_pos}')
            ax.axvline(best_pos, color='green', linestyle='--', label=f'Final pos {best_pos}')
            ax.set_title(f'Arm {arm_idx + 1}')
            ax.set_xlabel('Position')
            ax.set_ylabel('Power (dBm)')
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()

    return positions


def benchmark_optimizers(n_runs=1, start_positions=None):
    """Run all polarization optimizers multiple times and report runtime and final power."""
    start_positions = list(start_positions) if start_positions is not None else [0, 77, 77]
    if len(start_positions) != 3:
        raise ValueError("start_positions must contain [arm1, arm2, arm3].")

    optimizers = []
    for coarse_iterations in [1, 2, 3, 4, 5]:
        for fine_iterations in [1, 2, 3, 4, 5]:
            optimizers.append((
                f"fast_sweep_c{coarse_iterations}_f{fine_iterations}",
                lambda coarse_iterations=coarse_iterations, fine_iterations=fine_iterations: fast_sweep(
                    coarse_iterations=coarse_iterations,
                    fine_iterations=fine_iterations,
                    start_positions=start_positions,
                    sweep_start=0,
                    sweep_end=154,
                    sample_interval=0.01,
                    fine_window=5,
                    fine_settle_time=0.1,
                    plot=False,
                ),
            ))

    results = {}
    for name, optimizer_fn in optimizers:
        print(f"\nBenchmarking {name}")
        run_results = []

        for run_idx in range(n_runs):
            set_arms(*start_positions)
            start_time = time.perf_counter()
            final_positions = optimizer_fn()
            elapsed_seconds = time.perf_counter() - start_time

            time.sleep(1.0)
            final_power_w = get_power()
            final_power_dbm = 10 * np.log10(max(final_power_w * 1e3, 1e-30))

            run_result = {
                "run": run_idx + 1,
                "elapsed_seconds": elapsed_seconds,
                "final_power_w": final_power_w,
                "final_power_dbm": final_power_dbm,
                "final_positions": final_positions,
            }
            run_results.append(run_result)

            print(
                f"  Run {run_idx + 1}: "
                f"{elapsed_seconds:.2f} s, "
                f"{final_power_dbm:.2f} dBm, "
                f"positions {final_positions}"
            )

        results[name] = run_results

    return results

benchmark_optimizers()
