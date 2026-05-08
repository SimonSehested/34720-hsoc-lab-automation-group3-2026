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
import os
import time
import threading
import matplotlib.pyplot as plt
from datetime import datetime
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
               sweep_start=0, sweep_end=154, sample_interval=0.01,
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
        return any(pos < 5 or pos > 149 for pos in current_positions)

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

def measure_power_avg(duration_seconds=2.0):
    """Measure optical power over a duration and return the average in dBm."""
    start_time = time.time()
    readings = []
    while time.time() - start_time < duration_seconds:
        readings.append(get_power())
        time.sleep(0.01)
    avg_power_w = np.mean(readings)
    avg_power_dbm = 10 * np.log10(max(avg_power_w * 1e3, 1e-30))
    return avg_power_dbm


def write_benchmark_row(writer, csv_file, row):
    """Write one benchmark row and force it to disk before the next run starts."""
    writer.writerow(row)
    csv_file.flush()
    os.fsync(csv_file.fileno())


def current_timestamp():
    """Return a local timestamp suitable for benchmark logs."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_benchmark_csv_columns(csv_path, fieldnames):
    """Add missing columns to an existing benchmark CSV without dropping rows."""
    import csv

    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return

    with open(csv_path, "r", newline="") as csv_f:
        reader = csv.DictReader(csv_f)
        existing_fieldnames = reader.fieldnames or []
        rows = list(reader)

    if existing_fieldnames == fieldnames:
        return

    merged_fieldnames = list(existing_fieldnames)
    for fieldname in fieldnames:
        if fieldname not in merged_fieldnames:
            merged_fieldnames.append(fieldname)

    temp_path = f"{csv_path}.tmp"
    with open(temp_path, "w", newline="") as temp_f:
        writer = csv.DictWriter(temp_f, fieldnames=merged_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: row.get(fieldname, "") for fieldname in merged_fieldnames})
        temp_f.flush()
        os.fsync(temp_f.fileno())

    os.replace(temp_path, csv_path)


def make_ordered_benchmark_schedule(coarse_values, fine_values, n_trials):
    """Return C1F1, C2F1, ... ordering repeated until every setup has n_trials."""
    existing_counts = {
        (coarse, fine): 0
        for coarse in coarse_values
        for fine in fine_values
    }
    return make_remaining_ordered_benchmark_schedule(
        coarse_values=coarse_values,
        fine_values=fine_values,
        n_trials=n_trials,
        existing_counts=existing_counts,
    )


def make_remaining_ordered_benchmark_schedule(coarse_values, fine_values, n_trials,
                                              existing_counts=None):
    """Return the ordered schedule for the trials still missing."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")

    combinations = [(coarse, fine) for fine in fine_values for coarse in coarse_values]
    if not combinations:
        raise ValueError("At least one coarse/fine combination is required.")

    existing_counts = existing_counts or {}
    schedule = []
    for trial_number in range(1, n_trials + 1):
        round_order = [
            combo for combo in combinations
            if existing_counts.get(combo, 0) < trial_number
        ]
        if not round_order:
            continue

        schedule.extend(round_order)

    return schedule


def make_interleaved_benchmark_schedule(coarse_values, fine_values, n_trials, rng=None):
    """Backward-compatible wrapper for the deterministic benchmark order."""
    return make_ordered_benchmark_schedule(coarse_values, fine_values, n_trials)


def make_remaining_interleaved_benchmark_schedule(coarse_values, fine_values, n_trials,
                                                  existing_counts=None, rng=None):
    """Backward-compatible wrapper for the deterministic remaining order."""
    return make_remaining_ordered_benchmark_schedule(
        coarse_values=coarse_values,
        fine_values=fine_values,
        n_trials=n_trials,
        existing_counts=existing_counts,
    )


def summarize_benchmark_csv(raw_csv_path="benchmark_raw.csv",
                            summary_csv_path="benchmark_summary.csv",
                            coarse_values=None,
                            fine_values=None):
    """Recompute benchmark_summary.csv from benchmark_raw.csv."""
    import csv

    summary_data = {}
    with open(raw_csv_path, "r", newline="") as raw_f:
        reader = csv.DictReader(raw_f)
        for row in reader:
            key = (int(row["coarse"]), int(row["fine"]))
            if key not in summary_data:
                summary_data[key] = {"elapsed": [], "power": []}
            summary_data[key]["elapsed"].append(float(row["elapsed_seconds"]))
            summary_data[key]["power"].append(float(row["final_power_dbm"]))

    if coarse_values is None:
        coarse_values = sorted({key[0] for key in summary_data})
    if fine_values is None:
        fine_values = sorted({key[1] for key in summary_data})

    summary_fieldnames = [
        "coarse", "fine", "n_trials",
        "elapsed_mean", "elapsed_std",
        "power_mean", "power_std",
        "power_median", "power_min", "power_max",
    ]

    with open(summary_csv_path, "w", newline="") as summary_f:
        writer = csv.DictWriter(summary_f, fieldnames=summary_fieldnames)
        writer.writeheader()

        for coarse in coarse_values:
            for fine in fine_values:
                key = (coarse, fine)
                if key not in summary_data:
                    continue

                elapsed = np.asarray(summary_data[key]["elapsed"], dtype=float)
                power = np.asarray(summary_data[key]["power"], dtype=float)

                writer.writerow({
                    "coarse": coarse,
                    "fine": fine,
                    "n_trials": int(power.size),
                    "elapsed_mean": round(float(np.mean(elapsed)), 3),
                    "elapsed_std": round(float(np.std(elapsed)), 3),
                    "power_mean": round(float(np.mean(power)), 4),
                    "power_std": round(float(np.std(power)), 4),
                    "power_median": round(float(np.median(power)), 4),
                    "power_min": round(float(np.min(power)), 4),
                    "power_max": round(float(np.max(power)), 4),
                })

    return summary_data


def benchmark_fast_sweep(n_trials=120, coarse_values=None, fine_values=None,
                         settle_time=3.0, measure_duration=2.0,
                         sweep_start=0, sweep_end=154, sample_interval=0.01,
                         fine_window=5, fine_settle_time=0.1,
                         raw_csv_path="benchmark_raw.csv",
                         summary_csv_path="benchmark_summary.csv",
                         random_seed=None,
                         resume=True):
    """Benchmark fast_sweep with all combinations of coarse and fine iterations.

    The runs are ordered in repeated rounds as C1F1, C2F1, C3F1, ...,
    then C1F2, C2F2, and so on. This makes the measurement order predictable
    while still distributing repeated trials over the whole run.

    For each (coarse, fine) combination:
      - Run n_trials with random start positions in deterministic C/F order
      - After optimization: wait settle_time, then measure for measure_duration
      - Save raw trial data to raw_csv_path
      - Save summary (mean, std) to summary_csv_path

    If resume=True and raw_csv_path already exists, existing rows are kept and
    only missing trials are appended until each setup reaches n_trials.
    """
    if coarse_values is None:
        coarse_values = [1, 2, 3, 4, 5]
    if fine_values is None:
        fine_values = [1, 2, 3, 4, 5]

    import csv

    rng = np.random.default_rng(random_seed)
    combinations = [(coarse, fine) for fine in fine_values for coarse in coarse_values]
    trial_counts = {combo: 0 for combo in combinations}
    last_run_order = 0

    if resume and os.path.exists(raw_csv_path):
        with open(raw_csv_path, "r", newline="") as existing_f:
            reader = csv.DictReader(existing_f)
            for row_idx, row in enumerate(reader, start=1):
                key = (int(row["coarse"]), int(row["fine"]))
                if key in trial_counts:
                    trial_counts[key] = max(trial_counts[key], int(row["trial"]))
                if "run_order" in row and row["run_order"]:
                    last_run_order = max(last_run_order, int(row["run_order"]))
                else:
                    last_run_order = max(last_run_order, row_idx)

    schedule = make_ordered_benchmark_schedule(
        coarse_values=coarse_values,
        fine_values=fine_values,
        n_trials=n_trials,
    ) if not resume else make_remaining_ordered_benchmark_schedule(
        coarse_values=coarse_values,
        fine_values=fine_values,
        n_trials=n_trials,
        existing_counts=trial_counts,
    )
    total_runs = len(schedule)

    raw_fieldnames = [
        "run_order", "test_started_at", "test_finished_at",
        "trial", "coarse", "fine",
        "start_arm1", "start_arm2", "start_arm3",
        "final_arm1", "final_arm2", "final_arm3",
        "elapsed_seconds", "final_power_dbm"
    ]

    file_exists = os.path.exists(raw_csv_path)
    if resume and file_exists:
        ensure_benchmark_csv_columns(raw_csv_path, raw_fieldnames)

    write_header = not (resume and file_exists and os.path.getsize(raw_csv_path) > 0)
    mode = "a" if resume and file_exists else "w"
    writer_fieldnames = raw_fieldnames
    if resume and file_exists and os.path.getsize(raw_csv_path) > 0:
        with open(raw_csv_path, "r", newline="") as existing_f:
            existing_fieldnames = csv.DictReader(existing_f).fieldnames
        if existing_fieldnames:
            writer_fieldnames = existing_fieldnames

    if total_runs == 0:
        print(f"\nAll requested benchmark trials already exist in {raw_csv_path}.")
        summarize_benchmark_csv(
            raw_csv_path=raw_csv_path,
            summary_csv_path=summary_csv_path,
            coarse_values=coarse_values,
            fine_values=fine_values,
        )
        return

    with open(raw_csv_path, mode, newline="") as raw_f:
        writer = csv.DictWriter(raw_f, fieldnames=writer_fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
            raw_f.flush()
            os.fsync(raw_f.fileno())

        for run_order, (coarse, fine) in enumerate(schedule, start=1):
            absolute_run_order = last_run_order + run_order
            key = (coarse, fine)
            trial_counts[key] += 1
            trial_idx = trial_counts[key]

            start_positions = [int(rng.integers(0, 155)) for _ in range(3)]
            test_started_at = current_timestamp()

            print(
                f"\nRun {run_order}/{total_runs} (row {absolute_run_order}): "
                f"C{coarse}F{fine}, trial {trial_idx}/{n_trials}"
            )

            set_arms(*start_positions)
            start_time = time.perf_counter()

            final_positions = fast_sweep(
                coarse_iterations=coarse,
                fine_iterations=fine,
                start_positions=start_positions,
                sweep_start=sweep_start,
                sweep_end=sweep_end,
                sample_interval=sample_interval,
                fine_window=fine_window,
                fine_settle_time=fine_settle_time,
                plot=False,
            )

            elapsed_seconds = time.perf_counter() - start_time

            time.sleep(settle_time)
            final_power_dbm = measure_power_avg(duration_seconds=measure_duration)
            test_finished_at = current_timestamp()

            write_benchmark_row(writer, raw_f, {
                "run_order": absolute_run_order,
                "test_started_at": test_started_at,
                "test_finished_at": test_finished_at,
                "trial": trial_idx,
                "coarse": coarse,
                "fine": fine,
                "start_arm1": start_positions[0],
                "start_arm2": start_positions[1],
                "start_arm3": start_positions[2],
                "final_arm1": final_positions[0],
                "final_arm2": final_positions[1],
                "final_arm3": final_positions[2],
                "elapsed_seconds": round(elapsed_seconds, 3),
                "final_power_dbm": round(final_power_dbm, 4),
            })

            print(
                f"  {elapsed_seconds:.2f} s, "
                f"{final_power_dbm:.2f} dBm, "
                f"positions {final_positions}"
            )

    summarize_benchmark_csv(
        raw_csv_path=raw_csv_path,
        summary_csv_path=summary_csv_path,
        coarse_values=coarse_values,
        fine_values=fine_values,
    )

    print(f"\nRaw data saved to {raw_csv_path}")
    print(f"Summary saved to {summary_csv_path}")


def run_top_benchmark(
    n_top=10,
    n_extra_trials=10,
    summary_path="benchmark_summary.csv",
    raw_path="benchmark_raw.csv",
    settle_time=3.0,
    measure_duration=2.0,
    sweep_start=0,
    sweep_end=154,
    sample_interval=0.01,
    fine_window=5,
    fine_settle_time=0.1,
):
    import csv

    # Step 1: Read summary and compute efficiency score = power_mean / elapsed_mean
    summary_rows = []
    with open(summary_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            summary_rows.append({
                "coarse":       int(row["coarse"]),
                "fine":         int(row["fine"]),
                "elapsed_mean": float(row["elapsed_mean"]),
                "power_mean":   float(row["power_mean"]),
            })

    summary_rows.sort(key=lambda r: r["power_mean"])
    top_combos = summary_rows[:n_top]

    # Step 2: Print selection table
    print(f"\n{'='*55}")
    print(f"Top {n_top} kombinationer (laveste power_mean dBm)")
    print(f"{'='*55}")
    print(f"{'Rank':>4}  {'Coarse':>6}  {'Fine':>4}  {'Power (dBm)':>11}  {'Tid (s)':>8}")
    print(f"{'-'*55}")
    for rank, row in enumerate(top_combos, start=1):
        print(
            f"{rank:>4}  {row['coarse']:>6}  {row['fine']:>4}  "
            f"{row['power_mean']:>11.4f}  {row['elapsed_mean']:>8.3f}"
        )
    print(f"{'='*55}\n")

    # Step 3: Find the highest existing trial number per (coarse, fine)
    last_trial = {}
    last_run_order = 0
    with open(raw_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            key = (int(row["coarse"]), int(row["fine"]))
            t = int(row["trial"])
            if key not in last_trial or t > last_trial[key]:
                last_trial[key] = t
            if "run_order" in row and row["run_order"]:
                last_run_order = max(last_run_order, int(row["run_order"]))
            else:
                last_run_order = max(last_run_order, row_idx)

    raw_fieldnames = [
        "run_order", "test_started_at", "test_finished_at",
        "trial", "coarse", "fine",
        "start_arm1", "start_arm2", "start_arm3",
        "final_arm1", "final_arm2", "final_arm3",
        "elapsed_seconds", "final_power_dbm",
    ]

    ensure_benchmark_csv_columns(raw_path, raw_fieldnames)

    # Step 4: Run extra trials and append to raw CSV (no header re-write)
    with open(raw_path, "a", newline="") as raw_f:
        writer = csv.DictWriter(raw_f, fieldnames=raw_fieldnames)

        for combo_rank, combo in enumerate(top_combos, start=1):
            coarse = combo["coarse"]
            fine   = combo["fine"]
            key    = (coarse, fine)
            next_trial = last_trial.get(key, 0) + 1

            print(
                f"[{combo_rank}/{n_top}] C{coarse}F{fine}  "
                f"power_mean={combo['power_mean']:.4f} dBm  "
                f"(trials {next_trial}-{next_trial + n_extra_trials - 1})"
            )

            for trial_offset in range(n_extra_trials):
                trial_num = next_trial + trial_offset
                last_run_order += 1
                start_positions = [int(np.random.randint(0, 155)) for _ in range(3)]
                test_started_at = current_timestamp()
                set_arms(*start_positions)

                t0 = time.perf_counter()
                final_positions = fast_sweep(
                    coarse_iterations=coarse,
                    fine_iterations=fine,
                    start_positions=start_positions,
                    sweep_start=sweep_start,
                    sweep_end=sweep_end,
                    sample_interval=sample_interval,
                    fine_window=fine_window,
                    fine_settle_time=fine_settle_time,
                    plot=False,
                )
                elapsed_seconds = time.perf_counter() - t0

                time.sleep(settle_time)
                final_power_dbm = measure_power_avg(duration_seconds=measure_duration)
                test_finished_at = current_timestamp()

                write_benchmark_row(writer, raw_f, {
                    "run_order":       last_run_order,
                    "test_started_at": test_started_at,
                    "test_finished_at": test_finished_at,
                    "trial":           trial_num,
                    "coarse":          coarse,
                    "fine":            fine,
                    "start_arm1":      start_positions[0],
                    "start_arm2":      start_positions[1],
                    "start_arm3":      start_positions[2],
                    "final_arm1":      final_positions[0],
                    "final_arm2":      final_positions[1],
                    "final_arm3":      final_positions[2],
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "final_power_dbm": round(final_power_dbm, 4),
                })

                print(
                    f"  Trial {trial_num}: "
                    f"{elapsed_seconds:.1f} s, "
                    f"{final_power_dbm:.2f} dBm, "
                    f"pos={final_positions}"
                )

            last_trial[key] = next_trial + n_extra_trials - 1

    summarize_benchmark_csv(raw_csv_path=raw_path, summary_csv_path=summary_path)

    print(f"\nFaerdig. {n_top * n_extra_trials} nye raekker tilfoejet til {raw_path}")
    print(f"Summary genberegnet og gemt i {summary_path}")


if __name__ == "__main__":
    benchmark_fast_sweep(n_trials=120)
