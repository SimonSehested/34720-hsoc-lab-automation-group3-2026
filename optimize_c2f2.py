# optimize_c2f2.py
# Automation of the High-Speed Optical Communications (HSOC) lab
# Course: 34720 | Group: 3 | Year: 2026
#
# C2F2 polarization optimizer — clean submission file.
# Public API: optimize_polarization()

import random
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


_SWEEP_START      = 0
_SWEEP_END        = 154
_SAMPLE_INTERVAL  = 0.01   # seconds between readings during coarse sweep
_FINE_WINDOW      = 5      # positions either side for fine refinement
_FINE_SETTLE      = 0.1    # settle time before each fine measurement (s)
_EDGE_MARGIN      = 5      # distance from 0/154 that counts as edge zone
_SIGNAL_THRESHOLD = -25.0  # dBm — minimum power required to trigger edge restart
_SIGNAL_TIMEOUT   = 5.0    # seconds to wait for valid signal before skipping edge check


def _get_power(power_meter) -> float:
    return float(power_meter.query('READ?'))


def _set_arms(arm1: int, arm2: int, arm3: int) -> None:
    if pol_ctrl is None:
        raise RuntimeError("MPC320 not connected.")
    pol_ctrl.set_position(1, int(arm1))
    pol_ctrl.set_position(2, int(arm2))
    pol_ctrl.set_position(3, int(arm3))


def _set_arm(arm_idx: int, position: int) -> None:
    if pol_ctrl is None:
        raise RuntimeError("MPC320 not connected.")
    pol_ctrl.set_position(arm_idx + 1, int(position))


def _needs_edge_check(positions: list) -> bool:
    return any(p < _EDGE_MARGIN or p > _SWEEP_END - _EDGE_MARGIN for p in positions)


def _has_signal(power_meter) -> bool:
    """Poll power meter for up to _SIGNAL_TIMEOUT seconds.
    Returns True only if a reading below _SIGNAL_THRESHOLD is observed.
    Edge restarts are skipped when no valid signal is detected."""
    deadline = time.time() + _SIGNAL_TIMEOUT
    while time.time() < deadline:
        if _get_power(power_meter) < _SIGNAL_THRESHOLD:
            return True
        time.sleep(0.05)
    return False


def _coarse_sweep(arm_idx: int, positions: list, power_meter) -> int:
    """Sweep arm from _SWEEP_START to _SWEEP_END while sampling power.
    Returns the position with the lowest measured power."""
    readings = []
    timestamps = []
    stop_event = threading.Event()

    def _sample(t0: float) -> None:
        while not stop_event.is_set():
            timestamps.append(time.time() - t0)
            readings.append(_get_power(power_meter))
            time.sleep(_SAMPLE_INTERVAL)

    positions[arm_idx] = _SWEEP_START
    _set_arms(*positions)

    t0 = time.time()
    sampler = threading.Thread(target=_sample, args=(t0,), daemon=True)
    sampler.start()
    try:
        _set_arm(arm_idx, _SWEEP_END)
    finally:
        stop_event.set()
        sampler.join()

    positions[arm_idx] = _SWEEP_END

    ts = np.asarray(timestamps, dtype=float)
    pwr = np.asarray(readings, dtype=float)
    if ts.size == 0:
        return _SWEEP_END // 2  # fallback to midpoint if no samples collected

    pos_trace = (
        _SWEEP_START + (_SWEEP_END - _SWEEP_START) * (ts / ts[-1])
        if ts[-1] > 0
        else np.full_like(ts, float(_SWEEP_START))
    )
    return int(round(pos_trace[int(np.argmin(pwr))]))


def _fine_refine(arm_idx: int, positions: list, power_meter) -> int:
    """Measure power at center ± _FINE_WINDOW. Expands window if the minimum
    lands at the boundary. Returns the position with the lowest power."""
    center = positions[arm_idx]
    measured: dict = {}

    def _measure_range(lo: int, hi: int) -> None:
        for pos in range(lo, hi + 1):
            if pos in measured:
                continue
            scan = list(positions)
            scan[arm_idx] = pos
            _set_arms(*scan)
            time.sleep(_FINE_SETTLE)
            measured[pos] = _get_power(power_meter)

    lo = max(0, center - _FINE_WINDOW)
    hi = min(_SWEEP_END, center + _FINE_WINDOW)
    _measure_range(lo, hi)

    while True:
        pos_arr = np.array(sorted(measured), dtype=int)
        pwr_arr = np.array([measured[p] for p in pos_arr], dtype=float)
        best_idx = int(np.argmin(pwr_arr))
        best_pos = int(pos_arr[best_idx])

        expanded = False
        if best_idx == 0 and lo > 0:
            new_lo = max(0, lo - _FINE_WINDOW)
            _measure_range(new_lo, lo - 1)
            lo = new_lo
            expanded = True
        elif best_idx == len(pos_arr) - 1 and hi < _SWEEP_END:
            new_hi = min(_SWEEP_END, hi + _FINE_WINDOW)
            _measure_range(hi + 1, new_hi)
            hi = new_hi
            expanded = True

        if not expanded:
            return best_pos


def optimize_polarization(
    power_meter,
    coarse_iterations: int = 2,
    fine_iterations: int = 2,
    max_restarts: int = 3,
    start_positions=None,
    arm_order: list = None,
) -> list:
    """Optimize laser polarization using the C2F2 algorithm.

    power_meter may be a VISA resource address string or an already-opened
    pyvisa resource object.

    arm_order controls the paddle sweep sequence: [0,1,2] = QHQ (default),
    [1,0,2] = HQQ (middle paddle first).

    Runs coarse_iterations full sweeps (0-154) then fine_iterations
    refinements per attempt. Restarts up to max_restarts times if any arm
    lands in the edge zone (< 5 or > 149). When the cap is reached, takes
    the best positions found so far and runs one final fine pass.

    Returns [arm1, arm2, arm3] -- final positions (0-154).
    """
    if isinstance(power_meter, str):
        power_meter = visa.ResourceManager().open_resource(power_meter)

    _order = arm_order if arm_order is not None else [0, 1, 2]

    positions = (
        list(start_positions)
        if start_positions is not None
        else [random.randint(0, _SWEEP_END) for _ in range(3)]
    )
    best_result = None  # (positions, power)
    restarts_used = 0

    while True:
        _set_arms(*positions)

        # --- Coarse phase ---
        for _ in range(coarse_iterations):
            for arm_idx in _order:
                positions[arm_idx] = _coarse_sweep(arm_idx, positions, power_meter)
                _set_arms(*positions)

        # Edge check #1 (after coarse) — only restart if signal < -25 dBm within 5 s
        if _needs_edge_check(positions) and _has_signal(power_meter):
            if restarts_used < max_restarts:
                restarts_used += 1
                positions = [random.randint(0, _SWEEP_END) for _ in range(3)]
                continue
            fallback = list(best_result[0]) if best_result else list(positions)
            for arm_idx in _order:
                fallback[arm_idx] = _fine_refine(arm_idx, fallback, power_meter)
                _set_arms(*fallback)
            return fallback

        # --- Fine phase ---
        for _ in range(fine_iterations):
            for arm_idx in _order:
                positions[arm_idx] = _fine_refine(arm_idx, positions, power_meter)
                _set_arms(*positions)

        # Track the best result by power
        power = _get_power(power_meter)
        if best_result is None or power < best_result[1]:
            best_result = (list(positions), power)

        # Edge check #2 (after fine) — only restart if signal < -25 dBm within 5 s
        if _needs_edge_check(positions) and _has_signal(power_meter):
            if restarts_used < max_restarts:
                restarts_used += 1
                positions = [random.randint(0, _SWEEP_END) for _ in range(3)]
                continue
            fallback = list(best_result[0])
            for arm_idx in range(3):
                fallback[arm_idx] = _fine_refine(arm_idx, fallback, power_meter)
                _set_arms(*fallback)
            return fallback

        return positions
    
optimize_polarization('USB0::0x1313::0x8078::P0028333::INSTR')