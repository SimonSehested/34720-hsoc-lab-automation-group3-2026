# optimize_c2f2_max.py
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
_SAMPLE_INTERVAL  = 0.002  # seconds between readings during coarse sweep
_FINE_WINDOW      = 5      # positions either side for fine refinement
_FINE_SETTLE      = 0.1    # settle time before each fine measurement (s)
_THRESHOLD_SETTLE = 2.0    # seconds to settle before the final -20 dBm check
VISA_ADDR         = 'USB0::0x1313::0x8078::P0028333::INSTR'


def _get_power(power_meter) -> float:
    watts = float(power_meter.query('READ?'))
    return 10.0 * np.log10(watts / 1e-3)  # W → dBm


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


def _coarse_sweep(arm_idx: int, positions: list, power_meter) -> int:
    """Sweep arm from _SWEEP_START to _SWEEP_END while sampling power.
    Returns the position with the highest measured power."""
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
    return int(round(pos_trace[int(np.argmax(pwr))]))


def _fine_refine(arm_idx: int, positions: list, power_meter) -> int:
    """Measure power at center ± _FINE_WINDOW. Expands window if the maximum
    lands at the boundary. Returns the position with the highest power."""
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
        best_idx = int(np.argmax(pwr_arr))
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
    start_positions=None,
    threshold_dbm: float = -20.0,
) -> list:
    """Optimize laser polarization using the C2F2 algorithm (HQQ paddle order).

    Parameters
    ----------
    power_meter : str or pyvisa resource
        VISA address string (e.g. 'USB0::0x1313::...::INSTR') or an already-
        opened pyvisa resource object for the optical power meter.
    coarse_iterations : int, default 2
        Number of full coarse sweeps (positions 0-154) per attempt. More
        iterations improve the starting estimate for fine refinement but take
        longer.
    fine_iterations : int, default 2
        Number of fine refinement passes per attempt. Each pass steps through
        ±_FINE_WINDOW positions around the current best and expands if the
        maximum sits at the boundary.
    start_positions : list of 3 ints or None, default None
        Initial [arm1, arm2, arm3] positions (0-154). If None, positions are
        chosen randomly at the start of each attempt.
    threshold_dbm : float, default -20.0
        Acceptance threshold in dBm. The function returns as soon as the
        measured power is at or above this value. Higher (less negative) values
        require stronger received power before accepting (e.g. -15 is stricter
        than -20).

    Returns
    -------
    list
        Final [arm1, arm2, arm3] paddle positions (0-154).
    """
    if isinstance(power_meter, str):
        power_meter = visa.ResourceManager().open_resource(power_meter)

    _order = [1, 0, 2]  # HQQ: middle paddle first

    positions = (
        list(start_positions)
        if start_positions is not None
        else [random.randint(0, _SWEEP_END) for _ in range(3)]
    )

    while True:
        _set_arms(*positions)

        # --- Coarse phase ---
        for _ in range(coarse_iterations):
            for arm_idx in _order:
                positions[arm_idx] = _coarse_sweep(arm_idx, positions, power_meter)
                _set_arms(*positions)

        # --- Fine phase ---
        for _ in range(fine_iterations):
            for arm_idx in _order:
                positions[arm_idx] = _fine_refine(arm_idx, positions, power_meter)
                _set_arms(*positions)

        time.sleep(_THRESHOLD_SETTLE)
        final_power = _get_power(power_meter)
        if final_power >= threshold_dbm:
            return positions

        positions = [random.randint(0, _SWEEP_END) for _ in range(3)]


def main() -> None:
    power_meter = visa.ResourceManager().open_resource(VISA_ADDR)
    try:
        positions = optimize_polarization(power_meter)
        power = _get_power(power_meter)
        print(f"Positioner : {positions}")
        print(f"Målt power : {power:.2f} dBm")
    finally:
        power_meter.close()


if __name__ == "__main__":
    main()
