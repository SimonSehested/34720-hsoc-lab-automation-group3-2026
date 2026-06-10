# optimize_c2f2.py
# Automation of the High-Speed Optical Communications (HSOC) lab
# Course: 34720 | Group: 3 | Year: 2026
#
# C2F2 polarization optimizer — clean submission file.
# Public API: optimize_polarization()

import sys
import time
import threading
import random
import numpy as np
import pyvisa as visa

# NOTE: update this path to match your local installation
sys.path.append(r"C:\Users\FTNK-LocalAdm\Desktop\mikael_speciale\thjalfe_instrument_control\InstrumentControl\LabInstruments")
from pol_cons import ThorlabsMPC320

pol_ctrl = ThorlabsMPC320()
power_meter = visa.ResourceManager().open_resource('USB0::0x1313::0x8078::P0024464::INSTR')

_SWEEP_START     = 0
_SWEEP_END       = 154
_SAMPLE_INTERVAL = 0.01   # seconds between readings during coarse sweep
_FINE_WINDOW     = 5      # positions either side for fine refinement
_FINE_SETTLE     = 0.1    # settle time before each fine measurement (s)
_EDGE_MARGIN     = 5      # distance from 0/154 that counts as edge zone


def _get_power() -> float:
    return float(power_meter.query('READ?'))


def _set_arms(arm1: int, arm2: int, arm3: int) -> None:
    pol_ctrl.set_position(1, int(arm1))
    pol_ctrl.set_position(2, int(arm2))
    pol_ctrl.set_position(3, int(arm3))


def _set_arm(arm_idx: int, position: int) -> None:
    pol_ctrl.set_position(arm_idx + 1, int(position))


def _needs_edge_check(positions: list) -> bool:
    return any(p < _EDGE_MARGIN or p > _SWEEP_END - _EDGE_MARGIN for p in positions)


def _coarse_sweep(arm_idx: int, positions: list) -> int:
    """Sweep arm from _SWEEP_START to _SWEEP_END while sampling power.
    Returns the position with the lowest measured power."""
    readings = []
    timestamps = []
    stop_event = threading.Event()

    def _sample(t0: float) -> None:
        while not stop_event.is_set():
            timestamps.append(time.time() - t0)
            readings.append(_get_power())
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
