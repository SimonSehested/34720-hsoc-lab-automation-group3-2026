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
