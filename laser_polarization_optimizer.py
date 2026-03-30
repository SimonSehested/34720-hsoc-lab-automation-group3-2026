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
    """Read current optical power from the power meter and return it in dBm."""
    power_dbm = 10 * np.log10(float(power_meter.query('READ?')) * 1000)
    return power_dbm


def set_arms(arm1, arm2, arm3):
    """Set the position of all three polarization controller arms (range: 0-154)."""
    pol_ctrl.set_position(1, arm1)
    pol_ctrl.set_position(2, arm2)
    pol_ctrl.set_position(3, arm3)


def optimize():
    """
    Optimize polarization by scanning each arm one at a time.
    For each arm, all 155 positions are tested and the one with
    minimum power is kept before moving on to the next arm.
    """
    # Starting positions for arms 1, 2, 3
    positions = [0, 77, 77]
    set_arms(*positions)

    for arm_idx in range(3):
        # Scan all positions for this arm while keeping the others fixed
        power_scan = []
        for step in range(155):
            scan_positions = positions.copy()
            scan_positions[arm_idx] = step
            set_arms(*scan_positions)
            power_scan.append(get_power())

        # Set this arm to the position with minimum power
        positions[arm_idx] = power_scan.index(min(power_scan))
        set_arms(*positions)

        # Debug: print power values around the minimum (uncomment to use)
        '''
        for i in range(-5, 5):
            print(power_scan[power_scan.index(min(power_scan)) + i])
        '''

optimize()
