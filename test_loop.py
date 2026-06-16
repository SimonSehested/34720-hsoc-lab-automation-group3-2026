from optimize_c2f2 import optimize_polarization, _get_power
import pyvisa as visa

VISA_ADDR = 'USB0::0x1313::0x8078::P0028333::INSTR'

pm = visa.ResourceManager().open_resource(VISA_ADDR)

run = 1
while True:
    print(f"\n--- Kørsel {run} ---")
    positions = optimize_polarization(pm)
    power = _get_power(pm)
    print(f"Positioner : {positions}")
    print(f"Målt power : {power:.2f} dBm")
    run += 1
