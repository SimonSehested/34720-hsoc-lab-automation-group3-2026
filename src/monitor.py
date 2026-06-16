"""
monitor.py
Continuous polarization monitor for the HSOC lab.

Every POLL_INTERVAL seconds the power is measured and logged as a drift point.
If the power exceeds THRESHOLD_DBM the optimizer is triggered immediately;
the result (achieved dBm, paddle positions, duration) is logged separately.

Output: monitor_log.csv  (appended on every run, so data survives restarts)
Stop with Ctrl+C.
"""

import csv
import time
import pyvisa as visa
from datetime import datetime
from optimize_c2f2 import optimize_polarization, _get_power

# ── configuration ────────────────────────────────────────────────────────────
VISA_ADDR      = 'USB0::0x1313::0x8078::P0028333::INSTR'
THRESHOLD_DBM  = -20.0   # dBm — trigger optimizer above this
POLL_INTERVAL  = 60      # seconds between drift measurements
LOG_FILE       = 'monitor_log.csv'
# ─────────────────────────────────────────────────────────────────────────────

FIELDNAMES = ['timestamp', 'event', 'power_dbm', 'arm1', 'arm2', 'arm3', 'duration_s']


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _log(writer, event: str, power: float, positions=None, duration=None):
    row = {
        'timestamp':  _now(),
        'event':      event,
        'power_dbm':  f'{power:.3f}',
        'arm1':       positions[0] if positions else '',
        'arm2':       positions[1] if positions else '',
        'arm3':       positions[2] if positions else '',
        'duration_s': f'{duration:.1f}' if duration is not None else '',
    }
    writer.writerow(row)
    print(
        f"[{row['timestamp']}]  {event:<20s}  {power:>8.3f} dBm"
        + (f"  pos={positions}  ({duration:.1f}s)" if positions else "")
    )


def main():
    pm = visa.ResourceManager().open_resource(VISA_ADDR)

    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if f.tell() == 0:
            writer.writeheader()

        print(f"Monitoring started. Logging to {LOG_FILE}  (Ctrl+C to stop)\n")

        next_poll = time.time()

        while True:
            now = time.time()

            if now >= next_poll:
                power = _get_power(pm)
                _log(writer, 'drift', power)
                f.flush()

                if power > THRESHOLD_DBM:
                    print(f"  → above threshold ({THRESHOLD_DBM} dBm), optimizing...")
                    t0 = time.time()
                    positions = optimize_polarization(pm, threshold_dbm=THRESHOLD_DBM)
                    duration = time.time() - t0
                    achieved = _get_power(pm)
                    _log(writer, 'optimization', achieved, positions, duration)
                    f.flush()

                next_poll = time.time() + POLL_INTERVAL
            else:
                time.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
