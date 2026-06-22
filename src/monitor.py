"""
monitor.py
Continuous polarization monitor for the HSOC lab.

The power meter reports optical power in dBm, where lower / more negative is
better for this setup. The lab requirement is phrased as a positive extinction
value: reach above 24 dB and repolarize when it drifts below 20 dB.

That maps to:
  extinction_db = -power_dbm
  target extinction > 24 dB  -> optimizer threshold <= -24 dBm
  repolarize below 20 dB     -> measured power > -20 dBm

Output: monitor_log.csv (appended on every run, so data survives restarts)
Stop with Ctrl+C.
"""

import csv
import time
from datetime import datetime

import pyvisa as visa

from optimize_c2f2 import _get_power, optimize_polarization

# Configuration
VISA_ADDR = 'USB0::0x1313::0x8078::P0028333::INSTR'

TARGET_EXTINCTION_DB = 24.0
REPOLARIZE_BELOW_EXTINCTION_DB = 20.0
EXPECTED_DRIFT_DB_PER_MIN = 0.4
MIN_HOLD_TIME_MIN = 10.0

TARGET_POWER_DBM = -TARGET_EXTINCTION_DB
REPOLARIZE_POWER_DBM = -REPOLARIZE_BELOW_EXTINCTION_DB

POLL_INTERVAL = 60
LOG_FILE = 'monitor_log.csv'

FIELDNAMES = ['timestamp', 'event', 'power_dbm', 'arm1', 'arm2', 'arm3', 'duration_s']


def extinction_db(power_dbm: float) -> float:
    """Convert measured dBm to the positive extinction value used in the lab."""
    return -power_dbm


def expected_hold_time_min(
    target_extinction_db: float = TARGET_EXTINCTION_DB,
    repolarize_below_extinction_db: float = REPOLARIZE_BELOW_EXTINCTION_DB,
    drift_db_per_min: float = EXPECTED_DRIFT_DB_PER_MIN,
) -> float:
    """Return expected minutes between target and repolarization threshold."""
    if drift_db_per_min <= 0:
        raise ValueError("drift_db_per_min must be positive.")
    return (target_extinction_db - repolarize_below_extinction_db) / drift_db_per_min


def should_repolarize(power_dbm: float) -> bool:
    """Return True when extinction has drifted below the allowed threshold."""
    return extinction_db(power_dbm) < REPOLARIZE_BELOW_EXTINCTION_DB


def _validate_hold_window() -> None:
    hold_time = expected_hold_time_min()
    if hold_time < MIN_HOLD_TIME_MIN:
        raise ValueError(
            "Target/repolarization thresholds do not cover the requested "
            f"{MIN_HOLD_TIME_MIN:.1f} min hold time at "
            f"{EXPECTED_DRIFT_DB_PER_MIN:.3f} dB/min drift. "
            f"Expected hold time is {hold_time:.1f} min."
        )


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _log(writer, event: str, power: float, positions=None, duration=None):
    row = {
        'timestamp': _now(),
        'event': event,
        'power_dbm': f'{power:.3f}',
        'arm1': positions[0] if positions else '',
        'arm2': positions[1] if positions else '',
        'arm3': positions[2] if positions else '',
        'duration_s': f'{duration:.1f}' if duration is not None else '',
    }
    writer.writerow(row)
    print(
        f"[{row['timestamp']}]  {event:<20s}  {power:>8.3f} dBm"
        f"  extinction={extinction_db(power):>6.3f} dB"
        + (f"  pos={positions}  ({duration:.1f}s)" if positions else "")
    )


def optimize_to_target(power_meter, writer=None, file_handle=None):
    """Run polarization optimization until the 24 dB extinction target is met."""
    t0 = time.time()
    positions = optimize_polarization(power_meter, threshold_dbm=TARGET_POWER_DBM)
    duration = time.time() - t0
    achieved = _get_power(power_meter)

    if writer is not None:
        _log(writer, 'optimization', achieved, positions, duration)
        if file_handle is not None:
            file_handle.flush()

    return achieved, positions, duration


def main():
    _validate_hold_window()

    pm = visa.ResourceManager().open_resource(VISA_ADDR)

    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if f.tell() == 0:
            writer.writeheader()

        print(f"Monitoring started. Logging to {LOG_FILE} (Ctrl+C to stop)")
        print(
            "Config: target extinction "
            f">= {TARGET_EXTINCTION_DB:.1f} dB, repolarize below "
            f"{REPOLARIZE_BELOW_EXTINCTION_DB:.1f} dB, expected hold "
            f"{expected_hold_time_min():.1f} min\n"
        )

        next_poll = time.time()

        while True:
            now = time.time()

            if now >= next_poll:
                power = _get_power(pm)
                _log(writer, 'drift', power)
                f.flush()

                if should_repolarize(power):
                    print(
                        "  -> extinction below "
                        f"{REPOLARIZE_BELOW_EXTINCTION_DB:.1f} dB, "
                        f"optimizing to >= {TARGET_EXTINCTION_DB:.1f} dB..."
                    )
                    optimize_to_target(pm, writer, f)

                next_poll = time.time() + POLL_INTERVAL
            else:
                time.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
