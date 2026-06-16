# run_continuous_test.py
# Automation of the High-Speed Optical Communications (HSOC) lab
# Course: 34720 | Group: 3 | Year: 2026
#
# Long-running test harness: compares QHQ vs HQQ paddle orderings.
# Runs optimize_polarization every 10 minutes (alternating orderings),
# measures power 5 s after each run, and logs continuous drift data.

import csv
import itertools
import math
import os
import shutil
import sys
import threading
import time
from datetime import datetime

import pyvisa as visa

from optimize_c2f2 import optimize_polarization

VISA_ADDRESS        = 'USB0::0x1313::0x8078::P0028333::INSTR'
OPT_INTERVAL_SEC    = 5 * 60
POST_OPT_SETTLE_SEC = 5
DRIFT_INTERVAL_SEC  = 60.0
DISK_FREE_MIN_BYTES = 100 * 1024 * 1024
OPT_CSV             = 'opt_results.csv'
DRIFT_CSV           = 'drift.csv'

ORDERINGS = [
    ('QHQ', [0, 1, 2]),
    ('HQQ', [1, 0, 2]),
]

OPT_FIELDS   = ['timestamp_optimized', 'arm_order', 'arm1', 'arm2', 'arm3',
                 'timestamp_measured', 'power_dbm']
DRIFT_FIELDS = ['timestamp', 'power_dbm']


def watts_to_dbm(watts: float) -> float:
    return 10.0 * math.log10(watts * 1000.0)


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def open_csv_writer(path: str, fieldnames: list):
    file_is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    fh = open(path, 'a', newline='')
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if file_is_new:
        writer.writeheader()
        fh.flush()
        os.fsync(fh.fileno())
    return fh, writer


def write_row(writer, fh, row: dict) -> None:
    writer.writerow(row)
    fh.flush()
    os.fsync(fh.fileno())


def safe_read_power(power_meter) -> float | None:
    try:
        return float(power_meter.query('READ?'))
    except Exception as exc:
        print(f"ERROR reading power meter: {exc}", file=sys.stderr)
        return None


def check_disk_space() -> None:
    free = shutil.disk_usage('.').free
    if free < DISK_FREE_MIN_BYTES:
        print(
            f"WARNING: disk space below 100 MB ({free // 1024 // 1024} MB free). Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)


def drift_loop(power_meter, visa_lock, stop_event, writer, fh) -> None:
    while not stop_event.is_set():
        t0 = time.monotonic()

        with visa_lock:
            watts = safe_read_power(power_meter)

        if watts is not None:
            check_disk_space()
            write_row(writer, fh, {
                'timestamp': current_timestamp(),
                'power_dbm': round(watts_to_dbm(watts), 4),
            })

        elapsed = time.monotonic() - t0
        stop_event.wait(timeout=max(0.0, DRIFT_INTERVAL_SEC - elapsed))


def optimization_loop(power_meter, visa_lock, stop_event, writer, fh) -> None:
    for label, order in itertools.cycle(ORDERINGS):
        if stop_event.is_set():
            break
        cycle_start = time.monotonic()

        with visa_lock:
            positions = optimize_polarization(power_meter, arm_order=order)
        ts_optimized = current_timestamp()

        stop_event.wait(timeout=POST_OPT_SETTLE_SEC)
        if stop_event.is_set():
            break

        with visa_lock:
            watts = safe_read_power(power_meter)
        ts_measured = current_timestamp()

        if watts is not None:
            check_disk_space()
            write_row(writer, fh, {
                'timestamp_optimized': ts_optimized,
                'arm_order':           label,
                'arm1':                positions[0],
                'arm2':                positions[1],
                'arm3':                positions[2],
                'timestamp_measured':  ts_measured,
                'power_dbm':           round(watts_to_dbm(watts), 4),
            })

        remaining = OPT_INTERVAL_SEC - (time.monotonic() - cycle_start)
        stop_event.wait(timeout=max(0.0, remaining))


def main() -> None:
    rm = visa.ResourceManager()
    power_meter = rm.open_resource(VISA_ADDRESS)

    stop_event = threading.Event()
    visa_lock  = threading.Lock()

    drift_fh, drift_writer = open_csv_writer(DRIFT_CSV, DRIFT_FIELDS)
    opt_fh,   opt_writer   = open_csv_writer(OPT_CSV,   OPT_FIELDS)

    drift_thread = threading.Thread(
        target=drift_loop,
        args=(power_meter, visa_lock, stop_event, drift_writer, drift_fh),
        daemon=True,
        name='drift-monitor',
    )
    drift_thread.start()
    print(f"Started. Logging to '{OPT_CSV}' and '{DRIFT_CSV}'. Press Ctrl+C to stop.")

    try:
        optimization_loop(power_meter, visa_lock, stop_event, opt_writer, opt_fh)
    except KeyboardInterrupt:
        print("\nCtrl+C received — stopping.", file=sys.stderr)
    finally:
        stop_event.set()
        drift_thread.join(timeout=5.0)
        drift_fh.close()
        opt_fh.close()
        power_meter.close()
        print("Clean shutdown complete.", file=sys.stderr)


if __name__ == '__main__':
    main()
