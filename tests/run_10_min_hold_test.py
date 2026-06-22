# Manual continuous hardware test for the HSOC polarization monitor.
#
# It optimizes to the 24 dB extinction target, logs once per minute, and
# repolarizes whenever extinction drifts below 20 dB. It runs until Ctrl+C.

import csv
import sys
import time
from pathlib import Path

import pyvisa as visa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from monitor import (
    FIELDNAMES,
    LOG_FILE,
    MIN_HOLD_TIME_MIN,
    POLL_INTERVAL,
    REPOLARIZE_BELOW_EXTINCTION_DB,
    TARGET_EXTINCTION_DB,
    VISA_ADDR,
    _get_power,
    _log,
    _validate_hold_window,
    extinction_db,
    optimize_to_target,
    should_repolarize,
)

def main() -> int:
    _validate_hold_window()

    power_meter = visa.ResourceManager().open_resource(VISA_ADDR)
    log_path = LOG_FILE

    try:
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if f.tell() == 0:
                writer.writeheader()

            print(
                "Starting continuous monitor test. "
                f"Target >= {TARGET_EXTINCTION_DB:.1f} dB extinction; "
                f"repolarize below {REPOLARIZE_BELOW_EXTINCTION_DB:.1f} dB; "
                f"logging every {POLL_INTERVAL} s."
            )
            print(f"Logging to {log_path}")
            print("Press Ctrl+C to stop.")

            last_optimized_at = None
            next_poll = time.monotonic()

            while True:
                now = time.monotonic()

                if last_optimized_at is None:
                    achieved, positions, duration = optimize_to_target(power_meter, writer, f)
                    last_optimized_at = time.monotonic()
                    print(
                        "Optimized: "
                        f"{extinction_db(achieved):.3f} dB extinction, "
                        f"positions={positions}, duration={duration:.1f}s"
                    )
                    next_poll = last_optimized_at + POLL_INTERVAL
                    continue

                if now >= next_poll:
                    power = _get_power(power_meter)
                    _log(writer, "drift", power)
                    f.flush()

                    if should_repolarize(power):
                        elapsed_min = (now - last_optimized_at) / 60.0
                        if elapsed_min < MIN_HOLD_TIME_MIN:
                            print(
                                "WARNING: extinction dropped below "
                                f"{REPOLARIZE_BELOW_EXTINCTION_DB:.1f} dB "
                                f"after {elapsed_min:.2f} min "
                                f"({extinction_db(power):.3f} dB).",
                                file=sys.stderr,
                            )
                        else:
                            print(
                                "Repolarizing after "
                                f"{elapsed_min:.2f} min "
                                f"({extinction_db(power):.3f} dB extinction)."
                            )

                        achieved, positions, duration = optimize_to_target(power_meter, writer, f)
                        last_optimized_at = time.monotonic()
                        print(
                            "Optimized: "
                            f"{extinction_db(achieved):.3f} dB extinction, "
                            f"positions={positions}, duration={duration:.1f}s"
                        )
                        next_poll = last_optimized_at + POLL_INTERVAL
                        continue

                    next_poll = now + POLL_INTERVAL
                else:
                    time.sleep(min(1.0, max(0.0, next_poll - now)))
    except KeyboardInterrupt:
        print("\nContinuous monitor test stopped.")
        return 0
    finally:
        power_meter.close()


if __name__ == "__main__":
    raise SystemExit(main())
