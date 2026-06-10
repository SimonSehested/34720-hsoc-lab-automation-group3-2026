# C2F2 Clean Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `optimize_c2f2.py` — a single clean Python file with one public function `optimize_polarization()` using the C2F2 algorithm.

**Architecture:** All logic in one file. Private helpers (`_coarse_sweep`, `_fine_refine`, etc.) prefixed with `_`. Module-level instrument connections. Tests live in `tests/test_optimize_c2f2.py` and mock hardware at import time via `sys.modules`.

**Tech Stack:** Python 3.12, pyvisa, numpy, threading, pytest

---

## File Map

| File | Role |
|---|---|
| `optimize_c2f2.py` | The deliverable — one public function + private helpers |
| `tests/test_optimize_c2f2.py` | Unit tests (hardware mocked) |

---

### Task 1: File skeleton, constants, and I/O helpers

**Files:**
- Create: `optimize_c2f2.py`
- Create: `tests/test_optimize_c2f2.py`

- [ ] **Step 1: Create `optimize_c2f2.py` with imports, instrument connections, and constants**

```python
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
```

- [ ] **Step 2: Add I/O helpers and `_needs_edge_check` to `optimize_c2f2.py`**

Append below the constants block:

```python
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
```

- [ ] **Step 3: Write failing tests for `_needs_edge_check`**

Create `tests/test_optimize_c2f2.py`:

```python
import sys
from unittest.mock import MagicMock, patch

# Mock hardware modules before optimize_c2f2 is imported
_pyvisa_mock = MagicMock()
_pyvisa_mock.ResourceManager.return_value.open_resource.return_value = MagicMock()
sys.modules['pyvisa'] = _pyvisa_mock
sys.modules['pol_cons'] = MagicMock()

import optimize_c2f2


# _needs_edge_check

def test_needs_edge_check_safe():
    assert optimize_c2f2._needs_edge_check([10, 77, 100]) is False

def test_needs_edge_check_low_edge():
    assert optimize_c2f2._needs_edge_check([3, 77, 77]) is True

def test_needs_edge_check_high_edge():
    assert optimize_c2f2._needs_edge_check([77, 77, 150]) is True

def test_needs_edge_check_exactly_at_margin():
    # 5 and 149 are exactly at the boundary — NOT in edge zone
    assert optimize_c2f2._needs_edge_check([5, 77, 149]) is False
```

- [ ] **Step 4: Run tests — verify they fail (function not yet imported cleanly)**

```
pytest tests/test_optimize_c2f2.py -v
```

Expected: 4 failures or import errors (depends on whether `optimize_c2f2.py` is importable yet)

- [ ] **Step 5: Run tests — verify they pass after Step 2**

```
pytest tests/test_optimize_c2f2.py -v
```

Expected:
```
PASSED tests/test_optimize_c2f2.py::test_needs_edge_check_safe
PASSED tests/test_optimize_c2f2.py::test_needs_edge_check_low_edge
PASSED tests/test_optimize_c2f2.py::test_needs_edge_check_high_edge
PASSED tests/test_optimize_c2f2.py::test_needs_edge_check_exactly_at_margin
```

- [ ] **Step 6: Commit**

```bash
git add optimize_c2f2.py tests/test_optimize_c2f2.py
git commit -m "feat: add optimize_c2f2 skeleton with I/O helpers and edge check"
```

---

### Task 2: Implement `_coarse_sweep`

**Files:**
- Modify: `optimize_c2f2.py`

Adapted from `measure_during_sweep` in `laser_polarization_optimizer.py:452-488`.

- [ ] **Step 1: Append `_coarse_sweep` to `optimize_c2f2.py`**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add optimize_c2f2.py
git commit -m "feat: add _coarse_sweep to optimize_c2f2"
```

> Manual test (requires connected instruments): open a Python REPL, `import optimize_c2f2`, then `optimize_c2f2._coarse_sweep(0, [0, 77, 77])` — should return an integer 0–154.

---

### Task 3: Implement `_fine_refine`

**Files:**
- Modify: `optimize_c2f2.py`

Adapted from `refine_position` in `laser_polarization_optimizer.py:490-526`.

- [ ] **Step 1: Append `_fine_refine` to `optimize_c2f2.py`**

```python
def _fine_refine(arm_idx: int, positions: list) -> int:
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
            measured[pos] = _get_power()

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
```

- [ ] **Step 2: Commit**

```bash
git add optimize_c2f2.py
git commit -m "feat: add _fine_refine to optimize_c2f2"
```

> Manual test (requires connected instruments): `optimize_c2f2._fine_refine(0, [77, 77, 77])` — should return an integer 0–154.

---

### Task 4: Implement `optimize_polarization` and restart/fallback tests

**Files:**
- Modify: `optimize_c2f2.py`
- Modify: `tests/test_optimize_c2f2.py`

- [ ] **Step 1: Add failing tests for `optimize_polarization` to `tests/test_optimize_c2f2.py`**

Append to the test file:

```python
# optimize_polarization restart and fallback behaviour

def test_returns_positions_when_no_edge():
    with patch.multiple('optimize_c2f2',
        _coarse_sweep=MagicMock(return_value=77),
        _fine_refine=MagicMock(return_value=77),
        _get_power=MagicMock(return_value=0.001),
        _set_arms=MagicMock(),
        _needs_edge_check=MagicMock(return_value=False),
    ), patch('optimize_c2f2.random.randint', return_value=50):
        result = optimize_c2f2.optimize_polarization(
            coarse_iterations=1, fine_iterations=1, max_restarts=3,
            start_positions=[77, 77, 77],
        )
    assert result == [77, 77, 77]


def test_restarts_on_coarse_edge_then_succeeds():
    # attempt 1: edge after coarse → restart
    # attempt 2: safe after coarse, safe after fine → return
    edge_values = iter([True, False, False])
    fine_mock = MagicMock(return_value=77)

    with patch.multiple('optimize_c2f2',
        _coarse_sweep=MagicMock(return_value=77),
        _fine_refine=fine_mock,
        _get_power=MagicMock(return_value=0.001),
        _set_arms=MagicMock(),
        _needs_edge_check=MagicMock(side_effect=lambda p: next(edge_values)),
    ), patch('optimize_c2f2.random.randint', return_value=50):
        result = optimize_c2f2.optimize_polarization(
            coarse_iterations=1, fine_iterations=1, max_restarts=3,
            start_positions=[77, 77, 77],
        )
    assert result == [77, 77, 77]
    # fine ran once for attempt 2 (3 arms × 1 iteration)
    assert fine_mock.call_count == 3


def test_fallback_when_cap_hit_before_fine():
    # max_restarts=0, edge always True → cap hit immediately after coarse
    # best_result is None → fallback uses current positions and runs final fine
    fine_mock = MagicMock(return_value=77)

    with patch.multiple('optimize_c2f2',
        _coarse_sweep=MagicMock(return_value=77),
        _fine_refine=fine_mock,
        _get_power=MagicMock(return_value=0.001),
        _set_arms=MagicMock(),
        _needs_edge_check=MagicMock(return_value=True),
    ), patch('optimize_c2f2.random.randint', return_value=50):
        optimize_c2f2.optimize_polarization(
            coarse_iterations=1, fine_iterations=1, max_restarts=0,
            start_positions=[77, 77, 77],
        )
    # fallback: one fine pass over all 3 arms
    assert fine_mock.call_count == 3


def test_fallback_uses_best_result_when_cap_hit_after_fine():
    # attempt 1: safe coarse, fine runs, power=0.001, edge after fine → restart
    # attempt 2: safe coarse, fine runs, power=0.002, edge after fine → cap hit
    # fallback should start from attempt 1 positions (best power)
    power_calls = iter([0.001, 0.002])
    edge_calls = iter([False, True, False, True])
    fine_mock = MagicMock(return_value=42)  # distinct return to verify source

    with patch.multiple('optimize_c2f2',
        _coarse_sweep=MagicMock(return_value=77),
        _fine_refine=fine_mock,
        _get_power=MagicMock(side_effect=lambda: next(power_calls)),
        _set_arms=MagicMock(),
        _needs_edge_check=MagicMock(side_effect=lambda p: next(edge_calls)),

    ), patch('optimize_c2f2.random.randint', return_value=50):
        optimize_c2f2.optimize_polarization(
            coarse_iterations=1, fine_iterations=1, max_restarts=1,
            start_positions=[77, 77, 77],
        )
    # 3 fine calls (attempt 1 fine phase)
    # + 3 fine calls (attempt 2 fine phase)
    # + 3 fine calls (fallback final fine pass)
    # = 9
    assert fine_mock.call_count == 9
```

- [ ] **Step 2: Run tests — verify they fail**

```
pytest tests/test_optimize_c2f2.py -v -k "optimize_polarization or returns_positions or restarts or fallback"
```

Expected: all 4 new tests fail with `AttributeError` or `NameError` (`optimize_polarization` not defined yet).

- [ ] **Step 3: Append `optimize_polarization` to `optimize_c2f2.py`**

```python
def optimize_polarization(
    coarse_iterations: int = 2,
    fine_iterations: int = 2,
    max_restarts: int = 3,
    start_positions=None,
) -> list:
    """Optimize laser polarization using the C2F2 algorithm.

    Runs coarse_iterations full sweeps (0–154) then fine_iterations
    refinements per attempt. Restarts up to max_restarts times if any arm
    lands in the edge zone (< 5 or > 149). When the cap is reached, takes
    the best positions found so far and runs one final fine pass.

    Returns [arm1, arm2, arm3] — final positions (0–154).
    """
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
            for arm_idx in range(3):
                positions[arm_idx] = _coarse_sweep(arm_idx, positions)
                _set_arms(*positions)

        # Edge check #1 (after coarse)
        if _needs_edge_check(positions):
            if restarts_used < max_restarts:
                restarts_used += 1
                positions = [random.randint(0, _SWEEP_END) for _ in range(3)]
                continue
            fallback = list(best_result[0]) if best_result else list(positions)
            for arm_idx in range(3):
                fallback[arm_idx] = _fine_refine(arm_idx, fallback)
                _set_arms(*fallback)
            return fallback

        # --- Fine phase ---
        for _ in range(fine_iterations):
            for arm_idx in range(3):
                positions[arm_idx] = _fine_refine(arm_idx, positions)
                _set_arms(*positions)

        # Track the best result by power
        power = _get_power()
        if best_result is None or power < best_result[1]:
            best_result = (list(positions), power)

        # Edge check #2 (after fine)
        if _needs_edge_check(positions):
            if restarts_used < max_restarts:
                restarts_used += 1
                positions = [random.randint(0, _SWEEP_END) for _ in range(3)]
                continue
            fallback = list(best_result[0])
            for arm_idx in range(3):
                fallback[arm_idx] = _fine_refine(arm_idx, fallback)
                _set_arms(*fallback)
            return fallback

        return positions
```

- [ ] **Step 4: Run all tests — verify they pass**

```
pytest tests/test_optimize_c2f2.py -v
```

Expected:
```
PASSED tests/test_optimize_c2f2.py::test_needs_edge_check_safe
PASSED tests/test_optimize_c2f2.py::test_needs_edge_check_low_edge
PASSED tests/test_optimize_c2f2.py::test_needs_edge_check_high_edge
PASSED tests/test_optimize_c2f2.py::test_needs_edge_check_exactly_at_margin
PASSED tests/test_optimize_c2f2.py::test_returns_positions_when_no_edge
PASSED tests/test_optimize_c2f2.py::test_restarts_on_coarse_edge_then_succeeds
PASSED tests/test_optimize_c2f2.py::test_fallback_when_cap_hit_before_fine
PASSED tests/test_optimize_c2f2.py::test_fallback_uses_best_result_when_cap_hit_after_fine
```

- [ ] **Step 5: Commit**

```bash
git add optimize_c2f2.py tests/test_optimize_c2f2.py
git commit -m "feat: add optimize_polarization with C2F2 algorithm, restart cap, and fallback"
```

---

### Task 5: Final review and commit

**Files:**
- Modify: `optimize_c2f2.py` (cleanup only if needed)

- [ ] **Step 1: Confirm `optimize_c2f2.py` contains only what belongs**

The file should have exactly:
- Module docstring / header comment
- Imports: `sys`, `time`, `threading`, `random`, `numpy`, `pyvisa`
- `sys.path.append(...)` + `from pol_cons import ThorlabsMPC320`
- `pol_ctrl` and `power_meter` connections
- 6 constants (`_SWEEP_START`, `_SWEEP_END`, `_SAMPLE_INTERVAL`, `_FINE_WINDOW`, `_FINE_SETTLE`, `_EDGE_MARGIN`)
- 6 private helpers: `_get_power`, `_set_arms`, `_set_arm`, `_needs_edge_check`, `_coarse_sweep`, `_fine_refine`
- 1 public function: `optimize_polarization`

Remove anything else (no `if __name__ == "__main__"` block, no benchmark code, no plotting).

- [ ] **Step 2: Run full test suite one final time**

```
pytest tests/test_optimize_c2f2.py -v
```

Expected: 8 passed, 0 failed.

- [ ] **Step 3: Final commit**

```bash
git add optimize_c2f2.py
git commit -m "chore: finalize optimize_c2f2 for submission"
```
