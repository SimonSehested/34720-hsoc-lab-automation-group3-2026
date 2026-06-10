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
    # fallback should use attempt 1 positions (best power = 0.001)
    power_calls = iter([0.001, 0.002])
    edge_calls = iter([False, True, False, True])
    fine_mock = MagicMock(return_value=42)

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
    # 3 (attempt 1 fine) + 3 (attempt 2 fine) + 3 (fallback final fine) = 9
    assert fine_mock.call_count == 9
