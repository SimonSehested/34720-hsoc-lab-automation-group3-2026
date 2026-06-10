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
