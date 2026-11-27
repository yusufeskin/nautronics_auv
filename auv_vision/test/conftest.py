"""Make auv_vision importable without a colcon build."""

import importlib.util
import os
import sys

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

# The ament linter tests need a sourced ROS 2 environment. Skip collecting
# them on plain Python so `python -m pytest auv_vision/test` still works;
# under colcon test the ament modules are present and nothing is skipped.
collect_ignore = [
    name for name, module in (
        ('test_copyright.py', 'ament_copyright'),
        ('test_flake8.py', 'ament_flake8'),
        ('test_pep257.py', 'ament_pep257'),
    )
    if importlib.util.find_spec(module) is None
]
