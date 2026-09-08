"""Development convenience: let `pytest` run without a sourced colcon install.

Under colcon the sibling packages are already installed and importable; this
appends the workspace paths to the end of sys.path, never the front, so it
cannot shadow them.
"""

import os
import sys

_WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for _sibling in ("auv_core",):
    _path = os.path.join(_WORKSPACE, _sibling)
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.append(_path)
