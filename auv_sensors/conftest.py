"""Development convenience: let `pytest` run without a sourced colcon install.

Under colcon the sibling packages are already installed and importable, and
this appends nothing that shadows them - the workspace path goes on the end of
sys.path, not the front. Its only job is to make `pytest` work straight from a
clone while someone is iterating on the validation logic.
"""

import os
import sys

_WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for _sibling in ("auv_core",):
    _path = os.path.join(_WORKSPACE, _sibling)
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.append(_path)
