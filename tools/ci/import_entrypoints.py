#!/usr/bin/env python3
"""Import every console_script module of the built workspace.

Run after `colcon build` with the workspace sourced. Importing a node module
does not start the node (`main()` is not called), but it does execute every
top-level import, so it catches typos, wrong module paths and missing
dependencies that `colcon build` never notices for ament_python packages.

Modules that need a third-party library CI does not install (for example the
YOLO stack) are reported as skipped instead of failing; see OPTIONAL below.

Usage: python3 tools/ci/import_entrypoints.py [repo_root]
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_ros_entrypoints import build_type, console_scripts  # noqa: E402

# Top-level module names whose absence only skips a module instead of failing.
# Keep this list short: every entry is a dependency CI does not exercise.
OPTIONAL = {
    'ultralytics',  # rosdep key skipped in CI: pulls in PyTorch (several GB)
    'gz',           # Gazebo Harmonic python bindings, simulation only
    'pymavlink',    # not declared in package.xml yet; remove once it is
    'brping',       # not declared in package.xml yet; remove once it is
}


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()
    failed, skipped, ok = [], [], 0
    for package_xml in sorted(root.glob('*/package.xml')):
        pkg_dir = package_xml.parent
        if build_type(package_xml) != 'ament_python' or not (pkg_dir / 'setup.py').is_file():
            continue
        for entry in console_scripts(pkg_dir / 'setup.py'):
            exe, target = (s.strip() for s in entry.split('=', 1))
            module = target.split(':', 1)[0].strip()
            label = f'{pkg_dir.name}/{exe} ({module})'
            try:
                importlib.import_module(module)
                ok += 1
            except ModuleNotFoundError as exc:
                top = (exc.name or '').split('.')[0]
                if top in OPTIONAL:
                    skipped.append(f'{label}: needs optional "{exc.name}"')
                else:
                    failed.append(f'{label}: {exc!r}')
            except Exception as exc:  # noqa: BLE001 - report anything import raises
                failed.append(f'{label}: {exc!r}')

    for line in skipped:
        print(f'skipped: {line}')
    for line in failed:
        print(f'::error::import failed: {line}')
    print(f'{ok} imported, {len(skipped)} skipped (optional dependency), {len(failed)} failed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
