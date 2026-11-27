#!/usr/bin/env python3
"""Statically check ROS 2 entry points and launch files, without ROS.

`colcon build` of an ament_python package only copies files, so it never
notices that a console_script points at a module that does not exist, or
that a launch file starts an executable no package provides. This script
catches both:

1. every ``console_scripts`` entry in each ``setup.py`` points at an
   existing module that defines the named function;
2. every ``Node(package=..., executable=...)`` in a launch file that names a
   package from this repository uses an executable that package declares.

Known problems can be listed in ``tools/ci/known_issues.txt`` so CI stays
green while they are being fixed. A listed issue that no longer occurs is
an error too, so the list cannot go stale.

Usage: python tools/ci/check_ros_entrypoints.py [repo_root]
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

KNOWN_ISSUES_FILE = Path(__file__).with_name('known_issues.txt')


def read_package_name(package_xml: Path) -> str | None:
    import xml.etree.ElementTree as ET

    try:
        root = ET.parse(package_xml).getroot()
    except ET.ParseError:
        return None
    name = root.findtext('name')
    return name.strip() if name else None


def build_type(package_xml: Path) -> str:
    import xml.etree.ElementTree as ET

    root = ET.parse(package_xml).getroot()
    bt = root.find('./export/build_type')
    return bt.text.strip() if bt is not None and bt.text else 'ament_cmake'


def console_scripts(setup_py: Path) -> list[str]:
    """Return the raw console_scripts strings from a setup.py, via AST."""
    tree = ast.parse(setup_py.read_text(encoding='utf-8'), str(setup_py))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != 'entry_points' or not isinstance(kw.value, ast.Dict):
                continue
            for key, value in zip(kw.value.keys, kw.value.values):
                if (isinstance(key, ast.Constant) and key.value == 'console_scripts'
                        and isinstance(value, (ast.List, ast.Tuple))):
                    return [e.value for e in value.elts
                            if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def module_file(pkg_dir: Path, module: str) -> Path | None:
    base = pkg_dir.joinpath(*module.split('.'))
    for candidate in (base.with_suffix('.py'), base / '__init__.py'):
        if candidate.is_file():
            return candidate
    return None


def defines_function(path: Path, func: str) -> bool:
    tree = ast.parse(path.read_text(encoding='utf-8'), str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func:
            return True
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == func for t in node.targets):
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                (a.asname or a.name) == func for a in node.names):
            return True
    return False


def launch_nodes(launch_file: Path):
    """Yield (lineno, package, executable) for literal Node(...) calls."""
    tree = ast.parse(launch_file.read_text(encoding='utf-8'), str(launch_file))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kws = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        pkg, exe = kws.get('package'), kws.get('executable')
        if (isinstance(pkg, ast.Constant) and isinstance(pkg.value, str)
                and isinstance(exe, ast.Constant) and isinstance(exe.value, str)):
            yield node.lineno, pkg.value, exe.value


def load_known_issues() -> set[str]:
    if not KNOWN_ISSUES_FILE.is_file():
        return set()
    lines = KNOWN_ISSUES_FILE.read_text(encoding='utf-8').splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith('#')}


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()
    problems: dict[str, str] = {}  # key -> human readable message

    packages: dict[str, Path] = {}
    for package_xml in sorted(root.glob('*/package.xml')):
        name = read_package_name(package_xml)
        if name:
            packages[name] = package_xml.parent

    executables: dict[str, set[str]] = {}
    for name, pkg_dir in packages.items():
        setup_py = pkg_dir / 'setup.py'
        if build_type(pkg_dir / 'package.xml') != 'ament_python' or not setup_py.is_file():
            continue
        executables[name] = set()
        for entry in console_scripts(setup_py):
            if '=' not in entry or ':' not in entry:
                problems[f'{name}/setup.py: malformed {entry.strip()}'] = (
                    f'{name}/setup.py: malformed console_script "{entry}"')
                continue
            exe, target = (s.strip() for s in entry.split('=', 1))
            module, func = (s.strip() for s in target.split(':', 1))
            executables[name].add(exe)
            path = module_file(pkg_dir, module)
            key = f'{name}/setup.py: {exe}'
            if path is None:
                problems[key] = (f'{name}/setup.py: "{exe}" points at module "{module}", '
                                 'which does not exist')
            elif not defines_function(path, func):
                problems[key] = (f'{name}/setup.py: "{exe}" points at {module}:{func}, '
                                 f'but {path.relative_to(root).as_posix()} defines no "{func}"')

    for launch_file in sorted(root.glob('*/launch/**/*.py')):
        rel = launch_file.relative_to(root).as_posix()
        try:
            nodes = list(launch_nodes(launch_file))
        except SyntaxError as exc:
            problems[f'{rel}: syntax'] = f'{rel}: syntax error: {exc}'
            continue
        for lineno, pkg, exe in nodes:
            if pkg not in executables:
                continue  # external package or ament_cmake package
            if exe not in executables[pkg]:
                problems[f'{rel}: {pkg} {exe}'] = (
                    f'{rel}:{lineno}: executable "{exe}" is not declared by package "{pkg}"')

    known = load_known_issues()
    new = sorted(k for k in problems if k not in known)
    tolerated = sorted(k for k in problems if k in known)
    stale = sorted(known - problems.keys())

    for key in tolerated:
        print(f'known issue (tolerated): {problems[key]}')
    for key in new:
        print(f'::error::{problems[key]}')
    for key in stale:
        print(f'::error::"{key}" is listed in {KNOWN_ISSUES_FILE.name} but no longer '
              'happens - remove it from the list')

    checked = sum(len(v) for v in executables.values())
    print(f'checked {checked} console_scripts in {len(executables)} python packages; '
          f'{len(new)} new problem(s), {len(tolerated)} known, {len(stale)} stale')
    return 1 if new or stale else 0


if __name__ == '__main__':
    sys.exit(main())
