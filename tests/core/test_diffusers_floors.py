"""Every declared min_diffusers_version must be high enough to actually work.

A floor only ever names an upgrade target in an error message, so one set too high
merely over-states the requirement. One set too low is a real defect: the user
installs the version we recommended and the load still fails, or worse the load
succeeds and generation dies on a symbol that arrived in a later release.

Models whose support is not released yet are marked DIFFUSERS_FROM_SOURCE instead of
carrying a version, and that marker goes stale in the opposite direction: once upstream
ships the symbols it should name the release. Both directions are checked here.

The check is static and needs no downloads. It resolves the diffusers symbols each
model reaches against the installed diffusers source tree, and for every model whose
floor the installed version already satisfies, demands that they all resolve. Point
XFUSER_DIFFUSERS_ROOT at an unpacked wheel to audit a floor against another release
without installing it.
"""

import ast
import os
import pathlib
import unittest
from importlib.util import find_spec

from packaging.version import InvalidVersion, Version

RUNNER_PACKAGE = "xfuser.model_executor.models.runner_models"


def _from_source_marker(repo_root):
    """base_model's DIFFUSERS_FROM_SOURCE, read from source rather than imported.

    Importing it would pull in torch, and this audit only ever reads source.
    """
    path = repo_root / RUNNER_PACKAGE.replace(".", "/") / "base_model.py"
    for node in ast.parse(path.read_text(errors="replace")).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if any(
                isinstance(t, ast.Name) and t.id == "DIFFUSERS_FROM_SOURCE"
                for t in node.targets
            ):
                return node.value.value
    raise AssertionError("DIFFUSERS_FROM_SOURCE is not defined in base_model.py")


def _satisfies(installed, floor):
    """True when installed is at least floor, ignoring any prerelease suffix.

    Mirrors xfuser.compat.version_at_least. It is repeated here so this audit runs
    without torch: importing anything from the xfuser package pulls torch in, while
    the audit itself only ever reads source.
    """
    try:
        return Version(installed).release >= Version(floor).release
    except InvalidVersion:
        return False


def _repo_root():
    return pathlib.Path(__file__).resolve().parents[2]


def _diffusers_root():
    """Path to the diffusers source tree, or None when it is unavailable."""
    override = os.environ.get("XFUSER_DIFFUSERS_ROOT")
    if override:
        return pathlib.Path(override)
    try:
        spec = find_spec("diffusers")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return pathlib.Path(list(spec.submodule_search_locations)[0]).parent


def _diffusers_version(root):
    """Read __version__ out of the tree without importing it (importing needs torch)."""
    for node in ast.parse((root / "diffusers" / "__init__.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets
        ):
            if isinstance(node.value, ast.Constant):
                return node.value.value
    return None


class DiffusersTree:
    """Answers "does diffusers.a.b define Symbol?" by reading the source tree."""

    def __init__(self, root):
        self.root = root
        self._names = {}

    def _module_path(self, dotted):
        rel = dotted.replace(".", "/")
        for candidate in (f"{rel}.py", f"{rel}/__init__.py"):
            path = self.root / candidate
            if path.exists():
                return path
        return None

    def has_module(self, dotted):
        return self._module_path(dotted) is not None

    def has_symbol(self, dotted, symbol):
        path = self._module_path(dotted)
        if path is None:
            return False
        if path not in self._names:
            self._names[path] = _top_level_names(path)
        if symbol in self._names[path]:
            return True
        # diffusers packages are _LazyModule; their exports live in _import_structure
        # as strings, so a plain name scan misses them.
        if path.name == "__init__.py":
            source = path.read_text(errors="replace")
            if f'"{symbol}"' in source or f"'{symbol}'" in source:
                return True
        return self.has_module(f"{dotted}.{symbol}")


def _top_level_names(path):
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
    return names


def _guarded(tree):
    """Imports whose ImportError is swallowed, so they impose no floor."""
    handled = ("ImportError", "Exception")
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        swallows = any(
            h.type is None
            or (isinstance(h.type, ast.Name) and h.type.id in handled)
            or (
                isinstance(h.type, ast.Tuple)
                and any(
                    isinstance(e, ast.Name) and e.id in handled for e in h.type.elts
                )
            )
            for h in node.handlers
        )
        if swallows:
            for stmt in node.body:
                out.update(
                    n for n in ast.walk(stmt) if isinstance(n, (ast.Import, ast.ImportFrom))
                )
    return out


def _imports(node, package, guarded):
    """(module, symbol) pairs for unguarded imports anywhere under node."""
    out = []
    for sub in ast.walk(node):
        if sub in guarded:
            continue
        if isinstance(sub, ast.ImportFrom):
            module = sub.module or ""
            if sub.level:
                parts = package.split(".")
                module = ".".join(
                    parts[: len(parts) - sub.level + 1] + ([module] if module else [])
                )
            out.extend((module, a.name) for a in sub.names)
        elif isinstance(sub, ast.Import):
            out.extend((a.name, None) for a in sub.names)
    return out


class FloorAuditor:
    def __init__(self, repo_root):
        self.repo_root = repo_root

    def _path_for(self, dotted):
        rel = dotted.replace(".", "/")
        for candidate in (f"{rel}.py", f"{rel}/__init__.py"):
            path = self.repo_root / candidate
            if path.exists():
                return path
        return None

    def requirements_of_xfuser_module(self, dotted, seen=None):
        """diffusers symbols an xfuser module needs, following its xfuser imports."""
        seen = seen if seen is not None else set()
        if dotted in seen:
            return set()
        seen.add(dotted)
        path = self._path_for(dotted)
        if path is None:
            return set()
        package = dotted if path.name == "__init__.py" else dotted.rsplit(".", 1)[0]
        tree = ast.parse(path.read_text(errors="replace"))
        reqs = set()
        for module, symbol in _imports(tree, package, _guarded(tree)):
            root = module.split(".")[0]
            if root == "diffusers":
                reqs.add((module, symbol))
            elif root == "xfuser" and RUNNER_PACKAGE not in module:
                reqs |= self.requirements_of_xfuser_module(module, seen)
        return reqs

    def _class_requirements(self, class_node, guarded):
        """diffusers symbols reached from a class body, following its xfuser imports."""
        reqs = set()
        for module, symbol in _imports(class_node, RUNNER_PACKAGE, guarded):
            root = module.split(".")[0]
            if root == "diffusers":
                reqs.add((module, symbol))
            elif root == "xfuser":
                reqs |= self.requirements_of_xfuser_module(module)
        return reqs

    def declared_floors(self):
        """(class name, file name, effective floor, requirements) per model class.

        A subclass inherits both its parent's floor and its parent's imports, since it
        may load through an inherited _load_model, so each class is resolved through
        its base chain. Every runner base class is declared in the same file as its
        subclasses, so walking within the file is enough.
        """
        runners = self.repo_root / RUNNER_PACKAGE.replace(".", "/")
        out = []
        for path in sorted(runners.glob("*.py")):
            tree = ast.parse(path.read_text(errors="replace"))
            guarded = _guarded(tree)

            # module-scope imports bind every class in the file; imports inside a
            # class body belong to that class and its subclasses
            shared = set()
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    continue
                for module, symbol in _imports(node, RUNNER_PACKAGE, guarded):
                    if module.split(".")[0] == "diffusers":
                        shared.add((module, symbol))

            classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
            own_floor = {n: _declared_floor_of(c) for n, c in classes.items()}
            own_reqs = {
                n: self._class_requirements(c, guarded) for n, c in classes.items()
            }

            for name in classes:
                ancestry, cursor = [], name
                while cursor in classes and cursor not in ancestry:
                    ancestry.append(cursor)
                    cursor = next(
                        (
                            b.id
                            for b in classes[cursor].bases
                            if isinstance(b, ast.Name) and b.id in classes
                        ),
                        None,
                    )
                floor = next((own_floor[a] for a in ancestry if own_floor[a]), None)
                if floor is None:
                    continue
                reqs = set(shared)
                for ancestor in ancestry:
                    reqs |= own_reqs[ancestor]
                out.append((name, path.name, floor, reqs))
        return out


def _declared_floor_of(class_node):
    for node in class_node.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if any(
            isinstance(t, ast.Name) and t.id == "min_diffusers_version"
            for t in node.targets
        ):
            return node.value.value
    return None


class TestDeclaredFloorsAreReachable(unittest.TestCase):
    def setUp(self):
        self.root = _diffusers_root()
        if self.root is None:
            self.skipTest("diffusers is not installed")
        self.installed = _diffusers_version(self.root)
        if self.installed is None:
            self.skipTest("cannot read diffusers.__version__")
        self.tree = DiffusersTree(self.root)
        self.from_source = _from_source_marker(_repo_root())
        self.floors = FloorAuditor(_repo_root()).declared_floors()
        self.assertNotEqual(
            self.floors, [], "no min_diffusers_version declarations found"
        )

    def _missing(self, reqs):
        return sorted(
            f"{module}:{symbol or '*'}"
            for module, symbol in reqs
            if not (
                self.tree.has_symbol(module, symbol)
                if symbol
                else self.tree.has_module(module)
            )
        )

    def test_from_source_markers_are_still_accurate(self):
        """A model marked as needing source diffusers must still be unreleased.

        Once upstream ships the symbols, the marker is stale and should name the
        release instead, which is how a floor ends up pointing at a version that never
        existed. Only a final release can settle this: a source install of diffusers
        legitimately has the symbols while the marker is still correct.
        """
        version = Version(self.installed)
        if version.is_prerelease or version.is_devrelease or version.local:
            self.skipTest(f"diffusers {self.installed} is not a final release")

        stale = []
        for name, filename, floor, reqs in self.floors:
            if floor != self.from_source:
                continue
            if not self._missing(reqs):
                stale.append(f"{filename}:{name}")
        self.assertEqual(
            stale,
            [],
            f"diffusers {self.installed} now provides everything these models need, so "
            f"replace {self.from_source!r} with that version:\n  " + "\n  ".join(stale),
        )

    def test_no_floor_is_too_low(self):
        installed = self.installed
        checked, failures = 0, []
        for name, filename, floor, reqs in self.floors:
            if floor == self.from_source or not _satisfies(installed, floor):
                continue
            checked += 1
            missing = self._missing(reqs)
            if missing:
                failures.append(
                    f"{filename}:{name} uses min_diffusers_version={floor!r} but "
                    f"diffusers {installed} cannot provide: {', '.join(missing)}"
                )
        if not checked:
            self.skipTest(f"diffusers {installed} satisfies no declared floor")
        self.assertEqual(
            failures,
            [],
            "raise these floors to the release that ships the missing symbols:\n  "
            + "\n  ".join(failures),
        )


if __name__ == "__main__":
    unittest.main()
