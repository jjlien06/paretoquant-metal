"""In-process CPU CLI: forbid MLX/network/signals; ordered execution ledger."""

import ast
import importlib.abc
import json
import os
import sys
import time
from pathlib import Path

FORBIDDEN = {"mlx", "mlx_lm", "mlx_vlm"}


class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise RuntimeError("CPU-only import guard: " + fullname)


def install_guard():
    sys.meta_path.insert(0, NoModelImports())

    def audit(event, args):
        if event in ("socket.__new__", "os.kill", "os.killpg"):
            raise RuntimeError("CPU-only no network/signals guard: " + event)

    sys.addaudithook(audit)


class CPUPlugin:
    def __init__(self):
        root = Path(__file__).parent
        imports = {}
        for path in root.glob("test*.py"):
            names = []
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, ast.Import):
                    names.extend(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    names.append((node.module or "").split(".")[0])
            imports[path.stem] = names
        excluded = {name for name, deps in imports.items() if FORBIDDEN.intersection(deps)}
        while True:
            more = {
                name for name, deps in imports.items() if excluded.intersection(deps)
            } - excluded
            if not more:
                break
            excluded |= more
        self.excluded_modules = sorted(name + ".py" for name in excluded)
        self.unsafe_functions = set()
        for path in root.glob("test*.py"):
            for fn in ast.walk(ast.parse(path.read_text())):
                if not isinstance(
                    fn, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) or not fn.name.startswith("test_"):
                    continue
                for node in ast.walk(fn):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [a.name.split(".")[0] for a in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names = [(node.module or "").split(".")[0]]
                    if (FORBIDDEN | excluded).intersection(names):
                        self.unsafe_functions.add((path.name, fn.name))
        self.counts = {}

    def pytest_ignore_collect(self, collection_path, config):
        return collection_path.name in self.excluded_modules

    def pytest_collection_modifyitems(self, items):
        import pytest

        for item in items:
            # These original watchdog fixtures may signal actual subprocess groups.
            if (
                (item.path.name, getattr(item, "originalname", item.name)) in self.unsafe_functions
                or item.get_closest_marker("gpu")
                or item.get_closest_marker("metal")
                or item.name
                in (
                    "test_large_schedule_deadline_and_explicit_log_overflow",
                    "test_owned_snapshot_includes_exact_schedule_and_preserves_cleanup",
                )
            ):
                item.add_marker(
                    pytest.mark.skip(reason="Owner-only fixture; CPU import/no-signals guard")
                )

    def pytest_sessionfinish(self, session, exitstatus):
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        self.counts = {
            key: len(values)
            for key, values in reporter.stats.items()
            if key in ("passed", "failed", "error", "skipped", "xfailed", "xpassed")
        }


def main():
    os.environ["TMPDIR"] = "/Users/jeremylien/.hermes/cache/scratch"
    install_guard()
    began = time.time()
    original = list(sys.argv[1:])
    plugin = None
    if original[:1] == ["--guarded-cli"]:
        import runpy

        script = Path(original[1]).resolve()
        sys.argv = [str(script), *original[2:]]
        try:
            runpy.run_path(str(script), run_name="__main__")
            result = 0
        except SystemExit as exc:
            result = exc.code or 0
    else:
        import pytest

        arguments = list(original)
        plugins = []
        if "--all-cpu" in arguments:
            arguments.remove("--all-cpu")
            plugin = CPUPlugin()
            plugins = [plugin]
            print(
                "CPU-only explicitly excluded MLX-dependent modules:",
                ",".join(plugin.excluded_modules),
            )
        result = pytest.main(arguments, plugins=plugins)
    record = {
        "timestamp": began,
        "argv": [sys.executable, __file__, *original],
        "exit_code": int(result),
        "guard": "forbid mlx/mlx_lm/mlx_vlm/network/signals",
        "mlx_imported": any(k.split(".")[0] in FORBIDDEN for k in sys.modules),
    }
    if plugin:
        record.update(test_counts=plugin.counts, excluded_mlx_modules=plugin.excluded_modules)
    with Path(__file__).with_name("overlap-command-history.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    raise SystemExit(result)


if __name__ == "__main__":
    main()
