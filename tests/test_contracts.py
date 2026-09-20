"""Contract tests: the drivers must stay in sync with the stage modules.

Stage 5 exists twice -- once as build_identities.main(), and once re-implemented
in each track_*.py driver, which calls the module's functions individually. Only
the drivers actually run. When a module function gains a parameter, a driver
that does not pass it fails SILENTLY: the default is used and a column is
written empty. That is exactly how `p1_max` came to be NaN for all 523,145 rows,
disabling the stage-6 collision tiebreak with no error anywhere.

Run:  python tests/test_contracts.py
"""
import ast
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "idtrackerai" / "src"))

# Parameters that carry data a later stage depends on. Omitting one is silent,
# so they cannot be left to the default.
REQUIRED = {"write_identities": {"p1_max"}}


def drivers():
    return sorted(ROOT.glob("track_*.py"))


def calls_to(tree, func_name):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name == func_name:
                out.append(node)
    return out


def test_drivers_pass_every_load_bearing_argument():
    import build_identities

    problems = []
    for path in drivers():
        tree = ast.parse(path.read_text())
        for func_name, needed in REQUIRED.items():
            fn = getattr(build_identities, func_name, None)
            if fn is None:
                continue
            params = list(inspect.signature(fn).parameters)
            for call in calls_to(tree, func_name):
                supplied = set(params[: len(call.args)])
                supplied |= {k.arg for k in call.keywords if k.arg}
                missing = needed - supplied
                if missing:
                    problems.append(
                        f"{path.name}:{call.lineno} calls {func_name}() without "
                        f"{sorted(missing)} -- the default will be used and the "
                        "data silently lost"
                    )
    assert not problems, "\n        " + "\n        ".join(problems)


def test_every_driver_snapshots_p1_before_the_cascade():
    """p1_max_snapshot must run BEFORE assign_p2_identities.

    assign_identity collapses P1_vector to a one-hot (fragment.py:449-450), so
    a snapshot taken afterwards records 1.0 for every assigned fragment and
    carries no information.
    """
    problems = []
    for path in drivers():
        src = path.read_text()
        if "assign_p2_identities" not in src:
            continue
        tree = ast.parse(src)
        snap = [c.lineno for c in calls_to(tree, "p1_max_snapshot")]
        casc = [c.lineno for c in calls_to(tree, "assign_p2_identities")]
        if not snap:
            problems.append(f"{path.name}: runs the P2 cascade but never calls "
                            "p1_max_snapshot()")
        elif min(snap) > min(casc):
            problems.append(f"{path.name}: p1_max_snapshot() at line {min(snap)} "
                            f"runs AFTER the cascade at line {min(casc)}")
    assert not problems, "\n        " + "\n        ".join(problems)


if __name__ == "__main__":
    print(f"drivers found: {[p.name for p in drivers()] or 'NONE (see note below)'}")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            fails += 1; print(f"  FAIL  {t.__name__}{e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
