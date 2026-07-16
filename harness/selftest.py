"""Zero-dependency runner for the deterministic harness tests (no pytest needed).

Runs every ``test_*`` function in ``harness.tests.test_harness`` and reports the
result. Exit code is 0 iff all pass. ``pytest harness/tests/`` also works when
pytest is installed; this runner exists so the suite can run inside the vanilla
application image, which does not ship pytest.
"""
from __future__ import annotations

import sys
import traceback


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from harness.tests import test_harness as mod

    tests = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
    passed = failed = 0
    for name in tests:
        try:
            getattr(mod, name)()
            print(f"PASS {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
