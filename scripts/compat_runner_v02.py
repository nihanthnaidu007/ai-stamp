#!/usr/bin/env python3
"""Standalone runner for the aistamp v0.2 compatibility kit.

Runs ``tests/compat`` through pytest and prints a PASS / PENDING / FAIL
table. Checks for surfaces that are not on this branch yet SKIP with a
``PENDING (PR #N)`` reason; the runner surfaces those as PENDING rows, not
failures. Exit code is 0 unless a check actually FAILs.

Usage (from the repo root, with the project venv active):

    python scripts/compat_runner_v02.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_PENDING_RE = re.compile(r"PENDING \(PR #(\d+)\): (.+)")
_CHECK_ID_RE = re.compile(r"[Vv]2[_-]\d\d")


class _Collector:
    """pytest plugin collecting one outcome row per compat check."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, str]] = {}

    def _record(self, nodeid: str, status: str, reason: str) -> None:
        found = _CHECK_ID_RE.search(nodeid)
        check_id = found.group(0) if found else nodeid
        self.rows[check_id] = {"nodeid": nodeid, "status": status, "reason": reason}

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when not in ("setup", "call"):
            return
        nodeid = report.nodeid
        if report.failed:
            last = str(report.longrepr).splitlines()[-1] if report.longrepr else ""
            self._record(nodeid, "FAIL", last)
        elif report.skipped:
            # Skip longreprs arrive as (path, lineno, "Skipped: <reason>").
            text = str(report.longrepr or "")
            match = _PENDING_RE.search(text)
            if match:
                reason = f"PR #{match.group(1)} — {match.group(2)}"
                self._record(nodeid, "PENDING", reason)
            elif "Skipped: " in text:
                reason = text.split("Skipped: ", 1)[1].rstrip("')\n ")
                self._record(nodeid, "SKIP", reason)
            else:
                detail = text.splitlines()[-1] if text else "skipped"
                self._record(nodeid, "SKIP", detail)
        elif report.when == "call" and report.passed:
            self._record(nodeid, "PASS", "")


def main() -> int:
    collector = _Collector()
    tests_dir = Path(__file__).resolve().parent.parent / "tests" / "compat"
    exit_code = pytest.main(
        [str(tests_dir), "-q", "--no-header", "-p", "no:cacheprovider"],
        plugins=[collector],
    )

    rows = dict(sorted(collector.rows.items()))
    print("\n=== aistamp v0.2 compatibility kit ===")
    print(f"{'CHECK':<8}{'STATUS':<10}{'DETAIL':<58}TEST")
    for check_id, row in rows.items():
        detail = row["reason"][:56]
        print(f"{check_id:<8}{row['status']:<10}{detail:<58}{row['nodeid'][:72]}")

    counts = {"PASS": 0, "PENDING": 0, "SKIP": 0, "FAIL": 0}
    for row in rows.values():
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    total = sum(counts.values())
    print(
        f"\n{total} checks: {counts['PASS']} pass, {counts['PENDING']} pending-by-PR, "
        f"{counts['SKIP']} skipped, {counts['FAIL']} failed"
    )
    if exit_code not in (0, pytest.ExitCode.OK):
        print(f"pytest exit: {exit_code}")
    # PENDING/SKIP are expected while PRs #3/#4 are open; only FAILs are fatal.
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
