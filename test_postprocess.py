#!/usr/bin/env python3
"""Plain-python tests for postprocess.collapse_repeats (pytest not installed).

Run: python test_postprocess.py
"""
from postprocess import collapse_repeats


def check(name, got, want):
    status = "ok" if got == want else "FAIL"
    print(f"[{status}] {name}")
    if got != want:
        print(f"   want: {want!r}")
        print(f"   got:  {got!r}")
    return got == want


def main() -> int:
    cases = [
        # single-word loop collapses to one
        ("single-word loop", collapse_repeats("the the the the the the"), "the"),
        # two-word loop collapses to one occurrence
        ("two-word loop", collapse_repeats("the cat the cat the cat the cat"), "the cat"),
        # loop embedded in real speech: only the loop collapses
        (
            "embedded loop",
            collapse_repeats("please send the the the the the report"),
            "please send the report",
        ),
        # case-insensitive matching
        ("case-insensitive", collapse_repeats("No no NO no no"), "No"),
        # genuine short repeat under threshold is preserved
        ("short repeat kept", collapse_repeats("that is so so good"), "that is so so good"),
        # normal sentence untouched
        ("normal sentence", collapse_repeats("scope this terminal to Mimir"), "scope this terminal to Mimir"),
        # empty input
        ("empty", collapse_repeats(""), ""),
        # exactly at threshold (3 reps) is kept; over threshold collapses
        ("at threshold kept", collapse_repeats("ha ha ha"), "ha ha ha"),
        ("over threshold collapsed", collapse_repeats("ha ha ha ha"), "ha"),
    ]
    passed = sum(check(n, g, w) for n, g, w in cases)
    total = len(cases)
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
