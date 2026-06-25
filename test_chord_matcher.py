#!/usr/bin/env python3
"""Plain-python tests for chord_matcher.ChordMatcher (pytest not installed).

Run: python test_chord_matcher.py
"""
from chord_matcher import Action, ChordMatcher, REQUIRED_CTRL_ALT_CMD


def check(name, got, want):
    status = "ok" if got == want else "FAIL"
    print(f"[{status}] {name}")
    if got != want:
        print(f"   want: {want!r}")
        print(f"   got:  {got!r}")
    return got == want


def hold():
    return ChordMatcher(REQUIRED_CTRL_ALT_CMD, mode="hold")


def toggle():
    return ChordMatcher(REQUIRED_CTRL_ALT_CMD, mode="toggle")


def main() -> int:
    ok = True

    # --- hold mode -----------------------------------------------------------
    m = hold()
    ok &= check("hold: partial press, no trigger", m.press("ctrl"), Action.NONE)
    ok &= check("hold: second partial, no trigger", m.press("alt"), Action.NONE)
    ok &= check("hold: completing chord starts", m.press("cmd"), Action.START)
    ok &= check("hold: active after start", m.active, True)
    ok &= check("hold: releasing one member stops", m.release("alt"), Action.STOP)
    ok &= check("hold: inactive after stop", m.active, False)
    ok &= check("hold: releasing the rest is a no-op", m.release("ctrl"), Action.NONE)
    ok &= check("hold: releasing last is a no-op", m.release("cmd"), Action.NONE)

    # repeated press of an already-complete chord must not re-fire (auto-repeat safety)
    m = hold()
    m.press("ctrl")
    m.press("alt")
    m.press("cmd")
    ok &= check("hold: repeat press while complete is a no-op", m.press("cmd"), Action.NONE)

    # non-required keys are ignored entirely
    m = hold()
    ok &= check("hold: shift press ignored", m.press("shift"), Action.NONE)
    ok &= check("hold: shift release ignored", m.release("shift"), Action.NONE)

    # --- toggle mode ---------------------------------------------------------
    # a tap (press all, release all) flips on, the next tap flips off
    m = toggle()
    m.press("ctrl")
    m.press("alt")
    ok &= check("toggle: first full chord turns on", m.press("cmd"), Action.START)
    ok &= check("toggle: active after first tap", m.active, True)
    ok &= check("toggle: release does not stop in toggle mode", m.release("cmd"), Action.NONE)
    m.release("alt")
    m.release("ctrl")
    ok &= check("toggle: still active after releasing the tap", m.active, True)
    # second tap
    m.press("ctrl")
    m.press("alt")
    ok &= check("toggle: second full chord turns off", m.press("cmd"), Action.STOP)
    ok &= check("toggle: inactive after second tap", m.active, False)

    # --- reset (watchdog / cancel) ------------------------------------------
    m = hold()
    m.press("ctrl"); m.press("alt"); m.press("cmd")
    ok &= check("reset: returns STOP when active", m.reset(), Action.STOP)
    ok &= check("reset: inactive afterwards", m.active, False)
    ok &= check("reset: when inactive returns NONE", m.reset(), Action.NONE)
    # after reset the held set is clear, so a fresh chord starts again
    m.press("ctrl"); m.press("alt")
    ok &= check("reset: chord re-arms after reset", m.press("cmd"), Action.START)

    # --- unknown mode coerces to hold ---------------------------------------
    m = ChordMatcher(REQUIRED_CTRL_ALT_CMD, mode="bogus")
    m.press("ctrl"); m.press("alt")
    ok &= check("mode: unknown coerces to hold (starts on complete)", m.press("cmd"), Action.START)
    ok &= check("mode: unknown coerces to hold (stops on release)", m.release("cmd"), Action.STOP)

    print("\nAll passed." if ok else "\nSOME FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
