"""Pure state machine for a multi-modifier chord push-to-talk trigger.

No pynput, no threads, no I/O, so it is unit-testable on its own. The
InputController normalises each modifier key it sees to a "family" string
(``ctrl`` / ``alt`` / ``cmd`` / ``shift``) and feeds press/release events here;
the returned :class:`Action` tells the controller to start or stop recording.

Two modes:

- ``hold`` (push-to-talk): recording is active while every required modifier is
  held; releasing any one of them stops it. Right for a key that *holds* the
  chord down while pressed.
- ``toggle``: each completed press of the full chord flips recording on/off;
  releases are ignored. Right for a key that *taps* the chord (press and release
  the modifiers in one quick burst).
"""
from __future__ import annotations

from enum import Enum


class Action(Enum):
    NONE = "none"
    START = "start"
    STOP = "stop"


# Convenience constants for the registered chords. Nicholas's Logitech F5 key
# emits Control+Shift+Option (⌃⇧⌥).
REQUIRED_CTRL_SHIFT_ALT = frozenset({"ctrl", "shift", "alt"})
REQUIRED_CTRL_ALT_CMD = frozenset({"ctrl", "alt", "cmd"})

_MODES = ("hold", "toggle")


class ChordMatcher:
    def __init__(self, required, *, mode: str = "hold"):
        self._required = frozenset(required)
        self._mode = mode if mode in _MODES else "hold"
        self._held: set[str] = set()
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def mode(self) -> str:
        return self._mode

    def _complete(self) -> bool:
        return self._required.issubset(self._held)

    def press(self, family: str) -> Action:
        if family not in self._required:
            return Action.NONE
        was_complete = self._complete()
        self._held.add(family)
        if was_complete or not self._complete():
            # Either the chord was already complete (auto-repeat / extra press) or
            # it is still incomplete; in neither case does pressing change state.
            return Action.NONE
        # The chord just transitioned to complete.
        if self._mode == "toggle":
            self._active = not self._active
            return Action.START if self._active else Action.STOP
        # hold mode
        if not self._active:
            self._active = True
            return Action.START
        return Action.NONE

    def release(self, family: str) -> Action:
        if family not in self._required:
            return Action.NONE
        was_complete = self._complete()
        self._held.discard(family)
        if self._mode != "hold":
            return Action.NONE
        if was_complete and not self._complete() and self._active:
            self._active = False
            return Action.STOP
        return Action.NONE

    def reset(self) -> Action:
        """Force inactive and clear held state (watchdog missed-release or cancel).

        Returns STOP if recording was active, else NONE.
        """
        self._held.clear()
        if self._active:
            self._active = False
            return Action.STOP
        return Action.NONE
