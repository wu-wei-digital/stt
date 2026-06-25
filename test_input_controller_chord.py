#!/usr/bin/env python3
"""Integration test for the chord path in InputController (pytest not installed).

Drives _on_press/_on_release with real pynput Key objects against a stub app,
confirming the ⌃⌥⌘ chord starts and stops recording. PromptOverlay and the
pynput/mouse Listeners are stubbed so it runs headless.

Run: python test_input_controller_chord.py
"""
import os
import threading
import time

os.environ.pop("HOTKEY_MODE", None)  # default hold

import input_controller as ic  # noqa: E402
from pynput import keyboard  # noqa: E402


class StubOverlay:
    def set_shift_held(self, _v):
        pass


class StubApp:
    def __init__(self):
        self.recording = False
        self._starting = False
        self._processing = False
        self.hotkey_id = None
        self._overlay = StubOverlay()
        self.started = threading.Event()
        self.processed = threading.Event()
        self.process_args = None

    def start_recording(self):
        self.recording = True
        self.started.set()

    def process_recording(self, send_enter=False):
        self.recording = False
        self.process_args = send_enter
        self.processed.set()

    def cancel_recording(self):
        pass

    def cancel_transcription(self):
        pass


class StubListener:
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass

    def stop(self):
        pass


def check(name, cond):
    print(f"[{'ok' if cond else 'FAIL'}] {name}")
    return bool(cond)


def main() -> int:
    ok = True

    # Stub the GUI/listener pieces before constructing the controller.
    ic.PromptOverlay = lambda **k: StubOverlay()
    ic.keyboard.Listener = StubListener
    ic.mouse.Listener = StubListener

    app = StubApp()
    controller = ic.InputController(app, hotkey_id="ctrl_shift_alt")

    ok &= check("hotkey resolves to chord mode", controller._chord_required is not None)
    ok &= check("display name is the chord glyphs", controller.hotkey_name == "⌃⇧⌥")
    ok &= check("watchdog mask requires all members", controller._trigger_flag_require_all is True)

    # Hold the chord: ctrl, shift, then alt completes it -> recording starts.
    controller._on_press(keyboard.Key.ctrl_l)
    controller._on_press(keyboard.Key.shift_l)
    ok &= check("not recording on partial chord", not app.started.is_set())
    controller._on_press(keyboard.Key.alt_l)
    ok &= check("recording started on full chord", app.started.wait(2))

    # Release one member -> recording stops and processes.
    controller._on_release(keyboard.Key.shift_l)
    ok &= check("recording processed on release", app.processed.wait(2))
    ok &= check("send_enter false (shift is part of the trigger)", app.process_args is False)

    # Left/right mix counts too (chord matches modifier families, not sides).
    app2 = StubApp()
    c2 = ic.InputController(app2, hotkey_id="ctrl_shift_alt")
    c2._on_press(keyboard.Key.ctrl_r)   # right control
    c2._on_press(keyboard.Key.shift_r)  # right shift
    c2._on_press(keyboard.Key.alt_r)    # right option
    ok &= check("right-side chord also starts recording", app2.started.wait(2))

    # Single-key mode still untouched: cmd_r alone triggers, chord state absent.
    app3 = StubApp()
    c3 = ic.InputController(app3, hotkey_id="cmd_r")
    ok &= check("single-key mode has no chord matcher", c3._chord_required is None)
    c3._on_press(keyboard.Key.cmd_r)
    ok &= check("single-key cmd_r still starts recording", app3.started.wait(2))

    # --- mouse-trigger gating ----------------------------------------------
    from pynput import mouse as _mouse

    os.environ["MOUSE_TRIGGER"] = "false"
    app4 = StubApp()
    c4 = ic.InputController(app4, hotkey_id="cmd_r")
    c4._on_click(0, 0, _mouse.Button.middle, True)
    ok &= check("mouse trigger off: middle click does not record", not app4.started.wait(0.3))

    os.environ["MOUSE_TRIGGER"] = "true"
    app5 = StubApp()
    c5 = ic.InputController(app5, hotkey_id="cmd_r")
    c5._on_click(0, 0, _mouse.Button.middle, True)
    ok &= check("mouse trigger on: middle click records", app5.started.wait(2))

    c5.set_mouse_trigger_enabled(False)
    app5.started.clear()
    c5._on_click(0, 0, _mouse.Button.middle, True)
    ok &= check("mouse trigger live-disabled: middle click ignored", not app5.started.wait(0.3))
    os.environ.pop("MOUSE_TRIGGER", None)

    time.sleep(0.05)
    print("\nAll passed." if ok else "\nSOME FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
