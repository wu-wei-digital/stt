from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Optional

from pynput import keyboard, mouse

from chord_matcher import Action, ChordMatcher
from prompt_overlay import PromptOverlay
from stt_app import STTApp
from text_injector import paste_text


HOTKEYS: dict[str, dict[str, object]] = {
    "cmd_r": {"key": keyboard.Key.cmd_r, "name": "Right ⌘"},
    "cmd_l": {"key": keyboard.Key.cmd_l, "name": "Left ⌘"},
    "alt_r": {"key": keyboard.Key.alt_r, "name": "Right ⌥"},
    "alt_l": {"key": keyboard.Key.alt_l, "name": "Left ⌥"},
    "ctrl_r": {"key": keyboard.Key.ctrl_r, "name": "Right ⌃"},
    "ctrl_l": {"key": keyboard.Key.ctrl_l, "name": "Left ⌃"},
    "shift_r": {"key": keyboard.Key.shift_r, "name": "Right ⇧"},
    "shift_l": {"key": keyboard.Key.shift_l, "name": "Left ⇧"},
}


# Chord hotkeys: a set of modifier *families* that must all be held at once
# (left/right agnostic). Handled by ChordMatcher rather than the single-key path.
CHORD_HOTKEYS: dict[str, dict[str, object]] = {
    "ctrl_alt_cmd": {"families": frozenset({"ctrl", "alt", "cmd"}), "name": "⌃⌥⌘"},
}


# Map every concrete pynput modifier key to its family, built defensively so a
# key name missing on this platform is simply skipped.
_KEY_FAMILY: dict[object, str] = {}
for _fam, _names in (
    ("ctrl", ("ctrl", "ctrl_l", "ctrl_r")),
    ("alt", ("alt", "alt_l", "alt_r", "alt_gr")),
    ("cmd", ("cmd", "cmd_l", "cmd_r")),
    ("shift", ("shift", "shift_l", "shift_r")),
):
    for _name in _names:
        _k = getattr(keyboard.Key, _name, None)
        if _k is not None:
            _KEY_FAMILY[_k] = _fam


def _family_of(key) -> Optional[str]:
    """The modifier family (ctrl/alt/cmd/shift) for a key, or None if not a modifier."""
    return _KEY_FAMILY.get(key)


VK_TO_CHAR: dict[int, str] = {
    18: "1",
    19: "2",
    20: "3",
    21: "4",
    23: "5",
    22: "6",
    26: "7",
    28: "8",
    25: "9",
    29: "0",
    0: "a",
    11: "b",
    8: "c",
    2: "d",
    14: "e",
    3: "f",
    5: "g",
    4: "h",
    34: "i",
    38: "j",
    40: "k",
    37: "l",
    46: "m",
    45: "n",
    31: "o",
    35: "p",
    12: "q",
    15: "r",
    1: "s",
    17: "t",
    32: "u",
    9: "v",
    13: "w",
    7: "x",
    16: "y",
    6: "z",
    36: "enter",
}


class InputController:
    def __init__(self, app: STTApp, *, hotkey_id: str):
        self._app = app

        self._lock = threading.Lock()
        self._key_pressed = False
        self._mouse_pressed = False
        self._record_source: Optional[str] = None
        self._shift_held = False
        self._send_enter_flag = False
        self._prompt_overlay_active = False

        self._trigger_key = keyboard.Key.cmd_r
        self._trigger_is_shift = False
        self._trigger_is_alt = False
        self._trigger_flag_mask = None
        self._trigger_flag_require_all = False
        self._trigger_key_name = "Right ⌘"

        # Chord mode: None for the single-key path, else a frozenset of required
        # families driven by a ChordMatcher.
        self._chord_required: Optional[frozenset] = None
        self._chord_matcher: Optional[ChordMatcher] = None

        self._prompt_overlay = PromptOverlay(on_select=self._on_prompt_select)
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._mouse_listener = mouse.Listener(on_click=self._on_click)
        self._fallback_thread: threading.Thread | None = None

        self.set_hotkey_id(hotkey_id)

    @property
    def prompt_overlay(self) -> PromptOverlay:
        return self._prompt_overlay

    @property
    def hotkey_name(self) -> str:
        return self._trigger_key_name

    def set_hotkey_id(self, hotkey_id: str) -> None:
        with self._lock:
            chord = CHORD_HOTKEYS.get(hotkey_id)
            if chord:
                families = chord["families"]  # type: ignore[assignment]
                mode = os.environ.get("HOTKEY_MODE", "hold")
                self._chord_required = families  # type: ignore[assignment]
                self._chord_matcher = ChordMatcher(families, mode=mode)
                self._trigger_key = None  # single-key paths stay dormant in chord mode
                self._trigger_is_shift = False
                self._trigger_is_alt = False
                self._trigger_key_name = str(chord["name"])
                self._app.hotkey_id = hotkey_id
                self._trigger_flag_mask, self._trigger_flag_require_all = self._combined_flag_mask(families)
                return

            self._chord_required = None
            self._chord_matcher = None
            self._trigger_flag_require_all = False

            entry = HOTKEYS.get(hotkey_id)
            if entry:
                self._trigger_key = entry["key"]  # type: ignore[assignment]
                self._trigger_key_name = str(entry["name"])
                self._app.hotkey_id = hotkey_id
            else:
                self._trigger_key = keyboard.Key.cmd_r
                self._trigger_key_name = "Right ⌘"
                self._app.hotkey_id = "cmd_r"

            self._trigger_is_shift = self._trigger_key in (keyboard.Key.shift_l, keyboard.Key.shift_r)
            self._trigger_is_alt = self._trigger_key in (keyboard.Key.alt_l, keyboard.Key.alt_r)

            self._trigger_flag_mask = None
            try:
                from Quartz import (
                    kCGEventFlagMaskCommand,
                    kCGEventFlagMaskShift,
                    kCGEventFlagMaskAlternate,
                    kCGEventFlagMaskControl,
                )

                if self._trigger_key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
                    self._trigger_flag_mask = kCGEventFlagMaskCommand
                elif self._trigger_key in (keyboard.Key.shift_l, keyboard.Key.shift_r):
                    self._trigger_flag_mask = kCGEventFlagMaskShift
                elif self._trigger_key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
                    self._trigger_flag_mask = kCGEventFlagMaskAlternate
                elif self._trigger_key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                    self._trigger_flag_mask = kCGEventFlagMaskControl
            except Exception:
                self._trigger_flag_mask = None

    @staticmethod
    def _combined_flag_mask(families):
        """OR the Quartz flag masks for a set of families. Returns (mask, require_all).

        require_all is True so the watchdog treats the chord as released the moment
        any one member is no longer held. (None, False) if Quartz is unavailable.
        """
        try:
            from Quartz import (
                kCGEventFlagMaskCommand,
                kCGEventFlagMaskShift,
                kCGEventFlagMaskAlternate,
                kCGEventFlagMaskControl,
            )
        except Exception:
            return None, False
        fam_to_mask = {
            "cmd": kCGEventFlagMaskCommand,
            "shift": kCGEventFlagMaskShift,
            "alt": kCGEventFlagMaskAlternate,
            "ctrl": kCGEventFlagMaskControl,
        }
        mask = 0
        for fam in families:
            bit = fam_to_mask.get(fam)
            if bit is None:
                return None, False
            mask |= bit
        return mask, True

    def start(self) -> None:
        self._listener.start()
        self._mouse_listener.start()
        self._start_release_fallback()

    def stop(self) -> None:
        try:
            self._listener.stop()
        except Exception:
            pass
        try:
            self._mouse_listener.stop()
        except Exception:
            pass

    def _on_prompt_select(self, text: str, send_enter: bool = False):
        def do_paste():
            # Delete the special char that Option+key produced.
            backspace_script = """
            tell application "System Events"
                key code 51
            end tell
            """
            subprocess.run(["osascript", "-e", backspace_script], timeout=2)
            time.sleep(0.03)
            paste_text(text, send_enter=send_enter, method="cgevent")

        threading.Thread(target=do_paste, daemon=True).start()

    def _on_press(self, key):
        try:
            with self._lock:
                trigger_key = self._trigger_key
                trigger_is_shift = self._trigger_is_shift
                trigger_is_alt = self._trigger_is_alt
                chord_required = self._chord_required

            if chord_required is not None:
                fam = _family_of(key)
                if fam in chord_required:
                    start = stop = False
                    send_enter = False
                    with self._lock:
                        action = self._chord_matcher.press(fam)
                        if action == Action.START and not self._key_pressed:
                            self._key_pressed = True
                            self._record_source = "keyboard"
                            self._send_enter_flag = self._shift_held
                            if self._send_enter_flag:
                                self._app._overlay.set_shift_held(True)
                            start = True
                        elif action == Action.STOP and self._key_pressed:
                            self._key_pressed = False
                            send_enter = self._send_enter_flag
                            self._send_enter_flag = False
                            self._record_source = None
                            stop = True
                    if start:
                        threading.Thread(target=self._app.start_recording, daemon=True).start()
                    if stop:
                        threading.Thread(target=self._app.process_recording, args=(send_enter,)).start()
                    return

            if key == trigger_key:
                with self._lock:
                    if self._key_pressed and not self._app.recording and not self._app._starting:
                        self._key_pressed = False
                        self._send_enter_flag = False
                    if not self._key_pressed:
                        self._key_pressed = True
                        self._record_source = "keyboard"
                        self._send_enter_flag = self._shift_held and not trigger_is_shift
                        if self._send_enter_flag:
                            self._app._overlay.set_shift_held(True)
                        threading.Thread(target=self._app.start_recording, daemon=True).start()
                return

            if key in (keyboard.Key.shift_l, keyboard.Key.shift_r) and not trigger_is_shift:
                with self._lock:
                    self._shift_held = True
                    if self._app.recording and not self._send_enter_flag:
                        self._send_enter_flag = True
                        self._app._overlay.set_shift_held(True)
                return

            if key == keyboard.Key.esc:
                with self._lock:
                    prompt_overlay_active = self._prompt_overlay_active

                if prompt_overlay_active:
                    self._prompt_overlay.hide()
                    with self._lock:
                        self._prompt_overlay_active = False
                elif self._app.recording:
                    with self._lock:
                        self._key_pressed = False
                        self._mouse_pressed = False
                        self._send_enter_flag = False
                        self._record_source = None
                    threading.Thread(target=self._app.cancel_recording, daemon=True).start()
                else:
                    threading.Thread(target=self._app.cancel_transcription, daemon=True).start()
                return

            # Prompt overlay (Right Option) unless Option is the trigger key.
            if key == keyboard.Key.alt_r and not trigger_is_alt:
                with self._lock:
                    if not self._prompt_overlay_active:
                        self._prompt_overlay_active = True
                        self._prompt_overlay.show()
                return

            with self._lock:
                prompt_overlay_active = self._prompt_overlay_active
            if prompt_overlay_active and hasattr(key, "vk") and key.vk is not None:
                char = VK_TO_CHAR.get(int(key.vk))
                if char and self._prompt_overlay.handle_key(char):
                    with self._lock:
                        self._prompt_overlay_active = False
        except Exception as e:
            print(f"⚠️  Error in key press handler: {e}")
            with self._lock:
                self._key_pressed = False
                self._mouse_pressed = False
                self._send_enter_flag = False
                self._record_source = None

    def _on_release(self, key):
        try:
            with self._lock:
                trigger_key = self._trigger_key
                trigger_is_shift = self._trigger_is_shift
                trigger_is_alt = self._trigger_is_alt
                chord_required = self._chord_required

            if chord_required is not None:
                fam = _family_of(key)
                if fam in chord_required:
                    stop = False
                    send_enter = False
                    with self._lock:
                        action = self._chord_matcher.release(fam)
                        if action == Action.STOP and self._key_pressed:
                            self._key_pressed = False
                            send_enter = self._send_enter_flag
                            self._send_enter_flag = False
                            self._record_source = None
                            stop = True
                    if stop:
                        threading.Thread(target=self._app.process_recording, args=(send_enter,)).start()
                    return

            if key in (keyboard.Key.shift_l, keyboard.Key.shift_r):
                with self._lock:
                    self._shift_held = False
                if not trigger_is_shift:
                    return

            if key == keyboard.Key.alt_r:
                with self._lock:
                    if self._prompt_overlay_active:
                        self._prompt_overlay.hide()
                        self._prompt_overlay_active = False
                if not trigger_is_alt:
                    return

            is_cmd_trigger = trigger_key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r)
            is_cmd_release = key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r)
            if key == trigger_key or (is_cmd_trigger and is_cmd_release):
                with self._lock:
                    if not self._key_pressed:
                        return
                    self._key_pressed = False
                    send_enter = self._send_enter_flag
                    self._send_enter_flag = False
                    self._record_source = None

                threading.Thread(target=self._app.process_recording, args=(send_enter,)).start()
        except Exception as e:
            print(f"⚠️  Error in key release handler: {e}")
            with self._lock:
                self._key_pressed = False
                self._mouse_pressed = False
                self._send_enter_flag = False
                self._record_source = None

    def _on_click(self, x, y, button, pressed):
        if button != mouse.Button.middle:
            return

        if pressed:
            with self._lock:
                if self._mouse_pressed and not self._app.recording and not self._app._starting:
                    self._mouse_pressed = False
                if not self._mouse_pressed:
                    self._mouse_pressed = True
                    self._record_source = "mouse"
                    threading.Thread(target=self._app.start_recording, daemon=True).start()
        else:
            with self._lock:
                if not self._mouse_pressed:
                    return
                self._mouse_pressed = False
                self._record_source = None
            threading.Thread(target=self._app.process_recording, args=(True,), daemon=True).start()

    def _start_release_fallback(self) -> None:
        if self._fallback_thread and self._fallback_thread.is_alive():
            return

        try:
            from Quartz import CGEventSourceFlagsState, kCGEventSourceStateHIDSystemState
        except Exception:
            return

        def loop() -> None:
            last_forced = 0.0
            while True:
                time.sleep(0.2)
                if not self._app.recording or self._app._starting or self._app._processing:
                    continue
                with self._lock:
                    if self._record_source != "keyboard":
                        continue
                    trigger_flag_mask = self._trigger_flag_mask
                    if trigger_flag_mask is None:
                        continue
                    flags = CGEventSourceFlagsState(kCGEventSourceStateHIDSystemState)
                    if self._trigger_flag_require_all:
                        still_held = (flags & trigger_flag_mask) == trigger_flag_mask
                    else:
                        still_held = bool(flags & trigger_flag_mask)
                    if still_held:
                        continue
                    now = time.time()
                    if now - last_forced < 1.0:
                        continue
                    last_forced = now
                    send_enter = self._send_enter_flag
                    self._send_enter_flag = False
                    self._key_pressed = False
                    self._record_source = None
                    if self._chord_matcher is not None:
                        self._chord_matcher.reset()
                threading.Thread(target=self._app.process_recording, args=(send_enter,), daemon=True).start()

        self._fallback_thread = threading.Thread(target=loop, daemon=True)
        self._fallback_thread.start()

