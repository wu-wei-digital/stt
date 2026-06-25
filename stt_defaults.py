from __future__ import annotations


HOTKEY_DISPLAY_NAMES: dict[str, str] = {
    "cmd_r": "Right ⌘",
    "cmd_l": "Left ⌘",
    "alt_r": "Right ⌥",
    "alt_l": "Left ⌥",
    "ctrl_r": "Right ⌃",
    "ctrl_l": "Left ⌃",
    "shift_r": "Right ⇧",
    "shift_l": "Left ⇧",
    "ctrl_alt_cmd": "⌃⌥⌘",
}


class NullOverlay:
    def show(self):
        pass

    def hide(self):
        pass

    def update_waveform(self, values, above_threshold: bool = False):
        pass

    def set_transcribing(self, transcribing: bool):
        pass

    def set_shift_held(self, held: bool):
        pass


def noop_text_injector(_text: str, _send_enter: bool = False) -> None:
    return


def noop_sound(_path: str) -> None:
    return

