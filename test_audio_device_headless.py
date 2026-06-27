#!/usr/bin/env python3
"""Plain-python tests for stt._select_audio_device headless fallback (pytest not installed).

Under launchd there is no controlling terminal, so the interactive device picker
cannot run. _select_audio_device must fall back to the system default device
(return None) instead of calling input() and raising EOFError.

Run: python test_audio_device_headless.py
"""
import sys
import types


def _install_fake_sounddevice(devices):
    """Stub the sounddevice module so importing/calling stt needs no audio hardware."""
    fake = types.ModuleType("sounddevice")
    fake.query_devices = lambda *a, **k: devices

    class _Default:
        device = (0, 0)

    fake.default = _Default()
    sys.modules["sounddevice"] = fake


def check(name, got, want):
    status = "ok" if got == want else "FAIL"
    print(f"[{status}] {name}")
    if got != want:
        print(f"   want: {want!r}")
        print(f"   got:  {got!r}")
    return got == want


def main() -> int:
    _install_fake_sounddevice([
        {"name": "MacBook Air Microphone", "max_input_channels": 1},
        {"name": "External USB Mic", "max_input_channels": 2},
    ])
    import stt

    ok = True

    # Force the "no controlling terminal" condition (as under launchd).
    real_isatty = sys.stdin.isatty
    sys.stdin.isatty = lambda: False
    try:
        # Empty saved device + no TTY: must return None (system default), never prompt.
        got = stt._select_audio_device(saved_device_name="", save_device_fn=lambda *_: None)
        ok &= check("empty AUDIO_DEVICE headless -> system default (None)", got, None)

        # Saved device that exists is honoured headlessly without prompting.
        got = stt._select_audio_device(
            saved_device_name="External USB Mic", save_device_fn=lambda *_: None
        )
        ok &= check("saved device honoured headlessly", got, "External USB Mic")

        # Saved device that no longer exists + no TTY: fall back, never prompt.
        got = stt._select_audio_device(
            saved_device_name="Unplugged Mic", save_device_fn=lambda *_: None
        )
        ok &= check("missing saved device headless -> system default (None)", got, None)
    finally:
        sys.stdin.isatty = real_isatty

    print("\n" + ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
