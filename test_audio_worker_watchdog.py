#!/usr/bin/env python3
"""Regression test for the 'frozen grey waveform' bug.

A wedged CoreAudio device can open an InputStream without error whose callback
never fires. Recorder.start() must detect zero frames and RAISE (so the client
retry/backoff recycles the worker) instead of silently reporting success.

Run: python test_audio_worker_watchdog.py
"""
import sys, types, threading, time
import numpy as np

import audio_worker

# Speed the watchdog up for the test.
audio_worker.FIRST_FRAME_TIMEOUT_S = 0.3


def _install_fake_sounddevice(fire_frames: bool):
    """Inject a fake `sounddevice` whose stream opens fine but only delivers a
    frame when fire_frames=True."""
    mod = types.ModuleType("sounddevice")

    class InputStream:
        def __init__(self, **kw):
            self._cb = kw.get("callback")

        def start(self):
            # Stream opens with no error either way. Only a healthy device fires
            # the callback; a wedged one opens but stays silent (no frames).
            if fire_frames and self._cb:
                def _pump():
                    for _ in range(5):
                        self._cb(np.zeros((160, 1), dtype=np.float32), 160, None, None)
                        time.sleep(0.02)
                threading.Thread(target=_pump, daemon=True).start()

        def abort(self, *a, **k):
            pass

        def close(self, *a, **k):
            pass

    mod.InputStream = InputStream
    mod.query_devices = lambda: []
    sys.modules["sounddevice"] = mod


def check(name, ok):
    print(f"[{'ok' if ok else 'FAIL'}] {name}")
    return ok


def main() -> int:
    results = []

    # 1. Dead stream (opens, never delivers frames) -> start() must raise.
    _install_fake_sounddevice(fire_frames=False)
    rec = audio_worker.Recorder()
    raised = False
    try:
        rec.start(device_name=None, sample_rate=16000, channels=1)
    except Exception:
        raised = True
    results.append(check("dead stream (no frames) -> start() raises", raised))
    results.append(check("dead stream -> worker flags force_exit for recycle", rec.should_exit()))

    # 2. Healthy stream (delivers frames) -> start() succeeds, no raise.
    _install_fake_sounddevice(fire_frames=True)
    rec2 = audio_worker.Recorder()
    ok = False
    try:
        rec2.start(device_name=None, sample_rate=16000, channels=1)
        ok = rec2._stream is not None
    except Exception as e:
        print("   unexpected raise on healthy stream:", e)
    results.append(check("healthy stream (frames flow) -> start() succeeds", ok))
    try:
        rec2.cancel()
    except Exception:
        pass

    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
