#!/usr/bin/env python3
"""Regression test for the 48kHz-mic 'waveform freezes after one frame' bug.

The model needs 16kHz, so the worker used to open the input stream at a fixed
16000. On a device whose native rate is 48000 (the MacBook Air mic), CoreAudio
opens the stream, delivers a frame or two, then stalls (the waveform shifts once
then freezes). The fix: capture at the device's NATIVE rate and resample the
captured audio down to the 16kHz target before writing the wav, so any mic works.

Run: python test_audio_worker_resample.py
"""
import sys, types, time, tempfile, os
import numpy as np
from scipy.io import wavfile

import audio_worker

audio_worker.FIRST_FRAME_TIMEOUT_S = 0.3

_OPENED = {}


def _install_fake_sounddevice(native_rate: float):
    """Fake sounddevice: device reports `native_rate`; InputStream records the
    samplerate it was opened with and pumps one chunk of frames synchronously."""
    mod = types.ModuleType("sounddevice")

    class InputStream:
        def __init__(self, **kw):
            self._cb = kw.get("callback")
            _OPENED["samplerate"] = kw.get("samplerate")
            _OPENED["channels"] = kw.get("channels")

        def start(self):
            # 4800 frames at 48k = 0.1s of audio; fire synchronously so the
            # first-frame watchdog passes deterministically.
            if self._cb:
                self._cb(np.zeros((4800, 1), dtype=np.float32), 4800, None, None)

        def abort(self, *a, **k):
            pass

        def close(self, *a, **k):
            pass

    def query_devices(*args, **kwargs):
        return {"name": "Fake", "default_samplerate": native_rate, "max_input_channels": 1}

    mod.InputStream = InputStream
    mod.query_devices = query_devices
    sys.modules["sounddevice"] = mod


def check(name, ok):
    print(f"[{'ok' if ok else 'FAIL'}] {name}")
    return ok


def main() -> int:
    results = []

    # 1. Pure resampler: 48k -> 16k is a 1/3 length change; identity when equal.
    sig = np.zeros((4800, 1), dtype=np.float32)
    out = audio_worker.resample_to(sig, 48000, 16000)
    results.append(check("resample_to 48k->16k yields ~1/3 length", abs(out.shape[0] - 1600) <= 2))
    same = audio_worker.resample_to(sig, 16000, 16000)
    results.append(check("resample_to is identity when rates match", same.shape[0] == 4800))

    # 2. start() must open the stream at the device's NATIVE 48k, not 16k.
    _install_fake_sounddevice(native_rate=48000.0)
    rec = audio_worker.Recorder()
    rec.start(device_name=None, sample_rate=16000, channels=1)
    results.append(check("start() opens stream at native 48000", _OPENED.get("samplerate") == 48000))

    # 3. stop() resamples to the 16k target: wav rate is 16000 and ~1/3 frames.
    with tempfile.TemporaryDirectory() as d:
        wav_path = os.path.join(d, "out.wav")
        frames, peak = rec.stop(wav_path=wav_path)
        rate, data = wavfile.read(wav_path)
        results.append(check("stop() writes a 16000 Hz wav", rate == 16000))
        results.append(check("stop() reports ~1/3 frames (downsampled)", abs(frames - 1600) <= 2))

    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
