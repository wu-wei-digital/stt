from __future__ import annotations

import atexit
import json
import os
import queue
import select
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Optional


class AudioWorkerClient:
    """Client for the audio recording worker subprocess (audio_worker.py)."""

    _WORKER_STARTUP_TIMEOUT_S = 10
    _START_TIMEOUT_S = 10
    _STOP_TIMEOUT_S = 10
    _CANCEL_TIMEOUT_S = 5
    _WRITE_TIMEOUT_S = 2.0
    _WRITE_LOCK_TIMEOUT_S = 2.0

    # After a recording whose stream-close hung, the worker force-exits and the
    # macOS CoreAudio input device stays busy for a beat. The next recording
    # then has to cold-start a stream on a still-busy device and drops out (the
    # "works every second time" symptom). Retry start on a fresh worker with a
    # growing backoff so the device has time to release. Tunable via env.
    _START_ATTEMPTS = max(1, int(os.environ.get("STT_START_ATTEMPTS", "4")))
    _START_BACKOFF_S = max(0.0, float(os.environ.get("STT_START_BACKOFF_S", "0.4")))

    def __init__(self):
        self._proc: subprocess.Popen[str] | None = None
        self._messages: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._next_id = 1
        self._cleanup_registered = False

        self._waveform_callback: Optional[Callable[[list[float], float], None]] = None

    def set_waveform_callback(self, callback: Optional[Callable[[list[float], float], None]]):
        """Set callback for waveform updates (values, raw_peak)."""
        self._waveform_callback = callback

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def ensure_running(self) -> None:
        with self._lock:
            self._ensure_running_locked()

    def stop(self, force: bool = False) -> None:
        with self._lock:
            self._stop_locked(force=force)

    def start_recording(self, *, device_name: str | None, sample_rate: int, channels: int) -> None:
        with self._lock:
            last_error: Exception | None = None
            attempts = self._START_ATTEMPTS
            for attempt in range(attempts):
                try:
                    self._ensure_running_locked()
                    req_id = self._next_id
                    self._next_id += 1

                    payload = json.dumps(
                        {
                            "type": "start",
                            "id": req_id,
                            "device_name": device_name,
                            "sample_rate": sample_rate,
                            "channels": channels,
                        }
                    )
                    if not self._write_lock.acquire(timeout=self._WRITE_LOCK_TIMEOUT_S):
                        raise TimeoutError("Timed out waiting for audio worker write lock")
                    try:
                        self._write_line(payload + "\n", timeout_s=self._WRITE_TIMEOUT_S)
                    finally:
                        self._write_lock.release()

                    message = self._wait_for_locked(
                        lambda m: m.get("type") in {"started", "error"} and m.get("id") == req_id,
                        timeout_s=self._START_TIMEOUT_S,
                    )
                    if not message:
                        raise TimeoutError("Timed out starting audio recording")
                    if message.get("type") == "error":
                        raise RuntimeError(message.get("error") or "Failed to start recording")
                    return
                except Exception as e:
                    last_error = e
                    self._stop_locked(force=True)
                    if attempt < attempts - 1:
                        backoff = self._START_BACKOFF_S * (attempt + 1)
                        print(
                            f"[stt:audio-worker-client] start attempt {attempt + 1}/{attempts} "
                            f"failed ({e}); retrying in {backoff:.1f}s",
                            file=sys.stderr,
                            flush=True,
                        )
                        if backoff > 0:
                            time.sleep(backoff)
                        continue
                    raise
            if last_error:
                raise last_error

    def stop_recording(self, *, wav_path: str) -> tuple[int, float]:
        with self._lock:
            self._ensure_running_locked()
            req_id = self._next_id
            self._next_id += 1

            payload = json.dumps({"type": "stop", "id": req_id, "wav_path": wav_path})
            if not self._write_lock.acquire(timeout=self._WRITE_LOCK_TIMEOUT_S):
                raise TimeoutError("Timed out waiting for audio worker write lock")
            try:
                self._write_line(payload + "\n", timeout_s=self._WRITE_TIMEOUT_S)
            finally:
                self._write_lock.release()

            message = self._wait_for_locked(
                lambda m: m.get("type") in {"stopped", "error"} and m.get("id") == req_id,
                timeout_s=self._STOP_TIMEOUT_S,
            )
            if not message:
                raise TimeoutError("Timed out stopping audio recording")
            if message.get("type") == "error":
                raise RuntimeError(message.get("error") or "Failed to stop recording")

            frames = message.get("frames")
            peak = message.get("peak")
            try:
                return int(frames or 0), float(peak or 0.0)
            except (TypeError, ValueError):
                return 0, 0.0

    def cancel_recording(self) -> None:
        with self._lock:
            if not self.is_running():
                return
            req_id = self._next_id
            self._next_id += 1

            payload = json.dumps({"type": "cancel", "id": req_id})
            if not self._write_lock.acquire(timeout=self._WRITE_LOCK_TIMEOUT_S):
                raise TimeoutError("Timed out waiting for audio worker write lock")
            try:
                self._write_line(payload + "\n", timeout_s=self._WRITE_TIMEOUT_S)
            finally:
                self._write_lock.release()

            message = self._wait_for_locked(
                lambda m: m.get("type") in {"canceled", "error"} and m.get("id") == req_id,
                timeout_s=self._CANCEL_TIMEOUT_S,
            )
            if not message:
                raise TimeoutError("Timed out cancelling audio recording")
            if message.get("type") == "error":
                raise RuntimeError(message.get("error") or "Failed to cancel recording")

    def _read_stdout(self, proc: subprocess.Popen[str], messages: "queue.Queue[dict[str, Any]]") -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if msg.get("type") == "waveform":
                    cb = self._waveform_callback
                    if cb:
                        try:
                            cb(msg.get("values", []), float(msg.get("raw_peak", 0.0) or 0.0))
                        except Exception:
                            pass
                else:
                    messages.put(msg)
            except json.JSONDecodeError:
                messages.put({"type": "stdout", "line": line})
        messages.put({"type": "eof"})

    def _write_line(self, line: str, timeout_s: float) -> None:
        assert self._proc is not None
        assert self._proc.stdin is not None
        fd = self._proc.stdin.fileno()
        data = line.encode("utf-8")
        total = 0
        deadline = time.time() + timeout_s
        while total < len(data):
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("Timed out writing to audio worker stdin")
            _, writable, _ = select.select([], [fd], [], remaining)
            if not writable:
                raise TimeoutError("Timed out writing to audio worker stdin")
            written = os.write(fd, data[total:])
            if written <= 0:
                raise RuntimeError("Failed to write to audio worker stdin")
            total += written

    def _wait_for_locked(self, predicate, timeout_s: int) -> dict[str, Any] | None:
        deadline = time.time() + timeout_s if timeout_s > 0 else None
        while True:
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
            else:
                remaining = None

            try:
                message = self._messages.get(timeout=remaining)
            except queue.Empty:
                return None

            if message.get("type") == "eof":
                return {"type": "error", "error": "Audio worker exited unexpectedly"}

            if predicate(message):
                return message

    def _ensure_running_locked(self) -> None:
        if self.is_running():
            return

        self._stop_locked(force=True)

        worker_path = os.path.join(os.path.dirname(__file__), "audio_worker.py")
        if not os.path.exists(worker_path):
            raise FileNotFoundError(f"Missing audio worker at {worker_path}")

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env["STT_PARENT_PID"] = str(os.getpid())

        last_error: Exception | None = None
        for attempt in range(2):
            proc = subprocess.Popen(
                [sys.executable, "-u", worker_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,  # inherit stderr to avoid pipe deadlocks
                text=True,
                bufsize=1,
                env=env,
            )
            messages: "queue.Queue[dict[str, Any]]" = queue.Queue()
            thread = threading.Thread(target=self._read_stdout, args=(proc, messages), daemon=True)
            thread.start()

            self._proc = proc
            self._messages = messages
            self._reader_thread = thread

            ready = self._wait_for_locked(
                lambda m: m.get("type") in {"ready", "error"},
                timeout_s=self._WORKER_STARTUP_TIMEOUT_S,
            )
            if ready and ready.get("type") == "ready":
                if not self._cleanup_registered:
                    atexit.register(self.stop)
                    self._cleanup_registered = True
                return

            if not ready:
                last_error = TimeoutError("Audio worker did not become ready in time")
            else:
                last_error = RuntimeError(ready.get("error") or "Audio worker failed to start")

            self._stop_locked(force=True)
            if attempt == 0:
                time.sleep(0.1)
                continue

        if last_error:
            raise last_error
        raise RuntimeError("Audio worker failed to start")

    def _stop_locked(self, force: bool = False) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return

        # Close stdin first to signal worker to stop and unblock any writes.
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass

        try:
            if not force and proc.poll() is None:
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass

            if proc.poll() is None:
                proc.terminate()
                # Close stdout to unblock reader thread before waiting.
                try:
                    if proc.stdout:
                        proc.stdout.close()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass

            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        except Exception:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
