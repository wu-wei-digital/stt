from __future__ import annotations

import os
import re
import tempfile
import threading
import time
from collections import deque
from enum import Enum
from typing import Callable, Optional

from audio_worker_client import AudioWorkerClient
from issue_capture import maybe_capture_mlx_issue
from postprocess import collapse_repeats
from providers import get_provider
from recordings import DEFAULT_RECORDINGS_DIR, DEFAULT_RECORDINGS_MAX_BYTES, archive_recording
from stt_defaults import HOTKEY_DISPLAY_NAMES, NullOverlay, noop_sound, noop_text_injector


SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHANNELS = 1
SILENCE_THRESHOLD = 0.01  # Skip transcription if peak below this

class AppState(Enum):
    """Application state for menu bar icon."""

    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"

class STTApp:
    # How long stop_recording waits for an in-flight start before giving up and
    # restarting the worker. Must exceed the client's start retry budget
    # (STT_START_ATTEMPTS x backoff) so a key release mid-retry doesn't abort a
    # recovery that is about to succeed. Tunable via env.
    _MAX_STARTING_TIME_S = float(os.environ.get("STT_MAX_STARTING_TIME_S", "9"))

    def __init__(
        self,
        device_name: str | None = None,
        provider=None,
        *,
        overlay=None,
        sound_player: Callable[[str], None] | None = None,
        text_injector: Callable[[str, bool], None] | None = None,
        language: str | None = None,
        prompt: str | None = None,
        hotkey_id: str | None = None,
        keep_recordings: bool | None = None,
        recordings_dir: str | None = None,
        recordings_max_bytes: int | None = None,
        audio_worker: AudioWorkerClient | None = None,
    ):
        self.recording = False
        self.device_name = device_name  # Store name, resolve to index at record time.
        self.provider = provider or get_provider(os.environ.get("PROVIDER", "mlx"))
        self._audio_worker = audio_worker or AudioWorkerClient()
        self._overlay = overlay or NullOverlay()

        self.language = language if language is not None else os.environ.get("LANGUAGE", "en")
        self.prompt = prompt if prompt is not None else os.environ.get("PROMPT", "")
        self.hotkey_id = hotkey_id if hotkey_id is not None else os.environ.get("HOTKEY", "cmd_r")

        if keep_recordings is None:
            keep_recordings = os.environ.get("KEEP_RECORDINGS", "false").lower() == "true"
        self.keep_recordings = keep_recordings
        self.recordings_dir = recordings_dir or DEFAULT_RECORDINGS_DIR
        self.recordings_max_bytes = recordings_max_bytes or DEFAULT_RECORDINGS_MAX_BYTES

        self._sound_player = sound_player or noop_sound
        self._text_injector = text_injector or noop_text_injector

        # Set up waveform callback to update overlay.
        self._audio_worker.set_waveform_callback(self._on_waveform)

        # Thread synchronization.
        self._lock = threading.Lock()
        self._processing = False  # Guard against concurrent process_recording calls.
        self._starting = False  # Guard against concurrent start_recording calls.
        self._event_log = deque(maxlen=200)

        # Used to invalidate stale work (cancel/reset while worker thread is still running).
        self._op_id = 0

        # State management for menu bar.
        self._state = AppState.IDLE
        self._state_callback: Optional[Callable[[AppState], None]] = None

    def _on_waveform(self, values: list[float], raw_peak: float):
        """Handle waveform data from audio worker."""
        above_threshold = raw_peak >= SILENCE_THRESHOLD
        self._overlay.update_waveform(values, above_threshold)

    def set_state_callback(self, callback: Callable[[AppState], None]):
        """Register callback for state changes (called from any thread)."""
        self._state_callback = callback

    def _set_state(self, new_state: AppState):
        """Update state and notify callback."""
        self._state = new_state
        self._log_event(f"state:{new_state.value}")
        if self._state_callback:
            self._state_callback(new_state)

    def _log_event(self, message: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "ts": ts,
            "message": message,
            "state": self._state.value,
            "recording": self.recording,
            "processing": self._processing,
            "starting": self._starting,
        }
        self._event_log.append(entry)
        if os.environ.get("STT_DEBUG"):
            print(f"[debug] {entry}")

    def start_recording(self):
        """Start recording audio from microphone."""
        with self._lock:
            if self._processing:
                self._log_event("start_ignored_processing")
                return
            if self.recording or self._starting:
                self._log_event("start_ignored_busy")
                return
            self._starting = True
            self.recording = True
            self._log_event("start_recording")

        self._set_state(AppState.RECORDING)
        self._overlay.show()
        self._sound_player("/System/Library/Sounds/Tink.aiff")
        print("Recording...")

        try:
            self._audio_worker.start_recording(
                device_name=self.device_name, sample_rate=SAMPLE_RATE, channels=CHANNELS
            )
            with self._lock:
                if not self.recording:
                    try:
                        self._audio_worker.cancel_recording()
                    except Exception:
                        self._audio_worker.stop(force=True)
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            self._audio_worker.stop(force=True)
            self._overlay.hide()
            with self._lock:
                self.recording = False
            self._set_state(AppState.IDLE)
        finally:
            with self._lock:
                self._starting = False

    def stop_recording(self):
        """Stop recording and return (wav_path, frames, peak)."""
        with self._lock:
            if not self.recording:
                return None, 0, 0.0
            self.recording = False
            starting = self._starting

        self._overlay.set_transcribing(True)
        self._sound_player("/System/Library/Sounds/Pop.aiff")
        print("Stopped")

        if starting:
            deadline = time.time() + self._MAX_STARTING_TIME_S
            while time.time() < deadline:
                with self._lock:
                    if not self._starting:
                        break
                time.sleep(0.01)
            with self._lock:
                if self._starting:
                    print("⚠️  Recording start still pending; restarting audio worker...")
                    self._starting = False
                    self._processing = False
                    try:
                        self._audio_worker.stop(force=True)
                    except Exception:
                        pass
                    return None, 0, 0.0

        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            frames, peak = self._audio_worker.stop_recording(wav_path=wav_path)
            return wav_path, frames, peak
        except TimeoutError:
            print("❌ Audio recording stop timed out. Restarting audio worker...")
            self._audio_worker.stop(force=True)
            try:
                if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                    archived = archive_recording(
                        wav_path,
                        keep_recordings=self.keep_recordings,
                        recordings_dir=self.recordings_dir,
                        recordings_max_bytes=self.recordings_max_bytes,
                        text=None,
                    )
                    if not archived:
                        print(f"⚠️  Kept timed-out wav for debugging: {wav_path}")
                        return wav_path, 0, 0.0
                else:
                    os.unlink(wav_path)
            except OSError:
                pass
            return None, 0, 0.0
        except Exception as e:
            print(f"❌ Failed to stop recording: {e}")
            self._audio_worker.stop(force=True)
            try:
                os.unlink(wav_path)
            except OSError:
                pass
            return None, 0, 0.0

    def cancel_recording(self):
        """Cancel recording without processing."""
        with self._lock:
            if not self.recording:
                if self._state == AppState.RECORDING:
                    self._set_state(AppState.IDLE)
                    self._overlay.hide()
                return
            self.recording = False

        self._set_state(AppState.IDLE)
        self._overlay.hide()
        self._sound_player("/System/Library/Sounds/Basso.aiff")
        print("❌ Recording cancelled")
        try:
            self._audio_worker.cancel_recording()
        except TimeoutError:
            print("⚠️  Audio cancel timed out. Restarting audio worker...")
            self._audio_worker.stop(force=True)
        except Exception as e:
            print(f"⚠️  Error cancelling audio: {e}")
            self._audio_worker.stop(force=True)

    def cancel_transcription(self):
        """Cancel an in-progress transcription (best-effort)."""
        with self._lock:
            if not self._processing:
                return
            self._op_id += 1
            self._processing = False

        cancel = getattr(self.provider, "cancel", None)
        if callable(cancel):
            print("Cancelling...")
            try:
                cancel()
            except Exception as e:
                print(f"⚠️  Error cancelling transcription: {e}")
        self._overlay.set_transcribing(False)
        self._overlay.hide()
        self._set_state(AppState.IDLE)

    def transcribe_audio(self, audio_file_path: str, max_retries: int = 2) -> str | None:
        """Transcribe audio using the configured provider (no thread wrapper)."""
        for attempt in range(max_retries + 1):
            try:
                return self.provider.transcribe(audio_file_path, self.language, self.prompt)
            except TimeoutError:
                cancel = getattr(self.provider, "cancel", None)
                if callable(cancel):
                    try:
                        cancel()
                    except Exception:
                        pass
                if attempt < max_retries:
                    print(f"⚠️  Transcription timed out, retrying ({attempt + 2}/{max_retries + 1})...")
                    continue
                print("❌ Transcription timed out after all retries")
                return None
            except Exception as e:
                if attempt < max_retries:
                    print(f"⚠️  Transcription failed, retrying ({attempt + 2}/{max_retries + 1})...")
                    continue
                print(f"❌ Transcription error after all retries: {e}")
                return None

        return None

    def print_ready_prompt(self):
        """Print the ready prompt with hotkey name."""
        from rich.console import Console

        console = Console()
        hotkey_name = HOTKEY_DISPLAY_NAMES.get(self.hotkey_id, self.hotkey_id)
        console.print(
            f"\n[bold green]Ready[/bold green] [dim]│[/dim] Hold [cyan]{hotkey_name}[/cyan] to record, +Shift ↵, Esc ✗"
        )

    def transform_text(self, text: str) -> str:
        """Apply text transformations."""
        text = re.sub(r"^[Ss]lash\s+", "/", text)
        text = collapse_repeats(text)  # drop Whisper repetition loops (all providers)
        return text

    def type_text(self, text: str, send_enter: bool = False) -> None:
        """Type text into the active text field (injected backend)."""
        if not text:
            return
        print(f"Typing: {text}" + (" ↵" if send_enter else ""))
        try:
            self._text_injector(text, send_enter)
        except Exception as e:
            print(f"❌ Failed to type text: {e}")

    def process_recording(self, send_enter: bool = False):
        """Process the recorded audio: transcribe and type."""
        with self._lock:
            if self._processing:
                self._log_event("process_ignored_processing")
                return
            if self._starting:
                self._log_event("process_wait_starting")
            self._processing = True
            self._op_id += 1
            op_id = self._op_id

        wav_path = None
        transcribed_text = None
        try:
            wav_path, frames, peak = self.stop_recording()

            with self._lock:
                if op_id != self._op_id or not self._processing:
                    return

            if not wav_path:
                print("⚠️  No audio captured, skipping...")
            elif frames < int(SAMPLE_RATE * 0.5):  # Less than 0.5 seconds.
                print("⚠️  Recording too short, skipping...")
            elif peak < SILENCE_THRESHOLD:
                print("⚠️  Audio too quiet (silence), skipping...")
            else:
                self._set_state(AppState.TRANSCRIBING)
                text = self.transcribe_audio(wav_path)

                with self._lock:
                    if op_id != self._op_id or not self._processing:
                        return

                if text:
                    text = self.transform_text(text)
                    transcribed_text = text
                    self.type_text(text, send_enter=send_enter)
                    print(f"✓ {text}")
                else:
                    if maybe_capture_mlx_issue(
                        provider=self.provider,
                        wav_path=wav_path,
                        language=self.language,
                        prompt=self.prompt,
                    ):
                        wav_path = None
                    print("No transcription returned")

            self.print_ready_prompt()
        except Exception as e:
            print(f"❌ Error processing recording: {e}")
        finally:
            # Archive or clean up temp file.
            if wav_path:
                archived = archive_recording(
                    wav_path,
                    keep_recordings=self.keep_recordings,
                    recordings_dir=self.recordings_dir,
                    recordings_max_bytes=self.recordings_max_bytes,
                    text=transcribed_text,
                )
                if not archived:
                    try:
                        os.unlink(wav_path)
                    except OSError:
                        pass

            with self._lock:
                is_current = op_id == self._op_id
                if is_current:
                    self._processing = False

            # Hide overlay and reset state (only if this is still the active op).
            if is_current:
                self._overlay.set_transcribing(False)
                self._overlay.hide()
                self._set_state(AppState.IDLE)
