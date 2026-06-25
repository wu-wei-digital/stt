from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from dotenv import dotenv_values
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


CONFIG_DIR = os.path.expanduser("~/.config/stt")
CONFIG_FILE = os.path.join(CONFIG_DIR, ".env")
INITIALIZED_MARKER = os.path.join(CONFIG_DIR, ".initialized")
LOCK_FILE = os.path.join(tempfile.gettempdir(), "stt.lock")

_BASE_ENV_KEYS = frozenset(os.environ.keys())
_MANAGED_ENV_KEYS: set[str] = set()


@dataclass
class Config:
    provider: str = "mlx"
    groq_api_key: str = ""
    whisper_model: str = ""
    parakeet_model: str = ""
    whisper_cpp_http_url: str = "http://localhost:8080"
    audio_device: str = ""  # device name
    language: str = "en"
    hotkey: str = "cmd_r"
    hotkey_mode: str = "hold"  # "hold" (push-to-talk) or "toggle"; only used by chord hotkeys
    mouse_trigger: bool = True  # middle-mouse-button starts/stops recording
    prompt: str = ""
    sound_enabled: bool = True
    keep_recordings: bool = False

    @staticmethod
    def from_env() -> "Config":
        return Config(
            provider=os.environ.get("PROVIDER", "mlx"),
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            whisper_model=os.environ.get("WHISPER_MODEL", ""),
            parakeet_model=os.environ.get("PARAKEET_MODEL", ""),
            whisper_cpp_http_url=os.environ.get("WHISPER_CPP_HTTP_URL", "http://localhost:8080"),
            audio_device=os.environ.get("AUDIO_DEVICE", ""),
            language=os.environ.get("LANGUAGE", "en"),
            hotkey=os.environ.get("HOTKEY", "cmd_r"),
            hotkey_mode=os.environ.get("HOTKEY_MODE", "hold"),
            mouse_trigger=os.environ.get("MOUSE_TRIGGER", "true").lower() == "true",
            prompt=os.environ.get("PROMPT", ""),
            sound_enabled=os.environ.get("SOUND_ENABLED", "true").lower() == "true",
            keep_recordings=os.environ.get("KEEP_RECORDINGS", "false").lower() == "true",
        )

    def to_env_dict(self) -> dict[str, str]:
        return {
            "PROVIDER": self.provider,
            "GROQ_API_KEY": self.groq_api_key,
            "WHISPER_MODEL": self.whisper_model,
            "PARAKEET_MODEL": self.parakeet_model,
            "WHISPER_CPP_HTTP_URL": self.whisper_cpp_http_url,
            "AUDIO_DEVICE": self.audio_device,
            "LANGUAGE": self.language,
            "HOTKEY": self.hotkey,
            "HOTKEY_MODE": self.hotkey_mode,
            "MOUSE_TRIGGER": str(self.mouse_trigger).lower(),
            "PROMPT": self.prompt,
            "SOUND_ENABLED": str(self.sound_enabled).lower(),
            "KEEP_RECORDINGS": str(self.keep_recordings).lower(),
        }


def load_env_startup() -> None:
    """Load environment variables from config files.

    Precedence (highest to lowest):
    - OS env
    - local .env (cwd)
    - global ~/.config/stt/.env
    """
    _apply_env_files(is_startup=True)


def reload_env_files() -> None:
    """Reload env files and apply changes (without overriding base OS env keys)."""
    _apply_env_files(is_startup=False)


def _read_env_files() -> dict[str, str]:
    """Read env files without mutating the process environment."""
    data: dict[str, str] = {}
    if os.path.exists(CONFIG_FILE):
        for k, v in dotenv_values(CONFIG_FILE).items():
            if v is None:
                continue
            data[str(k)] = str(v)

    local_env = os.path.join(os.getcwd(), ".env")
    if os.path.exists(local_env):
        for k, v in dotenv_values(local_env).items():
            if v is None:
                continue
            data[str(k)] = str(v)

    return data


def _apply_env_files(*, is_startup: bool) -> None:
    """Apply env file values into os.environ, without overriding base OS env keys."""
    global _MANAGED_ENV_KEYS

    data = _read_env_files()
    next_managed = {k for k in data.keys() if k not in _BASE_ENV_KEYS}

    if not is_startup:
        for key in list(_MANAGED_ENV_KEYS):
            if key not in next_managed:
                os.environ.pop(key, None)

    for key in next_managed:
        os.environ[key] = data[key]

    _MANAGED_ENV_KEYS = next_managed


def is_first_run() -> bool:
    return not os.path.exists(INITIALIZED_MARKER)


def mark_initialized() -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(INITIALIZED_MARKER, "w", encoding="utf-8") as f:
        f.write("")


def mask_api_key(key: str) -> str:
    if not key or len(key) < 8:
        return ""
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def save_config(key: str, value: str, *, force_global: bool = False) -> str:
    """Save a config value to local .env (if present) or global ~/.config/stt/.env."""
    local_env = os.path.join(os.getcwd(), ".env")
    if not force_global and os.path.exists(local_env):
        env_path = local_env
    else:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        env_path = CONFIG_FILE

    lines: list[str] = []
    found = False

    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                found = True
                break

    if not found:
        lines.append(f"{key}={value}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return env_path


class ConfigWatcher:
    """Watch config files for changes and trigger reload."""

    def __init__(self, on_config_change: Callable[[Config, dict[str, Any]], None]):
        self._on_config_change = on_config_change
        self._observer: Optional[Observer] = None
        self._watched_files: set[str] = set()
        self._last_mtime: dict[str, float] = {}
        self._debounce_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def start(self):
        local_env = os.path.join(os.getcwd(), ".env")
        if os.path.exists(local_env):
            self._watched_files.add(local_env)
        if os.path.exists(CONFIG_FILE):
            self._watched_files.add(CONFIG_FILE)
        elif os.path.exists(CONFIG_DIR):
            # Watch the directory for file creation.
            self._watched_files.add(CONFIG_FILE)

        if not self._watched_files:
            return

        for path in self._watched_files:
            if os.path.exists(path):
                self._last_mtime[path] = os.path.getmtime(path)

        self._observer = Observer()
        handler = _ConfigFileHandler(self._on_file_changed, self._watched_files)

        watched_dirs = set()
        for path in self._watched_files:
            dir_path = os.path.dirname(path)
            if dir_path and dir_path not in watched_dirs:
                watched_dirs.add(dir_path)
                self._observer.schedule(handler, dir_path, recursive=False)

        self._observer.start()

    def stop(self):
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None
        with self._lock:
            if self._debounce_timer:
                self._debounce_timer.cancel()
                self._debounce_timer = None

    def _on_file_changed(self, path: str):
        with self._lock:
            if os.path.exists(path):
                new_mtime = os.path.getmtime(path)
                old_mtime = self._last_mtime.get(path, 0)
                if new_mtime == old_mtime:
                    return
                self._last_mtime[path] = new_mtime

            if self._debounce_timer:
                self._debounce_timer.cancel()
            self._debounce_timer = threading.Timer(0.1, self._reload_config)
            self._debounce_timer.start()

    def _reload_config(self):
        old = Config.from_env().to_env_dict()

        # Reload env files. local wins over global; OS env wins over both.
        _apply_env_files(is_startup=False)

        new_cfg = Config.from_env()
        new = new_cfg.to_env_dict()

        changes: dict[str, Any] = {}
        for k, old_v in old.items():
            if old_v != new.get(k, ""):
                # Preserve booleans for a couple of known keys.
                if k in ("SOUND_ENABLED", "KEEP_RECORDINGS", "MOUSE_TRIGGER"):
                    changes[k] = new.get(k, "false").lower() == "true"
                else:
                    changes[k] = new.get(k, "")

        if changes:
            print(f"Config reloaded: {', '.join(changes.keys())}")
            self._on_config_change(new_cfg, changes)


class _ConfigFileHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[str], None], watched_files: set[str]):
        self._callback = callback
        self._watched_files = watched_files

    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path in self._watched_files:
            self._callback(event.src_path)

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path in self._watched_files:
            self._callback(event.src_path)
