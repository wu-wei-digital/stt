#!/usr/bin/env python3
"""
STT - Speech-to-text for macOS.

Entry-point module. Keep import-time dependencies lightweight so `import stt`
works in headless contexts (tests, harnesses, etc).
"""

from __future__ import annotations

import atexit
import fcntl
import os
import sys
import tempfile
import threading
from typing import Optional

from stt_app import AppState, STTApp
from stt_defaults import HOTKEY_DISPLAY_NAMES


HEADLESS = os.environ.get("STT_HEADLESS") == "1"

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("stt")
except Exception:
    __version__ = "0.0.0"

RELEASES_URL = "https://api.github.com/repos/bokan/stt/releases/latest"
LOCK_FILE = os.path.join(tempfile.gettempdir(), "stt.lock")

_lock_file: Optional[object] = None


def acquire_lock() -> bool:
    """Ensure only one instance is running."""
    global _lock_file
    _lock_file = open(LOCK_FILE, "w")
    try:
        fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_file.write(str(os.getpid()))
        _lock_file.flush()
        return True
    except (BlockingIOError, OSError):
        try:
            _lock_file.close()
        except Exception:
            pass
        _lock_file = None
        return False


def release_lock() -> None:
    global _lock_file
    if _lock_file:
        try:
            fcntl.flock(_lock_file, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            _lock_file.close()
        except Exception:
            pass
        try:
            os.unlink(LOCK_FILE)
        except OSError:
            pass
        _lock_file = None


def check_for_updates() -> None:
    """Check if a newer version is available on GitHub releases."""
    try:
        import requests
        from packaging.version import parse as parse_version

        response = requests.get(RELEASES_URL, timeout=5)
        if response.status_code != 200:
            return
        latest = str(response.json().get("tag_name", "")).lstrip("v")
        if latest and parse_version(latest) > parse_version(__version__):
            print(f"\n📦 Update available: {__version__} → {latest}", flush=True)
            print(
                "   Run: uv tool install --reinstall git+https://github.com/bokan/stt.git",
                flush=True,
            )
    except Exception:
        pass


def check_accessibility_permissions() -> bool:
    """Check and request accessibility permissions on macOS."""
    try:
        from ApplicationServices import AXIsProcessTrustedWithOptions

        options = {"AXTrustedCheckOptionPrompt": True}
        return bool(AXIsProcessTrustedWithOptions(options))
    except ImportError:
        print("⚠️  Could not check accessibility permissions")
        return True


def _select_audio_device(*, saved_device_name: str, save_device_fn) -> str | None:
    """List and select an audio input device. Returns device NAME (not index)."""
    import sounddevice as sd

    devices = sd.query_devices()
    input_devices = []

    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            input_devices.append((i, dev))

    if saved_device_name:
        for _, dev in input_devices:
            if dev["name"] == saved_device_name:
                return saved_device_name
        print(f"⚠️  Saved device '{saved_device_name}' not found, please select again")

    print("\nAvailable input devices:")
    for i, dev in input_devices:
        marker = "*" if i == sd.default.device[0] else " "
        print(f"  {marker} [{i}] {dev['name']}")
    print("\n  (* = default)")

    while True:
        choice = input("\nSelect device number (or press Enter for default): ").strip()
        if choice == "":
            return None
        try:
            device_idx = int(choice)
            matching = [(i, d) for i, d in input_devices if i == device_idx]
            if matching:
                device_name = matching[0][1]["name"]
                save = input("Save this device for future use? [y/N]: ").strip().lower()
                if save == "y":
                    save_device_fn(device_name)
                return device_name
            print("Invalid device number")
        except ValueError:
            print("Please enter a number")


def main() -> None:
    if HEADLESS:
        print("STT_HEADLESS=1 set; UI disabled")
        raise SystemExit(1)

    # Heavy imports live inside main.
    import subprocess
    import time

    from providers import get_provider
    from prompts_config import ensure_default_prompts
    from stt_config import (
        CONFIG_FILE,
        Config,
        ConfigWatcher,
        is_first_run,
        load_env_startup,
        reload_env_files,
        mark_initialized,
        save_config,
    )
    from text_injector import paste_text

    load_env_startup()
    cfg = Config.from_env()

    # --config: interactive wizard.
    if "--config" in sys.argv:
        from onboarding import run_setup

        def save(key: str, value: str):
            save_config(key, value, force_global=True)
            os.environ[str(key)] = str(value)

        current_config = {
            "provider": cfg.provider,
            "model": cfg.whisper_model,
            "groq_api_key": cfg.groq_api_key,
            "hotkey": cfg.hotkey,
            "audio_device": cfg.audio_device,
        }
        run_setup(save, current_config=current_config, reconfigure=True)
        return

    # Dev-only permission flow testing.
    if "--test-permissions" in sys.argv:
        from onboarding import (
            console,
            get_terminal_app,
            open_accessibility_settings,
            open_input_monitoring_settings,
            show_permission_error,
        )
        from rich.prompt import Confirm

        show_permission_error()
        if Confirm.ask("Open Accessibility settings?", default=True):
            open_accessibility_settings()
            console.print("\n[dim]Enable the permission, then come back here.[/dim]\n")
            Confirm.ask("Done with Accessibility?", default=True)
        if Confirm.ask("Open Input Monitoring settings?", default=True):
            open_input_monitoring_settings()
            console.print("\n[dim]Enable the permission, then come back here.[/dim]\n")
            Confirm.ask("Done with Input Monitoring?", default=True)

        terminal = get_terminal_app()
        console.print("\n[green]Permission setup complete.[/green]")
        console.print(f"[yellow]Restart {terminal} and run STT again.[/yellow]\n")
        return

    # Ensure only one instance.
    if not acquire_lock():
        from rich.console import Console

        Console().print("[red]Another instance of STT is already running[/red]")
        raise SystemExit(1)
    atexit.register(release_lock)

    # First-run onboarding.
    if is_first_run():
        from onboarding import run_first_time_setup

        def save_and_update(key: str, value: str):
            save_config(key, value, force_global=True)
            os.environ[str(key)] = str(value)

        run_first_time_setup(save_and_update)
        mark_initialized()
        reload_env_files()
        cfg = Config.from_env()

    from rich.console import Console
    from rich.status import Status

    console = Console()
    console.print()
    console.print("[bold]STT[/bold] [dim]Voice-to-text for macOS[/dim]")
    console.print("[dim]https://github.com/bokan/stt[/dim]")
    console.print()

    threading.Thread(target=check_for_updates, daemon=True).start()

    # Initialize provider (may be slow due to imports).
    provider = None
    init_error = None
    provider_available = False

    status = Status("[dim]Initializing...[/dim]", console=console, spinner="dots")
    status.start()

    slow_timer = threading.Timer(2.0, lambda: status.update("[dim]Initializing... first run may take ~30s[/dim]"))
    slow_timer.start()
    try:
        os.environ["PROVIDER"] = cfg.provider
        provider = get_provider(cfg.provider)
        provider_available = provider.is_available()
    except ValueError as e:
        init_error = e
    finally:
        slow_timer.cancel()
        status.stop()

    if init_error:
        console.print(f"[red]✗[/red] {init_error}")
        raise SystemExit(1)

    if not provider_available:
        if cfg.provider == "groq" and not cfg.groq_api_key:
            from onboarding import run_setup

            def save(key: str, value: str):
                save_config(key, value, force_global=True)
                os.environ[str(key)] = str(value)

            current_config = {
                "provider": cfg.provider,
                "model": cfg.whisper_model,
                "groq_api_key": cfg.groq_api_key,
                "hotkey": cfg.hotkey,
                "audio_device": cfg.audio_device,
            }
            run_setup(save, current_config=current_config, reconfigure=True)
            reload_env_files()
            cfg = Config.from_env()
            provider = get_provider(cfg.provider)
        else:
            console.print(f"[red]✗[/red] Provider '{cfg.provider}' is not available")
            if cfg.provider == "mlx":
                console.print("  [dim]Install with: pip install mlx-whisper[/dim]")
            raise SystemExit(1)

    assert provider is not None
    console.print(f"[green]✓[/green] Provider: [cyan]{provider.name}[/cyan]")
    provider.warmup()

    # Check accessibility permissions.
    if not check_accessibility_permissions():
        from onboarding import (
            get_terminal_app,
            open_accessibility_settings,
            prompt_open_settings,
            show_permission_error,
        )

        show_permission_error()
        if prompt_open_settings():
            open_accessibility_settings()
        terminal = get_terminal_app()
        console.print(f"\n[yellow]Restart {terminal} and run STT again.[/yellow]")
        raise SystemExit(1)

    # Select audio device (uses saved device or prompts).
    def save_device(device_name: str) -> None:
        env_path = save_config("AUDIO_DEVICE", device_name)
        os.environ["AUDIO_DEVICE"] = device_name
        print(f"  (saved to {env_path})")

    device_name = _select_audio_device(saved_device_name=cfg.audio_device, save_device_fn=save_device)
    if device_name:
        console.print(f"[green]✓[/green] Device: [cyan]{device_name}[/cyan]")
    else:
        console.print("[green]✓[/green] Device: [cyan]System default[/cyan]")

    console.print("[bright_black]Hint:[/bright_black] [dim]Run[/dim] [grey70]stt --config[/grey70] [dim]to change settings[/dim]")
    console.print()

    # Ensure default prompts exist before PromptOverlay loads.
    ensure_default_prompts()

    # UI imports (AppKit/Quartz/rumps) after config and provider are ready.
    from overlay import get_overlay
    from input_controller import InputController
    from menubar import STTMenuBar

    overlay = get_overlay()

    def play_sound(sound_path: str) -> None:
        if not cfg.sound_enabled:
            return
        subprocess.Popen(["afplay", sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def type_text(text: str, send_enter: bool = False) -> None:
        paste_text(text, send_enter=send_enter, method="osascript")

    app = STTApp(
        device_name=device_name,
        provider=provider,
        overlay=overlay,
        sound_player=play_sound,
        text_injector=type_text,
        language=cfg.language,
        prompt=cfg.prompt,
        hotkey_id=cfg.hotkey,
        keep_recordings=cfg.keep_recordings,
    )

    hotkey_name = HOTKEY_DISPLAY_NAMES.get(cfg.hotkey, cfg.hotkey)
    console.print(f"[bold green]Ready[/bold green] [dim]│[/dim] Hold [cyan]{hotkey_name}[/cyan] to record, +Shift ↵, Esc ✗")
    console.print()

    controller = InputController(app, hotkey_id=cfg.hotkey)

    config_watcher: Optional[ConfigWatcher] = None

    def cleanup():
        if config_watcher:
            config_watcher.stop()
        controller.stop()

    atexit.register(cleanup)

    def on_sound_toggle(enabled: bool):
        nonlocal cfg
        cfg.sound_enabled = enabled
        save_config("SOUND_ENABLED", str(enabled).lower())
        os.environ["SOUND_ENABLED"] = str(enabled).lower()

    menubar = STTMenuBar(
        stt_app=app,
        hotkey_name=hotkey_name,
        provider_name=provider.name,
        sound_enabled=cfg.sound_enabled,
        config_file=CONFIG_FILE,
        on_sound_toggle=on_sound_toggle,
        on_quit=cleanup,
    )

    # Config change handler (watcher starts only after menubar exists).
    def on_config_change(new_cfg: Config, changes: dict):
        nonlocal cfg, provider, hotkey_name
        cfg = new_cfg

        if "AUDIO_DEVICE" in changes:
            app.device_name = cfg.audio_device or None
            print(f"   Audio device: {cfg.audio_device or 'default'}")
        if "LANGUAGE" in changes:
            app.language = cfg.language
            print(f"   Language: {cfg.language}")
        if "HOTKEY" in changes or "HOTKEY_MODE" in changes:
            controller.set_hotkey_id(cfg.hotkey)
            hotkey_name = HOTKEY_DISPLAY_NAMES.get(cfg.hotkey, cfg.hotkey)
            menubar.update_hotkey_name(hotkey_name)
            print(f"   Hotkey: {hotkey_name}")
        if "PROMPT" in changes:
            app.prompt = cfg.prompt
            print(f"   Prompt: {cfg.prompt or '(empty)'}")
        if "KEEP_RECORDINGS" in changes:
            app.keep_recordings = cfg.keep_recordings
            print(f"   Keep recordings: {'enabled' if cfg.keep_recordings else 'disabled'}")
        if "SOUND_ENABLED" in changes:
            menubar.update_sound_enabled(cfg.sound_enabled)
            print(f"   Sound: {'enabled' if cfg.sound_enabled else 'disabled'}")

        if (
            "PROVIDER" in changes
            or "WHISPER_MODEL" in changes
            or "GROQ_API_KEY" in changes
            or "WHISPER_CPP_HTTP_URL" in changes
            or "PARAKEET_MODEL" in changes
        ):
            try:
                provider = get_provider(cfg.provider)
                if provider.is_available():
                    provider.warmup()
                    app.provider = provider
                    menubar.update_provider_name(provider.name)
                    print(f"   Provider: {provider.name}")
                else:
                    print(f"   Provider '{cfg.provider}' not available")
            except Exception as e:
                print(f"   Failed to reinitialize provider: {e}")

    config_watcher = ConfigWatcher(on_config_change)
    config_watcher.start()

    controller.start()
    menubar.run()


if __name__ == "__main__":
    main()
