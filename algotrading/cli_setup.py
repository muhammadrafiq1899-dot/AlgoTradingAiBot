"""First-run configuration helpers for the AlgoTrading bot.

Shared by the interactive setup wizard (scripts/setup.py) and the `algobot`
command. All logic lives here so there is a single source of truth for how
secrets are read from / written to .env, and for how the bot is started /
stopped / inspected on the terminal.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
RUN_BOT = PROJECT_ROOT / "scripts" / "run_bot.sh"
PID_PATH = PROJECT_ROOT / "data" / "algobot.pid"
LOG_PATH = PROJECT_ROOT / "logs" / "algotrading.log"

# Field order shown to the user during setup.
ENV_FIELDS: list[dict] = [
    {
        "key": "TELEGRAM_BOT_TOKEN",
        "label": "Telegram bot token",
        "help": "From @BotFather in Telegram (looks like 123456:ABC...). "
                "This is how the bot talks to you.",
        "secret": True,
    },
    {
        "key": "TELEGRAM_ALLOWED_USERS",
        "label": "Your Telegram user ID(s)",
        "help": "Your numeric Telegram ID (ask @userinfobot). Comma-separated "
                "for multiple users. Only these IDs can control the bot.",
        "secret": False,
    },
    {
        "key": "AI_API_KEY",
        "label": "AI assistant API key",
        "help": "Optional. Enables the daily AI strategy suggestions. "
                "Leave blank to run without the AI assistant.",
        "secret": True,
        "optional": True,
    },
    {
        "key": "AI_BASE_URL",
        "label": "AI base URL",
        "help": "OpenAI-compatible endpoint. Default works for OpenAI; change "
                "it for other providers.",
        "secret": False,
        "optional": True,
        "default": "https://api.openai.com/v1",
    },
    {
        "key": "AI_MODEL",
        "label": "AI model",
        "help": "Model name to use for suggestions.",
        "secret": False,
        "optional": True,
        "default": "gpt-4o-mini",
    },
]


# ---------------------------------------------------------------------------
# .env read / write
# ---------------------------------------------------------------------------

def read_env() -> dict[str, str]:
    """Parse the .env file into {KEY: value}. Missing file -> {}."""
    if not ENV_PATH.exists():
        return {}
    values: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip()
    return values


def write_env(values: dict[str, str]) -> None:
    """Write a .env file preserving the original comments/order where possible,
    updating any keys present in `values`. Keys not present keep old values."""
    existing = read_env()
    merged = {**existing, **values}

    # Preserve header + any non-secret comments from the template.
    header = []
    if ENV_PATH.exists():
        src = ENV_PATH.read_text(encoding="utf-8")
        header = [l for l in src.splitlines()
                  if l.strip().startswith("#") or (l.strip() and "=" not in l)]

    lines = list(header)
    if lines and not lines[-1].startswith("#"):
        lines.append("")
    for key in sorted(merged):
        lines.append(f"{key}={merged[key]}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# interactive prompts
# ---------------------------------------------------------------------------

def _mask(value: str) -> str:
    """Shorten a secret for display, e.g. 1234...wxyz."""
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "..." + value[-4:]


def _input(prompt: str, secret: bool = False) -> str:
    """Read a line from stdin. Always plain input() so long-press paste works.

    getpass() is NOT used for secrets: on Termux it opens /dev/tty directly and
    reads there, bypassing stdin — pasted tokens (inserted into stdin) would
    never arrive. Echo of the value is unavoidable in the Termux terminal.
    """
    return input(prompt)


def prompt_for_fields(existing: dict[str, str]) -> dict[str, str]:
    """Ask the user for each field, showing current value and letting them
    keep it by pressing Enter. Returns a dict of changed values."""
    print()
    print("AlgoTrading first-time setup")
    print("============================")
    print("Press Enter to keep the current value shown in brackets. "
          "Type 'skip' to leave a value blank.")
    print()

    updates: dict[str, str] = {}

    for field in ENV_FIELDS:
        key = field["key"]
        current = existing.get(key, "")
        shown = _mask(current) if (field.get("secret") and current) else current
        if not current and field.get("default"):
            shown = field["default"]

        print(f"-- {field['label']} --")
        print(f"   {field['help']}")
        prompt = f"   [{shown}]> "
        while True:
            raw = _input(prompt, secret=bool(field.get("secret"))).strip()
            if raw == "":
                # keep current (or default if none set)
                new = current or field.get("default", "")
                break
            if raw.lower() == "skip":
                new = ""
                break
            # non-secret typed value overrides default too
            new = raw
            break

        updates[key] = new

    return updates


def validate_telegram(token: str) -> bool:
    """Call Telegram's getMe to confirm the token is valid. Returns True on
    success, False on any failure (network or bad token)."""
    if not token:
        return False
    import urllib.request
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            import json
            data = json.loads(resp.read().decode())
            return bool(data.get("ok"))
    except Exception:
        return False


def apply_updates(updates: dict[str, str]) -> None:
    """Persist the setup fields into .env, then init the DB."""
    write_env(updates)
    print("\nSaved .env")
    # Keep AI disabled until a key is actually present.
    # Initialize the database so first run has tables ready.
    _run([str(PYTHON), str(PROJECT_ROOT / "scripts" / "init_db.py")],
         "initializing database")


# ---------------------------------------------------------------------------
# bot lifecycle (used by algobot command)
# ---------------------------------------------------------------------------

def _run(cmd: list[str], what: str) -> int:
    print(f"==> {what} ...")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"!! {what} failed (exit {proc.returncode})")
    return proc.returncode


def get_pid() -> int | None:
    try:
        return int(PID_PATH.read_text().strip())
    except Exception:
        return None


def is_running() -> bool:
    pid = get_pid()
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start(args: list[str]) -> None:
    if is_running():
        print(f"Bot is already running (pid {get_pid()}).")
        return
    if not PYTHON.exists():
        print("No virtualenv found. Run:  bash scripts/setup_termux.sh")
        return

    # Launch the supervisor loop detached from this terminal.
    log = open(LOG_PATH, "a", encoding="utf-8") if LOG_PATH.parent.exists() else subprocess.DEVNULL
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_PATH, "a", encoding="utf-8")
    devnull = open(os.devnull, "w")
    cmd = ["bash", str(RUN_BOT)] + args
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            stdin=devnull, start_new_session=True)
    PID_PATH.write_text(str(proc.pid))
    print("Bot starting in the background.")
    print(f"  watch logs :  algobot logs")
    print(f"  stop       :  algobot stop")


def stop() -> None:
    pid = get_pid()
    if not pid or not is_running():
        print("Bot is not running.")
        return
    import signal
    # The bot runs as a supervisor process group (start_new_session=True), so
    # the group leader's pid == its pgid. Signal the whole group to take down
    # both the supervisor loop and the bot child it spawns.
    try:
        os.killpg(pid, signal.SIGTERM)
        print(f"Sent stop signal to bot (pid {pid}).")
    except OSError as exc:
        print(f"Could not signal pid {pid}: {exc}")


def status() -> None:
    from algotrading.config import load_settings
    s = load_settings()
    running = is_running()
    print(f"mode          : {s.mode}")
    print(f"running       : {'yes (pid ' + str(get_pid()) + ')' if running else 'no'}")
    print(f"telegram      : {'configured' if os.getenv('TELEGRAM_BOT_TOKEN') else 'missing'}")
    print(f"ai assistant  : {'enabled' if os.getenv('AI_API_KEY') else 'disabled'}")
    print(f"symbols       : {', '.join(s.market.symbols)}")


def show_logs(tail: int = 40) -> None:
    if not LOG_PATH.exists():
        print("No log file yet.")
        return
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    for line in lines[-tail:]:
        print(line)


def menu() -> None:
    while True:
        print()
        print("AlgoTrading — what do you want to do?")
        print("  1) Start the bot")
        print("  2) Stop the bot")
        print("  3) Status")
        print("  4) Show recent logs")
        print("  5) Run configuration setup")
        print("  0) Exit")
        choice = _input("  > ")
        if choice == "1":
            start([])
        elif choice == "2":
            stop()
        elif choice == "3":
            status()
        elif choice == "4":
            show_logs()
        elif choice == "5":
            run_setup()
        elif choice == "0":
            break
        else:
            print("  Invalid choice.")


def run_setup() -> None:
    """Interactive first-run configuration."""
    existing = read_env()
    updates = prompt_for_fields(existing)
    apply_updates(updates)

    # Validate the Telegram token if one was set/changed.
    token = updates.get("TELEGRAM_BOT_TOKEN", existing.get("TELEGRAM_BOT_TOKEN", ""))
    if token:
        if validate_telegram(token):
            print("Telegram token OK.")
        else:
            print("Could not verify the Telegram token (network or wrong token). "
                  "You can re-run setup later.")

    print()
    print("Setup complete. Start the bot with:   algobot")


def main() -> None:
    if sys.argv and sys.argv[0].endswith("algobot"):
        # Invoked as the `algobot` command
        cmd = sys.argv[1] if len(sys.argv) > 1 else "menu"
        if cmd == "start":
            start(sys.argv[2:])
        elif cmd == "stop":
            stop()
        elif cmd == "status":
            status()
        elif cmd in ("logs", "log"):
            tail = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 40
            show_logs(tail)
        elif cmd == "setup":
            run_setup()
        else:
            menu()
    else:
        # Invoked as the setup wizard (scripts/setup.py)
        run_setup()


if __name__ == "__main__":
    main()
