"""Locate bundled assets, whether running from source or from a PyInstaller bundle.

`selftest_panel.npz` has to stay a **whole frame** shrunk down, not a crop: the
detector's geometry is a fraction of screen width, so a crop's own width is the
wrong ruler and the self-check would be testing a panel that cannot exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_NAME = "宁姆韦德的雨"
APP_NAME_ASCII = "NimvedNoAme"      # for the exe and the window class


def build_id() -> str:
    """Which build this is, without anyone having to remember to bump a number.

    Recorded as the first line of a diagnostics session: a folder sent in from a
    machine is only interpretable if you know which executable wrote it, and the
    last-modified time of the running binary is the one stamp that cannot go stale.
    """
    from datetime import datetime

    if getattr(sys, "frozen", False):
        target, kind = Path(sys.executable), "exe"
    else:
        target, kind = Path(__file__), "source"
    try:
        when = datetime.fromtimestamp(target.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        when = "unknown"
    return "%s %s" % (kind, when)


DEFAULT_TRACK = "雨爱.mp3"


def asset_root() -> Path:
    """The assets/ dir next to the script, or the one unpacked into _MEIPASS."""
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return Path(bundle) / "assets"
    return Path(__file__).resolve().parents[1] / "assets"


def app_dir() -> Path:
    """Where the user's music/ and config.json live: next to the exe, not inside it."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def bundled_music() -> Path:
    bundle = getattr(sys, "_MEIPASS", None)
    return (Path(bundle) / "music") if bundle else app_dir() / "music"


def ensure_default_music() -> Path | None:
    """On the first run, drop the bundled track next to the exe so it is editable.

    Only when the folder does not exist yet: if the user clears it out later that
    is a decision, not something to undo on the next launch.
    """
    target = app_dir() / "music"
    if target.exists():
        return None
    source = bundled_music()
    if not source.is_dir():
        return None
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file():
            (target / item.name).write_bytes(item.read_bytes())
    return target
