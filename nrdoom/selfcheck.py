"""Startup self-check, so a packaged exe can be diagnosed without a console.

Build the app with `宁姆韦德的雨.exe --selftest` and it writes 自检.txt next to the
exe instead of opening a window. Useful because a onefile bundle can silently
lose an optional native extension, and the app then quietly falls back to the
screen-capture path.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from . import __version__
from .assets import app_dir, asset_root, bundled_music


def _line(label: str, ok: bool, detail: str = "") -> str:
    return "%-5s %-22s %s" % ("OK" if ok else "FAIL", label, detail)


def _panel():
    import numpy as np

    path = asset_root() / "selftest_panel.npz"
    if not path.is_file():
        return None
    return np.load(path)["frame"]


def _check_detection() -> list[str]:
    """Actually run the detector on a bundled picture of two three-bar gauges.

    Importing numpy proves very little; a onefile bundle can lose a dependency
    and still import fine. This exercises the whole colour-space, blob-labelling
    and ring-sampling path end to end, so trimming the package cannot silently
    break detection. The fixture is an .npz rather than a PNG so that nothing
    here needs an image decoder.
    """
    frame = _panel()
    if frame is None:
        return [_line("detection", False, "missing assets/selftest_panel.npz")]
    try:
        from .detector import count_rows, gauge_spots, hud_rect, propose_rings, read_ring

        lines = []
        rect = hud_rect(frame)
        bars = [r.bars for r in propose_rings(frame)]
        ok = bars == [3, 3]
        lines.append(_line("detection", ok,
                           "两张计量表，各 3 格" if ok else f"读到 {bars or '空'}，应为 [3, 3]"))
        spots = gauge_spots(frame.shape, rect=rect)
        fixed = [read_ring(frame, *s) for s in spots]
        fbars = [None if r is None else r.bars for r in fixed]
        lines.append(_line("fixed spots", fbars == [3, 3],
                           "固定位置直读 %s，应为 [3, 3]" % fbars if fbars != [3, 3]
                           else "固定位置直读两张表，各 3 格"))
        rows = count_rows(frame, rect)
        lines.append(_line("row count", rows == 2,
                           f"数到 {rows} 行队友，应为 2" if rows != 2 else "2 行队友"))
        return lines
    except Exception as exc:                            # noqa: BLE001 - reporting
        return [_line("detection", False, f"{type(exc).__name__}: {exc}")]


def _revive_frame(frame):
    """The same panel with both allies back on their feet.

    Painted rather than bundled: the gauge is erased *and* a health bar is drawn,
    because a missing gauge on its own is also what a failed read looks like, and
    the stop machine has to want both.
    """
    from .detector import gauge_spots, hud_rect

    out = frame.copy()
    w = out.shape[1]
    for cx, cy, r in gauge_spots(out.shape, rect=hud_rect(out)):
        top, rad = int(cy), int(r)
        out[top - rad:top + rad + 1, int(cx) - rad:int(cx) + rad + 1] = (38, 37, 65)
        out[int(cy - 0.0080 * w):int(cy - 0.0030 * w),
            int(w * 0.082):int(w * 0.140)] = (30, 45, 185)
    return out


def _check_temporal() -> list[str]:
    """Run a short scene through the shipped engine: all downed, then revived.

    The static check above proves one frame reads. Everything that decides *when*
    to play and stop is written in seconds and lives in the state machine, and a
    onefile bundle can drop one of those modules and still import cleanly - so
    drive the sequence and look for the two events.

    The same run writes a real diagnostics folder, because the one thing that can
    only be found out on the user's machine is worth being able to reconstruct
    afterwards: if a session cannot be explained from its own log, a live report
    cannot be diagnosed at all.
    """
    frame = _panel()
    if frame is None:
        return [_line("temporal", False, "missing assets/selftest_panel.npz")]
    try:
        import shutil
        import tempfile
        from .diag import Diagnostics
        from .engine import Engine
        from .replay import EventRecorder, ReplaySource
        from .settings import Settings

        down, up = frame.copy(), _revive_frame(frame)
        scratch = Path(tempfile.mkdtemp(prefix="nrdoom-selftest-"))
        st = Settings(slots=[], hold_frames=4)
        src = ReplaySource(fps=20.0)
        rec = EventRecorder(src.clock)
        diag = Diagnostics(scratch, clock=src.clock, before_s=3.0, after_s=1.0,
                           stride=4)
        eng = Engine(st, source=src, player=rec, clock=src.clock, diag=diag)
        if not eng.set_listening(True):
            return [_line("temporal", False, "回放引擎起不来")]
        for _ in range(20):
            src.feed(down)
            eng._tick()
        for _ in range(80):
            src.feed(up)
            eng._tick()
        acts = [e["action"] for e in rec.events]
        ok = acts[:1] == ["play"] and "stop" in acts
        lines = [_line("temporal", ok,
                       "倒地→播放，救起→停歌" if ok
                       else f"事件序列是 {acts or '空'}，应为 play 后接 stop")]
        lines.append(_check_session_log(diag, scratch))
        shutil.rmtree(scratch, ignore_errors=True)
        return lines
    except Exception as exc:                            # noqa: BLE001 - reporting
        return [_line("temporal", False, f"{type(exc).__name__}: {exc}")]


def _check_session_log(diag: "Diagnostics", folder: Path) -> str:
    """The log has to be enough on its own: events, and evidence in the look-back."""
    import json

    diag.close()
    path = folder / "events.jsonl"
    if not path.is_file():
        return _line("session log", False, f"{path} was not written")
    records = [json.loads(raw) for raw in
               path.read_text(encoding="utf-8").splitlines() if raw.strip()]
    kinds = [r.get("kind") for r in records]
    stop = next((r for r in records if r.get("kind") == "stop"), None)
    near = [r for r in records if r.get("kind") == "frame"
            and stop is not None and stop["t"] - 0.6 <= r["t"] <= stop["t"]]
    ok = "fire" in kinds and stop is not None and bool(near)
    return _line("session log", ok,
                 "停歌前 0.6 秒有 %d 条证据可离线复现" % len(near) if ok
                 else "fire=%s stop=%s 回看窗口里的记录=%d 条"
                 % ("fire" in kinds, stop is not None, len(near)))


def run() -> int:
    lines = ["宁姆韦德的雨 自检", "=" * 46]

    for mod, why in (("windows_capture", "按窗口抓屏（首选）"),
                     ("mss", "整屏抓屏（兜底）"),
                     ("numpy", "数值计算")):
        try:
            if mod == "windows_capture":
                # Same shim the capture path installs, so the check exercises the
                # real import rather than failing on a deliberately absent cv2.
                from .capture import _stub_cv2_when_frozen
                _stub_cv2_when_frozen()
            importlib.import_module(mod)
            lines.append(_line(mod, True, why))
        except Exception as exc:                        # noqa: BLE001 - reporting
            lines.append(_line(mod, False, f"{type(exc).__name__}: {exc}"))

    try:
        from PySide6.QtMultimedia import QMediaPlayer   # noqa: F401
        lines.append(_line("QtMultimedia", True, "播放"))
    except Exception as exc:                            # noqa: BLE001 - reporting
        lines.append(_line("QtMultimedia", False, f"{type(exc).__name__}: {exc}"))

    root = asset_root()
    for name in ("background.jpg", "icon.png", "icon.ico", "cover.jpg",
                 "selftest_panel.npz"):
        path = root / name
        lines.append(_line(name, path.is_file(),
                           f"{path.stat().st_size} bytes" if path.is_file() else str(path)))

    lines += _check_detection()
    lines += _check_temporal()

    target = app_dir() / "music"
    bundled = bundled_music()
    # A missing user folder is not a fault: the first run creates it from the
    # bundled music. Only the bundled copy is a hard requirement.
    lines.append("%-5s %-22s %s" % ("info", "music dir",
                                    "present" if target.is_dir() else "will be created"))
    lines.append(_line("bundled music", bundled.is_dir(), str(bundled)))

    lines += ["", f"version     = {__version__}",
              f"frozen      = {getattr(sys, 'frozen', False)}",
              f"executable  = {sys.executable}",
              f"config.json = {app_dir() / 'config.json'}"]

    text = "\n".join(lines)
    out = app_dir() / "自检.txt"
    try:
        out.write_text(text, encoding="utf-8")
    except OSError:
        pass
    try:
        print(text)
    except UnicodeEncodeError:
        pass
    return 0
