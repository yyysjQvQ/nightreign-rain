"""Get game frames off the screen.

Two backends, tried in order:

* ``windows-capture`` (Windows.Graphics.Capture) captures a specific window, so
  it keeps working when the game is not on the primary monitor and never reads
  the overlay this app draws itself.
* ``mss`` grabs a whole monitor. It is the fallback for the case WGC handles
  badly, and the one it handles badly in return: a game in *exclusive* fullscreen
  bypasses the desktop compositor, so both paths can yield black frames. Elden
  Ring defaults to borderless windowed, where everything works.

Frames are only ever read, never written, and nothing touches the game process.
"""
from __future__ import annotations

import ctypes
import sys
import threading
import time
import types
from ctypes import wintypes
from dataclasses import dataclass

import numpy as np

user32 = ctypes.windll.user32
_dwm = ctypes.windll.dwmapi

GW_OWNER = 4
DWMWA_CLOAKED = 14

# Windows that carry a title but are never what the user means.
_CLASS_DENY = {
    "Shell_TrayWnd", "Shell_SecondaryTrayWnd", "WorkerW", "Progman",
    "Windows.UI.Core.CoreWindow", "XamlExplorerHostIslandWindow",
    "NotifyIconOverflowWindow", "TopLevelWindowForOverflowXamlIsland",
    "TaskListThumbnailWnd", "MSCTFIME UI", "Default IME",
}


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    width: int
    height: int

    @property
    def label(self) -> str:
        return f"{self.title}  ({self.width}x{self.height})"


def _class_name(hwnd) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value or ""


def _is_cloaked(hwnd) -> bool:
    flag = ctypes.c_int(0)
    _dwm.DwmGetWindowAttribute(ctypes.c_void_p(hwnd), DWMWA_CLOAKED,
                               ctypes.byref(flag), ctypes.sizeof(flag))
    return bool(flag.value)


def list_windows() -> list[WindowInfo]:
    """Visible, owned, uncloaked top-level windows. Roughly what the taskbar shows."""
    out: list[WindowInfo] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        if _class_name(hwnd) in _CLASS_DENY:
            return True
        if user32.GetWindow(hwnd, GW_OWNER):
            return True
        if _is_cloaked(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n == 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        w, h = rect.right - rect.left, rect.bottom - rect.top
        if w < 160 or h < 120:
            return True
        out.append(WindowInfo(int(hwnd), buf.value, w, h))
        return True

    user32.EnumWindows(cb, 0)
    return out


class CaptureError(RuntimeError):
    pass


def _stub_cv2_when_frozen() -> None:
    """Let windows_capture import in a bundle that deliberately has no cv2.

    windows_capture does `import cv2` at module scope but only calls
    cv2.imwrite from Frame.save_as_image, which this app never uses. Shipping
    cv2 costs 26 MB of compressed exe for that one unused convenience, so the
    packaged build excludes it and this satisfies the import instead.

    Guarded on sys.frozen so a source checkout always gets the real cv2 if it
    has one; otherwise this would poison tests/test_cv2_equivalence.py.
    """
    if not getattr(sys, "frozen", False) or "cv2" in sys.modules:
        return
    stub = types.ModuleType("cv2")

    def _unavailable(*_args, **_kwargs):
        raise NotImplementedError(
            "cv2 is not bundled; Frame.save_as_image is unavailable. "
            "The rest of windows_capture does not need cv2.")

    stub.imwrite = _unavailable                     # type: ignore[attr-defined]
    sys.modules["cv2"] = stub


class FrameSource:
    """Hands the caller the newest captured frame.

    The capture thread overwrites a single slot rather than queueing: a stale
    frame is worse than a dropped one when the point is "is he down *now*".
    """

    def __init__(self, target_fps: float = 20.0):
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        self.target_fps = float(target_fps)
        self._min_interval = 1.0 / self.target_fps
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._control = None
        self.backend = "none"
        self.error: str | None = None
        self.frames = 0
        self._last_pub = 0.0

    # -- publishing ---------------------------------------------------------
    def _publish(self, frame: np.ndarray) -> None:
        with self._lock:
            self._latest = frame
            self.frames += 1

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._latest

    def _throttled(self) -> bool:
        now = time.monotonic()
        if now - self._last_pub < self._min_interval:
            return True
        self._last_pub = now
        return False

    # -- lifecycle ----------------------------------------------------------
    def start_window(self, hwnd: int) -> str:
        """Capture one window through Windows.Graphics.Capture."""
        self.stop()
        try:
            _stub_cv2_when_frozen()
            from windows_capture import WindowsCapture
        except Exception as exc:                      # pragma: no cover - env dependent
            self.error = f"windows-capture unavailable: {exc}"
            return self.start_screen()

        def on_frame_arrived(frame, capture_control):
            if self._stop.is_set() or self._throttled():
                return
            # The buffer is a view onto native memory that is recycled as soon as
            # this callback returns, so anything kept past it has to be copied.
            # Copy the whole BGRA block: slicing off the alpha channel turns a
            # flat memcpy into a strided gather and costs 32 ms a frame instead
            # of 6 ms at 2560x1600. Downstream handles four channels directly.
            self._publish(np.ascontiguousarray(frame.frame_buffer))

        def on_closed():
            self.error = "capture closed by the system"
            self.backend = "closed"

        try:
            cap = WindowsCapture(window_hwnd=int(hwnd), cursor_capture=False,
                                 draw_border=False,
                                 minimum_update_interval=max(1, int(1000 / self.target_fps)))
            cap.event(on_frame_arrived)
            cap.event(on_closed)
            # stop() above latched the flag; leaving it set makes every callback
            # return early and the session looks connected but never delivers a frame.
            self._stop.clear()
            self._last_pub = 0.0
            self._control = cap.start_free_threaded()
        except Exception as exc:
            self.error = f"windows-capture failed: {exc}"
            return self.start_screen()

        self.backend = "windows-capture"
        return self.backend

    def start_screen(self, monitor_index: int = 1) -> str:
        """Capture a whole monitor with mss. Works when WGC is unavailable."""
        self.stop()

        def loop():
            try:
                import mss
            except Exception as exc:                  # pragma: no cover - env dependent
                self.error = f"mss unavailable: {exc}"
                self.backend = "failed"
                return
            try:
                with mss.MSS() if hasattr(mss, "MSS") else mss.mss() as sct:
                    monitors = sct.monitors
                    if monitor_index >= len(monitors):
                        self.error = f"no monitor {monitor_index}"
                        self.backend = "failed"
                        return
                    mon = monitors[monitor_index]
                    while not self._stop.is_set():
                        shot = sct.grab(mon)
                        # .copy() matters: np.asarray(shot) is a view onto the
                        # screenshot's own buffer, which dies with `shot`.
                        self._publish(np.asarray(shot).copy())
                        self._stop.wait(self._min_interval)
            except Exception as exc:
                self.error = f"mss failed: {exc}"
                self.backend = "failed"

        self._stop.clear()
        self.backend = "mss"
        self._thread = threading.Thread(target=loop, name="mss-capture", daemon=True)
        self._thread.start()
        return self.backend

    def stop(self) -> None:
        self._stop.set()
        if self._control is not None:
            try:
                self._control.stop()
            except Exception:
                pass
            self._control = None
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self.backend = "none"

    def wait_for_frame(self, timeout: float = 3.0) -> bool:
        """Block until a first frame lands. False means the backend is silent."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.latest() is not None:
                return True
            if self.backend in ("failed", "closed"):
                return False
            time.sleep(0.05)
        return self.latest() is not None

    @property
    def frame_size(self):
        frame = self.latest()
        return None if frame is None else (frame.shape[1], frame.shape[0])

    @property
    def running(self) -> bool:
        return self.backend not in ("none", "failed")
