import time

import numpy as np
import pytest

from nrdoom.capture import FrameSource, list_windows


def test_list_windows_returns_titled_windows():
    wins = list_windows()
    assert isinstance(wins, list)
    for w in wins:
        assert w.title.strip()
        assert w.width > 0 and w.height > 0
        assert w.hwnd > 0


def test_labels_are_human_readable():
    wins = list_windows()
    if wins:
        assert "x" in wins[0].label


def test_rejects_a_nonsense_frame_rate():
    with pytest.raises(ValueError):
        FrameSource(target_fps=0)


def test_cv2_stub_is_confined_to_the_frozen_build(monkeypatch):
    """The stub exists only so the bundle can omit cv2.

    In a source checkout it must not shadow the real cv2, or
    tests/test_cv2_equivalence.py would silently compare cv2 against a stub.
    """
    import sys

    from nrdoom import capture

    saved = sys.modules.pop("cv2", None)
    try:
        monkeypatch.delattr(sys, "frozen", raising=False)
        capture._stub_cv2_when_frozen()
        assert "cv2" not in sys.modules, "must not install a stub when running from source"

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        capture._stub_cv2_when_frozen()
        stub = sys.modules.get("cv2")
        assert stub is not None, "frozen build needs the stub for windows_capture"
        assert not hasattr(stub, "__file__"), "must be a synthetic stub, not the real cv2"
        assert stub.imwrite is not None
        with pytest.raises(NotImplementedError):
            stub.imwrite("x.png", None)
    finally:
        sys.modules.pop("cv2", None)
        if saved is not None:
            sys.modules["cv2"] = saved


def test_no_frame_before_start():
    src = FrameSource()
    assert src.latest() is None
    assert src.frame_size is None
    assert src.running is False


def test_stop_is_safe_when_never_started():
    src = FrameSource()
    src.stop()
    src.stop()


def test_throttle_drops_frames_beyond_the_target_rate():
    src = FrameSource(target_fps=20.0)
    src._last_pub = time.monotonic()
    assert src._throttled() is True
    src._last_pub = time.monotonic() - 1.0
    assert src._throttled() is False


def test_publish_replaces_instead_of_queueing():
    src = FrameSource()
    src._publish(np.zeros((4, 4, 3), dtype=np.uint8))
    src._publish(np.ones((8, 8, 3), dtype=np.uint8))
    assert src.frames == 2
    assert src.latest().shape == (8, 8, 3)
    assert src.frame_size == (8, 8)


def test_screen_backend_produces_frames():
    """Real desktop capture. Skips on a headless box rather than lying."""
    src = FrameSource(target_fps=10.0)
    try:
        backend = src.start_screen(monitor_index=1)
        if backend != "mss":
            pytest.skip(f"fell back to {backend}: {src.error}")
        assert src.wait_for_frame(timeout=5.0), f"no frame arrived: {src.error}"
        frame = src.latest()
        # Both backends now hand over BGRA: slicing off the alpha channel during
        # capture turned a flat memcpy into a strided gather and cost 32 ms a frame.
        assert frame.ndim == 3 and frame.shape[2] == 4
        assert frame.dtype == np.uint8
        assert frame.shape[0] >= 480
    finally:
        src.stop()
    assert src.running is False


def test_window_backend_rejects_a_bogus_handle():
    src = FrameSource()
    try:
        backend = src.start_window(0x7FFFFFFF)
        # Either it refuses outright or it falls back to the screen; it must not
        # claim to be streaming a window that does not exist.
        assert backend in ("mss", "windows-capture")
        if backend == "windows-capture":
            assert not src.wait_for_frame(timeout=2.0)
    finally:
        src.stop()


def test_start_window_does_not_leave_the_stop_flag_latched():
    """stop() latches the flag that every frame callback checks.

    start_window used to leave it set, so the session reported "connected" while
    the callback dropped every single frame. Only the mss path cleared it.
    """
    wins = list_windows()
    if not wins:
        pytest.skip("no windows to capture on this desktop")
    src = FrameSource(target_fps=10.0)
    try:
        src.start_window(wins[0].hwnd)
        assert src._stop.is_set() is False, "start_window left the stop flag set"
        assert src.backend in ("windows-capture", "mss")
    finally:
        src.stop()
    assert src._stop.is_set() is True


def test_restart_after_stop_can_receive_frames():
    """A stop/start cycle has to leave the source able to publish again."""
    src = FrameSource(target_fps=10.0)
    src._publish(np.zeros((4, 4, 3), dtype=np.uint8))
    src.stop()
    src._stop.clear()          # what start_window must do
    src._last_pub = 0.0
    assert src._throttled() is False
    src._publish(np.ones((4, 4, 3), dtype=np.uint8))
    assert src.frames == 2
