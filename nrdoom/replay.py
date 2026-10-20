"""Drive recorded footage through the production engine on a clock we control.

Phase 0 of the V2 plan: nothing here changes how a frame is read. It makes the
reading *reproducible and explainable* - events stamped with time, and the
evidence that produced each one - so a claimed fix can be measured against a
baseline instead of argued about.

The source and player stand-ins implement the same interfaces the real ones do,
which is the whole point: `Engine`, `DoomTrigger`, `ReviveWatcher` and
`PartyCounter` under replay are the shipped objects, not a simplified copy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


class ManualClock:
    """A clock the caller advances.

    `time.monotonic()` cannot be stepped, and a replay that runs on wall time
    cannot assert on "the candidate expired after 0.25 s of no good evidence".
    Callable, so it drops straight into `Engine(clock=...)`.
    """

    def __init__(self, t: float = 0.0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += float(dt)
        return self.t

    def set(self, t: float) -> float:
        self.t = float(t)
        return self.t


@dataclass(frozen=True)
class FramePacket:
    """One captured frame plus what the capture thread knows about it.

    `frame_id` is what makes a frozen feed detectable: the same buffer read
    twice is not two observations, and a state machine that counts observations
    will stop the music on a picture that has not changed for ten seconds.
    """

    frame_id: int
    capture_monotonic: float
    frame: np.ndarray
    backend: str = "replay"
    source_ok: bool = True


class ReplaySource:
    """A `capture.FrameSource` stand-in that serves whatever the caller feeds it.

    `feed()` is a new frame from the screen. `hold()` is the same buffer again -
    the frame counter does not move, which is exactly how a stalled WGC callback
    looks to the loop.
    """

    backend = "replay"

    def __init__(self, fps: float = 20.0, clock: ManualClock | None = None):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = float(fps)
        self.target_fps = float(fps)
        self._min_interval = 1.0 / float(fps)
        self.clock = clock or ManualClock()
        self.frames = 0
        self.running = True
        self.error = None
        self._frame: np.ndarray | None = None

    # -- FrameSource interface ------------------------------------------------
    def latest(self) -> np.ndarray | None:
        return self._frame

    def stop(self) -> None:
        self.running = False

    def wait_for_frame(self, timeout: float = 0.0) -> bool:
        return self._frame is not None

    @property
    def frame_size(self):
        return None if self._frame is None else (self._frame.shape[1], self._frame.shape[0])

    # -- driving --------------------------------------------------------------
    def feed(self, frame: np.ndarray) -> FramePacket:
        self.clock.advance(1.0 / self.fps)
        self._frame = frame
        self.frames += 1
        return FramePacket(self.frames, self.clock.t, frame, self.backend)

    def hold(self) -> FramePacket:
        """One more tick on the same buffer: time passes, `frames` does not."""
        self.clock.advance(1.0 / self.fps)
        return FramePacket(self.frames, self.clock.t, self._frame, self.backend)


class EventRecorder:
    """A `MusicPlayer` stand-in that records what the engine asked for, and when.

    Tracks a nominal playhead so a resume can be told apart from a restart -
    "it started over" and "it carried on" are the two things the user can hear
    and the test suite cannot see.
    """

    def __init__(self, clock: ManualClock, track: str = "doom.mp3"):
        self.clock = clock
        self.track = track
        self.events: list[dict] = []
        self.playing = False
        self.position = 0.0
        self._resume_to: float | None = None
        self.volume = None

    def play_next(self, resume: bool = True):
        started_from = self._resume_to if resume else None
        self._resume_to = None
        self.playing = True
        self.position = started_from if started_from is not None else 0.0
        if self.track is None:
            return None
        self.events.append({"t": self.clock.t, "action": "play", "track": self.track,
                            "from_ms": round(1000 * self.position),
                            "resumed": started_from is not None})
        return Path(self.track)

    def stop(self) -> None:
        if self.playing:
            self._resume_to = self.position
            self.events.append({"t": self.clock.t, "action": "stop",
                                "at_ms": round(1000 * self.position)})
        self.playing = False

    def set_volume(self, value: float) -> None:
        self.volume = value

    def advance_playback(self, dt: float) -> None:
        """Let a test say "the song ran for N seconds" without a media backend."""
        if self.playing:
            self.position += dt


# -- video ------------------------------------------------------------------
def video_frames(path: str | Path, fps: float = 20.0, start: float = 0.0,
                 stop: float | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp, frame) resampled to `fps`, using cv2.

    cv2 stays a dev-only import (the shipped bundle has no OpenCV), so it is
    imported here rather than at module scope.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(1, int(round(src_fps / fps)))
    n = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if n % step:
                n += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = n / src_fps
            if t >= start and (stop is None or t <= stop):
                yield t, frame
            n += 1
    finally:
        cap.release()


def run_frames(engine, source: ReplaySource, frames: Iterable[tuple[float, np.ndarray]],
               tick: int = 20) -> list[dict]:
    """Feed frames through `engine._tick()` and return the recorder's events.

    `tick` is the frames-per-second the engine is driven at; the clock keeps
    real time so a hold measured in seconds means what it says.
    """
    for _t, frame in frames:
        source.feed(frame)
        engine._tick()
    return list(getattr(engine.player, "events", []))
