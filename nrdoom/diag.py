"""Event-level diagnostics, so a stop can be explained after the fact.

Two products, both off unless `Settings.diagnostics` is on:

* `events.jsonl` - one line per state change and per low-confidence frame, with
  the evidence the decision was made from rather than just the decision.
* `roi/` - the last ~10 s of HUD crops around each event, so "why did it stop"
  can be answered by looking rather than by guessing.

Deliberately Qt-free and cv2-free: this runs inside the packaged exe, which has
neither. Crops are `.npz` for that reason; `tools/` converts them to PNG when a
human wants to look at one.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .detector import detect_spots, gauge_spots

ROI_X_FRAC = 0.20            # the ally panel is inside the left fifth
ROI_Y_FRAC = 0.20            # ... and the top fifth of the picture
ROI_STRIDE_TICKS = 10        # ~2 crops a second at 20 fps


def hud_roi(frame: np.ndarray, rect=None) -> np.ndarray:
    """The one corner of the screen any of this ever looks at, as a copy."""
    w = frame.shape[1]
    side = int(w * ROI_X_FRAC)
    top = 0 if rect is None else int(rect[1])
    return np.ascontiguousarray(frame[top:top + side, 0:side, :3])


@dataclass
class Diagnostics:
    """Records what the engine saw and decided. `None` is the off switch.

    The engine holds either None or this, and never builds a record when it is
    None, so a disabled instance costs nothing on the frame path.
    """

    folder: Path | str
    clock: callable = time.monotonic
    before_s: float = 10.0
    after_s: float = 5.0
    low_conf: float = 0.6
    stride: int = ROI_STRIDE_TICKS
    heartbeat_s: float = 0.5

    _t0: float = field(default=0.0, init=False)
    _ring: deque = field(default=None, init=False)
    _last_key: tuple = field(default=None, init=False)
    _last_write_t: float = field(default=-1e9, init=False)
    _open_dir: Path | None = field(default=None, init=False)
    _open_at: float = field(default=0.0, init=False)
    _tick: int = field(default=0, init=False)
    _written: int = field(default=0, init=False)
    _last_frame_t: float | None = field(default=None, init=False)
    _handle: object = None

    def __post_init__(self):
        self.folder = Path(self.folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / "roi").mkdir(exist_ok=True)
        self._ring = deque()
        self._t0 = self.clock()
        self._handle = (self.folder / "events.jsonl").open("a", encoding="utf-8")

    # -- writing --------------------------------------------------------------
    def line(self, **fields) -> None:
        fields.setdefault("t", round(self.clock() - self._t0, 3))
        self._handle.write(json.dumps(fields, ensure_ascii=False, default=float) + "\n")
        self._written += 1
        if not self._written % 20:
            self._handle.flush()

    def flush(self) -> None:
        try:
            self._handle.flush()
        except (AttributeError, OSError):
            pass

    def frame(self, frame: np.ndarray, rect, spots, readings, party: int,
              frame_id: int, **extra) -> None:
        """One tick. Writes only when something changed or the frame is dubious."""
        self._tick += 1
        now = self.clock() - self._t0
        # The gap between ticks is the one thing no recording but a live run can
        # answer: a replay feeds a steady 20 fps and a real desktop does not, and
        # every stop rule here is written in seconds.
        dt = 0.0 if self._last_frame_t is None else round(now - self._last_frame_t, 4)
        self._last_frame_t = now
        bars = tuple(None if r is None else r.bars for r in readings)
        conf = tuple(0.0 if r is None else round(float(r.confidence), 3)
                     for r in readings)
        key = (bars, conf, party, frame.shape[1], None if rect is None else rect[1])
        dubious = any(c and c < self.low_conf for c in conf)
        # A heartbeat as well as a change: the frames before a stop are exactly the
        # ones where nothing changes (the panel reads blank while the ally stands up
        # with no visible fill), and a log that only records change cannot answer
        # "what was on screen just before it stopped" - which is the whole point.
        steady = (self._last_frame_t is not None
                  and now - self._last_write_t >= self.heartbeat_s)
        if key != self._last_key or dubious or steady:
            self._last_key = key
            self._last_write_t = now
            self.line(kind="frame", t=round(now, 3), frame_id=frame_id, dt=dt,
                      party=party, bars=bars, conf=conf,
                      rect=None if rect is None else list(rect),
                      grid_bars=[None if r is None else r.bars for r in detect_spots(
                          frame, gauge_spots(frame.shape, rect=rect))],
                      spots=[[round(s[0], 1), round(s[1], 1), round(s[2], 1)]
                             for s in spots], **extra)
        if self._tick % self.stride:
            return
        # The ring has to fill *before* anything happens, or there is no "ten
        # seconds before the stop" to write out when a stop finally comes.
        self._ring.append((now, hud_roi(frame, rect)))
        horizon = now - self.before_s
        while self._ring and self._ring[0][0] < horizon:
            self._ring.popleft()
        if self._open_dir is not None:
            # The seconds *after* a stop are the other half of the answer: did he
            # really get up, or did we stop on a flicker.
            if now - self._open_at > self.after_s:
                self._open_dir = None
            else:
                np.savez_compressed(
                    self._open_dir / ("after_%+06.2fs.npz" % (now - self._open_at)),
                    frame=hud_roi(frame, rect))

    def event(self, kind: str, frame: np.ndarray | None = None, rect=None,
              **evidence) -> Path | None:
        """Note a fire/stop/error and start saving the frames around it."""
        now = self.clock() - self._t0
        self.line(kind=kind, **evidence)
        # Frame lines can afford to be batched; an event cannot. Losing the record
        # of a stop because the process was killed on the way out is exactly the
        # case the log exists for.
        self.flush()
        tag = "%s_%07.2f" % (kind, now)
        out = self.folder / "roi" / tag
        if kind not in ("fire", "stop"):
            # A geometry change gets its own line and its own crop, and nothing else:
            # touching the window below would close the "five seconds after the stop"
            # collection that is the evidence for a different decision entirely.
            out.mkdir(exist_ok=True)
            if frame is not None:
                np.savez_compressed(out / "at_+00.00s.npz",
                                    frame=hud_roi(frame, rect))
            return out
        out.mkdir(exist_ok=True)
        if kind == "stop":
            # Only a stop needs the seconds around it: a fire is explained by the
            # frame that caused it, and keeping ten seconds of crops for every one
            # of them doubled what a session costs on disk.
            for t, crop in self._ring:
                np.savez_compressed(out / ("before_%+06.2fs.npz" % (t - now)),
                                    frame=crop)
            self._open_dir, self._open_at = out, now
        else:
            self._open_dir = None
        if frame is not None:
            np.savez_compressed(out / "at_+00.00s.npz",
                                frame=hud_roi(frame, rect))
        return out

    def close(self) -> None:
        try:
            self._handle.flush()
            self._handle.close()
        except (AttributeError, OSError):
            pass
        self._handle = None
