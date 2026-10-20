"""Is this frame news?

A capture backend can keep handing over the same picture - a stalled WGC callback,
a window that stopped compositing, a game that is minimised or paused. Every other
part of the pipeline counts observations, so a stalled feed would look like a long
run of evidence.

**This is diagnostic only. It deliberately does not gate the trigger or the stop.**
Measured on the user's screen recording, the whole picture never repeats (one
identical sample at most in 7291), which makes a freeze detector look free. But a
recording carries codec noise, while a live Windows.Graphics.Capture callback of a
genuinely static scene is byte-identical - so gating on this would suppress real
evidence on the machine that cannot be tested here, and it broke two of the suite's
own "a real revive must stop the song" tests when tried.

What it is good for is answering "why did nothing happen": a long `stale` count in
the diagnostics or the status bar says the frames stopped arriving, which is a
capture problem and not a recognition problem.

`frame_id` remains the signal the decisions are built on: it is what the source
claims, not a guess about pixels.

    the ally panel sits byte-identical for up to 46 sampled frames (2.3 s) during
    ordinary play - a quiet corner with a standing ally - so a freeze judged on the
    HUD region would call normal gameplay stalled. Quality questions belong to the
    HUD; this question belongs to the frame as a whole.
"""
from __future__ import annotations

import numpy as np

# Stride-sampled, not resized: the shipped bundle has no OpenCV, and every sixteenth
# pixel is enough to tell "the picture moved" from "it did not".
SAMPLE_STEP = 16


class FrameValidator:
    """Counts how many frames in a row have looked identical."""

    def __init__(self, hold: int = 5):
        if hold < 1:
            raise ValueError("hold must be >= 1")
        self.hold = int(hold)
        self.stale = 0
        self._last: np.ndarray | None = None

    def reset(self) -> None:
        self._last = None
        self.stale = 0

    @property
    def frozen(self) -> bool:
        return self.stale >= self.hold

    def check(self, frame: np.ndarray) -> bool:
        """True when this frame differs from the last one; counts the repeats."""
        sample = np.ascontiguousarray(frame[::SAMPLE_STEP, ::SAMPLE_STEP, :3])
        same = (self._last is not None and sample.shape == self._last.shape
                and np.array_equal(sample, self._last))
        self._last = sample
        self.stale = self.stale + 1 if same else 0
        return not same
