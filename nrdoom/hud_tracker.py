"""Where the rendered picture actually is, held across frames instead of re-guessed.

`detector.hud_rect` measures one frame, and one frame can be wrong for reasons that
have nothing to do with the game: a fade-in leaves the top bar measured at 312 px
where it is really 80, an overlay sitting in the bottom bar makes that one measure
0, a dropped frame lands mid-composite. The HUD geometry hangs off the top of this
box, so a single bad measurement is a single frame in which every gauge read is
aimed somewhere else - and if it changes the value the engine keys on, it also
throws the accumulated positions away.

So this votes. The picture box does not move because one frame says so; it moves
when a different box holds a clear majority of the last second.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .detector import hud_rect

ACQUIRING = "ACQUIRING"
LOCKED = "LOCKED"
OCCLUDED = "OCCLUDED"
LOST = "LOST"


@dataclass(frozen=True)
class HudPose:
    """The box to lay the HUD geometry against, and how much to trust it.

    `state` is for the human-facing side: "OCCLUDED" means the reads are still
    aimed at a box we believe in, on a frame that currently disagrees, and "LOST"
    means it has been disagreeing long enough that the engine should not infer
    anything about the game from what it finds there.
    """

    rect: tuple[int, int, int, int]
    state: str
    confidence: float

    @property
    def top(self) -> int:
        return self.rect[1]


class HudTracker:
    """Temporal majority vote on the letterbox measurement.

    `window` is in frames, and the vote is on the box's *origin* - the part the
    geometry hangs off. Width and height come from the newest measurement, so a
    frame whose bottom bar is hidden by an overlay still gets its correct top.
    """

    def __init__(self, window: int = 20, share: float = 0.6,
                 lost_after: int = 0, relocate_frames: int = 0):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        self.share = float(share)
        # How long the confirmed anchor can go unmeasured before we stop calling it
        # occlusion. It has to be shorter than the vote's own patience (a box that
        # disagrees for a fifth of the window is about to be adopted), or `LOST`
        # would be a state that exists for a single frame.
        self.lost_after = int(lost_after) or max(3, self.window // 5)
        # §5.3: while the capture size is the same, a confirmed picture box is only
        # given up for one that has held on its own for this long. A cinematic bar
        # crosses the screen in half a second; a real letterbox change does not.
        # Zero keeps the old, immediate behaviour.
        self.relocate_frames = int(relocate_frames)
        self._seen: list[tuple[int, int]] = []
        self._origin: tuple[int, int] | None = None
        self._confirmed = False
        self._missed = 0
        self._stable_origin: tuple[int, int] | None = None
        self._stable = 0
        self._shape: tuple[int, int] | None = None

    @property
    def stable(self) -> int:
        """How many frames running the measurement has agreed on."""
        return self._stable

    def reset(self) -> None:
        """Forget the vote, not the window it was taken on.

        Clearing `_shape` here would make every frame look like a resize and the
        vote would restart on each one, which is the same as having no tracker.
        """
        self._seen.clear()
        self._origin = None
        self._confirmed = False
        self._missed = 0
        self._stable_origin = None
        self._stable = 0

    @property
    def state(self) -> str:
        if not self._confirmed:
            return ACQUIRING
        if not self._seen:
            return ACQUIRING
        if self._seen[-1] != self._origin:
            return LOST if self._missed >= self.lost_after else OCCLUDED
        return LOCKED

    def _relocatable(self, winner, confidence: float, measured) -> bool:
        """May the confirmed box be given up for `winner` on this frame?

        Two things have to be true that the raw vote does not check: the box has to
        be the one being measured *now* (a majority left over from history is a tie
        being resolved, not an arrival), and once a box has been confirmed it has to
        have held on its own for `relocate_frames` frames. That second part is what
        stops a cinematic bar, which crosses the screen in half a second, from being
        taken for a new letterbox - and a real resolution change simply waits it out.
        """
        if winner == self._origin:
            return False
        if not (confidence >= self.share or self._origin not in self._seen[-3:]):
            return False
        if winner != measured:
            return False
        return not self._confirmed or self._stable >= self.relocate_frames

    def measure(self, frame) -> HudPose:
        rect = hud_rect(frame)
        origin = (rect[0], rect[1])
        h, w = frame.shape[:2]
        if (w, h) != self._shape:
            # A different capture size is a different window: nothing measured
            # before this frame describes it.
            self._shape = (w, h)
            self.reset()
        self._seen.append(origin)
        if origin == self._stable_origin:
            self._stable += 1
        else:
            self._stable_origin, self._stable = origin, 1
        if len(self._seen) > self.window:
            self._seen.pop(0)
        winner, votes = Counter(self._seen).most_common(1)[0]
        confidence = votes / len(self._seen)
        if self._origin is None:
            # Nothing is confirmed yet, so take the best guess going rather than
            # sit out the first second of the run.
            self._origin = winner
        elif self._relocatable(winner, confidence, origin):
            self._origin = winner
        # A box is "ours" once more than one frame has agreed on it, and it stays
        # ours while a frame or two disagrees - that is what OCCLUDED means. Falling
        # back to ACQUIRING on every bad frame would throw away the very agreement
        # that is holding the geometry still.
        if not self._confirmed and (len(self._seen) > 1
                                    and self._seen[-1] == self._origin
                                    and confidence >= self.share):
            self._confirmed = True
        self._missed = 0 if self._seen[-1] == self._origin else self._missed + 1
        return HudPose((self._origin[0], self._origin[1], rect[2], rect[3]),
                       self.state, round(confidence, 2))
