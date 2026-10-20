"""Where the rendered picture actually is, held across frames instead of re-guessed.

`detector.hud_rect` measures one frame, and one frame can be wrong for reasons that
have nothing to do with the game: a fade-in leaves the top bar measured at 312 px
where it is really 80, an overlay sitting in the bottom bar makes that one measure
0, a dropped frame lands mid-composite. The HUD geometry hangs off this box - and
not just off its top edge: `gauge_spots()` multiplies the panel's x, its row pitch
and the gauge radius out of the *rendered width*, so a width that arrives from a
single occluded frame resizes the whole ally layout while the origin sits still.

So this votes, on the whole box the geometry is derived from. The picture does not
move because one frame says so; it moves when a different box holds a clear
majority of the last second.
"""
from __future__ import annotations

from dataclasses import dataclass

from .detector import hud_rect
from .geometry import boxes_agree as _agrees

ACQUIRING = "ACQUIRING"
LOCKED = "LOCKED"
OCCLUDED = "OCCLUDED"
LOST = "LOST"


def _median(values) -> int:
    ordered = sorted(values)
    n = len(ordered)
    mid = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    return int(round(mid))


@dataclass(frozen=True)
class HudPose:
    """The box to lay the HUD geometry against, and how much to trust it.

    `rect` is (left, top, width, height). The first three are the voted box, so
    the two things the geometry is multiplied out of - where the panel starts and
    how wide the picture is - always describe the same picture. The height is this
    frame's own measurement: nothing here scales off it except a portrait window
    that tolerates a few pixels, and holding it would hide every bottom-bar change.

    `state` is for the human-facing side: "OCCLUDED" means the reads are still
    aimed at a box we believe in, on a frame that currently disagrees, and "LOST"
    means it has been disagreeing long enough that the engine should not infer
    anything about the game from what it finds there.
    """

    rect: tuple[int, int, int, int]
    state: str
    confidence: float
    # The box this frame's own measurement reported, un-voted. Kept for the log:
    # "the tracker moved" and "this frame disagrees with the tracker" look identical
    # in the adopted geometry alone, and telling them apart is what shows whether a
    # bad frame was followed or merely refused.
    measured: tuple[int, int, int, int] | None = None

    @property
    def top(self) -> int:
        return self.rect[1]

    @property
    def box(self) -> tuple[int, int, int]:
        """The voted part of the rect: (left, top, width)."""
        return (self.rect[0], self.rect[1], self.rect[2])


class HudTracker:
    """Temporal majority vote on the letterbox measurement.

    `window` is in frames, and the vote is on the (left, top, width) the geometry
    is derived from. The height comes from the newest measurement, so a frame whose
    bottom bar is hidden by an overlay still gets its correct top and width.
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
        self._seen: list[tuple[int, int, int]] = []
        self._box: tuple[int, int, int] | None = None
        self._confirmed = False
        self._missed = 0
        self._stable_box: tuple[int, int, int] | None = None
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
        self._box = None
        self._confirmed = False
        self._missed = 0
        self._stable_box = None
        self._stable = 0

    @property
    def state(self) -> str:
        if not self._confirmed or not self._seen or self._box is None:
            return ACQUIRING
        if not _agrees(self._seen[-1], self._box):
            return LOST if self._missed >= self.lost_after else OCCLUDED
        return LOCKED

    def _vote(self) -> tuple[tuple[int, int, int], int]:
        """The box the window agrees on most, settled, and how many frames say so.

        Settled means the median of every frame that agrees with the winning
        measurement, which is what turns a handful of one-pixel disagreements into
        one number that does not move. The oldest anchor wins a tie, as the counter
        it replaces did, so a 50/50 window is not a change.
        """
        best_anchor, best_votes = None, 0
        for cand in self._seen:
            members = [b for b in self._seen if _agrees(b, cand)]
            if len(members) > best_votes:
                best_anchor, best_votes = cand, len(members)
        agreed = [b for b in self._seen if _agrees(b, best_anchor)]
        settled = tuple(_median([b[i] for b in agreed]) for i in range(3))
        return settled, best_votes

    def _relocatable(self, winner, confidence: float, measured) -> bool:
        """May the confirmed box be given up for `winner` on this frame?

        Two things have to be true that the raw vote does not check: the box has to
        be the one being measured *now* (a majority left over from history is a tie
        being resolved, not an arrival), and once a box has been confirmed it has to
        have held on its own for `relocate_frames` frames. That second part is what
        stops a cinematic bar, which crosses the screen in half a second, from being
        taken for a new letterbox - and a real resolution change simply waits it out.
        """
        if self._box is not None and _agrees(winner, self._box):
            return False
        recent = self._seen[-3:]
        if not (confidence >= self.share
                or not any(_agrees(b, self._box) for b in recent)):
            return False
        if not _agrees(winner, measured):
            return False
        return self._box is None or not self._confirmed or self._stable >= self.relocate_frames

    def measure(self, frame) -> HudPose:
        rect = hud_rect(frame)
        box = (rect[0], rect[1], rect[2])
        h, w = frame.shape[:2]
        if (w, h) != self._shape:
            # A different capture size is a different window: nothing measured
            # before this frame describes it.
            self._shape = (w, h)
            self.reset()
        self._seen.append(box)
        if self._stable_box is not None and _agrees(box, self._stable_box):
            self._stable += 1
        else:
            self._stable_box, self._stable = box, 1
        if len(self._seen) > self.window:
            self._seen.pop(0)
        winner, votes = self._vote()
        confidence = votes / len(self._seen)
        if self._box is None:
            # Nothing is confirmed yet, so take the best guess going rather than
            # sit out the first second of the run.
            self._box = winner
        elif self._relocatable(winner, confidence, box):
            self._box = winner
        # A box is "ours" once more than one frame has agreed on it, and it stays
        # ours while a frame or two disagrees - that is what OCCLUDED means. Falling
        # back to ACQUIRING on every bad frame would throw away the very agreement
        # that is holding the geometry still.
        if (not self._confirmed and len(self._seen) > 1
                and self._box is not None and _agrees(self._seen[-1], self._box)
                and confidence >= self.share):
            self._confirmed = True
        seen_now = self._box is not None and _agrees(self._seen[-1], self._box)
        self._missed = 0 if seen_now else self._missed + 1
        return HudPose((self._box[0], self._box[1], self._box[2], rect[3]),
                       self.state, round(confidence, 2), rect)
