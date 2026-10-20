"""Where to look, and how much proof a move has to have before it is kept.

The detector has three ways to find the ally panel: lay it out from the picture box
(`seed`), slide it onto the portraits (`realign`), and jitter a slot until a gauge
appears in it (`recovery`). The last two are *discoveries*, and until now a single
frame's discovery was written straight into the calibration - which is how one
grazing read of a half-visible ring could outlive the frame it came from, and be
loaded again on the next launch.

But the same single frame is also the only reason a panel that really did move gets
recovered in about a fifth of a second, and that speed is measured, not assumed:
the dead-zone work in Phase 5 found positions that stayed wrong for thousands of
frames because the tool was too cautious to follow one. So the two are separated
here rather than traded off:

    candidate   a position some frame proposed; used at once, remembered by nobody
    verified    the same position held across separate captures, with better reads
                than the layout it replaced
    persisted   verified, and therefore written to the config for the next launch

One rule from the report this implements, kept deliberately literal: losing the
picture box is never evidence about the game, so a frame whose geometry is not
settled casts no vote, and a freeze or a re-served frame does not either.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

# How far two (left, top, width) measurements can be and still be called the same
# picture box. Every edge is measured by sampling every 8th line
# (`detector.LETTER_SAMPLE_STEP`), so a picture that has not moved at all does come
# back a pixel or two apart; comparing integers would call that a new layout. This
# is the one tolerance for that question, shared by the vote that decides it and the
# store that keys by it.
GEOM_TOL_PX = 8


def boxes_agree(a, b, tol: int = GEOM_TOL_PX) -> bool:
    """Do two (left, top, width) triples describe the same picture box?"""
    if a is None or b is None or len(a) != len(b):
        return False
    return all(abs(x - y) <= tol for x, y in zip(a, b))


@dataclass
class Candidate:
    """A layout waiting to be believed."""

    spots: tuple
    source: str
    frame_ids: set = field(default_factory=set)
    better_ids: set = field(default_factory=set)   # captures that read better here
    first_s: float = 0.0
    last_s: float = 0.0
    pitch_off: float = 0.0  # how far it moved the row pitch away, in pixels

    @property
    def votes(self) -> int:
        return len(self.frame_ids)


class GeometryManager:
    """Active positions, plus the evidence a change needs to be kept.

    `min_frames` votes must come from that many *different capture frames*, and the
    first and last of them must be at least `min_age_s` apart - a burst of ticks
    over one frozen picture is one observation, not several, and the capture clock
    is what tells them apart.
    """

    def __init__(self, min_frames: int = 3, min_age_s: float = 0.2,
                 spread_px: float = 4.0, pitch_px: float = 6.0,
                 rollback_frames: int = 4):
        if min_frames < 1:
            raise ValueError("min_frames must be >= 1")
        self.min_frames = int(min_frames)
        self.min_age_s = float(min_age_s)
        self.spread_px = float(spread_px)
        self.pitch_px = float(pitch_px)
        # How many fresh frames of reading nothing at the kept layout before the
        # tool stops defending it and goes back to the one before.
        self.rollback_frames = int(rollback_frames)
        self.active: list[tuple] = []
        self.verified: list[tuple] = []
        self.fallback: list[tuple] | None = None
        self.source = "seed"
        self.candidate: Candidate | None = None
        self.blind = 0
        # Bounded because the log is not the only reader: with diagnostics off nobody
        # drains this, and a five-hour session would otherwise grow without end.
        self.events: deque = deque(maxlen=16)

    # -- the two entry points the engine has ---------------------------------
    def settle(self, spots, source: str, t: float) -> None:
        """Adopt a layout outright: a new picture box, or one just loaded.

        There is no candidate to accumulate for the grid seed - it is a derivation,
        not a discovery - and a fresh box must be usable immediately or the first
        seconds of a session are blind.
        """
        self.active = [tuple(s) for s in spots]
        self.verified = [tuple(s) for s in spots]
        self.fallback = None
        self.source = source
        self.candidate = None
        self.blind = 0
        self.events.append(("settled", t, tuple(self.active)))

    def propose(self, spots, source: str, frame_id: int, t: float,
                score: float, base_score: float) -> list[tuple]:
        """Offer a discovered position, and say what to read this frame.

        The candidate is used at once: waiting for proof before *looking* there is
        how a moved panel stays unread for minutes. What waits for proof is the
        decision to keep it.
        """
        spots = tuple(tuple(s) for s in spots)
        self.active = list(spots)
        if self.candidate is None or self._replaced(self.candidate, spots):
            self.candidate = Candidate(spots=spots, source=source,
                                       frame_ids={frame_id}, first_s=t, last_s=t,
                                       pitch_off=self._pitch_change(spots))
            self.events.append(("proposed", t, spots))
        self._vote(self.candidate, frame_id, t, score, base_score)
        return self.active

    def confirm(self, frame_id: int, t: float, score: float, base_score: float) -> None:
        """Cast this frame's vote for the candidate already being collected.

        A position the tool has moved to stops being *proposed* - there is nothing
        left to move - which is exactly when the frames that agree with it start
        arriving. Without counting those, a move is never believed and never kept.
        """
        if self.candidate is not None:
            self._vote(self.candidate, frame_id, t, score, base_score)

    @staticmethod
    def _vote(cand: Candidate, frame_id: int, t: float, score: float,
              base_score: float) -> None:
        cand.frame_ids.add(frame_id)
        cand.last_s = max(cand.last_s, t)
        if score > base_score:
            cand.better_ids.add(frame_id)

    def earned(self, t: float) -> bool:
        """Has the candidate enough agreement to become the kept layout?"""
        cand = self.candidate
        if cand is None:
            return False
        if cand.votes < self.min_frames:
            return False
        if t - cand.first_s < self.min_age_s:
            return False
        if len(cand.better_ids) < self.min_frames:
            # Each agreeing capture has to read *better* than the layout it would
            # replace; a frame that merely tolerates the move is not a vote for it.
            return False
        return cand.pitch_off <= self.pitch_px

    def accept(self, t: float) -> list[tuple]:
        """Promote the candidate: it is now what the tool defends and persists."""
        cand = self.candidate
        if cand is None:
            return self.active
        self.fallback = list(self.verified)
        self.verified = list(cand.spots)
        self.active = list(cand.spots)
        self.source = cand.source
        self.candidate = None
        self.blind = 0
        self.events.append(("verified", t, tuple(self.verified)))
        return self.active

    def refuse(self, t: float) -> None:
        """Drop the candidate and keep reading where the last one said to."""
        if self.candidate is not None:
            self.events.append(("refused", t, tuple(self.candidate.spots)))
        self.candidate = None
        self.active = list(self.verified)

    # -- losing faith in what was kept ---------------------------------------
    def report_read(self, t: float, score: float) -> list[tuple] | None:
        """Tally one frame's read against the kept layout; roll back if it died.

        Returns the restored positions when a rollback happened, else None. Only
        frames that actually brought a new picture count: a stalled capture repeats
        the same blindness and would reach the threshold on its own.
        """
        if score > 0:
            self.blind = 0
            return None
        self.blind += 1
        if self.blind < self.rollback_frames or not self.fallback:
            return None
        restored = list(self.fallback)
        self.verified = restored
        self.active = restored
        self.fallback = None
        self.candidate = None
        self.blind = 0
        self.source = "rollback"
        self.events.append(("rolled_back", t, tuple(restored)))
        return restored

    # -- what the interface and the log are told ------------------------------
    def state(self) -> str:
        """One word for how much to trust the positions being drawn."""
        if self.candidate is not None:
            return "pending"
        return "fallback" if self.fallback else "verified"

    def drain(self) -> list:
        """The state changes since last time, for whoever is logging them."""
        out, self.events = self.events, []
        return out

    def restore(self, t: float) -> list | None:
        """Give up the current layout and go back to the one before it, on request.

        The automatic version of this is `report_read`: enough frames of reading
        nothing and the tool moves back on its own. This is the button for the case
        where the user decided it first.
        """
        if not self.fallback:
            return None
        self.verified = self.active = list(self.fallback)
        self.fallback = None
        self.candidate = None
        self.blind = 0
        self.source = "manual"
        self.events.append(("restored", t, tuple(self.verified)))
        return self.active

    # -- helpers --------------------------------------------------------------
    def _replaced(self, cand: Candidate, spots) -> bool:
        """Is this a different proposal than the one already being collected?"""
        if len(spots) != len(cand.spots):
            return True
        return any(max(abs(a[0] - b[0]), abs(a[1] - b[1])) > self.spread_px
                   for a, b in zip(spots, cand.spots))

    def _pitch_change(self, spots) -> float:
        """How much the candidate bends the row spacing, which it must not do:
        the ally list is one rigid sprite, so a proposal that only moved one row is
        a coincidence of two separate errors rather than a panel."""
        if len(spots) < 2 or len(self.verified) < 2:
            return 0.0
        was = abs(self.verified[1][1] - self.verified[0][1])
        now = abs(spots[1][1] - spots[0][1])
        return abs(now - was)
