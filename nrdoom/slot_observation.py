"""What one ally slot looked like in one captured frame, as data.

The old path passed the stop logic a three-valued flag (`True` downed, `False`
standing, `None` unclear) built from whichever frame the loop happened to hold.
That loses everything needed to decide whether the flag is worth counting: was
this a new frame or the same buffer again, was the geometry where it should be,
was it the ally's own health bar that said "standing" or just the absence of a
gauge.

The rules this encodes, from the V2 plan §4.2: **standing has to be positive
evidence, not "we failed to see a ring"**. Two things qualify as positive: the
health bar drawn where the HUD draws health bars, and - measured on the user's own
recording, where an ally stood for seven seconds with no visible fill at all - an
annulus that is demonstrably empty on a panel bright enough to read. The colour
rule failing to parse a gauge does not, on its own, ever become "standing".
"""
from __future__ import annotations

from dataclasses import dataclass

STRONG_DOWN = "down"
STRONG_ALIVE = "alive"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class SlotObservation:
    """One slot, one frame, one moment of the capture clock.

    `t` is the capture timestamp rather than a tick count, so a duration written
    in seconds means the same at 12 fps, at 60, and in a replay.

    `ring_share` / `panel_lit` come from `detector.ring_presence` and are what
    let "no gauge" mean something on its own: the ally on his feet after a revive
    often has no visible health-bar fill for seconds at a time, and waiting for
    the red would hold the song over a confirmed rescue. An empty annulus at a
    bright, correctly-placed slot is a positive reading of "not downed" - but only
    where the panel is lit, because on a dark panel every pixel passes the
    brightness test and the share says nothing.
    """

    slot_id: int
    t: float
    frame_id: int
    bars: int | None = None        # segments on a readable downed gauge
    confidence: float = 0.0        # how good that read looked (0 when there was none)
    standing: bool = False         # positive health-bar evidence at this slot
    position_ok: bool = True       # the geometry is where the picture box says
    fresh: bool = True             # a new buffer, not the last frame read again
    ring_share: float | None = None
    panel_lit: int = 0
    ring_absent_max: float = 0.10
    panel_lit_min: int = 70
    accept_empty_ring: bool = True

    @property
    def downed(self) -> bool:
        return self.bars is not None

    @property
    def gauge_absent_valid(self) -> bool:
        """The ring is demonstrably not being drawn here, not merely unreadable."""
        return (self.accept_empty_ring
                and self.ring_share is not None
                and self.panel_lit >= self.panel_lit_min
                and self.ring_share <= self.ring_absent_max)

    def evidence(self) -> str:
        """STRONG_DOWN / STRONG_ALIVE / UNKNOWN - what this frame is allowed to say.

        A frame that is stale or aimed at the wrong place proves nothing, and
        "proved nothing" must not be able to stop a song.
        """
        if not self.fresh or not self.position_ok:
            return UNKNOWN
        if self.downed:
            return STRONG_DOWN
        if self.standing or self.gauge_absent_valid:
            return STRONG_ALIVE
        return UNKNOWN
