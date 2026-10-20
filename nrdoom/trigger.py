"""Decide when "every ally is downed on their last segment" has really happened.

A single frame is never enough evidence: the gauge arcs flicker while an ally is
being revived, and a frame can be dropped or half-composited. So the condition
has to hold for a run of consecutive frames, and after firing the trigger stays
disarmed until the condition actually clears, otherwise it would re-fire every
frame of a long doomed fight.
"""
from __future__ import annotations


class DoomTrigger:
    def __init__(self, required_bars: int = 3, hold_frames: int = 8,
                 allies: int = 2, cooldown_s: float = 0.0):
        if hold_frames < 1:
            raise ValueError("hold_frames must be >= 1")
        if allies < 1:
            raise ValueError("allies must be >= 1")
        self.required_bars = required_bars
        self.hold_frames = hold_frames
        self.allies = allies
        self.cooldown_s = cooldown_s
        self.held = 0
        self.armed = True
        self.fires = 0
        self._last_fire = None

    def update(self, bars, now: float) -> bool:
        """Feed one frame's bar counts (None where nobody is downed).

        Returns True on the frame the music should start.
        """
        qualifying = sum(1 for b in bars if b is not None and b >= self.required_bars)
        met = len(bars) >= self.allies and qualifying >= self.allies
        if not met:
            self.held = 0
            self.armed = True
            return False
        if not self.armed:
            return False

        self.held += 1
        if self.held < self.hold_frames:
            return False
        if self._last_fire is not None and now - self._last_fire < self.cooldown_s:
            # Sit at the top of the hold so it fires the instant the cooldown ends
            # rather than needing another full hold.
            self.held -= 1
            return False

        self.armed = False
        self.held = 0
        self.fires += 1
        self._last_fire = now
        return True

    def reset(self):
        self.held = 0
        self.armed = True
        self.fires = 0
        self._last_fire = None
