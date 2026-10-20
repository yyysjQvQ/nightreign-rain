"""Decides when a track that is already playing should stop.

Deliberately not "everybody back to three segments" and not "somebody's segment
count changed": during a revive the gauge goes 3 -> 2 -> 1 and only then
disappears. Losing segments is the music's whole point, so it must keep playing.
What ends the song is seeing an ally standing again, which the caller spells out
as *positive* evidence - the gauge gone and their health bar back. A gauge that
merely failed to read (the panel over a purple sky, a fade, a menu) is not
evidence of anything and must not cut the track off.

A single frame is not trusted either: one dropped or half-composited frame must
not end the song.
"""
from __future__ import annotations


class ReviveWatcher:
    def __init__(self, hold_frames: int = 5):
        if hold_frames < 1:
            raise ValueError("hold_frames must be >= 1")
        self.hold_frames = hold_frames
        self.held = 0
        self._per_slot: list[int] = []
        self.watching = False

    def start(self) -> None:
        self.watching = True
        self._clear()

    def stop(self) -> None:
        self.watching = False
        self._clear()

    def _clear(self) -> None:
        self._per_slot = []
        self.held = 0

    def update(self, downed) -> bool:
        """Feed one frame. True = downed, False = seen standing, None = cannot tell.

        The three cases are not two: a frame that proves nothing must neither reset
        the count nor add to it, because a revived ally's health bar flickers while
        the camera moves. Counted per ally, because the usual revive is one ally
        standing while the other is still on the floor.
        """
        if not self.watching:
            return False
        flags = list(downed)
        if not flags:
            return False
        if len(self._per_slot) != len(flags):
            self._per_slot = [0] * len(flags)
        for i, flag in enumerate(flags):
            if flag is True:
                self._per_slot[i] = 0
            elif flag is False:
                self._per_slot[i] += 1
        self.held = max(self._per_slot)
        if self.held >= self.hold_frames:
            self.stop()
            return True
        return False
