"""Who is in the run, and what it takes for the engine to end a song.

Two jobs, both about identity rather than pixels:

* **The slots that triggered are the slots that matter.** Once a song starts, the
  allies whose gauges were on their last segment are locked in. A later frame that
  seems to show fewer rows must not quietly redefine who we are watching, and a
  slot we cannot read contributes nothing rather than a vote.
* **A stop is a confirmed event, not a counter overflow.** Each locked slot gets a
  `ReviveTracker`; the first one to confirm standing ends the song, and the
  evidence for it stays on the record so `diag` can say why.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .slot_observation import SlotObservation
from .temporal_filter import ReviveTracker


@dataclass
class EventArbiter:
    tracker_args: dict = field(default_factory=dict)
    _trackers: dict = field(default_factory=dict)
    _locked: tuple[int, ...] = ()

    def tracker(self, slot_id: int) -> ReviveTracker:
        got = self._trackers.get(slot_id)
        if got is None:
            got = self._trackers[slot_id] = ReviveTracker(**self.tracker_args)
        return got

    def start_song(self, slot_ids) -> None:
        """Lock the allies that took part and clear every slate."""
        self._locked = tuple(slot_ids)
        for tracker in self._trackers.values():
            tracker.reset()

    def end_song(self) -> None:
        self._locked = ()

    @property
    def locked(self) -> tuple[int, ...]:
        return self._locked

    def observe(self, obs: SlotObservation) -> bool:
        """Feed one slot's frame. True when that ally is confirmed standing."""
        if not self._locked or obs.slot_id not in self._locked:
            return False
        return self.tracker(obs.slot_id).update(obs)

    def ages(self) -> dict:
        """How far each locked ally's revive candidate has got, for diagnostics."""
        return {i: round(self._trackers[i].candidate_age, 2)
                for i in self._locked if i in self._trackers}
