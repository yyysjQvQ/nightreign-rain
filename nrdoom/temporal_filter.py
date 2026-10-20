"""When a song that is already playing is allowed to end.

`revive.ReviveWatcher` counts *frames*: thirty observations of "no gauge plus a
health bar" ends the track. Counting frames is the wrong unit for two reasons
the V2 plan puts first:

* the same buffer read thirty times is one observation, not thirty - a capture
  that stalls must never look like a revive;
* thirty scattered frames are not thirty consecutive ones. A bar that flickers
  for a fifth of a second every couple of seconds accumulates into a stop, even
  though no second of the game ever showed the ally on their feet.

So this works on the capture clock: a candidate has to *survive* its window, and
evidence that lapses takes the candidate with it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .slot_observation import STRONG_ALIVE, STRONG_DOWN, SlotObservation


@dataclass
class ReviveTracker:
    """One ally's road from "downed" to "confirmed standing".

    Defaults are the plan's starting values, in seconds. They are calibration
    seeds, not measured truths: `tools/baseline.py` is what decides whether they
    stop the real revives early enough and let the fake ones through.
    """

    window_s: float = 1.4          # how far back the share is measured over
    min_candidate_s: float = 0.8   # a candidate must have been alive this long
    min_samples: int = 12          # ... on at least this many distinct frames
    min_alive_share: float = 0.80  # ... of which this much says "standing"
    lapse_s: float = 0.25          # no usable evidence for this long -> expired
    conflict_s: float = 0.5        # a downed reading in here vetoes the stop

    _samples: deque = field(default_factory=deque, repr=False)
    _candidate_t: float | None = field(default=None, repr=False)
    _last_good_t: float | None = field(default=None, repr=False)
    _last_frame_id: int = field(default=-1, repr=False)
    _confirmed: bool = field(default=False, repr=False)

    def reset(self) -> None:
        self._samples.clear()
        self._candidate_t = self._last_good_t = None
        self._last_frame_id = -1
        self._confirmed = False

    @property
    def confirmed(self) -> bool:
        return self._confirmed

    @property
    def candidate_age(self) -> float:
        if self._candidate_t is None:
            return 0.0
        return 0.0 if not self._samples else self._samples[-1][0] - self._candidate_t

    def _trim(self, now: float) -> None:
        horizon = now - self.window_s
        while self._samples and self._samples[0][0] < horizon:
            self._samples.popleft()

    def update(self, obs: SlotObservation) -> bool:
        """Feed one observation. True on the frame this ally is confirmed up."""
        if self._confirmed:
            return False
        if obs.frame_id <= self._last_frame_id:
            return False                 # the same buffer, or a clock that went back
        self._last_frame_id = obs.frame_id

        kind = obs.evidence()
        self._samples.append((obs.t, kind, obs.frame_id))
        self._trim(obs.t)

        if kind == STRONG_DOWN:
            self._candidate_t = None     # he is still on the floor: nothing to build on
            self._last_good_t = obs.t
            return False
        if kind == STRONG_ALIVE:
            self._last_good_t = obs.t
            if self._candidate_t is None:
                self._candidate_t = obs.t

        if self._candidate_t is None:
            return False
        if self._last_good_t is None or obs.t - self._last_good_t > self.lapse_s:
            self._candidate_t = None     # the evidence lapsed; start over from scratch
            return False

        good = [s for s in self._samples if s[1] in (STRONG_ALIVE, STRONG_DOWN)]
        if len({f for _t, _k, f in good}) < self.min_samples:
            return False
        if obs.t - self._candidate_t < self.min_candidate_s:
            return False
        if good[-1][0] - good[0][0] < self.min_candidate_s:
            return False
        # The share counts every frame that could tell, not only the ones that
        # agreed: a window of nothing but hits would be self-selecting evidence.
        alive = sum(1 for _t, k, _f in good if k == STRONG_ALIVE)
        if alive / len(good) < self.min_alive_share:
            return False
        if any(t > obs.t - self.conflict_s for t, k, _f in good if k == STRONG_DOWN):
            return False
        self._confirmed = True
        return True
