"""Follow the live party size, which is not the same as what was calibrated.

Players leave mid-run and the ally list compacts, so a frozen count would either
wait forever for an ally who has gone or, worse, stop waiting for one who is still
there. The rows are counted from what the HUD draws for an ally (see
detector.count_rows); this turns that per-frame number into a decision.

The two directions used to be treated as asymmetric: raising was free, lowering
was what could start the music early, so only lowering got a hold. That turned out
to be wrong. A single bad frame can read a gauge on a row where nobody is sitting
(one frame in 5533 on the user's own recording), and an unearned rise then withholds
the music for the whole 60-frame lowering hold that follows it - the ally who really
was downed on their last segment gets nothing. So a rise above a count we already
hold has to be seen more than once, and until it is, the count is admittedly not
settled and `pending` says so.

Growing the party from nothing is exempt: that is arming mid-run, and making the
user wait for the allies they can already see is not a safety win.

A count of zero is not evidence either way. It means the panel gave us nothing -
which is what a hidden HUD, a scene change, or an ally standing on nearly no
health looks like from here - so it holds the roster and restarts any run toward a
shorter one. Only a *seen* shorter list may lower the count.
"""
from __future__ import annotations


class PartyCounter:
    def __init__(self, hold_frames: int = 60, rise_samples: int = 1,
                 rise_frames: int = 8):
        if hold_frames < 1:
            raise ValueError("hold_frames must be >= 1")
        if rise_samples < 1:
            raise ValueError("rise_samples must be >= 1")
        if rise_frames < rise_samples:
            raise ValueError("the corroboration window has to fit its samples")
        self.hold_frames = hold_frames
        self.rise_samples = rise_samples
        self.rise_frames = rise_frames
        self.count = 0
        self.pending = False
        self.last_rows = 0
        self._held = 0
        self._step = 0
        self._above: list[int] = []

    def update(self, rows: int) -> int:
        """Feed one frame's row count and return the party size to act on."""
        self._step += 1
        self.last_rows = rows
        if not rows:
            # Nothing was readable anywhere on the panel. That is the picture being
            # unhelpful, not the allies leaving - and on the user's own recordings
            # every single drop of the count came from a stretch like this, while
            # the panel legitimately reads blank for seconds after a revive. Hold
            # the roster, and break any run toward a shorter one: that count has to
            # be seen on its own.
            self._held = 0
            return self.count
        if rows > self.count:
            if self.count and self.rise_samples > 1:
                self._above = [s for s in self._above
                               if self._step - s < self.rise_frames]
                self._above.append(self._step)
                if len(self._above) < self.rise_samples:
                    self.pending = True
                    return self.count
            self._above = []
        self.pending = False
        if rows >= self.count:
            self.count = rows
            self._held = 0
            return self.count
        self._held += 1
        if self._held >= self.hold_frames:
            self.count = rows
            self._held = 0
        return self.count

    def reset(self) -> None:
        self.count = 0
        self.pending = False
        self._held = 0
        self._step = 0
        self._above = []
