"""The music folder: what counts as a track, and which one plays next.

Deliberately Qt-free so the ordering behaviour is testable without an event loop.
"""
from __future__ import annotations

import random
from pathlib import Path

SUPPORTED = {".mp3", ".ogg", ".oga", ".wav", ".flac", ".m4a", ".aac", ".opus", ".wma"}

SHUFFLE = "shuffle"        # a redraw bag: every track once per cycle
ORDERED = "ordered"        # the sequence the user arranged, wrapping at the end


class MusicLibrary:
    def __init__(self, folder: str | Path, seed: int | None = None,
                 order: str = SHUFFLE):
        self.folder = Path(folder)
        self._rng = random.Random(seed)
        self.order = order if order in (SHUFFLE, ORDERED) else SHUFFLE
        self._tracks: list[Path] = []
        self._bag: list[Path] = []
        self._last: Path | None = None
        self._wanted: list[str] = []
        self._cursor = 0

    def refresh(self) -> list[Path]:
        """Re-scan the folder. Call it after the user drops files in."""
        found = []
        if self.folder.is_dir():
            for path in sorted(self.folder.rglob("*")):
                if not path.is_file():
                    continue
                if path.name.startswith("."):
                    continue
                if path.suffix.lower() in SUPPORTED:
                    found.append(path)
        self._tracks = found
        # Drop anything that no longer exists, keep anything still queued.
        alive = set(self._tracks)
        self._bag = [t for t in self._bag if t in alive]
        if self._last not in alive:
            self._last = None
        return list(self._tracks)

    def _fill_bag(self):
        """Refill with a full shuffled cycle, only swapping to avoid a repeat."""
        refill = list(self._tracks)
        self._rng.shuffle(refill)
        if len(refill) > 1 and refill[0] == self._last:
            other = self._rng.randrange(1, len(refill))
            refill[0], refill[other] = refill[other], refill[0]
        self._bag = refill

    @property
    def tracks(self) -> list[Path]:
        return list(self._tracks)

    # -- the order the user arranged ------------------------------------------
    def set_sequence(self, names) -> None:
        """Remember the play order, by file name, as the list widget shows it.

        Re-arranging starts the run again from the top: a half-consumed queue of
        the old order is not what someone meant when they moved a song up.
        """
        self._wanted = [str(n) for n in names]
        self._cursor = 0

    def sequence(self) -> list[Path]:
        """What will play, in that order: what was listed first, then whatever the
        user never mentioned - a song dropped into the folder still gets played,
        just last, rather than becoming silent."""
        left: dict[str, list[Path]] = {}
        for path in self._tracks:
            left.setdefault(path.name, []).append(path)
        out: list[Path] = []
        for name in self._wanted:
            queue = left.get(name)
            if queue:
                out.append(queue.pop(0))
                if not queue:
                    del left[name]
        for name in sorted(left):
            out.extend(left[name])
        return out

    def pick(self) -> Path | None:
        """Next track: shuffled without immediate repeats, or in the chosen order.

        A redraw bag in shuffled mode: every track plays once per cycle before any
        track plays again, so a short folder of three songs still sounds random over
        a long session. Cycles stay exactly one track each so that stays true across
        a boundary; the only exception is the swap that stops the last track of one
        cycle being the first of the next.
        """
        if not self._tracks:
            return None
        if self.order == ORDERED:
            order = self.sequence()
            if not order:
                return None
            nxt = order[self._cursor % len(order)]
            self._cursor = (self._cursor + 1) % len(order)
            self._last = nxt
            return nxt
        if not self._bag:
            self._fill_bag()
        nxt = self._bag.pop(0)
        self._last = nxt
        return nxt

    def peek(self) -> Path | None:
        """The track that would play next, without consuming it."""
        if not self._tracks:
            return None
        if self.order == ORDERED:
            order = self.sequence()
            return order[self._cursor % len(order)] if order else None
        if not self._bag:
            self._fill_bag()
        return self._bag[0]

    def __len__(self) -> int:
        return len(self._tracks)
