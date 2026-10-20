"""Persisted settings, kept next to the code so they are easy to eyeball and edit."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .geometry import boxes_agree


def _default_path() -> Path:
    """Next to the exe when frozen.

    `__file__` inside a onefile bundle points into the _MEIPASS temp directory,
    so a config written there vanishes when the process exits and the user's
    calibration is lost on every launch.
    """
    import sys

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "config.json"
    return Path(__file__).resolve().parents[1] / "config.json"


DEFAULT_PATH = _default_path()


@dataclass
class Slot:
    """One ally gauge position, in captured-frame pixels."""
    cx: float
    cy: float
    radius: float


@dataclass
class Layout:
    """A geometry that was proved, and the picture it was proved on.

    The in-game UI size setting and the letterbox both move the panel inside the
    same captured frame, so a resolution is not enough to say where to look: two
    different renderings of 2560x1440 need two different layouts, and once one of
    them is filed under the shared resolution the other starts the session looking
    eighty pixels away from the panel.
    """

    capture: list[int]                 # [width, height] of the captured frame
    box: list[int] | None              # [left, top, width] of the game's picture
    slots: list[Slot] = field(default_factory=list)
    source: str = "seed"               # seed / cache / realign / recovery / rollback
    verified_frames: int = 0           # separate captures that agreed on it
    verified_at: float = 0.0           # wall clock of the moment it was earned

    @property
    def key(self):
        return (tuple(self.capture), tuple(self.box) if self.box else None)


# Bumped when the meaning of a stored layout changes, not when a field is added.
LAYOUT_SCHEMA = 1


@dataclass
class Settings:
    music_folder: str = "music"
    window_title: str = "ELDEN RING"
    target_fps: int = 20
    required_bars: int = 3
    hold_frames: int = 8
    revive_hold_frames: int = 30
    # "v1" counts frames of standing evidence (the rollback path, still tested);
    # "v2" needs a confirmed standing ally held on the capture clock, so a frozen
    # feed and scattered flickers cannot add up to a revive.
    stop_policy: str = "v2"
    revive_window_s: float = 1.4
    revive_min_evidence_s: float = 0.8
    revive_min_samples: int = 12
    revive_min_alive_share: float = 0.80
    revive_lapse_s: float = 0.25
    revive_conflict_s: float = 0.5
    # An empty gauge annulus on a bright, correctly-placed slot counts as "not
    # downed", which is what lets the song end while the ally's health bar is
    # still too short to see. Off, the stop waits for the red fill (the Phase 1
    # behaviour) - kept so the two can be measured against each other.
    revive_empty_ring_counts: bool = True
    # A picture box that only disagrees for a moment is a cinematic bar crossing the
    # screen, not a new letterbox: on the user's own recording a 0.6 s overlay moved
    # the geometry and reset the roster. With the same capture size, a confirmed box
    # is only given up once the new one has held this long.
    box_hold_s: float = 1.5
    party_hold_frames: int = 60
    # Which song plays next: "shuffle" is the redraw bag, "ordered" walks the list
    # below and wraps around. `track_order` holds file names as the music folder
    # shows them; anything in the folder that is not listed plays last.
    play_order: str = "shuffle"
    track_order: list[str] = field(default_factory=list)
    # A row count above the one we already hold needs this many separate frames
    # inside party_rise_s before it is believed. One fluke read on an empty row is
    # enough to raise the party, and the rise then withholds the music for the
    # whole party_hold_frames it takes to fall back down again.
    party_rise_samples: int = 2
    party_rise_s: float = 0.4
    cooldown_s: float = 0.0
    loop: bool = True
    volume: float = 0.8
    slots: list[Slot] = field(default_factory=list)
    # Geometries that were *proved*, one per (capture size, picture box). What this
    # replaces was keyed by resolution alone, which is how a layout measured on one
    # letterbox ended up steering a session inside another one - the in-game UI size
    # setting and a window drag both move the panel within the same 2560x1440.
    layouts: list[Layout] = field(default_factory=list)
    layout_schema: int = 0
    # Window dragging produces one layout per box it passes through. Twelve is far
    # more than a machine actually has and keeps the file from growing forever.
    MAX_LAYOUTS = 12
    show_pet: bool = True
    # Diagnostics cost nothing when off (the engine holds None and never builds a
    # record), but they write a folder next to the exe, so it stays opt-in.
    diagnostics: bool = False
    diagnostics_dir: str = "diagnostics"
    diagnostics_seconds: float = 10.0

    def layout_for(self, capture, box) -> Layout | None:
        """The proved geometry for this capture size and picture box, if there is one.

        A layout carried over from the old resolution-only key (`box is None`) is
        only used when nothing was ever proved for this box: it is a guess with the
        right resolution and no idea which letterbox it came from.
        """
        legacy = None
        for lay in self.layouts:
            if tuple(lay.capture) != tuple(capture):
                continue
            if lay.box is None:
                legacy = legacy or lay
            elif boxes_agree(lay.box, box):
                return lay
        return legacy

    def remember_layout(self, capture, box, spots, source: str = "cache",
                        frames: int = 0, when: float = 0.0) -> Layout:
        """File the geometry that is currently believed, for this exact picture box."""
        lay = Layout(capture=[int(capture[0]), int(capture[1])],
                     box=[int(v) for v in box] if box is not None else None,
                     slots=[Slot(cx, cy, radius) for cx, cy, radius in spots],
                     source=source, verified_frames=int(frames),
                     verified_at=float(when))
        for i, other in enumerate(self.layouts):
            if other.key == lay.key or (other.capture == lay.capture
                                         and other.box is not None
                                         and boxes_agree(other.box, lay.box)):
                self.layouts.pop(i)
                break
        self.layouts.insert(0, lay)
        del self.layouts[self.MAX_LAYOUTS:]
        return lay

    def forget_layout(self, capture, box) -> int:
        """Drop the geometry proved for one picture box, and only that one.

        The interface's "reset this layout" button needs this to be narrow: the
        other resolutions and letterboxes the same machine has been seen in are
        proved too, and wiping them would cost a session of relearning.
        """
        before = len(self.layouts)
        self.layouts = [lay for lay in self.layouts
                        if tuple(lay.capture) != tuple(capture)
                        or not boxes_agree(lay.box, box)]
        return before - len(self.layouts)

    def use_spots(self, spots):
        """Look here on this frame, and nothing more.

        `remember_layout` is the decision to keep; this is only "aim here now". A
        picture box the vote has not confirmed, or a position that is still a
        candidate, must never reach the store - or one frame measured during a fade
        becomes where the next launch looks too.
        """
        self.slots = [Slot(cx, cy, radius) for cx, cy, radius in spots]
        return self.slots

    def slot_tuples(self):
        return [(s.cx, s.cy, s.radius) for s in self.slots]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["slots"] = [asdict(s) for s in self.slots]
        data["layout_schema"] = LAYOUT_SCHEMA
        return data

    # Shipped defaults nobody was ever asked to choose. A config carrying one of
    # these was written by that version, not typed by the user, so the newer
    # default wins instead of the stale number pinning it forever.
    LEGACY = {"revive_hold_frames": 5}

    # Fields that are not plain scalars, so not something `from_dict` may hand to
    # the constructor as whatever happened to be in the file.
    DERIVED = {"slots", "layouts", "spots_cache", "layout_schema"}

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items()
                 if k in known and k not in cls.DERIVED}
        for key, was in cls.LEGACY.items():
            if clean.get(key) == was:
                clean.pop(key)
        st = cls(**clean)
        st.slots = cls._slots(data.get("slots", []))
        stored = data.get("layouts")
        if stored and int(data.get("layout_schema") or 0) >= LAYOUT_SCHEMA:
            st.layouts = cls._layouts(stored)
        else:
            st.layouts = cls._migrate(data.get("spots_cache") or {})
        st.layout_schema = LAYOUT_SCHEMA
        return st

    @staticmethod
    def _slots(raw):
        return [Slot(**{k: v for k, v in s.items() if k in {"cx", "cy", "radius"}})
                for s in raw]

    @staticmethod
    def _layouts(raw):
        fields = {f for f in Layout.__dataclass_fields__}
        out = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            lay = Layout(capture=item.get("capture") or [0, 0], box=item.get("box"))
            for k, v in item.items():
                if k in fields and k not in {"capture", "box", "slots"}:
                    setattr(lay, k, v)
            lay.slots = Settings._slots(item.get("slots") or [])
            out.append(lay)
        return out

    @staticmethod
    def _migrate(cache):
        """The old `"2560x1440" -> spots` map, as layouts with no box recorded.

        Nothing is thrown away: the geometry a user's earlier sessions earned is
        still the best guess available for that resolution, and it will be replaced
        by a box-specific one the first time one is proved.
        """
        out = []
        for key, slots in cache.items():
            try:
                w, h = (int(part) for part in str(key).lower().split("x"))
            except ValueError:
                continue
            out.append(Layout(capture=[w, h], box=None,
                               slots=Settings._slots(slots), source="cache"))
        return out

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "Settings":
        path = Path(path)
        if not path.is_file():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # A hand-edited file with a typo should not cost the user their
            # calibration; the previous copy is kept alongside on every save.
            backup = path.with_suffix(".json.bak")
            if backup.is_file():
                try:
                    raw = json.loads(backup.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    return cls()
            else:
                return cls()
        return cls.from_dict(raw)

    def save(self, path: Path | str = DEFAULT_PATH) -> None:
        """Write atomically, and keep the previous good copy.

        The old version moved `config.json` aside before writing the new one, so a
        crash or a power cut in between left no config at all and the loader fell
        back to defaults: music folder, volume, play order and layouts, all gone.
        Renaming a finished temporary file is the step that cannot be caught halfway.
        """
        path = Path(path)
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(text, encoding="utf-8")
        if path.is_file():
            path.with_suffix(".json.bak").write_bytes(path.read_bytes())
        os.replace(tmp, path)
