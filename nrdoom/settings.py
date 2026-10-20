"""Persisted settings, kept next to the code so they are easy to eyeball and edit."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


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
    # "2560x1440" -> the spots that worked there. The in-game UI size setting
    # moves the panel, so the geometry belongs to the resolution, not to you.
    spots_cache: dict[str, list[Slot]] = field(default_factory=dict)
    show_pet: bool = True
    # Diagnostics cost nothing when off (the engine holds None and never builds a
    # record), but they write a folder next to the exe, so it stays opt-in.
    diagnostics: bool = False
    diagnostics_dir: str = "diagnostics"
    diagnostics_seconds: float = 10.0

    def apply_spots(self, key: str, spots):
        """Make `spots` the live geometry and remember it for this frame size."""
        self.spots_cache[key] = [Slot(cx, cy, radius) for cx, cy, radius in spots]
        self.slots = list(self.spots_cache[key])
        return self.slots

    def slot_tuples(self):
        return [(s.cx, s.cy, s.radius) for s in self.slots]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["slots"] = [asdict(s) for s in self.slots]
        data["spots_cache"] = {k: [asdict(s) for s in v]
                               for k, v in self.spots_cache.items()}
        return data

    # Shipped defaults nobody was ever asked to choose. A config carrying one of
    # these was written by that version, not typed by the user, so the newer
    # default wins instead of the stale number pinning it forever.
    LEGACY = {"revive_hold_frames": 5}

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known and k not in {"slots", "spots_cache"}}
        for key, was in cls.LEGACY.items():
            if clean.get(key) == was:
                clean.pop(key)
        st = cls(**clean)
        st.slots = cls._slots(data.get("slots", []))
        st.spots_cache = {k: cls._slots(v) for k, v in (data.get("spots_cache") or {}).items()}
        return st

    @staticmethod
    def _slots(raw):
        return [Slot(**{k: v for k, v in s.items() if k in {"cx", "cy", "radius"}})
                for s in raw]

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
        path = Path(path)
        if path.is_file():
            path.replace(path.with_suffix(".json.bak"))
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")
