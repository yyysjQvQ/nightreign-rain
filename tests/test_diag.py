"""Diagnostics have to be able to answer "what did the two seconds before that
stop look like", which is the only reason to keep them at all."""
import json
import time
from pathlib import Path

import numpy as np
import pytest

from nrdoom.diag import Diagnostics
from nrdoom.replay import ManualClock


def frame(v=40):
    return np.zeros((200, 200, 3), dtype=np.uint8) + v


@pytest.fixture
def folder(tmp_path):
    return tmp_path / "diag"


def lines(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def test_an_event_writes_the_frames_that_came_before_it(folder):
    """The ring has to fill while nothing is happening; a buffer that only starts
    recording once the event fires can show what came after, which is never the
    interesting half."""
    clock = ManualClock()
    d = Diagnostics(folder, clock=clock, before_s=10.0, stride=1)
    for _ in range(20):
        clock.advance(0.1)
        d.frame(frame(), (0, 0, 200, 200), [(50, 60, 20)], [], 1, frame_id=1)
    d.event("stop", frame(), (0, 0, 200, 200), slot=0)
    d.close()
    saved = list((folder / "roi").glob("stop_*"))
    assert saved, "no event folder was written"
    crops = list(saved[0].glob("*.npz"))
    before = [c for c in crops if c.name.startswith("before_")]
    assert len(before) >= 10, [c.name for c in crops]
    assert max(float(c.name.split("_")[1][:-4].rstrip("s")) for c in before) <= 0.0


def test_a_stop_also_keeps_the_seconds_after_it(folder):
    """"Did he actually get up" is answered by what came after the stop, and
    `after_s` had been a field nothing ever read."""
    clock = ManualClock()
    d = Diagnostics(folder, clock=clock, before_s=2.0, after_s=2.0, stride=1)
    ring = [(50, 60, 20)]
    for i in range(20):
        clock.advance(0.1)
        d.frame(frame(), (0, 0, 200, 200), ring, [], 1, frame_id=i)
    d.event("stop", frame(), (0, 0, 200, 200), slot=0)
    for i in range(20, 50):
        clock.advance(0.1)
        d.frame(frame(), (0, 0, 200, 200), ring, [], 1, frame_id=i)
    d.close()
    saved = list((folder / "roi").glob("stop_*"))[0]
    after = [c for c in saved.glob("after_*.npz")]
    assert len(after) >= 15, [c.name for c in after]
    offsets = [float(c.name.split("_")[1][:-4].rstrip("s")) for c in after]
    assert min(offsets) > 0.0 and max(offsets) <= 2.0, offsets


def test_an_unrelated_event_does_not_close_the_stop_s_after_window(folder):
    """The seconds after a stop are the evidence that he really got up. The geometry
    events now log to the same file, and routing them through the same bookkeeping
    let a `geometry_verified` two seconds later switch the collection off mid-answer."""
    clock = ManualClock()
    d = Diagnostics(folder, clock=clock, before_s=2.0, after_s=2.0, stride=1)
    ring = [(50, 60, 20)]
    for i in range(20):
        clock.advance(0.1)
        d.frame(frame(), (0, 0, 200, 200), ring, [], 1, frame_id=i)
    d.event("stop", frame(), (0, 0, 200, 200), slot=0)
    for i in range(20, 50):
        clock.advance(0.1)
        d.frame(frame(), (0, 0, 200, 200), ring, [], 1, frame_id=i)
        if i == 30:
            d.event("geometry_verified", frame(), (0, 0, 200, 200), spots=ring)
    d.close()
    saved = list((folder / "roi").glob("stop_*"))[0]
    after = list(saved.glob("after_*.npz"))
    assert len(after) >= 15, \
        "a geometry event cut the stop's after-window short: %d" % len(after)
    geom = list((folder / "roi").glob("geometry_verified_*"))
    assert len(geom) == 1, geom
    assert list(geom[0].glob("at_*.npz")), "the geometry event still needs its own crop"
    kinds = [l["kind"] for l in lines(folder / "events.jsonl")]
    assert "geometry_verified" in kinds


def test_a_fire_does_not_bank_up_seconds_of_crops(folder):
    """A stop has to be explained; a fire is explained by the frame that caused it.
    Recording both windows for every fire doubled what a session costs on disk."""
    clock = ManualClock()
    d = Diagnostics(folder, clock=clock, before_s=2.0, stride=1)
    ring = [(50, 60, 20)]
    for i in range(20):
        clock.advance(0.1)
        d.frame(frame(), (0, 0, 200, 200), ring, [], 1, frame_id=i)
    d.event("fire", frame(), (0, 0, 200, 200), track="doom.mp3")
    for i in range(20, 40):
        clock.advance(0.1)
        d.frame(frame(), (0, 0, 200, 200), ring, [], 1, frame_id=i)
    d.close()
    saved = list((folder / "roi").glob("fire_*"))[0]
    assert list(saved.glob("before_*.npz")) == []
    assert list(saved.glob("after_*.npz")) == []
    assert list(saved.glob("at_*.npz")), "a fire still needs the frame it fired on"


def test_a_steady_panel_still_leaves_a_trail(folder):
    """Change-only logging left every stop unverifiable.

    The seconds before a revive are exactly the ones where nothing changes - the
    panel reads blank the whole time the ally stands up - so a log that only records
    change answers "what was on screen just before it stopped" with nothing. A quiet
    stretch is therefore sampled, not skipped, and still not written every frame."""
    clock = ManualClock()
    d = Diagnostics(folder, clock=clock, stride=1000)
    ring = [(50, 60, 20)]
    for i in range(30):                       # 1.5 s of an unchanged frame
        clock.advance(0.05)
        d.frame(frame(), (0, 0, 200, 200), ring, [], 1, frame_id=i)
    d.close()
    got = lines(folder / "events.jsonl")
    assert 2 <= len(got) <= 5, [g["t"] for g in got]


def test_the_half_second_before_a_stop_is_never_empty(folder):
    clock = ManualClock()
    d = Diagnostics(folder, clock=clock, stride=1000)
    ring = [(50, 60, 20)]
    for i in range(200):                      # 10 s of a steady, blank panel
        clock.advance(0.05)
        d.frame(frame(), (0, 0, 200, 200), ring, [], 1, frame_id=i)
    d.event("stop", frame(), (0, 0, 200, 200), slot=0)
    d.close()
    got = lines(folder / "events.jsonl")
    stop = [g for g in got if g["kind"] == "stop"][0]
    window = [g for g in got if g["kind"] == "frame"
              and stop["t"] - 0.6 <= g["t"] <= stop["t"]]
    assert window, "the look-back was empty, so the stop cannot be explained"


def test_a_low_confidence_read_is_written_even_when_nothing_changed(folder):
    clock = ManualClock()
    d = Diagnostics(folder, clock=clock, stride=1000, low_conf=0.6)
    ring = [(50, 60, 20)]
    good = [type("R", (), {"bars": 3, "confidence": 0.95})()]
    weak = [type("R", (), {"bars": 3, "confidence": 0.4})()]
    for i, reading in enumerate(good + weak * 4):
        clock.advance(0.05)
        d.frame(frame(), (0, 0, 200, 200), ring, [reading], 1, frame_id=i)
    d.close()
    logged = lines(folder / "events.jsonl")
    assert len(logged) == 5              # one change, then every dubious frame
    assert logged[0]["conf"] == [0.95] and logged[1]["conf"] == [0.4]
