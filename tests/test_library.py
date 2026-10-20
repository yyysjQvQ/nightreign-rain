from pathlib import Path

import pytest

from nrdoom.library import MusicLibrary


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    music = tmp_path / "music"
    (music / "artist").mkdir(parents=True)
    for name in ("a.mp3", "b.MP3", "c.ogg", "d.flac", "e.wav"):
        (music / name).write_bytes(b"x")
    (music / "artist" / "f.m4a").write_bytes(b"x")
    (music / "notes.txt").write_text("not music")
    (music / "cover.jpg").write_bytes(b"x")
    (music / ".hidden.mp3").write_bytes(b"x")
    return music


def test_scans_recursively_and_skips_non_audio(folder):
    lib = MusicLibrary(folder)
    names = sorted(p.name.lower() for p in lib.refresh())
    assert names == ["a.mp3", "b.mp3", "c.ogg", "d.flac", "e.wav", "f.m4a"]
    assert len(lib) == 6


def test_extension_match_is_case_insensitive(folder):
    lib = MusicLibrary(folder, seed=1)
    lib.refresh()
    assert any(p.name == "b.MP3" for p in lib.tracks)


def test_missing_folder_is_empty_not_an_error(tmp_path):
    lib = MusicLibrary(tmp_path / "nope")
    assert lib.refresh() == []
    assert lib.pick() is None


def test_pick_returns_none_when_empty(tmp_path):
    assert MusicLibrary(tmp_path / "nothing").pick() is None


def test_every_track_plays_once_before_any_repeats(folder):
    lib = MusicLibrary(folder, seed=7)
    lib.refresh()
    first_cycle = [lib.pick() for _ in range(6)]
    assert len(set(first_cycle)) == 6
    second_cycle = [lib.pick() for _ in range(6)]
    assert len(set(second_cycle)) == 6


def test_no_repeat_across_a_cycle_boundary(folder):
    lib = MusicLibrary(folder, seed=3)
    lib.refresh()
    played = [lib.pick() for _ in range(60)]
    repeats = [i for i in range(1, len(played)) if played[i] == played[i - 1]]
    assert repeats == []


def test_order_is_random_but_reproducible(folder):
    a = [p.name for p in _play_all(MusicLibrary(folder, seed=1))]
    b = [p.name for p in _play_all(MusicLibrary(folder, seed=1))]
    c = [p.name for p in _play_all(MusicLibrary(folder, seed=2))]
    assert a == b
    assert a != c


def _play_all(lib):
    lib.refresh()
    return [lib.pick() for _ in range(len(lib))]


def test_refresh_picks_up_added_and_deleted_files(folder):
    lib = MusicLibrary(folder, seed=5)
    lib.refresh()
    assert len(lib) == 6

    (folder / "g.mp3").write_bytes(b"x")
    (folder / "a.mp3").unlink()
    lib.refresh()
    names = {p.name for p in lib.tracks}
    assert "g.mp3" in names and "a.mp3" not in names


def test_refresh_evicts_a_deleted_track_from_the_queue(folder):
    lib = MusicLibrary(folder, seed=9)
    lib.refresh()
    lib.pick()                       # starts the bag
    (folder / "b.MP3").unlink()
    lib.refresh()
    remaining = [lib.pick() for _ in range(20)]
    assert all(p.name != "b.MP3" for p in remaining)


def test_deleted_file_does_not_stick_as_last_played(folder):
    lib = MusicLibrary(folder, seed=11)
    lib.refresh()
    lib.pick()
    only = folder / "c.ogg"
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p != only:
            p.unlink()
    lib.refresh()
    assert lib.pick() == only


def test_peek_does_not_consume_the_queue(folder):
    lib = MusicLibrary(folder, seed=13)
    lib.refresh()
    assert lib.peek() == lib.pick()


def test_single_track_folder_keeps_playing_that_track(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    (music / "only.mp3").write_bytes(b"x")
    lib = MusicLibrary(music, seed=1)
    lib.refresh()
    assert [lib.pick() for _ in range(3)] == [music / "only.mp3"] * 3


# -- playing in an order the user chose, not one we drew -----------------------
# The folder is a playlist once there is more than one song in it: "play 雨爱
# first, the other one only after that" is a thing a person wants to say, and a
# redraw bag cannot express it.
def test_ordered_mode_plays_the_list_in_that_order(folder):
    lib = MusicLibrary(folder, order="ordered")
    lib.refresh()
    lib.set_sequence(["d.flac", "a.mp3", "c.ogg"])
    assert [lib.pick().name for _ in range(3)] == ["d.flac", "a.mp3", "c.ogg"]


def test_ordered_mode_wraps_around_to_the_front(tmp_path):
    """With the whole folder listed, the order repeats exactly as arranged."""
    music = tmp_path / "music"
    music.mkdir()
    for name in ("a.mp3", "b.mp3"):
        (music / name).write_bytes(b"x")
    lib = MusicLibrary(music, order="ordered")
    lib.refresh()
    lib.set_sequence(["a.mp3", "b.mp3"])
    assert [lib.pick().name for _ in range(5)] == ["a.mp3", "b.mp3", "a.mp3",
                                                   "b.mp3", "a.mp3"]


def test_tracks_missing_from_the_order_are_appended_at_the_end(folder):
    """Dropping a song in the folder must not make it unplayable - it goes last."""
    lib = MusicLibrary(folder, order="ordered")
    lib.refresh()
    lib.set_sequence(["a.mp3"])
    played = [lib.pick().name for _ in range(7)]
    assert played[0] == "a.mp3"
    assert set(played[1:6]) == {"b.MP3", "c.ogg", "d.flac", "e.wav", "f.m4a"}
    assert played[6] == "a.mp3", "the cycle closes back onto the chosen order"


def test_the_order_survives_a_rescan_and_drops_what_is_gone(folder):
    lib = MusicLibrary(folder, order="ordered")
    lib.refresh()
    lib.set_sequence(["c.ogg", "b.MP3", "a.mp3"])
    (folder / "b.MP3").unlink()
    lib.refresh()
    assert [p.name for p in lib.sequence()][:2] == ["c.ogg", "a.mp3"]


def test_names_that_are_not_in_the_library_are_not_queued(folder):
    """A stale config must not create a phantom slot that plays nothing."""
    lib = MusicLibrary(folder, order="ordered")
    lib.refresh()
    lib.set_sequence(["gone.mp3", "a.mp3"])
    assert [p.name for p in lib.sequence() if p.name == "gone.mp3"] == []
    assert lib.pick().name == "a.mp3"


def test_ordered_mode_reopens_after_the_last_track_is_played(tmp_path):
    """Stopping mid-run and firing again should continue the order, not restart it."""
    music = tmp_path / "music"
    music.mkdir()
    for name in ("a.mp3", "b.mp3"):
        (music / name).write_bytes(b"x")
    lib = MusicLibrary(music, order="ordered")
    lib.refresh()
    lib.set_sequence(["a.mp3", "b.mp3"])
    assert lib.pick().name == "a.mp3"
    assert lib.pick().name == "b.mp3"
    assert lib.pick().name == "a.mp3"


def test_shuffle_mode_is_the_default_and_still_a_redraw_bag(folder):
    lib = MusicLibrary(folder, seed=7)
    lib.refresh()
    assert lib.order == "shuffle"
    assert len({lib.pick() for _ in range(6)}) == 6
