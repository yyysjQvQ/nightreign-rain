import json
from pathlib import Path

import pytest

from nrdoom.settings import Settings, Slot


@pytest.fixture
def cfg(tmp_path: Path) -> Path:
    return tmp_path / "config.json"


def test_defaults_when_no_file(cfg):
    st = Settings.load(cfg)
    assert st.slots == []
    assert st.required_bars == 3
    assert st.layouts == []


def test_round_trip_keeps_slots(cfg):
    st = Settings(music_folder=r"D:\tunes", volume=0.33,
                  slots=[Slot(128, 247, 27), Slot(128, 358, 27)])
    st.save(cfg)
    back = Settings.load(cfg)
    assert back.music_folder == st.music_folder
    assert back.volume == pytest.approx(0.33)
    assert back.slot_tuples() == st.slot_tuples()


def test_file_is_readable_json_with_chinese(cfg):
    Settings(music_folder=r"E:\音乐\黑夜").save(cfg)
    raw = json.loads(cfg.read_text(encoding="utf-8"))
    assert raw["music_folder"] == r"E:\音乐\黑夜"


def test_unknown_keys_from_an_older_version_are_ignored(cfg):
    cfg.write_text(json.dumps({"volume": 0.5, "legacy_toggle": True,
                               "slots": [{"cx": 1, "cy": 2, "radius": 3, "gone": 9}]}),
                   encoding="utf-8")
    st = Settings.load(cfg)
    assert st.volume == 0.5
    assert st.slots == [Slot(1, 2, 3)]


def test_a_corrupt_file_falls_back_to_the_previous_save(cfg):
    Settings(volume=0.2).save(cfg)
    Settings(volume=0.9).save(cfg)          # rotates the 0.2 copy into .bak
    assert cfg.with_suffix(".json.bak").is_file()
    cfg.write_text("{ not json", encoding="utf-8")
    assert Settings.load(cfg).volume == pytest.approx(0.2)


def test_a_corrupt_file_with_no_backup_starts_clean(cfg):
    cfg.write_text("{ nope", encoding="utf-8")
    assert Settings.load(cfg).volume == pytest.approx(Settings().volume)


def test_layouts_are_kept_per_picture_box(cfg):
    """The in-game UI size and the letterbox both move the panel, so a resolution is
    not enough to say where to look: two renderings of the same 2276x1280 need two
    layouts, and each has to find its own again."""
    st = Settings()
    st.remember_layout((2276, 1280), (0, 0, 2276), [(128, 247, 27), (128, 358, 27)],
                       source="recovery", frames=4, when=12.5)
    st.remember_layout((2276, 1280), (160, 0, 2116), [(280, 229, 25), (280, 333, 25)])
    st.remember_layout((1366, 768), (0, 0, 1366), [(77, 148, 14.5), (77, 215, 14.3)])
    st.save(cfg)
    back = Settings.load(cfg)
    got = back.layout_for((2276, 1280), (0, 0, 2276))
    assert [s.cy for s in got.slots] == [247, 358]
    assert (got.source, got.verified_frames, got.verified_at) == ("recovery", 4, 12.5)
    assert [s.cx for s in back.layout_for((2276, 1280), (160, 0, 2116)).slots] == [280, 280]
    # A measurement of the same box a pixel or two off is the same box.
    assert back.layout_for((2276, 1280), (2, 0, 2274)).box == [0, 0, 2276]
    assert back.layout_for((2276, 1280), (158, 0, 2118)).box == [160, 0, 2116]
    # ...and eighty pixels over is a different letterbox. Borrowing the other box's
    # layout is exactly the bug this store exists to fix.
    assert back.layout_for((2276, 1280), (80, 0, 2196)) is None
    assert back.layout_for((1920, 1080), (0, 0, 1920)) is None


def test_remembering_a_box_replaces_it_rather_than_piling_up(cfg):
    st = Settings()
    st.remember_layout((2276, 1280), (0, 0, 2276), [(128, 247, 27), (128, 358, 27)])
    st.remember_layout((2276, 1280), (1, 0, 2276), [(130, 250, 27), (130, 361, 27)])
    assert len(st.layouts) == 1
    assert st.layout_for((2276, 1280), (0, 0, 2276)).slots[0].cx == 130


def test_the_store_stops_growing_when_a_window_gets_dragged(cfg):
    """One layout per letterbox the window passed through is not a memory worth
    keeping; the newest few are what the machine actually uses."""
    st = Settings()
    for i in range(st.MAX_LAYOUTS + 9):
        st.remember_layout((2276, 1280), (i * 20, 0, 2276 - i * 40), [(1, 2, 3)])
    assert len(st.layouts) == st.MAX_LAYOUTS
    assert st.layouts[0].box == [(st.MAX_LAYOUTS + 8) * 20, 0, 2276 - (st.MAX_LAYOUTS + 8) * 40]


def test_resetting_one_layout_leaves_the_other_resolutions(cfg):
    """G18: the "forget this layout" button must be able to promise that narrowly."""
    st = Settings()
    st.remember_layout((2276, 1280), (0, 0, 2276), [(128, 247, 27)])
    st.remember_layout((2276, 1280), (160, 0, 2116), [(280, 229, 25)])
    st.remember_layout((1366, 768), (0, 0, 1366), [(77, 148, 14.5)])
    assert st.forget_layout((2276, 1280), (4, 0, 2272)) == 1
    assert st.layout_for((2276, 1280), (0, 0, 2276)) is None
    assert st.layout_for((2276, 1280), (160, 0, 2116)) is not None
    assert st.layout_for((1366, 768), (0, 0, 1366)) is not None


def test_an_old_config_moves_its_cache_into_the_store(cfg):
    """G13: `"2560x1440" -> spots` was the whole format once. Reading a config from
    that build has to keep the geometry it earned, and must not cost the user the
    music folder, the volume or the order they arranged."""
    cfg.write_text(json.dumps({
        "music_folder": r"E:\音乐\黑夜", "volume": 0.4,
        "track_order": ["a.mp3", "b.mp3"], "play_order": "ordered",
        "slots": [{"cx": 128, "cy": 247, "radius": 27}],
        "spots_cache": {"2276x1280": [{"cx": 128, "cy": 247, "radius": 27},
                                      {"cx": 128, "cy": 358, "radius": 27}],
                        "1366x768": [{"cx": 77, "cy": 148, "radius": 14.5}],
                        "nonsense": [{"cx": 1, "cy": 1, "radius": 1}]},
    }), encoding="utf-8")
    st = Settings.load(cfg)
    assert st.music_folder == r"E:\音乐\黑夜"
    assert st.track_order == ["a.mp3", "b.mp3"]
    assert st.play_order == "ordered"
    assert st.volume == pytest.approx(0.4)
    assert len(st.layouts) == 2, "the unreadable key must be dropped, not crash"
    got = st.layout_for((2276, 1280), (999, 0, 1234))
    assert got is not None and got.box is None, "an unattributed layout still serves its size"
    assert [s.cy for s in got.slots] == [247, 358]
    assert st.layout_for((1920, 1080), (0, 0, 1920)) is None


def test_a_proved_layout_wins_over_the_migrated_one(cfg):
    st = Settings.from_dict({"spots_cache": {"2276x1280": [{"cx": 9, "cy": 9, "radius": 9}]}})
    st.remember_layout((2276, 1280), (0, 0, 2276), [(128, 247, 27)])
    assert st.layout_for((2276, 1280), (0, 0, 2276)).slots[0].cx == 128


def test_a_failed_write_leaves_the_old_config_whole(cfg, monkeypatch):
    """G14: saving used to move `config.json` out of the way before writing the new
    one, so anything that went wrong in between left no config at all - and the
    loader's answer to that is defaults, which is the user's whole setup."""
    Settings(volume=0.2, music_folder=r"E:\音乐").save(cfg)
    before = cfg.read_text(encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("nrdoom.settings.json.dumps", boom)
    with pytest.raises(OSError):
        Settings(volume=0.9).save(cfg)
    monkeypatch.undo()
    assert cfg.read_text(encoding="utf-8") == before
    assert not cfg.with_suffix(".json.tmp").exists()


def test_a_temporary_file_is_never_left_behind(cfg):
    Settings(volume=0.5).save(cfg)
    assert not cfg.with_suffix(".json.tmp").exists()
    Settings(volume=0.6).save(cfg)
    assert cfg.with_suffix(".json.bak").is_file()
    assert json.loads(cfg.with_suffix(".json.bak").read_text(encoding="utf-8"))["volume"] == 0.5


def test_target_fps_drives_the_slot_read(cfg):
    st = Settings(slots=[Slot(1, 2, 3), Slot(4, 5, 6)])
    assert st.slot_tuples() == [(1, 2, 3), (4, 5, 6)]


def test_config_path_follows_the_exe_when_frozen(tmp_path, monkeypatch):
    """A onefile bundle unpacks into a temp dir; writing there loses the calibration."""
    import sys

    from nrdoom import settings as mod

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "宁姆韦德的雨.exe"))
    assert mod._default_path() == tmp_path / "config.json"


def test_config_path_is_the_project_dir_when_running_from_source(tmp_path, monkeypatch):
    import sys

    from nrdoom import settings as mod

    monkeypatch.delattr(sys, "frozen", raising=False)
    # The invariant is "next to the code", not a folder name - a checkout under any
    # other name is still a source run.
    assert mod._default_path() == Path(mod.__file__).resolve().parents[1] / "config.json"


def test_the_stop_hold_default_is_a_second_and_a_half():
    """Cutting the song off is the most visible thing the tool can get wrong, so a
    misread has to survive well over a second of it before it counts."""
    assert Settings().revive_hold_frames == 30


def test_a_config_carrying_the_old_shipped_default_is_raised(cfg):
    """5 frames was never something to choose - it was what every config was
    written with - so it must not pin the new default away from existing users."""
    st = Settings.from_dict({"revive_hold_frames": 5})
    assert st.revive_hold_frames == 30


def test_a_stop_hold_the_user_chose_is_left_alone(cfg):
    st = Settings.from_dict({"revive_hold_frames": 12})
    assert st.revive_hold_frames == 12
