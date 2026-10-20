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
    assert st.spots_cache == {}


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


def test_spots_are_remembered_per_resolution(cfg):
    """The in-game UI size moves the panel, so the geometry belongs to the
    resolution it was found at. Losing this costs the user a re-derivation every
    launch, and a stale one from another resolution reads nothing."""
    st = Settings()
    st.apply_spots("2276x1280", [(128, 247, 27), (128, 358, 27)])
    st.apply_spots("1366x768", [(77, 148, 14.5), (77, 215, 14.3)])
    st.save(cfg)
    back = Settings.load(cfg)
    assert [s.cy for s in back.spots_cache["2276x1280"]] == [247, 358]
    assert [s.cy for s in back.spots_cache["1366x768"]] == [148, 215]
    assert back.slot_tuples() == st.slot_tuples()


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
