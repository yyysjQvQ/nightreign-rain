"""The exe is windowed: an exception in a slot has to leave something behind."""
import sys

import main as entry


def test_a_slot_exception_is_written_to_the_crash_log(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(entry, "_log_handle", None)
    monkeypatch.setattr(sys, "__excepthook__", lambda *a: None)
    try:
        raise RuntimeError("boom in a slot")
    except RuntimeError as exc:
        entry._excepthook(RuntimeError, exc, exc.__traceback__)
    text = (tmp_path / entry.CRASH_LOG).read_text(encoding="utf-8")
    assert "运行期" in text and "boom in a slot" in text


def test_a_startup_failure_is_written_to_the_same_file(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(entry, "_log_handle", None)
    entry._log_crash(ValueError("no music folder"), "启动")
    text = (tmp_path / entry.CRASH_LOG).read_text(encoding="utf-8")
    assert "启动" in text and "no music folder" in text


def test_the_log_survives_a_directory_that_cannot_be_written(tmp_path, monkeypatch):
    """A diagnostic that raises on the way to reporting a crash is worse than none."""
    monkeypatch.setattr(entry, "app_dir", lambda: tmp_path / "not" / "there")
    monkeypatch.setattr(entry, "_log_handle", None)
    entry._log_crash(RuntimeError("still fine"))          # must not raise
