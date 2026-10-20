from nrdoom import selfcheck


def test_selfcheck_reports_every_dependency_available(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(selfcheck, "app_dir", lambda: tmp_path)
    assert selfcheck.run() == 0
    text = (tmp_path / "自检.txt").read_text(encoding="utf-8")
    assert "FAIL" not in text, text
    for label in ("windows_capture", "mss", "numpy", "QtMultimedia",
                  "background.jpg", "icon.ico", "selftest_panel.npz", "detection",
                  "temporal"):
        assert label in text, label
    assert "cv2" not in text


def test_selfcheck_temporal_is_a_real_result(tmp_path, monkeypatch):
    """The line has to depend on the stop actually happening, or it proves nothing
    about the state machine in the bundle."""
    from nrdoom.engine import Engine

    monkeypatch.setattr(selfcheck, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(Engine, "_song_ended", lambda self, obs: None)
    selfcheck.run()
    text = (tmp_path / "自检.txt").read_text(encoding="utf-8")
    assert "FAIL  temporal" in text, text
    assert "play 后接 stop" in text, text


def test_selfcheck_actually_runs_the_detector(tmp_path, monkeypatch):
    """The detection line must be a real result, not a hardcoded 'OK'."""
    from nrdoom import detector

    monkeypatch.setattr(selfcheck, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(detector, "propose_rings", lambda frame: [])
    selfcheck.run()
    text = (tmp_path / "自检.txt").read_text(encoding="utf-8")
    assert "FAIL  detection" in text, text


def test_selfcheck_actually_counts_the_rows(tmp_path, monkeypatch):
    from nrdoom import detector

    monkeypatch.setattr(selfcheck, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(detector, "count_rows", lambda frame, rect=None, spots=None: 1)
    selfcheck.run()
    text = (tmp_path / "自检.txt").read_text(encoding="utf-8")
    assert "FAIL  row count" in text, text
    assert "OK    row count" not in text


def test_selfcheck_flags_a_missing_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(selfcheck, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(selfcheck, "asset_root", lambda: tmp_path / "nothing")
    selfcheck.run()
    text = (tmp_path / "自检.txt").read_text(encoding="utf-8")
    assert "FAIL" in text


def test_selfcheck_proves_the_fixed_spots_read(tmp_path, monkeypatch):
    """The shipped path reads at hardcoded positions, so the bundle has to prove
    exactly that - not just that a search could find the gauges."""
    from nrdoom import detector

    monkeypatch.setattr(selfcheck, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(detector, "gauge_spots",
                        lambda shape, n=2, rect=None: [(0, 0, 1)] * n)
    selfcheck.run()
    text = (tmp_path / "自检.txt").read_text(encoding="utf-8")
    assert "FAIL  fixed spots" in text, text
