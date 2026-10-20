from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

from fakes import FakePlayer, FakeSource
from nrdoom.gui import GaugeBadge, MainWindow, PreviewWidget
from nrdoom.gui.preview import bgr_to_qimage
from nrdoom.settings import Settings, Slot

FIX = Path(__file__).resolve().parents[1] / "fixtures"
from shots import shot
SLOTS = [Slot(128, 247, 27), Slot(128, 358, 27)]


@pytest.fixture
def music(tmp_path):
    folder = tmp_path / "music"
    folder.mkdir()
    for name in ("one.mp3", "two.ogg"):
        (folder / name).write_bytes(b"x")
    return folder


@pytest.fixture
def window(qapp, tmp_path, music, monkeypatch):
    monkeypatch.setattr("nrdoom.gui.main_window.DEFAULT_PATH", tmp_path / "config.json")
    st = Settings(music_folder=str(music), slots=list(SLOTS), hold_frames=1,
                  show_pet=False)
    win = MainWindow(st)
    # The real FrameSource is never started in tests, so arming would refuse on
    # "not connected" before it ever got to the thing under test.
    win.engine.source = FakeSource(golden())
    win.resize(1100, 700)
    win.show()
    qapp.processEvents()
    yield win
    win.close()
    win.deleteLater()
    qapp.processEvents()


def golden():
    return shot("golden_both_3bars.jpg")


def feed(win, img):
    win.engine.source.set_frame(img)
    win.engine._last_frame_id = -1
    win.engine._tick()


def test_window_builds_and_persists_nothing_unprompted(window, tmp_path):
    assert window.isVisible()
    assert not (tmp_path / "config.json").exists() or window.settings.slots


def test_track_list_mirrors_the_folder(window):
    names = {window.track_list.item(i).text() for i in range(window.track_list.count())}
    assert names == {"one.mp3", "two.ogg"}
    assert window.track_count.text() == "2 首"


def test_readings_reach_the_labels_and_badges(window):
    feed(window, golden())
    assert "3 格" in window.slot_labels[0].text()
    assert "3 格" in window.slot_labels[1].text()
    assert "置信度" in window.slot_labels[0].text()


def test_the_row_of_an_ally_who_left_says_so(window):
    window.engine.party.hold_frames = 1
    window.arm_btn.click()
    feed(window, shot("synth_rows_duo.png"))
    assert "离场" in window.slot_labels[1].text()
    assert "3 格" in window.slot_labels[0].text()


def test_arm_button_stays_off_without_a_picture(window):
    """Nothing else can stop it: there is no calibration to do first."""
    window.engine.source.stop()
    window.arm_btn.click()
    assert window.arm_btn.isChecked() is False
    assert window.engine.listening is False
    assert "窗口" in window.statusBar().currentMessage()


def test_arming_works_once_calibrated(window):
    window.arm_btn.click()
    assert window.engine.listening is True
    assert window.arm_btn.text() == "停止监听"


def test_firing_updates_the_counter_and_the_status_line(window):
    window.arm_btn.click()
    feed(window, golden())
    assert window.engine.trigger.fires == 1
    assert "已播放" in window.statusBar().currentMessage()
    assert "已触发: 1 次" in window.trigger_label.text()


def test_the_spots_appear_without_any_click(window):
    window.settings.slots = []
    feed(window, golden())
    assert [(s.cx, s.cy) for s in window.settings.slots] ==                 [pytest.approx((128, 247), abs=1), pytest.approx((128, 358), abs=1)]
    assert "3 格" in window.slot_labels[0].text()


def test_the_labels_say_why_a_playing_song_has_not_stopped(window):
    """§11 Phase 2 asked for more than bars and confidence: the reason is the thing
    a player wants while the track is still running - and building it costs a crop
    per unread slot, so it is shown only then."""
    window.arm_btn.click()
    feed(window, golden())                      # this frame fires
    feed(window, golden())                      # this one the stop machine judged
    assert "倒地" in window.slot_labels[0].text()
    window.engine.set_paused(True)
    feed(window, golden())
    assert "倒地" not in window.slot_labels[0].text()


def test_add_button_imports_the_picked_files(window, tmp_path, monkeypatch):
    """Dragging onto a list is not something you can be told to do; the button has
    to reach the same import path."""
    from PySide6.QtWidgets import QFileDialog

    outside = tmp_path / "downloads"
    outside.mkdir()
    for name in ("new.mp3", "other.ogg"):
        (outside / name).write_bytes(b"x")
    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: ([str(outside / "new.mp3"),
                                                       str(outside / "other.ogg")],
                                                      "")))
    window.add_btn.click()
    shown = [window.track_list.item(i).text()
             for i in range(window.track_list.count())]
    assert "new.mp3" in shown and "other.ogg" in shown
    assert (window._music_dir() / "new.mp3").is_file()


def test_the_order_buttons_rearrange_the_playlist(window, tmp_path):
    music = tmp_path / "songs"
    music.mkdir()
    for name in ("a.mp3", "b.mp3", "c.mp3"):
        (music / name).write_bytes(b"x")
    window.folder_edit.setText(str(music))         # the text box is the source of truth
    window._on_imported()
    window.order_combo.setCurrentIndex(1)          # 按列表顺序
    window.track_list.setCurrentRow(2)             # c.mp3
    window.up_btn.click()

    shown = [window.track_list.item(i).text()
             for i in range(window.track_list.count())]
    assert shown == ["a.mp3", "c.mp3", "b.mp3"], shown
    assert window.settings.track_order == shown
    played = [window.library.pick().name for _ in range(3)]
    assert played == ["a.mp3", "c.mp3", "b.mp3"], played


def test_the_saved_order_survives_reopening(qapp, tmp_path, music, monkeypatch):
    monkeypatch.setattr("nrdoom.gui.main_window.DEFAULT_PATH", tmp_path / "config.json")
    win = MainWindow(Settings(music_folder=str(music), slots=list(SLOTS),
                              play_order="ordered", show_pet=False))
    try:
        win.track_list.setCurrentRow(1)
        win.up_btn.click()
        wanted = [win.track_list.item(i).text()
                  for i in range(win.track_list.count())]
        assert wanted == ["two.ogg", "one.mp3"], wanted
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()

    again = MainWindow(Settings.load(tmp_path / "config.json"))
    try:
        assert again.settings.play_order == "ordered"
        assert again.library.order == "ordered"
        assert [again.library.pick().name for _ in range(3)] == \
            ["two.ogg", "one.mp3", "two.ogg"]
    finally:
        again.close()
        again.deleteLater()
        qapp.processEvents()


def test_the_order_combo_reaches_the_library(window):
    window.order_combo.setCurrentIndex(1)
    assert window.library.order == "ordered"
    assert window.settings.play_order == "ordered"
    window.order_combo.setCurrentIndex(0)
    assert window.library.order == "shuffle"


def test_pet_can_be_shown_and_hidden(window):
    window.pet_check.setChecked(True)
    assert window.pet is not None and window.pet.isVisible()
    window.pet_check.setChecked(False)
    assert window.pet is None


def test_closing_the_pet_by_itself_leaves_the_window_usable(window, qapp):
    """The pet deletes its own C++ object on close, so Alt+F4 on it - or a closer
    that hits it as well as the main window - must leave the window pointing at
    nothing. Otherwise every later touch raises, closing the window included."""
    window.pet_check.setChecked(True)
    window.pet.close()
    qapp.processEvents()
    assert window.pet is None
    window._on_readings([])
    window._on_party(1)
    window.close()
    qapp.processEvents()


def test_a_dismissed_pet_stays_dismissed(window, qapp):
    """... and it must not come back by itself. `_rebuild_pet()` runs whenever the
    ally count or the calibration changes, and it creates a pet whenever the box is
    ticked - so dismissing the window had to uncheck the box, or the user's Alt+F4
    would be undone by the next event."""
    window.pet_check.setChecked(True)
    window.pet.close()
    qapp.processEvents()
    window._rebuild_pet()
    qapp.processEvents()
    assert window.pet is None, "the pet resurrected itself"
    assert window.pet_check.isChecked() is False
    assert window.settings.show_pet is False, "the dismissal must survive a restart"


def test_quitting_does_not_turn_the_pet_off(qapp, tmp_path, music, monkeypatch):
    """Quitting is not a pet preference.

    `closeEvent` disconnects the engine, which announces a party of zero, which
    rebuilds the pet and so destroys the old one - and the destroyed handler read
    that as "the user dismissed the pet" and unchecked the box. Measured: the
    packaged build came back with no pet on the next launch after a clean quit.
    """
    monkeypatch.setattr("nrdoom.gui.main_window.DEFAULT_PATH", tmp_path / "config.json")
    win = MainWindow(Settings(music_folder=str(music), slots=list(SLOTS), show_pet=True))
    win.engine.source = FakeSource(golden())
    assert win.pet is not None
    try:
        win.close()
        qapp.processEvents()
    finally:
        win.deleteLater()
        qapp.processEvents()
    assert win.settings.show_pet is True
    assert Settings.load(tmp_path / "config.json").show_pet is True


def test_saved_pet_preference_is_restored_on_startup(qapp, tmp_path, music, monkeypatch):
    """The checkbox is set before signals are wired, so this needs an explicit sync."""
    monkeypatch.setattr("nrdoom.gui.main_window.DEFAULT_PATH", tmp_path / "config.json")
    win = MainWindow(Settings(music_folder=str(music), slots=list(SLOTS), show_pet=True))
    try:
        assert win.pet is not None
    finally:
        win.pet.close()
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_the_pet_follows_the_live_ally_count(window):
    window.engine.party.hold_frames = 1
    window.pet_check.setChecked(True)
    feed(window, golden())
    assert len(window.pet.badges) == 2
    first = window.pet
    feed(window, shot("synth_rows_duo.png"))
    assert window.pet is not first, "rebuilt for one badge"
    assert len(window.pet.badges) == 1


def test_volume_control_reaches_the_player(window):
    window.volume.setValue(0.25)
    assert window.player.volume() == pytest.approx(0.25)


# -- duo mode in the UI -------------------------------------------------------
def test_arming_needs_no_stored_geometry(qapp, tmp_path, music, monkeypatch):
    """A first launch has never seen a frame, so it has no spots at all."""
    monkeypatch.setattr("nrdoom.gui.main_window.DEFAULT_PATH", tmp_path / "config.json")
    win = MainWindow(Settings(music_folder=str(music), hold_frames=1, show_pet=False))
    win.engine.source = FakeSource(golden())
    try:
        win.arm_btn.click()
        assert win.engine.listening is True
        win.engine._tick()
        assert win.engine.party.count == 2
        assert win.slot_labels[0].text().count("格") == 1
    finally:
        win.pet and win.pet.close()
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_a_duo_run_gets_a_single_badge_pet(qapp, tmp_path, music, monkeypatch):
    monkeypatch.setattr("nrdoom.gui.main_window.DEFAULT_PATH", tmp_path / "config.json")
    win = MainWindow(Settings(music_folder=str(music), show_pet=True,
                              party_hold_frames=1, hold_frames=1))
    win.engine.source = FakeSource(shot("synth_rows_duo.png"))
    try:
        win.arm_btn.click()
        assert len(win.pet.badges) == 1
    finally:
        win.pet.close()
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_settings_written_on_a_deliberate_change(window, tmp_path):
    window.bars_spin.setValue(2)
    saved = (tmp_path / "config.json").read_text(encoding="utf-8")
    assert '"required_bars": 2' in saved


# -- preview widget plumbing ---------------------------------------------------
def test_bgr_to_qimage_keeps_pixel_values(qapp):
    arr = np.zeros((4, 6, 3), dtype=np.uint8)
    arr[1, 2] = (10, 20, 30)
    img = bgr_to_qimage(arr)
    assert (img.width(), img.height()) == (6, 4)
    assert img.pixelColor(2, 1).red() == 30
    assert img.pixelColor(2, 1).green() == 20
    assert img.pixelColor(2, 1).blue() == 10


def test_bgr_to_qimage_keeps_pixel_values_for_bgra(qapp):
    """The capture path now hands over BGRA, so the mapping must not shift channels."""
    arr = np.zeros((4, 6, 4), dtype=np.uint8)
    arr[1, 2] = (10, 20, 30, 255)
    img = bgr_to_qimage(arr)
    c = img.pixelColor(2, 1)
    assert (c.red(), c.green(), c.blue()) == (30, 20, 10)


def test_coordinate_mapping_round_trips(qapp):
    view = PreviewWidget()
    view.resize(800, 450)
    view.set_frame(np.zeros((1280, 2276, 3), dtype=np.uint8))
    view.show()
    qapp.processEvents()
    x, y = view.to_frame(view.to_widget(128, 247))
    assert x == pytest.approx(128, abs=1.5)
    assert y == pytest.approx(247, abs=1.5)


def test_preview_paints_without_a_frame(qapp):
    view = PreviewWidget()
    view.resize(640, 360)
    pm = QPixmap(view.size())
    view.render(pm)
    assert not pm.isNull()


def test_gauge_badge_paints_every_state(qapp):
    for bars in (None, 0, 1, 2, 3):
        badge = GaugeBadge(0)
        badge.set_state(bars)
        pm = QPixmap(40, 40)
        badge.render(pm)
        assert not pm.isNull()


def test_gauge_badge_distinguishes_bar_counts(qapp):
    """Count the blue ink, not brightness: the widget paints an opaque background."""
    def blue_ink(bars):
        badge = GaugeBadge(0, side=60)
        badge.set_state(bars)
        img = QImage(60, 60, QImage.Format.Format_BGR888)
        img.fill(Qt.GlobalColor.black)
        badge.render(img)
        hits = 0
        for y in range(0, 60, 2):
            for x in range(0, 60, 2):
                c = img.pixelColor(x, y)
                if c.blue() > 150 and c.blue() - c.red() > 40 and c.blue() - c.green() > 60:
                    hits += 1
        return hits

    assert blue_ink(0) == 0
    assert blue_ink(1) < blue_ink(2) < blue_ink(3)


def test_qimage_survives_its_source_buffer_being_freed(qapp):
    """Regression: QImage(data=...) does not own the memory.

    The first version of bgr_to_qimage called detach() on the assumption that
    copied, which it does not for an externally-owned buffer, so the preview was
    reading freed memory a frame later.
    """
    import gc

    arr = np.full((200, 300, 3), 7, dtype=np.uint8)
    arr[50, 50] = (1, 2, 3)
    img = bgr_to_qimage(arr)
    del arr
    gc.collect()
    assert img.pixelColor(50, 50).red() == 3
    assert img.pixelColor(50, 50).green() == 2
    assert img.pixelColor(50, 50).blue() == 1


def test_an_ally_who_left_is_shown_as_gone(window):
    window.engine.party.hold_frames = 1
    window.arm_btn.click()
    feed(window, golden())
    assert "队友数: 2" in window.party_label.text()

    feed(window, shot("synth_rows_duo.png"))
    assert "队友数: 1" in window.party_label.text()
    assert "3 格" in window.slot_labels[0].text()
    assert "离场" in window.slot_labels[1].text()
    assert window.engine.trigger.allies == 1


def test_solo_says_there_is_no_teammate(window):
    """Armed into a solo run: nothing was ever counted, so the panel says so.

    This is deliberately not the same path as "my allies left" - a row count of
    zero is also what a blank panel reads like, and that must not shrink the
    roster (see test_party's reading-nothing rule).
    """
    window.engine.party.hold_frames = 1
    window.arm_btn.click()
    feed(window, shot("synth_rows_solo.png"))
    assert "没有队友" in window.party_label.text()
    assert "没有队友" in window.slot_labels[0].text()
    assert window.engine.readings == []


def test_slots_still_read_while_disarmed(window):
    """Calibration needs to see what a position reports before committing to it."""
    feed(window, golden())
    assert "3 格" in window.slot_labels[0].text()
    assert "3 格" in window.slot_labels[1].text()


# -- dropping files on the list ------------------------------------------------
def drop(win, *paths):
    from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
    from PySide6.QtGui import QDropEvent

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    win.track_list.dropEvent(QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, mime,
                                        Qt.MouseButton.LeftButton,
                                        Qt.KeyboardModifier.NoModifier))


def test_dropping_a_file_that_is_already_in_the_folder_does_not_crash(window, music):
    """Explorer lets you drop a file onto the folder it already sits in, and
    copying a file onto itself raises PermissionError on Windows."""
    inside = music / "雨爱.mp3"
    inside.write_bytes(b"abc")
    drop(window, inside)
    assert inside.read_bytes() == b"abc", "must be left alone, not clobbered"
    assert "雨爱.mp3" in {window.track_list.item(i).text()
                        for i in range(window.track_list.count())}


def test_dropping_a_name_clash_keeps_both_files(window, music):
    mine = music / "song.mp3"
    mine.write_bytes(b"existing")
    elsewhere = music.parent / "song.mp3"
    elsewhere.write_bytes(b"different")
    drop(window, elsewhere)
    assert mine.read_bytes() == b"existing", "the track already in the library survives"
    assert elsewhere.read_bytes() == b"different"
    assert (music / "song (1).mp3").read_bytes() == b"different"


def test_one_unreadable_file_does_not_stop_the_others(window, music):
    good = music.parent / "good.mp3"
    good.write_bytes(b"ok")
    drop(window, music.parent / "missing.mp3", good)
    names = {window.track_list.item(i).text() for i in range(window.track_list.count())}
    assert "good.mp3" in names
    assert "没拷进来" in window.statusBar().currentMessage()


def test_non_audio_files_are_ignored(window, music):
    txt = music.parent / "notes.txt"
    txt.write_text("hi", encoding="utf-8")
    drop(window, txt)
    assert not (music / "notes.txt").exists()


def test_choosing_a_folder_switches_the_library(window, tmp_path, monkeypatch):
    """The text box and the drop target both speak `folder_edit`, so the library
    has to follow that one and not the value the window was built with."""
    other = tmp_path / "别的曲库"
    other.mkdir()
    (other / "x.mp3").write_bytes(b"x")

    class Dialog:
        @staticmethod
        def getExistingDirectory(*_a, **_k):
            return str(other)

    monkeypatch.setattr("nrdoom.gui.main_window.QFileDialog", Dialog)
    window.pick_btn.click()
    assert window.track_count.text() == "1 首", window.track_count.text()
    assert Path(window.settings.music_folder) == other


def test_typing_a_folder_path_switches_the_library(window, tmp_path):
    other = tmp_path / "手打路径"
    other.mkdir()
    (other / "a.mp3").write_bytes(b"x")
    (other / "b.flac").write_bytes(b"x")
    window.folder_edit.setText(str(other))
    window.folder_edit.editingFinished.emit()
    assert window.track_count.text() == "2 首"
    assert Path(window.settings.music_folder) == other


# -- the settings strip has to stay clickable ----------------------------------
def _fields(win, qapp):
    """Every spin box and combo on every settings tab, laid out and visible."""
    from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox, QTabWidget

    tabs = win.findChild(QTabWidget)
    found = []
    for i in range(tabs.count()):
        tabs.setCurrentIndex(i)
        qapp.processEvents()
        page = tabs.widget(i)
        for cls in (QSpinBox, QDoubleSpinBox, QComboBox):
            found += [(page, w) for w in page.findChildren(cls) if w.isVisible()]
    return found


def test_every_setting_gets_the_height_it_asked_for(window, qapp):
    """A spin box that is 13px tall holds a number you can read and cannot click.

    The tab strip used to carry a maximumHeight below what the forms inside it
    needed, and Qt settled the difference by shaving each field's height - so the
    invariant is not "tall enough, roughly" but "as tall as the widget asked".
    """
    pairs = _fields(window, qapp)
    assert len(pairs) >= 8
    short = [("%s %dx%d want %d" % (type(w).__name__, w.width(), w.height(),
                                   w.sizeHint().height()))
             for _page, w in pairs if w.height() < w.sizeHint().height()]
    assert not short, short


def test_no_setting_runs_off_its_page(window, qapp):
    """The other way a squeezed strip shows up: rows past the bottom edge."""
    seen = set()
    for page, w in _fields(window, qapp):
        if page in seen:
            continue
        seen.add(page)
        page.layout().activate()
        bottom = page.mapFrom(w.parentWidget(), w.pos() + w.rect().bottomLeft()).y()
        assert bottom <= page.height(), (page.objectName(), bottom, page.height())


def test_the_settings_strip_cannot_be_capped_below_its_own_height(window, qapp):
    """A ceiling under the height the pages ask for is what caused the 13px boxes."""
    from PySide6.QtWidgets import QTabWidget

    tabs = window.findChild(QTabWidget)
    want = tabs.sizeHint().height()
    assert want > 0
    assert tabs.maximumHeight() >= want, "%dpx ceiling for %dpx of forms" % (
        tabs.maximumHeight(), want)
    assert tabs.minimumHeight() >= want


# -- 恢复默认 ------------------------------------------------------------------
def mess_up(win):
    """The exact accident the button exists for: numbers nudged past what works."""
    win.hold_spin.setValue(120)
    win.bars_spin.setValue(1)
    win.fps_spin.setValue(5)
    win.loop_check.setChecked(False)
    win.policy_combo.setCurrentIndex(1)


def agree_to_reset(monkeypatch, yes=True):
    from PySide6.QtWidgets import QMessageBox

    answer = (QMessageBox.StandardButton.Yes if yes
              else QMessageBox.StandardButton.No)
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: answer))


def test_restore_defaults_puts_the_rules_page_back(window, tmp_path, monkeypatch):
    from nrdoom.gui.main_window import RULE_FIELDS
    from nrdoom.settings import Settings

    agree_to_reset(monkeypatch)
    mess_up(window)
    assert window.settings.hold_frames == 120, "the mess has to be real first"
    window.reset_btn.click()
    fresh = Settings()
    for name in RULE_FIELDS:
        assert getattr(window.settings, name) == getattr(fresh, name), name
    # the page has to show it too, not just the object behind it
    assert window.hold_spin.value() == fresh.hold_frames
    assert window.policy_combo.currentIndex() == 0
    assert window.loop_check.isChecked()
    # and it survives a restart
    assert Settings.load(tmp_path / "config.json").hold_frames == fresh.hold_frames


def test_restore_defaults_uses_the_shipped_number_not_what_the_window_started_with(
        window, monkeypatch):
    """The fixture opens with hold_frames=1; 'default' means 8, not 'undo my click'."""
    agree_to_reset(monkeypatch)
    assert window.settings.hold_frames == 1
    window.reset_btn.click()
    assert window.hold_spin.value() == 8


def test_restore_defaults_leaves_the_music_and_the_calibration_alone(window,
                                                                    monkeypatch):
    from nrdoom.settings import Slot

    agree_to_reset(monkeypatch)
    window.settings.music_folder = "D:/我的歌"
    window.settings.track_order = ["b.mp3", "a.mp3"]
    window.settings.volume = 0.25
    window.settings.slots = [Slot(1, 2, 3)]
    window.settings.spots_cache = {"2560x1440": [Slot(4, 5, 6)]}
    mess_up(window)
    window.reset_btn.click()
    assert window.settings.music_folder == "D:/我的歌"
    assert window.settings.track_order == ["b.mp3", "a.mp3"]
    assert window.settings.volume == 0.25
    assert window.settings.slot_tuples() == [(1, 2, 3)]
    assert list(window.settings.spots_cache) == ["2560x1440"]


def test_restore_defaults_asks_before_touching_anything(window, monkeypatch):
    agree_to_reset(monkeypatch, yes=False)
    mess_up(window)
    window.reset_btn.click()
    assert window.settings.hold_frames == 120
    assert window.settings.target_fps == 5


def test_the_reset_list_cannot_reach_the_fields_it_promises_to_keep():
    from nrdoom.gui.main_window import RULE_FIELDS

    assert not {"music_folder", "track_order", "volume", "slots", "spots_cache",
                "window_title", "play_order"} & set(RULE_FIELDS)


# -- 悬浮窗不能被自己的重建流程关掉 --------------------------------------------------
def test_rebuilding_the_pet_is_not_the_user_closing_it(window, qapp):
    """The pet is closed and rebuilt whenever the slot count or the calibration moves.

    That close landed in the same handler as Alt+F4, which reads it as "the user
    dismissed it" and unchecks 显示游戏内悬浮窗 - so the rebuild saw an unchecked box
    and the pet was gone for the rest of the session, with `show_pet` saved off.
    """
    window.pet_check.setChecked(True)
    qapp.processEvents()
    assert window.pet is not None
    for _ in range(3):
        window._push_slots()          # what a party change or an adopted drift does
        qapp.processEvents()
    assert window.pet is not None, "the pet closed itself on the second rebuild"
    assert window.pet.isVisible(), "the pet exists but nothing is on screen"
    assert window.pet_check.isChecked()
    assert window.settings.show_pet is True


def test_a_pet_the_user_really_closed_stays_closed(window, qapp):
    """The other half: dismissing it by hand is a decision, not a state to undo."""
    window.pet_check.setChecked(True)
    qapp.processEvents()
    window.pet.close()
    qapp.processEvents()
    assert not window.pet_check.isChecked()
    window._push_slots()
    qapp.processEvents()
    assert window.pet is None, "it came back after the user closed it"


def test_hiding_the_pet_from_its_menu_is_a_decision(window, qapp):
    """The right-click 隐藏悬浮窗 must survive the next rebuild, not just look hidden."""
    window.pet_check.setChecked(True)
    qapp.processEvents()
    window.pet.hideRequested.emit()
    qapp.processEvents()
    assert window.pet is None and not window.pet_check.isChecked()
    window._push_slots()
    qapp.processEvents()
    assert window.pet is None, "a rebuild brought back a pet the user hid"


def test_the_readout_follows_a_track_that_advanced_on_its_own(window, qapp):
    """A song that ends mid-run starts the next one without re-firing the trigger,
    so the window has to learn the new name from the player."""
    window.player.track_started.emit("第二首.mp3")
    qapp.processEvents()
    assert "第二首.mp3" in window.statusBar().currentMessage()
