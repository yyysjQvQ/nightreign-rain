"""The control window."""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont, QIcon, QPainter
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
    QPushButton, QSpinBox, QSizePolicy, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from ..assets import APP_NAME, app_dir, asset_root
from ..audio import MusicPlayer
from ..capture import list_windows
from ..diag import Diagnostics
from ..engine import Engine
from ..hud_tracker import LOCKED
from ..library import SUPPORTED, MusicLibrary
from ..settings import DEFAULT_PATH, Settings, Slot
from ..slot_observation import STRONG_ALIVE, STRONG_DOWN
from .. import __version__
from . import theme
from .pet import PetWidget
from .preview import GaugeBadge, PreviewWidget


def _free_name(dst: Path) -> Path:
    """`song.mp3` -> `song (1).mp3` when the name is taken by another file, so a
    drop never replaces a track that may be playing right now."""
    if not dst.exists():
        return dst
    for i in range(1, 100):
        cand = dst.with_name(f"{dst.stem} ({i}){dst.suffix}")
        if not cand.exists():
            return cand
    return dst


# What "恢复默认" is allowed to touch: exactly the fields the rules page shows.
# The music folder, the playlist order, the volume and the calibrated spots are
# things the user picked or the tracker earned, and a button that quietly threw
# them away would be a far worse accident than a mistyped frame count.
RULE_FIELDS = ("required_bars", "hold_frames", "stop_policy", "revive_min_evidence_s",
               "revive_hold_frames", "cooldown_s", "target_fps", "loop", "show_pet",
               "diagnostics")


class TrackList(QListWidget):
    """Dropping audio files on the list copies them into the music folder."""

    def __init__(self, folder_getter, on_import, parent=None):
        super().__init__(parent)
        self._folder = folder_getter
        self._on_import = on_import
        self.import_note = ""
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        self.import_paths([Path(url.toLocalFile()) for url in event.mimeData().urls()])

    def import_paths(self, sources) -> None:
        """Copy the given files into the music folder, then re-scan.

        Shared by the drop target and the 添加音乐 button, so the two cannot drift
        apart on what counts as imported - the quiet case being a file that is
        already in the library, which raises on Windows if it is copied onto itself.
        """
        folder = Path(self._folder())
        folder.mkdir(parents=True, exist_ok=True)
        copied = skipped = failed = 0
        for src in sources:
            if src.suffix.lower() not in SUPPORTED:
                continue                      # not our business
            if not src.is_file():
                failed += 1                   # they meant to add it, but it is gone
                continue
            try:
                if src.parent.resolve() == folder.resolve():
                    skipped += 1              # already in the library; copying a
                else:                         # file onto itself raises on Windows
                    shutil.copy2(src, _free_name(folder / src.name))
                    copied += 1
            except OSError:
                failed += 1
        bits = []
        if copied:
            bits.append(f"导入 {copied} 首")
        if failed:
            bits.append(f"{failed} 首没拷进来")
        if skipped:
            bits.append(f"{skipped} 首本来就在曲库里")
        self.import_note = "，".join(bits)
        self._on_import()


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or Settings.load()
        self.setWindowTitle(APP_NAME)
        icon = asset_root() / "icon.png"
        if icon.is_file():
            self.setWindowIcon(QIcon(str(icon)))
        # 880 tall: the layout's own floor is ~710 (source bar + preview minimum +
        # the settings strip at full height), and a window that only just clears the
        # floor leaves nothing for the preview to grow into.
        self.resize(1180, 880)

        self.library = MusicLibrary(self._music_dir(), order=self.settings.play_order)
        self.library.set_sequence(self.settings.track_order)
        self.library.refresh()
        self.player = MusicPlayer(self.library)
        self.diag = self._open_diag()
        self.engine = Engine(self.settings, player=self.player, diag=self.diag)

        self.pet: PetWidget | None = None
        self._pet_pos = (60, 60)
        # Quitting destroys the pet as part of shutting down, which is not the user
        # saying "no pet" - see _on_pet_destroyed.
        self._shutting_down = False
        self._connected = False
        self._frames_seen = 0
        self._fps_clock = time.monotonic()

        self._build_ui()
        self._wire()
        self._refresh_windows()
        self._reload_tracks()
        # The checkbox was set during build, before the signal was connected, so a
        # saved "show pet" preference needs applying by hand.
        if self.pet_check.isChecked():
            self._make_pet()
        self.statusBar().showMessage("就绪。先连游戏窗口，再标定，最后开始监听。")

    # -- layout ---------------------------------------------------------------
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        outer = QVBoxLayout(root)
        outer.addWidget(self._source_bar())

        split = QSplitter(Qt.Orientation.Horizontal)
        self.preview = PreviewWidget()
        split.addWidget(self.preview)
        split.addWidget(self._side_panel())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([820, 330])
        # 3:1, not 1:0. With all the surplus going to the splitter the tab strip
        # stayed below its own minimum and Qt paid for it by shaving every spin
        # box's height - the arrows become impossible to hit.
        outer.addWidget(split, 3)
        outer.addWidget(self._tabs(), 1)
        self.setCentralWidget(root)

    def paintEvent(self, event):
        """The artwork sits behind everything; see theme.paint_background."""
        painter = QPainter(self)
        theme.paint_background(self, painter)
        painter.end()
        super().paintEvent(event)

    def _source_bar(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("游戏窗口"))
        self.window_combo = QComboBox()
        self.window_combo.setMinimumWidth(340)
        row.addWidget(self.window_combo, 1)
        self.refresh_btn = QPushButton("刷新")
        self.connect_btn = QPushButton("连接")
        self.connect_btn.setCheckable(True)
        row.addWidget(self.refresh_btn)
        row.addWidget(self.connect_btn)
        self.feed_label = QLabel("后端: 未连接")
        row.addWidget(self.feed_label)
        return bar

    def _side_panel(self) -> QWidget:
        panel = QWidget()
        box = QVBoxLayout(panel)

        read_group = QGroupBox("读数")
        read_lay = QVBoxLayout(read_group)
        self.badges: list[GaugeBadge] = []
        self.slot_labels: list[QLabel] = []
        for i in range(4):
            row = QHBoxLayout()
            badge = GaugeBadge(0, side=30)
            label = QLabel(f"槽位 {i + 1}: 等待画面")
            label.setFont(QFont("Microsoft YaHei", 10))
            row.addWidget(badge)
            row.addWidget(label, 1)
            read_lay.addLayout(row)
            self.badges.append(badge)
            self.slot_labels.append(label)
        self.party_label = QLabel("队友数: 等待监听")
        read_lay.addWidget(self.party_label)
        self.trigger_label = QLabel("已触发: 0 次")
        read_lay.addWidget(self.trigger_label)
        box.addWidget(read_group)

        self.arm_btn = QPushButton("开始监听")
        self.arm_btn.setCheckable(True)
        self.arm_btn.setObjectName("arm")
        box.addWidget(self.arm_btn)

        row = QHBoxLayout()
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setToolTip("暂停会停掉当前曲子，并停止自动播放，直到再次点击")
        self.pause_btn.setEnabled(False)
        self.test_btn = QPushButton("试听")
        self.test_btn.setToolTip("手动放一首，不走检测逻辑")
        row.addWidget(self.pause_btn)
        row.addWidget(self.test_btn)
        box.addLayout(row)
        box.addStretch(1)
        return panel

    def _tabs(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._music_tab(), "音乐")
        tabs.addTab(self._rules_tab(), "规则与高级")
        # This used to carry `setMaximumHeight(230)`, and the one-column form inside
        # it asked for 283, so Qt paid for the ceiling by shaving every spin box
        # down to 13px tall - readable, unhittable. The pages' own size is a floor.
        tabs.setMinimumHeight(tabs.sizeHint().height())
        tabs.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        return tabs

    def _music_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        row = QHBoxLayout()
        self.folder_edit = QLineEdit(str(self._music_dir()))
        row.addWidget(self.folder_edit, 1)
        self.pick_btn = QPushButton("选择文件夹…")
        self.add_btn = QPushButton("添加音乐…")
        self.rescan_btn = QPushButton("重新扫描")
        row.addWidget(self.pick_btn)
        row.addWidget(self.add_btn)
        row.addWidget(self.rescan_btn)
        lay.addLayout(row)

        body = QHBoxLayout()
        self.track_list = TrackList(lambda: self.folder_edit.text(), self._on_imported)
        self.track_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        body.addWidget(self.track_list, 1)

        side = QFormLayout()
        self.volume = QDoubleSpinBox()
        self.volume.setRange(0.0, 1.0)
        self.volume.setSingleStep(0.05)
        self.volume.setValue(self.settings.volume)
        self.track_count = QLabel("0 首")
        self.order_combo = QComboBox()
        self.order_combo.addItem("随机播放（每首先各一次）", "shuffle")
        self.order_combo.addItem("按上面的列表顺序", "ordered")
        self.order_combo.setCurrentIndex(
            1 if self.settings.play_order == "ordered" else 0)
        move = QHBoxLayout()
        self.up_btn = QPushButton("↑ 上移")
        self.down_btn = QPushButton("↓ 下移")
        move.addWidget(self.up_btn)
        move.addWidget(self.down_btn)
        hint = QLabel("把 mp3 / ogg / flac / wav 拖进来，或点“添加音乐…”。\n"
                      "选“按上面的列表顺序”后，用 ↑↓ 排出你要的顺序，播完最后一首回到第一首。")
        hint.setWordWrap(True)
        side.addRow("音量", self.volume)
        side.addRow("曲库", self.track_count)
        side.addRow("播放顺序", self.order_combo)
        side.addRow(move)
        body.addLayout(side)
        lay.addLayout(body)
        # QFormLayout hands a wrapped label the width of its own column and then
        # ignores the height that width needs, so the hint goes under the whole
        # page instead — full width, where it can never be clipped mid-line.
        lay.addWidget(hint)
        return page

    def _rules_tab(self) -> QWidget:
        page = QWidget()
        # Eleven rows in one column want ~280px of height and the tab strip only
        # ever gets ~200, so Qt squeezed every spin box down to 13px tall - a value
        # you can read but not hit. Two columns halve the height.
        shell = QVBoxLayout(page)
        left = QFormLayout()
        right = QFormLayout()
        cols = QHBoxLayout()
        cols.addLayout(left)
        cols.addLayout(right)
        shell.addLayout(cols)
        self.bars_spin = QSpinBox()
        self.bars_spin.setRange(1, 3)
        self.bars_spin.setSuffix(" 格")
        self.hold_spin = QSpinBox()
        self.hold_spin.setRange(1, 120)
        self.hold_spin.setSuffix(" 帧")
        self.cooldown_spin = QDoubleSpinBox()
        self.cooldown_spin.setRange(0.0, 600.0)
        self.cooldown_spin.setSuffix(" 秒")
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(5, 60)
        self.fps_spin.setSuffix(" 帧/秒")
        self.pet_check = QCheckBox("显示游戏内悬浮窗")
        self.revive_spin = QSpinBox()
        self.revive_spin.setRange(1, 60)
        self.revive_spin.setSuffix(" 帧")
        self.revive_spin.setToolTip("旧版判据：队友的计量表消失、原位置又出现血条，连续多少帧才判定为已被救起"
                                    "（只是读不到表不会停）")
        self.policy_combo = QComboBox()
        # The form would otherwise stretch this one box across half the page: the
        # spin boxes next to it are capped, and a 420px combo runs into the column
        # to its right.
        self.policy_combo.setMaximumWidth(300)
        self.policy_combo.addItems(["按时间（默认）", "按帧数（旧版，回退用）"])
        self.policy_combo.setToolTip(
            "按时间：站立证据必须在采集时钟上连续站得住（默认 0.8 秒、至少 12 个不同画面、"
            "窗口内八成以上证据都站着、最近 0.5 秒内没看到倒地），"
            "所以卡住的画面、零散的闪一下、隔几分钟的碎片都攒不成一次停歌。\n"
            "按帧数：旧版行为，连续 N 帧判站立即停，留作回退开关。")
        self.revive_secs_spin = QDoubleSpinBox()
        self.revive_secs_spin.setRange(0.4, 5.0)
        self.revive_secs_spin.setSingleStep(0.1)
        self.revive_secs_spin.setSuffix(" 秒")
        self.revive_secs_spin.setToolTip("站立证据要连续保持多久才确认「已被救起」（越大越不容易误停，停得越晚）")
        self.loop_check = QCheckBox("循环播放，直到有队友被救起或你手动暂停")
        self.diag_check = QCheckBox("记录诊断（exe 旁的 diagnostics/，用来事后查为什么停歌）")
        self.diag_check.setToolTip("每次状态变化和读表没把握的帧写一行 events.jsonl，"
                                   "并把事件前后约 10 秒的左上角 HUD 小图存下来。"
                                   "默认关闭；开着会占一点磁盘和内存，不影响识别结果。")
        left.addRow("触发条件：所有队友都达到", self.bars_spin)
        left.addRow("连续成立才触发", self.hold_spin)
        left.addRow("停歌判据", self.policy_combo)
        left.addRow("站立证据保持", self.revive_secs_spin)
        left.addRow("站立证据保持（旧版）", self.revive_spin)
        right.addRow("再次触发间隔", self.cooldown_spin)
        right.addRow("检测帧率", self.fps_spin)
        right.addRow(self.loop_check)
        right.addRow(self.pet_check)
        right.addRow(self.diag_check)
        self._show_rules()

        bottom = QHBoxLayout()
        self.reset_btn = QPushButton("恢复默认")
        self.reset_btn.setToolTip("把这一页的数值和开关改回出厂设置。\n"
                                  "曲库、播放顺序、音量和标定位置都不会动。")
        bottom.addWidget(self.reset_btn)
        # The four things to reach for when the box on the preview is not where the
        # panel is. They are here rather than in a menu because the moment you want
        # them is the moment you have already lost confidence in the automatic path.
        self.redetect_btn = QPushButton("重新检测当前位置")
        self.redetect_btn.setToolTip("丢掉现在用的计量表位置，按当前画面重新找一遍。\n"
                                     "已验证过的布局缓存保留，下次还是它。")
        bottom.addWidget(self.redetect_btn)
        self.forget_btn = QPushButton("重置当前布局缓存")
        self.forget_btn.setToolTip("只忘记**当前这套画面**（这个分辨率和信箱黑边组合）记住的位置，\n"
                                   "其他分辨率、其他黑边组合、曲库和音量都不受影响。")
        bottom.addWidget(self.forget_btn)
        self.restore_btn = QPushButton("恢复上次可靠定位")
        self.restore_btn.setToolTip("退回到上一个被验证过的位置。位置被一次误读带走时用。")
        bottom.addWidget(self.restore_btn)
        self.export_btn = QPushButton("导出诊断包")
        self.export_btn.setToolTip("打包最近一次诊断记录：events.jsonl、HUD 小图、\n"
                                   "版本、配置摘要和抓屏分辨率。不含整屏录制，也不含你的音乐文件。")
        bottom.addWidget(self.export_btn)
        bottom.addStretch(1)
        shell.addLayout(bottom)
        return page

    def _show_rules(self):
        """One path from the settings onto the page, so a reset cannot drift.

        The signals have to stay blocked while the page is being filled in:
        `_rules_changed` reads every field at once, so the first widget that fired
        wrote the still-untouched fields - old values and all - straight back into
        the settings, and the rest of the list became no-ops.
        """
        widgets = (self.bars_spin, self.hold_spin, self.policy_combo,
                   self.revive_secs_spin, self.revive_spin, self.cooldown_spin,
                   self.fps_spin, self.loop_check, self.pet_check, self.diag_check)
        for w in widgets:
            w.blockSignals(True)
        try:
            st = self.settings
            self.bars_spin.setValue(st.required_bars)
            self.hold_spin.setValue(st.hold_frames)
            self.policy_combo.setCurrentIndex(0 if st.stop_policy == "v2" else 1)
            self.revive_secs_spin.setValue(st.revive_min_evidence_s)
            self.revive_spin.setValue(st.revive_hold_frames)
            self.cooldown_spin.setValue(st.cooldown_s)
            self.fps_spin.setValue(st.target_fps)
            self.loop_check.setChecked(st.loop)
            self.pet_check.setChecked(st.show_pet)
            self.diag_check.setChecked(st.diagnostics)
        finally:
            for w in widgets:
                w.blockSignals(False)

    def _reset_rules(self):
        fresh = Settings()
        changed = [name for name in RULE_FIELDS
                   if getattr(self.settings, name) != getattr(fresh, name)]
        if not changed:
            self.statusBar().showMessage("这一页已经是默认设置了。")
            return
        answer = QMessageBox.question(
            self, "恢复默认",
            "要把「规则与高级」这一页改回出厂设置吗？现在有 %d 项和默认值不同。\n\n"
            "曲库、播放顺序、音量和标定位置不会动。" % len(changed),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        for name in changed:
            setattr(self.settings, name, getattr(fresh, name))
        self._show_rules()
        # The page was filled with the signals held back, so run each effect once
        # here instead of once per field.
        self._rules_changed()
        self._fps_changed(self.settings.target_fps)
        self._loop_changed(self.settings.loop)
        if "show_pet" in changed:
            self._toggle_pet(self.settings.show_pet)
        if "diagnostics" in changed:
            self._diagnostics_changed(self.settings.diagnostics)
        self._save()
        self.statusBar().showMessage(
            "已恢复默认（改动了 %d 项）；曲库、播放顺序和标定位置没动。" % len(changed))

    # -- reaching for the geometry by hand ------------------------------------
    def _redetect_layout(self):
        if not self.engine.redetect():
            self.statusBar().showMessage("重新检测需要一张画面：先连接游戏窗口。")
            return
        self.statusBar().showMessage("已丢掉现在的位置，下一帧重新找计量表；记住的布局没删。")

    def _forget_layout(self):
        if self.engine._geom_sig is None:
            self.statusBar().showMessage("还没有可重置的布局：先连接并看到画面。")
            return
        answer = QMessageBox.question(
            self, "重置当前布局缓存",
            "要忘记**当前这套画面**记住的计量表位置吗？\n"
            "其他分辨率、其他黑边组合记住的位置，以及曲库、播放顺序和音量都不受影响。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        gone = self.engine.reset_layout()
        self._save()
        self.statusBar().showMessage(
            "已忘记当前布局（%d 条），下一帧按画面重新找。" % gone
            if gone else "当前这套画面还没有记住的位置，已经重新检测了。")

    def _restore_layout(self):
        if not self.engine.restore_layout():
            self.statusBar().showMessage("没有可退回的上一个可靠位置：这次会话还没换过位置。")
            return
        self._save()
        self.statusBar().showMessage("已退回上一个被验证过的位置。")

    def _diagnostics_folder(self) -> Path | None:
        """The newest session folder next to the exe, which is the one being asked about."""
        root = app_dir() / self.settings.diagnostics_dir
        if not root.is_dir():
            return None
        runs = [p for p in root.iterdir() if p.is_dir()]
        return max(runs, key=lambda p: p.name) if runs else None

    def _export_diagnostics(self):
        src = self._diagnostics_folder()
        if src is None:
            self.statusBar().showMessage(
                "还没有诊断记录：勾上「诊断记录」再玩一段，才会写这个文件夹。")
            return
        meta = src / "meta.txt"
        try:
            sig = self.engine._geom_sig
            meta.write_text("\n".join([
                "%s %s" % (APP_NAME, __version__),
                "python  %s" % sys.version.split()[0],
                "时间    %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
                "抓屏    后端 %s，%d 帧，画面框 %s，本帧测量 %s，状态 %s"
                % (self.engine.source.backend, self.engine.source.frames,
                   self.engine._rect, self.engine._raw_rect, self.engine._pose_state),
                "几何    使用中 %s，已验证 %s，来源 %s，%s"
                % (self.engine.geom.active, self.engine.geom.verified,
                   self.engine.geom.source, self.engine.geom.state()),
                "布局    %s" % (self.settings.layouts or "（还没有记住的布局）"),
                "设置    %s" % json.dumps(
                    {k: v for k, v in self.settings.to_dict().items()
                     if k not in {"layouts"}}, ensure_ascii=False),
                "", "只包含这一段时间的事件与 HUD 小图：不含整屏录制，也不含音乐文件。",
            ]), encoding="utf-8")
        except OSError as exc:
            self.statusBar().showMessage("写 meta 失败：%s" % exc)
            return
        out = src.parent / (src.name + "_诊断包")
        try:
            archived = Path(shutil.make_archive(str(out), "zip", str(src)))
        except OSError as exc:
            self.statusBar().showMessage("打包失败：%s" % exc)
            return
        self.statusBar().showMessage("诊断包已导出：%s" % archived)
        return archived

    # -- wiring ---------------------------------------------------------------
    def _wire(self):
        self.refresh_btn.clicked.connect(self._refresh_windows)
        self.connect_btn.clicked.connect(self._toggle_connect)
        self.arm_btn.clicked.connect(self._toggle_arm)
        self.test_btn.clicked.connect(self._preview_track)

        self.pick_btn.clicked.connect(self._pick_folder)
        self.rescan_btn.clicked.connect(self._on_imported)
        self.add_btn.clicked.connect(self._add_tracks)
        self.up_btn.clicked.connect(lambda: self._move_track(-1))
        self.down_btn.clicked.connect(lambda: self._move_track(1))
        self.order_combo.currentIndexChanged.connect(self._order_mode_changed)
        self.folder_edit.editingFinished.connect(self._on_imported)
        self.pause_btn.clicked.connect(self._toggle_pause)

        for spin in (self.bars_spin, self.hold_spin, self.cooldown_spin, self.revive_spin):
            spin.valueChanged.connect(self._rules_changed)
        self.fps_spin.valueChanged.connect(self._fps_changed)
        self.policy_combo.currentIndexChanged.connect(self._rules_changed)
        self.revive_secs_spin.valueChanged.connect(self._rules_changed)
        self.pet_check.toggled.connect(self._toggle_pet)
        self.diag_check.toggled.connect(self._diagnostics_changed)
        self.loop_check.toggled.connect(self._loop_changed)
        self.reset_btn.clicked.connect(self._reset_rules)
        self.redetect_btn.clicked.connect(self._redetect_layout)
        self.forget_btn.clicked.connect(self._forget_layout)
        self.restore_btn.clicked.connect(self._restore_layout)
        self.export_btn.clicked.connect(self._export_diagnostics)
        self.volume.valueChanged.connect(self._volume_changed)


        self.engine.frame_ready.connect(self._on_frame)
        self.engine.preview_snapshot.connect(self._on_preview_snapshot)
        self.engine.readings_changed.connect(self._on_readings)
        self.engine.party_changed.connect(self._on_party)
        self.engine.calibrationChanged.connect(self._on_drift_adopted)
        self.engine.fired.connect(self._on_fired)
        self.engine.stopped.connect(self._on_stopped)
        self.engine.error.connect(self._on_error)
        self.player.error.connect(self._on_error)
        self.player.track_started.connect(self._on_track_started)
        self.player.set_loop(self.settings.loop)

    def _on_drift_adopted(self):
        self._push_slots()
        self._save()
        self.statusBar().showMessage("标定位置有漂移，已自动跟随计量表。")

    # -- music ----------------------------------------------------------------
    def _music_dir(self, raw: str | None = None) -> Path:
        """Resolve the folder the user is looking at. Relative means next to the
        exe, not next to the source."""
        raw = (raw if raw is not None else self.settings.music_folder).strip() or "music"
        path = Path(raw)
        return path if path.is_absolute() else app_dir() / raw

    def _open_diag(self) -> Diagnostics | None:
        """One folder per session, so a session that misbehaved stays findable."""
        if not self.settings.diagnostics:
            return None
        folder = (app_dir() / self.settings.diagnostics_dir
                  / time.strftime("%Y%m%d-%H%M%S"))
        return Diagnostics(folder, before_s=self.settings.diagnostics_seconds)

    def _diagnostics_changed(self, on: bool):
        self.settings.diagnostics = on
        if self.diag is not None:
            self.diag.close()
        self.diag = self._open_diag()
        self.engine.diag = self.diag
        self._save()
        self.statusBar().showMessage(
            "诊断记录已开启，写在 exe 旁的 %s/" % self.settings.diagnostics_dir if on
            else "诊断记录已关闭")

    def _on_imported(self):
        # The text box is the one source of truth: the folder picker and the drag
        # and drop target both write to it, so reading settings instead made the
        # picker look like it did nothing.
        self.library.folder = self._music_dir(self.folder_edit.text())
        self.library.refresh()
        self.settings.music_folder = str(self.library.folder)
        self._reload_tracks()
        note = self.track_list.import_note
        self.track_list.import_note = ""
        self.statusBar().showMessage(
            f"曲库 {len(self.library)} 首" + (f"，{note}" if note else ""))

    def _reload_tracks(self):
        """Show the list in the order it will actually play in."""
        selected = self.track_list.currentRow()
        self.track_list.clear()
        for path in self.library.sequence():
            self.track_list.addItem(path.name)
        if 0 <= selected < self.track_list.count():
            self.track_list.setCurrentRow(selected)
        self.track_count.setText(f"{len(self.library)} 首")

    def _push_order(self):
        """Take the order back out of the list widget and remember it."""
        names = [self.track_list.item(i).text()
                 for i in range(self.track_list.count())]
        self.library.set_sequence(names)
        self.settings.track_order = names
        self._save()

    def _add_tracks(self):
        pattern = "音频 " + " ".join(f"*{ext}" for ext in sorted(SUPPORTED))
        chosen, _ = QFileDialog.getOpenFileNames(self, "添加音乐",
                                                 str(self._music_dir()), pattern)
        if chosen:
            self.track_list.import_paths([Path(p) for p in chosen])

    def _move_track(self, step: int):
        row = self.track_list.currentRow()
        target = row + step
        if row < 0 or not 0 <= target < self.track_list.count():
            return
        item = self.track_list.takeItem(row)
        self.track_list.insertItem(target, item)
        self.track_list.setCurrentRow(target)
        self._push_order()

    def _order_mode_changed(self, index: int):
        mode = self.order_combo.itemData(index)
        self.library.order = mode
        self.settings.play_order = mode
        self._save()
        nxt = self.library.peek()
        self.statusBar().showMessage(
            "播放顺序：%s，下一首 %s" % ("按列表顺序" if mode == "ordered" else "随机",
                                          nxt.name if nxt else "（曲库为空）"))

    def _pick_folder(self):
        chosen = QFileDialog.getExistingDirectory(self, "选择音乐文件夹",
                                                  str(self._music_dir()))
        if chosen:
            self.folder_edit.setText(chosen)
            self._on_imported()

    def _preview_track(self):
        # resume=False: listening around must not spend the place the auto-play
        # stopped at, or the next trigger starts from the beginning again.
        track = self.player.play_next(resume=False)
        if track:
            self.statusBar().showMessage(f"试听: {track.name}")

    def _volume_changed(self, value: float):
        self.player.set_volume(value)
        self.settings.volume = value
        self._save()

    # -- capture --------------------------------------------------------------
    def _refresh_windows(self):
        self.window_combo.clear()
        wanted = self.settings.window_title.lower()
        best = -1
        for info in list_windows():
            self.window_combo.addItem(info.label, info.hwnd)
            if best < 0 and wanted and wanted in info.title.lower():
                best = self.window_combo.count() - 1
        if best >= 0:
            self.window_combo.setCurrentIndex(best)
        self.feed_label.setText(f"找到 {self.window_combo.count()} 个窗口")

    def _toggle_connect(self, checked: bool):
        if checked:
            hwnd = self.window_combo.currentData()
            if hwnd is None:
                QMessageBox.warning(self, "没有窗口", "列表里没有可选窗口，先点刷新。")
                self.connect_btn.setChecked(False)
                return
            backend = self.engine.connect_window(int(hwnd))
            self._connected = self.engine.source.running
            if not self._connected:
                QMessageBox.warning(self, "连接失败", self.engine.source.error or "抓屏失败")
                self.connect_btn.setChecked(False)
                return
            self.connect_btn.setText("断开")
            self.statusBar().showMessage(f"已连接（{backend}）。等待第一帧…")
            QTimer.singleShot(1500, self._check_feed)
        else:
            self.engine.disconnect()
            self._connected = False
            self.connect_btn.setText("连接")
            self.arm_btn.setChecked(False)
            self._toggle_arm(False)
            self.feed_label.setText("后端: 未连接")

    def _check_feed(self):
        if not self._connected:
            return
        if self.engine.source.frames == 0:
            QMessageBox.warning(
                self, "抓不到画面",
                "已连接但一帧都没收到。\n"
                "多半是游戏处在独占全屏：请在游戏内把显示模式改成「无边框窗口」，"
                "或用窗口化启动。")

    def _push_slots(self):
        # The preview is not filled from here: its circles come from the snapshot,
        # which carries the positions this frame actually read. Setting them from
        # `settings.slots` used to repaint stale circles over a frame that had
        # already been described - and while disarmed the two are different, because
        # the aligner moves the panel for the frame without keeping it.
        self._update_slot_labels([])
        self._on_party(self.engine.party.count)   # depends on the slot count too
        self._rebuild_pet()

    def _rebuild_pet(self):
        """The pet carries one badge per slot, so a calibration change means rebuilding it."""
        if self.pet is not None:
            self._pet_pos = (self.pet.x(), self.pet.y())
            old, self.pet = self.pet, None
            # Unhook the destroyed handler first: this close is ours, and leaving it
            # attached makes `_on_pet_destroyed` read it as the user dismissing the
            # window, uncheck the box, and the rebuild below finds nothing to do.
            old.destroyed.disconnect(self._on_pet_destroyed)
            old.close()
        if self.pet_check.isChecked():
            self._make_pet()

    def _make_pet(self):
        self.pet = PetWidget(self._pet_badges())
        self.pet.quitRequested.connect(self.close)
        self.pet.hideRequested.connect(lambda: self.pet_check.setChecked(False))
        # It deletes itself on close, so Alt+F4 on the pet (or a window closer that
        # hits it as well as this one) leaves the C++ object gone while self.pet
        # still points at it - and every later touch raises.
        self.pet.destroyed.connect(self._on_pet_destroyed)
        self.pet.move(*self._pet_pos)
        self.pet.show()

    def _on_pet_destroyed(self):
        """Dismissing the pet is a decision, not a state to undo.

        Without unchecking the box, the next ally-count or calibration event calls
        `_rebuild_pet()`, sees the box ticked and conjures the window back - so
        closing it looked like nothing happened except a resurrection seconds later.

        Quitting the app goes through here too, because `disconnect()` announces a
        party of zero and that rebuilds the pet, which destroys the old one. Treating
        that as a decision silently turned `show_pet` off for the next launch.
        """
        self.pet = None
        if self._shutting_down or not self.pet_check.isChecked():
            return
        self.pet_check.setChecked(False)

    def _toggle_pet(self, on: bool):
        self.settings.show_pet = on
        if on and self.pet is None:
            self._make_pet()
        elif not on and self.pet is not None:
            self.pet.close()
            self.pet = None
        self._save()

    def _toggle_arm(self, checked: bool):
        ok = self.engine.set_listening(checked)
        if checked and not ok:
            self.arm_btn.setChecked(False)
            return
        self.arm_btn.setText("停止监听" if self.engine.listening else "开始监听")
        self.pause_btn.setEnabled(self.engine.listening)
        if self.engine.listening:
            self.pause_btn.setChecked(False)
            self.pause_btn.setText("暂停")
        if self.pet:
            if self.engine.listening:
                self.pet.flash("监听中")
            else:
                self.pet.clear_flash()
        self._save()

    def _toggle_pause(self, checked: bool):
        self.engine.set_paused(checked)
        self.pause_btn.setText("已暂停，点击恢复" if checked else "暂停")
        if self.pet:
            (self.pet.clear_flash if checked else self.pet.flash)("已暂停" if checked
                                                                  else "监听中")
        self.statusBar().showMessage("已暂停自动播放" if checked else "恢复自动播放")
        self._save()

    def _loop_changed(self, on: bool):
        self.settings.loop = on
        self.player.set_loop(on)
        self._save()

    def _rules_changed(self, *_args):
        self.settings.required_bars = self.bars_spin.value()
        self.settings.hold_frames = self.hold_spin.value()
        self.settings.cooldown_s = self.cooldown_spin.value()
        self.settings.revive_hold_frames = self.revive_spin.value()
        self.settings.revive_min_evidence_s = self.revive_secs_spin.value()
        self.settings.stop_policy = "v2" if self.policy_combo.currentIndex() == 0 else "v1"
        self.engine.reconfigure()
        self._save()

    def _fps_changed(self, value: int):
        self.settings.target_fps = value
        self.engine.reconfigure()
        self._save()

    # -- engine callbacks -------------------------------------------------------
    def _on_preview_snapshot(self, snap):
        """The one place the preview is filled: picture, circles and counts, from
        the same frame, in one repaint."""
        self.preview.set_snapshot(snap)

    def _on_frame(self, frame):
        # No picture is drawn from here any more - `_on_preview_snapshot` owns the
        # preview, and drawing here is what let a new frame inherit old circles.
        now = time.monotonic()
        if now - self._fps_clock >= 1.0:
            # Count delivered captures, not preview repaints: the preview is
            # throttled well below the detection rate.
            total = self.engine.source.frames
            fps = (total - self._frames_seen) / (now - self._fps_clock)
            self._frames_seen = total
            self._fps_clock = now
            pose = self.engine.hud.state
            note = "" if pose == LOCKED else f"  画面框: {pose}"
            if self.engine.validator.frozen:
                note += f"  画面未变化 {self.engine.validator.stale} 帧(抓屏可能停了)"
            self.feed_label.setText(
                f"后端: {self.engine.source.backend}  实际 {fps:.0f} 帧/秒{note}")

    def _on_readings(self, readings):
        # The preview's own counts come with the snapshot; this is the text and the
        # pet, which are allowed to be a frame late and are not drawn over a picture.
        self._update_slot_labels(readings)
        # A frame has been read, so a count of zero now means "solo" rather than
        # "not yet" - and the header has to say it even for a player who armed the
        # tool alone, whose count never changes and so never gets announced.
        self._show_party(self.engine.party.count)
        if self.pet and self.pet.isVisible():
            self.pet.set_readings(readings)

    def _update_slot_labels(self, readings):
        # Only ever called once a frame has been read, so a count of zero means
        # "solo", not "not yet". Rows past the count belong to allies who left.
        party = self.engine.party.count
        for i, label in enumerate(self.slot_labels):
            if i >= len(self.settings.slots):
                label.setText(f"槽位 {i + 1}: 等待画面")
                self.badges[i].set_state(None)
                continue
            if i >= party:
                label.setText(f"槽位 {i + 1}: "
                              + ("没有队友" if not party else "队友已离场"))
                self.badges[i].set_state(None)
                continue
            got = readings[i] if i < len(readings) else None
            bars = None if got is None else got.bars
            conf = 0.0 if got is None else got.confidence
            obs = self.engine.observations
            note = self._evidence_note(obs[i]) if i < len(obs) else ""
            label.setText(f"槽位 {i + 1}: {bars if bars is not None else '没读到'} 格"
                          f"   置信度 {conf:.2f}{note}")
            self.badges[i].set_state(bars, conf)

    def _evidence_note(self, obs) -> str:
        """Why the stop machine is or is not counting this slot, in one phrase.

        Only available while a track is playing, which is when it is worth the room:
        bars and confidence cannot answer "why has it not stopped yet".
        """
        kind = obs.evidence()
        if kind == STRONG_DOWN:
            return "   倒地"
        if kind == STRONG_ALIVE:
            if obs.standing:
                return "   站着：血条"
            return "   站着：环区空 %.2f" % obs.ring_share
        if not obs.position_ok:
            return "   不确定：定位未确认"
        return "   不确定"

    def _on_party(self, allies: int):
        self._show_party(allies)
        want = self._pet_badges()
        if self.pet is not None and len(self.pet.badges) != want:
            self._rebuild_pet()

    def _show_party(self, allies: int):
        self.party_label.setText("队友数: 单人，没有队友" if allies == 0
                                 else f"队友数: {allies}")

    def _pet_badges(self) -> int:
        """One badge per ally actually in the run, not per spot we watch."""
        return max(1, self.engine.party.count or len(self.settings.slots))

    def _on_fired(self, track: str):
        self.trigger_label.setText(f"已触发: {self.engine.trigger.fires} 次")
        if self.pet and self.pet.isVisible():
            self.pet.flash(f"♪ {track}")
        self.statusBar().showMessage(f"队友全倒地，已播放 {track}")

    def _on_track_started(self, name: str):
        """The readout follows the player, not the trigger.

        A track that ends while the allies are still downed starts the next one, and
        nothing re-fires at that moment - without this the window and the pet keep
        naming the song that just finished.
        """
        if self.pet and self.pet.isVisible():
            self.pet.flash(f"♪ {name}")
        self.statusBar().showMessage(f"正在播放 {name}")

    def _on_stopped(self, slot: int = 0):
        if self.pet and self.pet.isVisible():
            self.pet.clear_flash()
            self.pet.flash("有人被救起")
        self.statusBar().showMessage("队友 %d 已经站起来，停止播放" % (slot + 1))

    def _on_error(self, message: str):
        if message:
            self.statusBar().showMessage(f"错误: {message}")

    # -- teardown ---------------------------------------------------------------
    def _save(self):
        try:
            self.settings.save(DEFAULT_PATH)
        except OSError as exc:
            self.statusBar().showMessage(f"配置没能保存: {exc}")

    def closeEvent(self, event):
        self._shutting_down = True
        self.player.stop()
        self.engine.disconnect()
        self._save()
        if self.diag is not None:
            self.diag.close()
            self.diag = None
            self.engine.diag = None
        if self.pet is not None:
            self.pet.close()          # WA_DeleteOnClose: this also frees the C++ object
            self.pet = None
        super().closeEvent(event)
