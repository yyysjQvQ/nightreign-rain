"""Window chrome: the artwork behind the controls, and a theme that keeps them readable.

A photograph behind a form is a legibility trap, so every surface the controls sit
on gets a translucent dark plate rather than the raw image.
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QWidget

from ..assets import asset_root

PLATE = "rgba(16, 14, 24, 214)"
PLATE_SOFT = "rgba(16, 14, 24, 170)"
EDGE = "rgba(255, 255, 255, 34)"
ACCENT = "rgba(112, 92, 226, 235)"
TEXT = "#e8e6f2"

_background: QPixmap | None = None
_background_key: tuple | None = None


def background_pixmap() -> QPixmap | None:
    path = asset_root() / "background.jpg"
    if not path.is_file():
        return None
    key = (str(path), path.stat().st_mtime_ns)
    global _background, _background_key
    if _background is None or _background_key != key:
        img = QImage(str(path))
        _background = None if img.isNull() else QPixmap.fromImage(img)
        _background_key = key
    return _background


def paint_background(widget: QWidget, painter: QPainter):
    """Scale-to-cover the artwork, then darken it so the controls stay legible."""
    pm = background_pixmap()
    rect = widget.rect()
    if pm is None or pm.isNull():
        painter.fillRect(rect, QColor(12, 11, 18))
        return
    scaled = pm.size()
    scale = max(rect.width() / scaled.width(), rect.height() / scaled.height())
    target = QSize(int(scaled.width() * scale), int(scaled.height() * scale))
    offset = QPoint((rect.width() - target.width()) // 2,
                    (rect.height() - target.height()) // 2)
    painter.drawPixmap(QRect(offset, target), pm, QRect(QPoint(0, 0), scaled))
    painter.fillRect(rect, QColor(8, 7, 14, 150))


STYLESHEET = f"""
QWidget {{ color: {TEXT}; font-family: "Microsoft YaHei", "Segoe UI", sans-serif; }}
QMainWindow, QDialog {{ background: transparent; }}
#root {{ background: transparent; }}

QGroupBox {{
    background: {PLATE}; border: 1px solid {EDGE}; border-radius: 7px;
    margin-top: 10px; padding: 6px 8px 8px 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 10px; padding: 0 5px;
    background: transparent; color: #b9b3d6;
}}

QLabel {{ background: transparent; }}
QStatusBar {{ background: {PLATE_SOFT}; }}
QStatusBar::item {{ border: none; }}

QTabWidget::pane {{ background: {PLATE}; border: 1px solid {EDGE}; border-radius: 6px; }}
QTabBar::tab {{
    background: {PLATE_SOFT}; padding: 6px 16px; border: 1px solid {EDGE};
    border-bottom: none; border-top-left-radius: 5px; border-top-right-radius: 5px;
}}
QTabBar::tab:selected {{ background: {ACCENT}; }}

QPushButton {{
    background: rgba(48, 43, 68, 225); border: 1px solid {EDGE};
    border-radius: 5px; padding: 6px 12px;
}}
QPushButton:hover {{ background: rgba(74, 65, 108, 235); }}
QPushButton:pressed {{ background: rgba(96, 84, 142, 240); }}
QPushButton:checked {{ background: {ACCENT}; border-color: rgba(255,255,255,80); }}
QPushButton:disabled {{ color: rgba(232,230,242,90); background: rgba(40,38,52,150); }}
QPushButton#arm {{
    background: rgba(38, 92, 58, 230); border: 1px solid rgba(120, 220, 150, 120);
    font-size: 13pt; font-weight: bold; padding: 10px;
}}
QPushButton#arm:checked {{ background: rgba(150, 46, 46, 235); }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: rgba(12, 11, 20, 225); border: 1px solid {EDGE};
    border-radius: 4px; padding: 3px 6px; min-height: 26px;
    selection-background-color: {ACCENT};
}}
/* A spin box that is allowed to fill its column turns into a runway: the value
   drifts a metre away from its label and the arrows end up 900px from the text.
   26px is the floor the arrows need to be hittable at all. */
QSpinBox, QDoubleSpinBox {{ max-width: 170px; }}
QComboBox QAbstractItemView {{
    background: rgba(18, 16, 28, 250); border: 1px solid {EDGE};
    selection-background-color: {ACCENT};
}}
QListWidget {{
    background: rgba(12, 11, 20, 210); border: 1px solid {EDGE}; border-radius: 5px;
}}

QCheckBox::indicator {{ width: 15px; height: 15px; }}
QCheckBox {{ background: transparent; spacing: 7px; }}
QSlider::groove:horizontal {{
    height: 5px; background: rgba(255,255,255,40); border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px; margin: -5px 0; background: {ACCENT}; border-radius: 7px;
}}
QToolTip {{ background: rgba(18,16,28,245); border: 1px solid {EDGE}; color: {TEXT}; }}
"""


def apply(app: QApplication) -> None:
    app.setStyleSheet(STYLESHEET)
