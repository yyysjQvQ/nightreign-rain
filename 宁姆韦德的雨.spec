# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for 宁姆韦德的雨.

The installed PySide6 alone is 655 MB and cv2 is 113 MB, so the default collection
would ship a bloated exe. This drops the Qt modules we never import and prunes the
plugin folders down to the four the app actually loads.
"""
import os

BLOCK_CIPHER = None

APP = "宁姆韦德的雨"

# Everything here is a PySide6 submodule the app never imports.
EXCLUDES = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel", "PySide6.QtWebSockets", "PySide6.QtPdf", "PySide6.QtPdfWidgets",
    "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQuickWidgets", "PySide6.QtQuickControls2",
    "PySide6.QtQml", "PySide6.QtQmlModels", "PySide6.QtQmlNetwork",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DInput", "PySide6.Qt3DAnimation",
    "PySide6.Qt3DExtras", "PySide6.Qt3DLogic", "PySide6.Qt3DPhysics",
    "PySide6.QtCharts", "PySide6.QtGraphs", "PySide6.QtGraphsWidgets",
    "PySide6.QtDataVisualization", "PySide6.QtDatavisualization",
    "PySide6.QtDesigner", "PySide6.QtUiTools", "PySide6.QtHelp", "PySide6.QtTest",
    "PySide6.QtRemoteObjects", "PySide6.QtScxml", "PySide6.QtStateMachine",
    "PySide6.QtSerialPort", "PySide6.QtSerialBus", "PySide6.QtNfc", "PySide6.QtBluetooth",
    "PySide6.QtPositioning", "PySide6.QtLocation", "PySide6.QtSensors",
    "PySide6.QtTextToSpeech", "PySide6.QtHttpServer", "PySide6.QtWebChannel",
    "PySide6.QtMultimediaWidgets", "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets",
    "PySide6.QtSvg", "PySide6.QtSvgWidgets", "PySide6.QtConcurrent",
    "PySide6.QtNetworkAuth", "PySide6.QtSpatialAudio", "PySide6.QtVirtualKeyboard",
    # 86 MB of image-processing code we no longer call: HSV and blob labelling
    # are reimplemented in numpy (nrdoom/color.py, nrdoom/label.py) and verified
    # against cv2 in tests/test_cv2_equivalence.py.
    "cv2",
    "tkinter", "matplotlib", "scipy", "PIL", "PyInstaller.utils",
    "pytest", "_pytest", "pydoc_data",
]

# Plugin families and loose DLLs that are never needed at runtime.
DROP_FRAGMENTS = (
    # 30.9 MB: cv2's ffmpeg backend, only used by VideoCapture on video files.
    # We grab the screen through WGC/mss, so nothing here ever loads it.
    "opencv_videoio_ffmpeg",
    # 20.6 MB: Mesa llvmpipe software GL, for QML/scene-graph. This app is
    # raster QtWidgets only.
    "opengl32sw",
    "qt6webengine", "qt6pdf", "qt6quick", "qt6qml", "qt63d", "qt6charts", "qt6graphs",
    "qt6datavisualization", "qt6designer", "qt6quick3d", "qt6remoteobjects", "qt6scxml",
    "qt6serialport", "qt6serialbus", "qt6nfc", "qt6bluetooth", "qt6positioning",
    "qt6location", "qt6sensors", "qt6texttospeech", "qt6httpserver", "qt6webchannel",
    "qt6websockets", "qt6help", "qt6test", "qt6svg", "qt6opengl",
    "imageformats/qtvlib", "canbus", "geoservices", "texttospeech", "webview",
    "assetimporters", "scx_statex", "qmllint", "qmlls", "renderers", "sensorgestures",
    "sqldrivers", "networkinformation", "tls", "printsupport",
    "qt6pdfquick", "qt6qmlcompact", "qt6qmlcore",
    "translations",
)

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=[("assets", "assets"), ("music", "music")],
    hiddenimports=["windows_capture", "numpy"],
    hookspath=[],
    excludes=EXCLUDES,
    runtime_hooks=[],
    noarchive=False,
)

kept, dropped = [], 0
dropped_bytes = 0
for triplet in a.binaries:
    name = str(triplet[0]).lower().replace("\\", "/")
    if any(frag in name for frag in DROP_FRAGMENTS):
        dropped += 1
        dropped_bytes += os.path.getsize(triplet[1])
        continue
    kept.append(triplet)
a.binaries = kept
print(f"[spec] pruned {dropped} binaries ({dropped_bytes / 1e6:.1f} MB), {len(kept)} kept")
print("[spec] largest kept:")
for t in sorted(kept, key=lambda t: -os.path.getsize(t[1]))[:10]:
    print("        %6.1f MB  %s" % (os.path.getsize(t[1]) / 1e6, t[0]))

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=APP,
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=os.path.join("assets", "icon.ico"),
)
