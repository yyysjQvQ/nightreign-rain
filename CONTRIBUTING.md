# 贡献指南

先说清楚这个项目在做什么：它只**看**屏幕左上角的 HUD，判断队友是不是都倒在最后一格，
然后放你自己选的歌。它不读游戏内存、不改游戏文件、不联网。任何"要碰游戏进程"的改动
都不在这个项目的范围内。

## 环境

- Windows 10/11，Python 3.11–3.13（开发用的是 3.13）。
- 依赖只有 `requirements.txt` 里那些；`opencv-python` 和 `pytest` 只在跑测试时需要，
  打包好的 exe 里没有 OpenCV（HSV 和连通域都用 numpy 重写过了，等价性由
  `tests/test_cv2_equivalence.py` 盯着）。

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

## 跑测试

```
.venv\Scripts\python.exe fixtures\make_synthetic.py     # 生成合成用例
set QT_QPA_PLATFORM=offscreen && .venv\Scripts\python.exe -m pytest -q
```

全部测试都是离屏跑的，不需要游戏在运行，也不需要真机抓屏。
没有跑 `make_synthetic.py`、或者没有真实截图时，需要它们的测试会**跳过**而不是失败
（见 `tests/shots.py`）——所以看到几十条 skipped 是正常的，红才是问题。

## 关于 `fixtures/`

仓库里只有一张真实截图（`golden_both_3bars.jpg`，合成用例的底图）。其余 `real_*` 是
不同分辨率、不同时段、不同光照下的真实画面，**不入库**：那是别人的版权画面，里面还有
队友的 ID。

**因此：如果你想改检测相关的任何常数，请先准备自己的一组截图。** 每个常数的出处都写在
它旁边的注释里（`nrdoom/detector.py` 的 `LETTER_*`、`GAUGE_*_FRAC_W`、
`REALIGN_OFF_FRAC_W`、`REALIGN_SETTLE_FRAC_W`、`RING_MIN_CONFIDENCE`），
它们不是调出来的手感数字，是从图上量出来的。改完请说明你在什么分辨率、什么画面上验的。

## 提 PR 之前

- 一个 PR 只做一件事。检测、音乐、界面三块的规则不一样，混在一起不好验。
- 改行为就改/加测试；改常数就在 PR 里写清测量过程。
- `python main.py --selftest` 要全绿（它会往旁边写 `自检.txt`）。
- 不要求你先跑真机游戏；跑不了就说一句，维护者会跑。
