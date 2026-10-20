测试图
======

这个目录里只随仓库走两张图：

- `golden_both_3bars.jpg` —— 2276×1280 的真实一帧，两个队友都在最后一格。
  它是 `make_synthetic.py` 的底图，也是自检样本的原料。
- `make_synthetic.py` —— 从上面那张生成全部 `synth_*.png` 用例（掉一格、掉两格、
  紫天空填充、单人/双人行、只留第一格……）：

      python fixtures/make_synthetic.py

其余 `real_*.png` / `real_*.jpg` 不在仓库里：那是不同分辨率、不同光照、不同时段
截的真实游戏画面（约 12 MB），也是检测器里每个常数的出处。它们既是别人的版权画面，
又包含玩家 ID，所以不上传。

需要这些图的测试在图不存在时会**跳过**而不是报错——见 `tests/shots.py`。
每个常数的测量过程写在它旁边的注释里，想复算就用自己的截图：

- 信箱黑边：`nrdoom/detector.py` 的 `LETTER_*`
- 面板位置随宽度的比例：同文件的 `GAUGE_X_FRAC_W` / `ROW0_Y_FRAC_W` / `ROW_PITCH_FRAC_W`
- 读数容差与对齐阈值：`REALIGN_OFF_FRAC_W`、`REALIGN_SETTLE_FRAC_W`、`RING_MIN_CONFIDENCE`

如果你打算改检测相关的任何东西，先准备好自己的一组截图，再动这些数。
