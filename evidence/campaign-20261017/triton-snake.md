# Triton Ascend SnakeBeta experiment

## 迁移结论

输入为 Qwen3-TTS PyTorch SnakeBeta，参考固定 SGLang-Omni main
`6bdd47967caa55c38430d18175b750008aecab61` 的
`sglang_omni/utils/snake_beta.py` 所描述的逐步 BF16 舍入语义。
这是 vector-only 逐元素算子，没有 `tl.dot`；没有复制 CUDA libdevice 路径。
当前结论仅为算子验证成功，尚未完成完整服务与音频质量验收。

## Triton-Ascend 实现

实现位于 `qwen3_tts_engine/snake_triton_ascend.py`，显式导入 `torch_npu`，
要求 NPU 上连续 NCT BF16 输入。alpha 指数及 beta 倒数由模型加载阶段缓存。
每个乘法、sin、平方、加法均保留 BF16 中间舍入，禁止 FP contraction。
初版平坦 1D 逻辑 grid 较慢，保留在代码及原始测量中。
改进版仍用 1D grid，将通道/时间逻辑 tile 放入 kernel 循环；从设备属性读取
vector core 数，最多绑定 48 个程序（本机 910B1），而非为每个逻辑块启动程序。

## 验证脚本

`tools/snake_triton_ascend_example.py` 已在现有容器内实际运行：

```bash
python tools/snake_triton_ascend_example.py --output runs/snake-triton-ascend-tiled-probe --tiled
```

实际解释器为 `/home/why/qwen3-tts-research/venv-engine/bin/python`；使用
`ASCEND_RT_VISIBLE_DEVICES=6`、jemalloc 与保留 CANN 的 `PYTHONPATH`。
10 组形状/幅度覆盖非整除尾块 `[1,7,37]`、小时间维及真实 decoder 大张量。
所有输出逐值一致且有限，NPUGraph 捕获与重放成功。ABBA 计时中，
`[4,1536,32]` 约 0.059 → 0.019 ms；`[4,96,15360]` 约 0.179 → 0.166 ms。
精确原始值见 `results/snake-triton-ascend-tiled-result.json`。

## 优化说明

每 tile 1024 元素，时间块 32–256、通道块补足 1024，使用二维连续读写及
通道广播，避免逐元素离散加载重复 alpha/beta。索引使用 Triton 默认 int32；
wrapper 限制总元素量，初版 coreDim 不超过 65535，改进版不超过实际向量核数。
未使用 block_ptr，因此无沿用错误 shape/stride/order 的问题；连续 NCT 和
时间方向访存是明确约束，尾块保留 mask。1024 元素子块控制 UB 工作集。
未使用 `care_padding=False`：尾块必须屏蔽无效地址。
未启用 `TRITON_ALL_BLOCKS_PARALLEL` 或 multibuffer：先以明确物理核循环取得
可验证收益，尚无证据需要额外编译选项。未保留 GPU num_warps/num_stages 调参。

## 风险与限制

随机算子逐值一致不等同于所有 BF16 编码、实际音频或整个服务质量一致。
已增加默认关闭的 `--triton-snake`，只对连续 BF16 激活使用 Triton；其他 dtype
或布局保留原算术。真实 decoder 的 101 帧、多流与槽位重用图对照正在运行。
首次预热可能产生较多形状编译，需保留启动成本；不以算子速度推算服务收益。
