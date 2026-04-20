# Part II 利用率与瓶颈归因

这个目录把 `SF-MLA` 论文实验计划中 Part II 需要的内容单独收拢出来，包括：

- 与当前 Part I 口径对齐的 `DeepSeek FlashMLA` 主方法 / `TT 主线 FlashMLA` proxy 说明
- `decode` full sweep 的阶段迁移与 bubble 分析
- `decode` 代表点的 source-level attribution
- `prefill` 控制对照与 direct PM util 曲线
- 论文可直接使用的图、表、总报告与正文段落草稿
- `DeepSeek / 4c / 8c` direct representative-point probe 的脚手架

说明：

- Part I 现在已经把 `DeepSeek FlashMLA` 固定为 `decode` 主方法。
- 但当前仓库中最完整的 dense stage/source/bubble profiler 资产仍然主要覆盖 `TT 主线 FlashMLA`。
- 因此本目录会额外生成一张 `Part I` 代理对齐图表，用于说明 `TT 主线 FlashMLA` 在 `4k~32k` 的 long-seq 区间可作为机理 proxy，但它不是 `DeepSeek FlashMLA` 的 direct profile 证据。

## 用法

在仓库根目录执行：

```bash
python3 mla_flash_attention_dev/experiments/part2_utilization_bottleneck_analysis/build_part2_results.py
```

生成结果位于：

- `outputs/report.md`
- `outputs/paper_ready_paragraphs.md`
- `outputs/dashboard.html`
- `outputs/visuals/`
- `outputs/tables/`
- `outputs/raw/`

## Direct Probe 脚手架

当前仓库已经补了两条 direct representative-point probe 入口：

- `mla_flash_attention_dev/experiments/profile_deepseek_flash_mla_part2.py`
  - 很薄的 child-run 执行器
  - 负责执行单个 `TT / DeepSeek-4c / DeepSeek-8c` decode 代表点
- `mla_flash_attention_dev/experiments/run_deepseek_flash_mla_part2_probe.py`
  - sweep runner
  - 默认走稳定的 `device_only capture + offline process_ops_logs.py` 链路
  - 负责 preflight、manifest、skip-existing、unsupported precheck
  - 如需实验性保留 `tracy` host trace/report，可显式使用 `--capture-mode tracy_report`

### 查看 preset 代表点

```bash
python3 mla_flash_attention_dev/experiments/profile_deepseek_flash_mla_part2.py --list-preset-cases
```

当前已内置：

- `A1/A2/A3/A4`：`DeepSeek vs TT` direct 对齐点
- `B1/B2/B3`：`q_shards=4` 的 `4c/8c` 几何比较点
- `C1/C2/C3`：`q_shards=3` 的 `4c/8c` 几何比较点
- `D1/D2`：`4c unsupported -> 8c supported` 容量边界点

### 最小 dry-run

```bash
python3 mla_flash_attention_dev/experiments/run_deepseek_flash_mla_part2_probe.py --minimal --dry-run
```

默认会展开：

- preset: `A1/A2/A3`
- variants: `tt + deepseek4c`
- capture mode: `device_only`

### 当前默认 direct 输出里有什么

默认 `device_only` 模式会稳定产出：

- `reports/ops_perf_results.csv`
- `reports/per_core_op_to_op_times.csv`
- `.logs/profile_log_device.csv`

它适合当前最关心的 direct 证据：

- `DEVICE KERNEL DURATION [ns]`
- `DEVICE COMPUTE CB WAIT FRONT [ns]`
- `DEVICE COMPUTE CB RESERVE BACK [ns]`
- `OP TO OP LATENCY [ns]`

也就是说，当前默认链路已经足够支撑：

- bubble 分析
- wait / reserve 主矛盾归因
- `TT` vs `DeepSeek-4c` 在代表点上的 direct device-time 对比

但它**默认不保证**拿到：

- `PM IDEAL [ns]`
- `PM COMPUTE [ns]`
- `PM BANDWIDTH [ns]`
- `PM FPU UTIL (%)`

也就是说，如果只需要稳定的 `kernel / wait_front / reserve_back` direct 证据，默认 `device_only` 链路已经足够；但如果要把 `PM` / perf-counter 列作为 direct evidence 进入正文，则需要显式切到修正 runtime 后的 `tracy_report` 路径。当前已经验证可用的是：

- `tracy_report + sum + sync`：可稳定拿到非-placeholder `PM IDEAL / COMPUTE / BANDWIDTH`
- `tracy_report + perf-fpu + sync`：可在同一批 artifact 中同时拿到非-placeholder PM 与 `FPU / SFPU / MATH` 利用率列

`A1` 的 corrected-runtime rerun 已在 `TT` 与 `DeepSeek-4c` 两侧得到 `PM IDEAL / COMPUTE / BANDWIDTH = 8704 / 8704 / 2129`，因此 decode 侧 `PM/perf-counter` 已不再只是未来工作。

### 容量边界 dry-run

```bash
python3 mla_flash_attention_dev/experiments/run_deepseek_flash_mla_part2_probe.py --groups D --variants tt deepseek4c deepseek8c --dry-run
```

这条命令会在 runner 层提前把 `deepseek4c` 的 `unsupported` 点标出来，例如：

- `D1`: `batch * deepseek_num_q_shards must be <= 24, got 32`
- `D2`: `batch * deepseek_num_q_shards must be <= 24, got 48`

因此 `4c unsupported -> 8c supported` 可以作为正式边界结果进入后续 `Part II`。
