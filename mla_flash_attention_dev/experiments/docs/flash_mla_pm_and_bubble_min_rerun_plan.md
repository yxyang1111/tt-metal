# FlashMLA PM Util And Compute Bubble Minimal Rerun Plan

## 目标

这份补充方案只解决一件事：

- 把当前结果里缺失的 `PM 利用率` 和 `compute 空泡` 真正跑出来

这里的目标列是：

- `PM IDEAL [ns]`
- `PM COMPUTE [ns]`
- `PM BANDWIDTH [ns]`
- `PM FPU UTIL (%)`
- `NOC UTIL (%)`
- `MULTICAST NOC UTIL (%)`
- `DRAM BW UTIL (%)`
- `DEVICE COMPUTE CB WAIT FRONT [ns]`
- `DEVICE COMPUTE CB RESERVE BACK [ns]`

## 最新验证状态

截至当前这轮 guarded rerun 之后，状态已经可以分得比较清楚：

- `decode_safe` 全 sweep 已完成，`DEVICE COMPUTE CB WAIT FRONT / RESERVE BACK` 在 `8/8` 个 decode 点上可用。
- 但这轮 `sum` 路线里的 `PM IDEAL / PM COMPUTE / PM BANDWIDTH` 仍固定在 `1.0 ns`，`PM FPU UTIL (%)` 只有 `0.000 ~ 0.002`，`NOC/MULTICAST/DRAM BW UTIL` 仍为空。
- 因此这轮结果应明确解读成：`bubble-complete / PM-incomplete`。

另外，后续做的两次安全验证给出了一个更谨慎的结论：

- `perf-fpu` smoke：`mla_flash_attention_dev/experiments/profiling/outputs/flash_mla_pm_bubble_probe_perf_fpu_smoke/run_manifest.json`
  - `preflight.status=failed`
  - `attempts=3`
  - 没有进入真正的 Tracy profiling
- 独立健康探测：`mla_flash_attention_dev/experiments/profiling/outputs/flash_mla_pm_bubble_probe_healthcheck_after_perf_fpu/run_manifest.json`
  - `preflight.status=failed`
  - `attempts=1`
  - 说明失败后设备仍未立即自行恢复

这两次失败的主错误签名是一致的：

- `RuntimeError: Timeout waiting for Ethernet core service remote IO request.`
- `TT_FATAL: Read unexpected run_mailbox value`

所以现在更准确的判断不是“`perf-fpu` 一定不可用”，而是：

- 当前机器状态下，`perf-fpu` 路线还没有拿到一次成功进入 profiling 的 smoke-run
- 现阶段应该先做独立 `preflight-only` 健康探测，确认设备恢复，再做下一次 `perf-fpu / NOC` 尝试

## 为什么当前 detailed 结果里拿不到

当前 `profile_flash_mla_wh_detailed.py` 读的是：

- `cpp_device_perf_report.csv`

但这批现有文件的表头里只有：

- `DEVICE FW/KERNEL/BRISC/NCRISC/TRISC*`
- `OP TO OP LATENCY`

并没有：

- `PM FPU UTIL (%)`
- `NOC UTIL (%)`
- `DRAM BW UTIL (%)`
- `DEVICE COMPUTE CB WAIT FRONT [ns]`
- `DEVICE COMPUTE CB RESERVE BACK [ns]`

这意味着问题不是“脚本没解析到”，而是“当前输出源根本没产出这些列”。

另外，`tools/tracy/process_ops_logs.py` 里已经明确写了：

- 当 post-process 直接使用 `cpp_device_perf_report.csv` 时，`device_analysis_types` 会被忽略

所以如果还沿用当前这条 `cpp_device_perf_report.csv` 路线，`compute bubble` 这两列还是拿不出来。

## 最小思路

最小 rerun 方案不是重跑整套 detailed sweep，而是补一条单独的 `PM/bubble probe` 路径。

当前建议分两层：

1. 安全默认路径：
   用 `python3 -m tracy -r` 生成 `ops_perf_results_*.csv`
   用 `--enable-sum-profiling` 触发 legacy Python post-process
   在同一次 run 里追加：
   `-a device_compute_cb_wait_front`
   `-a device_compute_cb_reserve_back`
2. 可选扩展路径：
   只有在设备状态稳定时，再额外打开 `--collect-noc-traces`
   只有在确认不会把卡拖脏时，再尝试 `--profiler-capture-perf-counters=fpu`

这样做的原因是：

- `ops_perf_results_*.csv` 会带上 host-side `performance_model` 字段，所以能得到 `PM IDEAL / PM COMPUTE / PM BANDWIDTH / PM FPU UTIL`
- `--enable-sum-profiling` 能稳定触发 legacy Python device-log parsing；这条路会真正处理 `device_compute_cb_wait_front` / `device_compute_cb_reserve_back`
- `--collect-noc-traces` 会让 post-process 尝试生成 `NOC UTIL / MULTICAST NOC UTIL / DRAM BW UTIL`，但这条路径在当前机器上需要谨慎启用
- `--profiler-capture-perf-counters=fpu` 理论上还能补更多 FPU 视角指标，但当前机器上曾触发不稳定初始化，因此不作为默认值

## 已封装好的 runner

已经补了一个可直接用的封装脚本：

- `mla_flash_attention_dev/experiments/profiling/run_flash_mla_pm_bubble_probe.py`

它默认就会跑完整 decode sweep：

- `decode_256`
- `decode_512`
- `decode_1k`
- `decode_2k`
- `decode_4k`
- `decode_8k`
- `decode_16k`
- `decode_32k`

同时它会自动：

- 在真正启动 profiler 之前先做 plain `child-run` 预检
- 预检失败时自动等待并重试
- 对单个 case 设置超时，超时后会清理整组子进程，避免留下残留锁占用
- 为每个 case 建单独输出目录
- 在 `reports/` 下寻找最新的 `ops_perf_results_*.csv`
- 检查关键列是否存在
- 生成 `run_manifest.json`
- 也可以只做 `preflight-only` 健康探测，不真正启动 Tracy profiling

### 当前安全默认值

runner 现在的默认配置是：

- 默认 `legacy-trigger=sum`
- 默认不打开 `--collect-noc-traces`
- 默认先做 preflight，再决定是否真正开始 sweep
- 默认对每个 case 启用超时保护
- 现在也支持 `--preflight-only`，可单独用于设备恢复探测

也就是说，当前直接执行默认命令时，优先目标是：

- 先确认不会死锁
- 先把 `PM` 和 `compute bubble` 主路径跑通
- 等机器稳定后，再额外补 `NOC/DRAM util`

### 当前建议的升级顺序

为了把“设备脏状态”和“参数组合是否可用”分开，建议现在按这个顺序推进：

1. `--preflight-only`
   先确认 plain `child-run` 能稳定打开设备。
2. 默认 `sum` smoke
   只验证 `bubble` 路线是否仍然健康。
3. `perf-fpu` smoke
   只有在 `preflight-only` 连续通过之后再尝试。
4. `--collect-noc-traces` smoke
   只有在 `perf-fpu` 或至少默认 smoke 已稳定时再加。

### 常用命令

只做 smoke-run：

```bash
python3 mla_flash_attention_dev/experiments/profiling/run_flash_mla_pm_bubble_probe.py --smoke
```

只做独立健康探测，不启动 Tracy profiling：

```bash
python3 mla_flash_attention_dev/experiments/profiling/run_flash_mla_pm_bubble_probe.py \
  --smoke \
  --preflight-only
```

跑你要的 8 个 decode S 长度：

```bash
python3 mla_flash_attention_dev/experiments/profiling/run_flash_mla_pm_bubble_probe.py
```

只跑指定 S 长度：

```bash
python3 mla_flash_attention_dev/experiments/profiling/run_flash_mla_pm_bubble_probe.py --seq-lens 256 512 1k 2k 4k 8k 16k 32k
```

先看命令不真正执行：

```bash
python3 mla_flash_attention_dev/experiments/profiling/run_flash_mla_pm_bubble_probe.py --dry-run
```

显式增加 preflight 重试：

```bash
python3 mla_flash_attention_dev/experiments/profiling/run_flash_mla_pm_bubble_probe.py \
  --smoke \
  --preflight-retries 5 \
  --preflight-wait-seconds 60
```

忽略已有结果，强制重跑：

```bash
python3 mla_flash_attention_dev/experiments/profiling/run_flash_mla_pm_bubble_probe.py --force
```

如果后面确认机器稳定，再额外尝试 NoC util：

```bash
python3 mla_flash_attention_dev/experiments/profiling/run_flash_mla_pm_bubble_probe.py \
  --collect-noc-traces
```

如果后面确认机器稳定，再尝试真正的 perf-fpu smoke：

```bash
python3 mla_flash_attention_dev/experiments/profiling/run_flash_mla_pm_bubble_probe.py \
  --smoke \
  --legacy-trigger perf-fpu \
  --force
```

## 最小 case 集合

建议不要一开始就重跑 13 个点，先跑 1 个 smoke case，然后只补 4 个代表点。

### Step 0: smoke case

- `decode_1k`

它是最合适的 smoke 点，因为：

- runtime 较短
- 当前结论里它代表短序列 compute-critical 区间
- 如果这一个点都拿不到新列，说明 rerun 路线本身有问题，不值得直接放大全 sweep

### Step 1: 最小代表集

- `decode_1k`
- `decode_4k`
- `decode_32k`
- `prefill_4k`

这 4 个点分别覆盖：

- 短 decode 的 compute-critical 区间
- decode 的阶段迁移点
- 长 decode 的 reader/writer 饱和区
- prefill 的重负载饱和区

如果你后面想更精确观察 `reader_close_to_critical_path` 的拐点，再补：

- `decode_16k`

## 建议命令

### 单点 smoke-run 模板（当前推荐）

```bash
python3 -m tracy -v -r -p \
  -o "mla_flash_attention_dev/experiments/profiling/outputs/flash_mla_pm_bubble_probe/decode_1k" \
  --check-exit-code \
  --op-support-count 4000 \
  --enable-sum-profiling \
  -a device_kernel_duration \
  -a device_brisc_kernel_duration \
  -a device_ncrisc_kernel_duration \
  -a device_trisc0_kernel_duration \
  -a device_trisc1_kernel_duration \
  -a device_trisc2_kernel_duration \
  -a device_compute_cb_wait_front \
  -a device_compute_cb_reserve_back \
  mla_flash_attention_dev/experiments/profiling/profile_flash_mla_wh_detailed.py --child-run --case decode_1k
```

### 仅在机器稳定后再加的可选参数

```bash
--collect-noc-traces
--profiler-capture-perf-counters=fpu
```

### 4 个代表点 rerun

把上面命令里的 `decode_1k` 依次替换成：

- `decode_1k`
- `decode_4k`
- `decode_32k`
- `prefill_4k`

建议每个 case 单独一个输出目录，不要把多个 case 混在同一个 output root 里，这样后面查 `ops_perf_results_*.csv` 更干净。

## 运行后应该看什么文件

不要再优先看：

- `cpp_device_perf_report.csv`

这次应该优先看每个 report folder 里的：

- `ops_perf_results_*.csv`

如果开了 NoC traces，也可以同时保留：

- `npe_viz/`

## smoke-run 成功判据

`decode_1k` 跑完后，先不要急着放大到 4 个点。只检查 `ops_perf_results_*.csv` 里这几列是否非空：

- `PM IDEAL [ns]`
- `PM COMPUTE [ns]`
- `PM BANDWIDTH [ns]`
- `PM FPU UTIL (%)`
- `DEVICE COMPUTE CB WAIT FRONT [ns]`
- `DEVICE COMPUTE CB RESERVE BACK [ns]`

如果 `tt-npe` 和 NoC traces 都工作正常，还应看到：

- `NOC UTIL (%)`
- `MULTICAST NOC UTIL (%)`
- `DRAM BW UTIL (%)`

## 当前阶段的停止条件

只要出现下面任意一种情况，就不建议继续放大到 `perf-fpu` 或 `NOC` rerun：

- `--preflight-only` 本身失败
- `preflight` 出现 `Timeout waiting for Ethernet core service remote IO request`
- `preflight` 出现 `Read unexpected run_mailbox value`

这时更好的做法是：

1. 停在健康探测层，不继续追加更激进的 profiling 参数
2. 等设备自行恢复
3. 重新跑一次 `--preflight-only`
4. 只有在它通过后，再恢复 `sum -> perf-fpu -> noc` 的升级顺序

如果还抓了 perf counter，还可能看到：

- `Avg FPU util on full grid (%)`

## 结果怎么解释

### PM 利用率

- `PM FPU UTIL (%)` 本质上是 performance model 视角下的 compute utilization
- 它不是原始硬件 perf counter，但对判断“当前 kernel 更接近 compute-limited 还是 bandwidth-limited”已经足够有用

### compute 空泡

- `DEVICE COMPUTE CB WAIT FRONT [ns] / DEVICE KERNEL DURATION [ns]`
  可以近似理解为 compute 在等输入
- `DEVICE COMPUTE CB RESERVE BACK [ns] / DEVICE KERNEL DURATION [ns]`
  可以近似理解为 compute 在等输出空间或下游释放

对这两个量的建议用法是：

- 先看绝对值 `[ns]`
- 再把它们除以 `DEVICE KERNEL DURATION [ns]` 得到 share

这样就能把现在只能从 reader/writer 侧间接推断的“空泡”，补成 compute 自身视角的直接计数。

## 推荐的两阶段执行顺序

### Phase A: 先验证列能不能出来

- 只跑 `decode_1k`
- 目标是验证 `ops_perf_results_*.csv` 里新列不为空

### Phase B: 再补最小代表集

- `decode_1k`
- `decode_4k`
- `decode_32k`
- `prefill_4k`

这一步完成后，就已经足够回答：

- PM utilization 在短 decode、长 decode、重 prefill 三种区间分别是什么状态
- compute 空泡是更偏 input-starved，还是更偏 output/backpressure-starved

## 如果 NoC/DRAM util 还是拿不到

优先排查这几件事：

- `tt-npe` 是否安装并可用
- rerun 时是否带了 `--collect-noc-traces`
- 输出目录里是否真的生成了 NoC trace 相关文件

如果这些仍然不工作，也不要阻塞整个 rerun。因为即使没有 `NOC/DRAM util`，你仍然可以先拿到：

- `PM FPU UTIL (%)`
- `DEVICE COMPUTE CB WAIT FRONT [ns]`
- `DEVICE COMPUTE CB RESERVE BACK [ns]`

这三项已经足够把“PM 利用率”和“compute 空泡”先补上大半。

## 对现有分析链路的最小影响

建议不要马上改写现有 `profile_flash_mla_wh_detailed.py` 的主路径。

最小改动方式是：

1. 保留当前 detailed sweep 继续产出 `stage-level` 结果
2. 单独增加这条 `PM/bubble probe` rerun
3. 先人工检查 `ops_perf_results_*.csv`
4. 确认列稳定后，再决定是否把 `ops_perf_results_*.csv` 合并进统一 JSON/Markdown

这样风险最低，也最符合“最小 rerun”的目标。

## 一句话结论

要把 `PM 利用率` 和 `compute 空泡` 真正跑出来，最小可行方案不是继续依赖 `cpp_device_perf_report.csv`，而是补一条基于 `python3 -m tracy -r` 的 `ops_perf_results_*.csv` rerun 路线，并优先使用更安全的默认组合：

- `--enable-sum-profiling`
- `-a device_compute_cb_wait_front`
- `-a device_compute_cb_reserve_back`

只有在设备状态稳定后，再额外打开：

- `--collect-noc-traces`
- `--profiler-capture-perf-counters=fpu`
