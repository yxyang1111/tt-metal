# Wormhole FlashMLA 8-Core S-Block 当前状态

阶段性总结速读：`flash-mla-wh-8core-phase-summary.md`

## 1. 目标

当前分支：`flash-mla-wh-8core`

本轮尝试的目标不是一次性把所有 Wormhole DeepSeek FlashMLA 路径都改成“真正稳定的 8 核版”，而是先回答两个更具体的问题：

1. 能不能把 Wormhole 的 S block 从 `4 cores/block` 扩到 `8 cores/block`，把总活跃核从 `24` 提到 `48`。
2. 这样做之后，之前因为 `batch * q_shards > 24` 而直接不支持的 case，是否至少能先跑起来。

## 2. 已完成的代码改动

本轮已经完成下面几类改动：

- `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py`
  - 新增 `FlashMLAOptimalGridNOC0_WH_8C`
  - 新增 `get_flash_mla_wormhole_grid(cores_per_block=4|8)`
- `mla_flash_attention_dev/experiments/run_flash_mla_wh_smoke.py`
  - 新增 `--wh-cores-per-block`
  - 新增 `--standalone-batch-size`
  - standalone smoke 改为按 `batch * q_shards` 分配 Q cores
  - `position_ids` 改为简单 `ROW_MAJOR_LAYOUT`，避免旧版单 batch height-shard 假设
- `mla_flash_attention_dev/experiments/part1_baselines/run_part1_benchmarks.py`
  - 新增 `--deepseek-wh-cores-per-block`
  - DeepSeek decode 输入构造、支持性检查、probe checkpoint 元数据都接入 4/8 核选项
- `models/demos/deepseek_v3_b1/tests/unit_tests/test_flash_mla_wh.py`
  - 4 核和 8 核 WH 布局都加入了 grid/dram-bank 校验

## 3. 当前已经验证通过的内容

### 3.1 布局与 DRAM bank 校验

命令：

```bash
python -m pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_flash_mla_wh.py -k "grid_layout or dram_bank_validation" -q
```

结果：

- `wh_4c` 和 `wh_8c` 都通过
- 8 核版确认使用 `6 S blocks x 8 cores = 48 active cores`
- 8 核版每个 block 的 multicast dest 数是 `7`
- 当前 8 核版仍复用 6-block 的 tree reduction 拓扑：`6 -> 3 -> 2 -> 1`

### 3.2 最小 smoke 已经打通

命令：

```bash
python mla_flash_attention_dev/experiments/run_flash_mla_wh_smoke.py \
  --cases standalone_decode_wh \
  --wh-cores-per-block 8 \
  --standalone-batch-size 8 \
  --standalone-num-heads 32 \
  --standalone-num-q-heads-per-core 8 \
  --standalone-max-seq-len 1024 \
  --standalone-decode-position 511
```

结果：

- `status = passed`
- `grid = FlashMLAOptimalGridNOC0_WH_8C`
- `output_shape = [1, 8, 32, 512]`
- `PCC = 0.9986749746991035`
- `iteration_latencies_ms = [6141.329450998455, 124.47912700008601]`

解读：

- 第一轮时延很大，主要是首次编译/首次程序建立的冷启动影响。
- 第二轮约 `124.48 ms`，说明在这个最小场景下，8 核版已经能稳定完成实际 decode，而不是只通过静态布局检查。

### 3.3 最小 capability probe：4 核版 vs 8 核版

比较场景固定为：

- `B=8`
- `H=32`
- `H_kv=1`
- `value_dim=512`
- `rope_dim=64`
- `seq_len=8k`
- `deepseek_num_q_heads_per_core=8`
- 此时 `q_shards = 4`
- 所需 Q cores = `batch * q_shards = 8 * 4 = 32`

#### 4 核版 probe

输出路径：

- `mla_flash_attention_dev/experiments/part1_baselines/outputs_wh8_probe_4c_b8_h32_s8k/raw/capability_probe_results.json`

结论：

- `reference_attention`: `ok`
- `flash_attention`: `ok`
- `flash_mla`: `ok`
- `deepseek_flash_mla`: `unsupported`

直接原因：

```text
batch * deepseek_num_q_shards must be <= 24, got 32
```

#### 8 核版 probe

输出路径：

- `mla_flash_attention_dev/experiments/part1_baselines/outputs_wh8_probe_8c_b8_h32_s8k/raw/capability_probe_results.json`

结论：

- 四个 baseline 全部 `ok`
- `supported_by_all = true`

这说明当前 8 核尝试版已经完成了最关键的一步：**把 4 核版在 `B=8, H=32` 下的硬容量限制打通了。**

### 3.4 单 case 正式 benchmark（8 核版）

命令：

```bash
PART1_OUTPUT_DIR=".../outputs_wh8_bench_8c_b8_h32_s8k" \
python mla_flash_attention_dev/experiments/part1_baselines/run_part1_benchmarks.py \
  --cases decode_8k \
  --batches 8 \
  --num-heads-list 32 \
  --num-kv-heads-list 1 \
  --value-dims 512 \
  --rope-dims 64 \
  --decode-seq-lens 8192 \
  --deepseek-num-q-heads-per-core 8 \
  --deepseek-wh-cores-per-block 8 \
  --warmup-device 1 \
  --iters-device 3 \
  --warmup-reference 0 \
  --iters-reference 1 \
  --skip-render
```

输出路径：

- `mla_flash_attention_dev/experiments/part1_baselines/outputs_wh8_bench_8c_b8_h32_s8k/raw/part1_four_method_results.json`

关键结果：

| baseline | mean_ms | min_ms | max_ms | throughput |
|---|---:|---:|---:|---:|
| Reference Attention | `1611.240589` | `1611.240589` | `1611.240589` | `4.965 tok/s` |
| Flash Attention | `0.614944` | `0.584531` | `0.674971` | `13009.315 tok/s` |
| FlashMLA (TT Mainline) | `0.583889` | `0.504163` | `0.737102` | `13701.227 tok/s` |
| DeepSeek FlashMLA (8c) | `0.508484` | `0.474783` | `0.556097` | `15733.052 tok/s` |

解读：

- 在这个单 case、单配置、热身后 3 次测量的口径下，8 核 DeepSeek 已经不只是“能跑”，而且 **当前比 Flash / TT 主线 FlashMLA 都更快**。
- 这里必须注意：
  - 这是单 case 结果，不代表整组 sweep 结论。
  - DeepSeek 口径仍然是 `device_core_only_after_one_time_adapter_conversion`，不包含当前 Python 包装层物化开销。
  - Reference 仍是 host torch control baseline，不能和 TT device path 做同量级解读。

### 3.5 正式 batch sweep：4 核版 vs 8 核版（`H=32, seq_len=8k, q_shards=4`）

命令：

```bash
PART1_OUTPUT_DIR=".../outputs_wh8_batchsweep_4c_h32_s8k" \
python mla_flash_attention_dev/experiments/part1_baselines/run_part1_benchmarks.py \
  --batches 1 2 4 6 8 12 \
  --num-heads-list 32 \
  --num-kv-heads-list 1 \
  --value-dims 512 \
  --rope-dims 64 \
  --decode-seq-lens 8192 \
  --cases decode_8k \
          decode_8k_b2_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8 \
          decode_8k_b4_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8 \
          decode_8k_b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8 \
          decode_8k_b8_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8 \
          decode_8k_b12_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8 \
  --deepseek-num-q-heads-per-core 8 \
  --deepseek-wh-cores-per-block 4 \
  --warmup-device 1 \
  --iters-device 3 \
  --warmup-reference 0 \
  --iters-reference 1 \
  --skip-render

PART1_OUTPUT_DIR=".../outputs_wh8_batchsweep_8c_h32_s8k" \
python mla_flash_attention_dev/experiments/part1_baselines/run_part1_benchmarks.py \
  ...同上...
  --deepseek-wh-cores-per-block 8
```

输出路径：

- `mla_flash_attention_dev/experiments/part1_baselines/outputs_wh8_batchsweep_4c_h32_s8k/raw/part1_four_method_results.json`
- `mla_flash_attention_dev/experiments/part1_baselines/outputs_wh8_batchsweep_8c_h32_s8k/raw/part1_four_method_results.json`
- 汇总表：`mla_flash_attention_dev/experiments/part1_baselines/outputs/tables/wh_4c_vs_8c_batch_summary.md`

DeepSeek FlashMLA 对照结果：

| batch | 所需 Q cores (`batch * q_shards`) | 4c | 8c |
|---|---:|---|---|
| `1` | `4` | `ok`, `0.464268 ms`, `2153.930 tok/s` | `ok`, `0.476736 ms`, `2097.598 tok/s` |
| `2` | `8` | `ok`, `0.418401 ms`, `4780.103 tok/s` | `ok`, `0.364708 ms`, `5483.844 tok/s` |
| `4` | `16` | `ok`, `0.416083 ms`, `9613.459 tok/s` | `ok`, `0.436574 ms`, `9162.257 tok/s` |
| `6` | `24` | `ok`, `0.517223 ms`, `11600.405 tok/s` | `ok`, `0.444241 ms`, `13506.173 tok/s` |
| `8` | `32` | `unsupported` (`batch * deepseek_num_q_shards > 24`) | `ok`, `0.500324 ms`, `15989.649 tok/s` |
| `12` | `48` | `unsupported` (`batch * deepseek_num_q_shards > 24`) | `ok`, `0.598540 ms`, `20048.774 tok/s` |

解读：

- 这轮 sweep 把容量边界从 **`24 Q cores` 提升到 `48 Q cores`** 的效果验证得很清楚：在 `q_shards=4` 固定时，4 核版最多只支持到 `B=6`，8 核版则刚好支持到 `B=12`。
- 在两者都支持的重叠区间（`B=1/2/4/6`），8 核版**不是每个点都更快**，而是呈现“基本持平到局部更优”的形态：`B=2` 和 `B=6` 有明显收益，`B=1` 和 `B=4` 则略慢一点。
- 这说明 8 核化当前更像是**先解决容量瓶颈，再在部分 batch 点顺带带来性能收益**；它已经不是单纯的“能跑版”，但也还不能说已经形成稳定、单调的 latency 优势。
- 在新解锁的点上，8 核版已经具备竞争力：
  - `B=8` 时，DeepSeek 8c `0.500324 ms`，快于 Flash Attention `0.639117 ms`，也快于 TT 主线 FlashMLA `0.524437 ms`。
  - `B=12` 时，DeepSeek 8c `0.598540 ms`，已经非常接近 TT 主线 FlashMLA `0.582850 ms`，明显快于 Flash Attention `0.779554 ms`。

### 3.6 正式 head sweep：4 核版 vs 8 核版（`B=8, seq_len=8k, dqhpc=8`）

输出路径：

- `mla_flash_attention_dev/experiments/part1_baselines/outputs_wh8_headsweep_4c_b8_s8k/raw/part1_four_method_results.json`
- `mla_flash_attention_dev/experiments/part1_baselines/outputs_wh8_headsweep_8c_b8_s8k/raw/part1_four_method_results.json`
- 汇总表：`mla_flash_attention_dev/experiments/part1_baselines/outputs/tables/wh_4c_vs_8c_head_summary.md`

关键结果：

- 固定 `deepseek_num_q_heads_per_core=8` 时，`H=8/16/24/32` 分别对应 `q_shards=1/2/3/4`。
- 在 `B=8` 固定下：
  - 4 核版支持到 `H=24 (q_shards=3)`，但 `H=32 (q_shards=4)` 不支持。
  - 8 核版则把 `H=32 (q_shards=4)` 也正式打通。
- 重叠支持区里，8 核版在 `H=8/16` 基本持平，但在 `H=24` 已有明显收益：
  - `H=24`：4c `0.495132 ms` -> 8c `0.421641 ms`，约 **提升 17.4%**。
- 新解锁点 `H=32` 上，8 核版不仅可跑，而且在该轮同口径 benchmark 里：
  - DeepSeek 8c `0.495343 ms`
  - Flash Attention `0.615567 ms`
  - TT 主线 FlashMLA `0.529264 ms`
  - 即 **DeepSeek 8c 最快**

### 3.7 正式 seq_len sweep：4 核版 vs 8 核版（`B=8, H=24, dqhpc=8`）

输出路径：

- `mla_flash_attention_dev/experiments/part1_baselines/outputs_wh8_seqsweep_4c_b8_h24/raw/part1_four_method_results.json`
- `mla_flash_attention_dev/experiments/part1_baselines/outputs_wh8_seqsweep_8c_b8_h24/raw/part1_four_method_results.json`
- 汇总表：`mla_flash_attention_dev/experiments/part1_baselines/outputs/tables/wh_4c_vs_8c_seq_summary.md`

关键结果：

- 这组固定 `B=8, H=24, dqhpc=8`，因此 `q_shards=3`，所需 Q cores 固定为 `24`。
- 这意味着它不是“4 核版本来就不支持、8 核版负责解锁”的 case，而是专门用来看 **8 核是否带来纯性能提升**。
- 实测结果显示：
  - `1k`：8 核略慢（`0.327004 -> 0.335564 ms`）
  - `4k`：8 核明显更快（`0.395547 -> 0.357776 ms`，约 **+10.6%**）
  - `8k`：8 核小幅更快（约 **+1.6%**）
  - `16k`：8 核几乎持平（约 **+0.5%**）
  - `32k`：两者几乎完全一致（`1.053104 vs 1.053284 ms`）
- 这说明在 `q_shards=3` 这条路线上，8 核收益**不是随着 seq_len 单调变大**，而是主要集中在中等序列长度附近。
- 但方法对比上，8 核版 DeepSeek 从 `4k` 开始已经稳定快于 Flash Attention 和 TT 主线 FlashMLA；其中 `32k` 时：
  - DeepSeek 8c `1.053284 ms`
  - Flash Attention `1.637969 ms`
  - TT 主线 FlashMLA `1.332051 ms`
  - 即 **DeepSeek 8c 仍是最快**

### 3.8 正式 seq_len sweep：4 核版 vs 8 核版（`B=6, H=32, dqhpc=8`, `q_shards=4`）

输出路径：

- `mla_flash_attention_dev/experiments/part1_baselines/outputs_wh8_qshard4_seqsweep_4c_b6_h32/raw/part1_four_method_results.json`
- `mla_flash_attention_dev/experiments/part1_baselines/outputs_wh8_qshard4_seqsweep_8c_b6_h32/raw/part1_four_method_results.json`
- 汇总表：`mla_flash_attention_dev/experiments/part1_baselines/outputs/tables/wh_4c_vs_8c_qshard4_seq_summary.md`
- `4k` 复测：
  - `mla_flash_attention_dev/experiments/part1_baselines/outputs_wh8_qshard4_seq4k_recheck_4c_b6_h32/raw/part1_four_method_results.json`
  - `mla_flash_attention_dev/experiments/part1_baselines/outputs_wh8_qshard4_seq4k_recheck_8c_b6_h32/raw/part1_four_method_results.json`
- `4k` 高迭代稳定性检查：
  - `mla_flash_attention_dev/experiments/part1_baselines/outputs_wh8_qshard4_seq4k_stability10_4c_b6_h32/raw/part1_four_method_results.json`
  - `mla_flash_attention_dev/experiments/part1_baselines/outputs_wh8_qshard4_seq4k_stability10_8c_b6_h32/raw/part1_four_method_results.json`

关键结果：

- 这组固定 `B=6, H=32, dqhpc=8`，因此 `q_shards=4`，所需 Q cores 固定为 `24`。
- 这条路线比 `H=24/q_shards=3` 更接近 4 核版的容量上边界，因此更适合观察 8 核化在“高压力映射”上的收益。
- 主 sweep 结果显示：
  - `1k`：8 核小幅更快（约 **+1.5%**）
  - `4k`：8 核看起来明显更慢（约 **-18.9%**）
  - `8k`：8 核小幅更快（约 **+1.7%**）
  - `16k`：8 核小幅更快（约 **+1.0%**）
  - `32k`：8 核明显更快（约 **+9.5%**）
- 由于 `4k` 点反常，我额外做了单 case 复测：
  - 主 sweep：4c `0.285465 ms`，8c `0.351972 ms`
  - 复测：4c `0.360118 ms`，8c `0.348120 ms`
- 之后又补了一轮高迭代稳定性检查（`warmup=2, iters=10`）：
  - 4c `0.348591 ms`
  - 8c `0.334179 ms`
- 这说明 `4k` 点当前确实对 run-to-run 状态敏感，但两次针对性复测都更偏向 `8c >= 4c`，因此不能机械地把主 sweep 的 `-18.9%` 当成稳定退化。
- 因此对 `q_shards=4` 这条路线，更合理的总结是：
  - 8 核版整体上**不差于** 4 核版。
  - `4k` 更接近“近似持平到略优”，而不是稳定退化。
  - 长序列端（尤其 `32k`）已经出现更明确的收益。
- 方法对比上，8 核版 DeepSeek 在该轮 sweep 里：
  - `1k` 仍落后 Flash/TT 主线
  - `4k` 起已经在主 sweep 中领先两者
  - `8k/16k/32k` 均为三种 device baseline 里最快
  - 其中 `32k` 时：DeepSeek 8c `0.882992 ms`，TT 主线 `1.295553 ms`，Flash `1.333998 ms`

## 4. 当前尚未打通的问题

### 4.1 单 batch 强行上 8 q_shards 仍未打通

尝试命令：

```bash
python mla_flash_attention_dev/experiments/run_flash_mla_wh_smoke.py \
  --cases standalone_decode_wh \
  --wh-cores-per-block 8 \
  --standalone-num-heads 64 \
  --standalone-num-q-heads-per-core 8 \
  --standalone-max-seq-len 1024 \
  --standalone-decode-position 511
```

失败点：

```text
Statically allocated circular buffers in program 1 clash with L1 buffers ...
```

这说明：

- 目前已经打通的是“通过增加总 active cores，支持更大 `batch * q_shards` 容量”的路线。
- 但“单个 head-group 本身直接扩到 `8 q_shards`”这一条，还会撞到 builtin WH backend 的更深层限制，当前还不能算解决。

## 5. 当前结论

到这一步，可以把当前状态总结成四句话：

1. **8-core/S block 的 Wormhole 布局已经实现，并通过了基础布局与 DRAM bank 校验。**
2. **它已经把 `q_shards=4` 这条路线下的 DeepSeek 容量上限，从 4 核版的 `B<=6` 扩到了 8 核版的 `B<=12`。**
3. **在重叠支持区间里，8 核版表现为“总体可比、部分更快”，说明这不是纯功能性 hack，而是已经具备实际性能价值。**
4. **但“单 batch / 单 head-group 直接扩到 8 q_shards”的路径还没有打通，后续还要继续查 builtin backend 的 CB/L1 约束。**

## 6. 下一步建议

最自然的下一步有两条：

### 路线 A：把当前结果扩成正式 slice

目标：

- 在已经完成的 `H=32, seq_len=8k` batch sweep 基础上，继续扩到更完整的 `batch / num_heads / seq_len / value_dim` 切片
- 生成新的 fair-aligned summary table / chart，区分“重叠区性能变化”和“新增覆盖区收益”

价值：

- 能更系统地回答“8 核版的收益主要来自容量扩展，还是也会稳定改善性能”
- 也能直接和之前 paper/report 里的结论衔接

### 路线 B：继续啃 `8 q_shards`

目标：

- 追 `H=64, dqhpc=8, q_shards=8` 失败时的 `CB/L1 clash`
- 进一步确认限制是 builtin SDPA backend 的静态 CB 规划、Q shard 布局，还是别的 L1 占用项

价值：

- 如果打通，8 核版不仅能扩 batch 容量，还可能直接改变单 config 的 head-parallel 上限
- 这是“真正意义上的 8-core/S block”更强版本
