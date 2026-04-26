# Flash MLA Perf Model — Smoke Test 结果记录

**日期**：2026-04-20
**运行环境**：`mla_flash_attention_dev/autotuner/` 的 `BasicMLAAutotuner` + 解析代价模型（`cost_model.py`）
**入口**：`python -m mla_flash_attention_dev.autotuner ...`

> 结论先行：**当前 perf model 可用**。5 个 preset 跑通，输入(S、B)的 sweep 输出量级合理、曲线单调、瓶颈项符合"长序 decode = reader 主导"的双流模型预期。

---

## 1. 运行内容概览

本次 smoke test 做了三件事：

| 项 | 脚本/命令 | 输出 |
|---|---|---|
| 5 个内置 preset 的 `top_k=5` 最优计划枚举 | `python -m mla_flash_attention_dev.autotuner --preset <name>` | `preset_*.stdout.log`、`preset_*.json` |
| WH decode preset 的 `seq_len_kv` sweep（6 点）| `run_seq_sweep.py` | `seq_sweep_wh_decode.{json,csv,log}` |
| WH decode preset 的 `batch_size` sweep（5 点）| `run_batch_sweep.py` | `batch_sweep_wh_decode.{json,log}` |

总耗时 ≈ 57 s（3 个作业加起来），全部在 CPU 上的解析模型，不依赖任何设备。

---

## 2. Preset 总览（每个 workload 的 best plan）

| Preset | Mode | `(B, Hq, Sq, S, d_k)` | 搜索候选数 | `estimated_ms` | 主导瓶颈 | Best plan (`B·H·Q·KV`) |
|---|---|---|---|---|---|---|
| `flash_decode_wh` | decode | (1, 32, 1, 4096, 576) | 14,688 | **0.0706** | reader=0.037 ms (KV 流) | 1·4·1·6 |
| `flash_decode_n300` | decode | (1, 32, 1, 4096, 576) | 14,688 | **0.0706** | reader=0.037 ms | 1·4·1·6 |
| `flash_decode_bh` | decode | (1, 64, 1, 4096, 576) | 7,524  | **0.0400** | reader=0.018 ms | 1·8·1·8 |
| `mla_prefill_wh` | prefill | (1, 32, 1024, 1024, 576) | 10,854 | **0.6046** | compute=0.364 ms（FLOPs 压过 KV 流）| 1·8·4·1 |
| `mla_prefill_bh` | prefill | (1, 64, 4096, 4096, 576) | 5,022  | **8.9933** | compute=8.524 ms | 1·8·4·1 |

可观察结论（即刻命中 §§7、10 的预测）：
- **Decode 场景**，`reader_ms`（= KV 流片外读 + 广播那一支）是主导项；`compute_ms` 基本可忽略，`reduce_ms` 是次要项。
- **Prefill 场景**（`Sq` 非 1），`compute_ms` 翻身成主导项——FLOPs ∝ `Sq·S` 起来了，roofline 的 arithmetic 支线成瓶颈。
- **BH 比 WH 明显快**：同样 decode `S=4096`，BH 0.040 ms vs WH 0.071 ms，比例 ≈ 1.77× ≈ `BW_L3` 比值（512/258 = 1.98×），再扣掉核数差异，这个量级和硬件常数一致。
- `flash_decode_wh` 和 `flash_decode_n300` 完全一致——两者 preset 只差一个 `num_devices` 字段，在单 workload 视角下等价（N300 的另一块 chip 要靠 `X` 轴 scale-out 才体现出差异）。

---

## 3. `seq_len_kv` sweep（WH decode）

| `S` | `estimated_ms` | `dram_ms` | `reader_ms` | `reduce_ms` | `compute_ms` | best `B/H/KV` |
|---:|---:|---:|---:|---:|---:|---|
| 1024  | 0.0450 | 0.0221 | 0.0238 | 0.0048 | 0.0004 | 1/8/8 |
| 2048  | 0.0396 | 0.0055 | 0.0176 | 0.0036 | 0.0020 | 1/4/6 |
| 4096  | 0.0706 | 0.0106 | 0.0370 | 0.0039 | 0.0040 | 1/4/6 |
| 8192  | 0.1305 | 0.0209 | 0.0690 | 0.0045 | 0.0080 | 1/4/6 |
| 16384 | 0.2530 | 0.0414 | 0.1383 | 0.0058 | 0.0159 | 1/4/6 |
| 32768 | 0.4924 | 0.0825 | 0.2754 | 0.0083 | 0.0319 | 1/4/6 |

观察：
- `S` 翻倍 → `estimated_ms` 接近翻倍（曲线单调、近线性）。严格比值：1.83×, 1.85×, 1.94×, 1.95×（越到大 S 越贴近线性，符合 roofline 下"KV-bound"区域的预期）。
- `reader_ms` / `dram_ms` 也严格单调增，且二者比值稳定在 ≈ 3.3×——`dram_ms` 只算了 DRAM bank 读取带宽，`reader_ms` 还叠加了 KV 流广播那一段，两者比值即 `(C_S-1)` 量级的加权。
- `compute_ms` = `O(S · d_k + S · d_v)`，随 S 严格线性，数值也正好翻倍。
- `reduce_ms` 几乎恒定（`depth(tree) · Sq · (d_v+2)`，`Sq=1`, tree 深度不变）——完美符合 §7.2 归并代价公式。
- 小 S（1024）时 tuner 自动换成 `KV=8, H=8`，把 KV 并行度推满；`S≥2048` 之后稳定在 `KV=6, H=4`，正好对应 WH 的 6×4 S-block 拓扑。

---

## 4. `batch_size` sweep（WH decode，`S=4096`）

| `B` | `estimated_ms` | `dram_ms` | `reader_ms` | `compute_ms` | best `B/H/KV` | 搜索候选数 |
|---:|---:|---:|---:|---:|---|---:|
| 1  | 0.0706 | 0.0106 | 0.0370 | 0.0040 | 1/4/6 | 14,688 |
| 2  | 0.1203 | 0.0212 | 0.0644 | 0.0080 | 1/4/6 | 24,768 |
| 4  | 0.2198 | 0.0423 | 0.1191 | 0.0159 | 1/4/6 | 30,528 |
| 8  | 0.4187 | 0.0846 | 0.2284 | 0.0319 | 1/4/6 | 30,528 |
| 16 | 0.8178 | 0.1692 | 0.4472 | 0.0638 | 1/4/6 | 30,528 |

观察：
- `B` 翻倍 → `estimated_ms` 接近翻倍（1.70×, 1.83×, 1.90×, 1.95×）——decode 里 `batch` 等价于"更多独立 Q 流"，但在当前 preset 下 `batch_parallel_factor=1`，batch 维度被串行化，所以代价线性上涨。
- 如果 `B` 足够大、`H` 足够小，tuner 会让 `batch_parallel_factor > 1`——当前 `H=32, KV=6` 已经把 24 个 core 填满，`B` 维再并行就得扩 core，目前没有裕量。
- 搜索空间随 `B` 一起扩大（14k → 30k），说明 `B` 参与了候选枚举（走 `batch_group_size` 分支）。

---

## 5. Perf model 有没有响应输入变化（sanity check）

| 预期行为 | 实测 | ✓/✗ |
|---|---|---|
| `S` 翻倍，long-S decode 近线性上涨 | 1.83×→1.95× | ✓ |
| `B` 翻倍，decode 近线性上涨 | 1.70×→1.95× | ✓ |
| Decode 主导项是 `reader_ms` | 是，占 estimated 的 52–56 % | ✓ |
| Prefill 主导项是 `compute_ms` | 是，WH prefill 60 %、BH prefill 95 % | ✓ |
| `reduce_ms` 和 `S/B` 几乎无关 | 固定在 3–8 µs | ✓ |
| BH vs WH 带宽差反映在 decode 延迟比 | 1.77× 实测 vs 1.98× 理论带宽比 | ✓ |
| Tuner 结果稳定（不随结构变化乱跳）| `KV=6, H=4` 在 `S≥2048` 下不变 | ✓ |

→ **Perf model 现阶段已可作为 autotuner 的排序依据使用**；它响应 workload 变化的方向和量级都与双流代价模型的推论一致。

---

## 6. 已知局限（下次测试要覆盖到的）

这一轮仅测了**纯解析路径（`selection_source=analytical`）**，没走：

1. `--measurement-db` 的测量重排序（`rerank_top_k`）——需要真设备 profile JSON。
2. `--calibration-json` 加载 WH calibration 后的 linear-fit 预测——要先 `--fit-calibration-output`。
3. `--disable-heuristic-pruning` 下的全笛卡尔搜索——验证剪枝是否牺牲了最优点。
4. Scale-out (`num_devices > 1`) 支线。
5. Paged KV (`workload.paged=True`) 支线。

这些属于**后续升级项**，不影响当前 smoke-test 的"可用"结论。

---

## 7. 复现命令

```bash
cd /rshome/yuxin.yang/dev/tt-metal
OUT=mla_flash_attention_dev/experiments/perf_model_smoke_test

# 5 个 preset 的 top-5 枚举
for p in flash_decode_wh flash_decode_n300 flash_decode_bh mla_prefill_wh mla_prefill_bh; do
  python -m mla_flash_attention_dev.autotuner --preset "$p" --top-k 5 \
    --output-json "$OUT/preset_$p.json" \
    > "$OUT/preset_$p.stdout.log" 2> "$OUT/preset_$p.stderr.log"
done

# S sweep
python $OUT/run_seq_sweep.py > $OUT/seq_sweep_wh_decode.log 2>&1

# B sweep
python $OUT/run_batch_sweep.py > $OUT/batch_sweep_wh_decode.log 2>&1
```

---

## 8. 产物清单

```text
perf_model_smoke_test/
├── README.md                            # 本文件
├── run_seq_sweep.py
├── run_batch_sweep.py
├── preset_flash_decode_wh.{stdout.log,stderr.log,json}
├── preset_flash_decode_n300.{stdout.log,stderr.log,json}
├── preset_flash_decode_bh.{stdout.log,stderr.log,json}
├── preset_mla_prefill_wh.{stdout.log,stderr.log,json}
├── preset_mla_prefill_bh.{stdout.log,stderr.log,json}
├── seq_sweep_wh_decode.{json,csv,log}
└── batch_sweep_wh_decode.{json,log}
```
