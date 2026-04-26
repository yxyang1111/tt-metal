# Development Log

## Goals

- Keep the project aligned with the current paper framing: `MLA decode as a spatial mapping problem`.
- Record implementation decisions, profiling notes, model assumptions, and test commands.
- Maintain a unified evidence chain for `Characterization -> Design -> Model + DSE -> Evaluation`.

## Notes

- Preferred model-side entry: `models/demos/deepseek_v3/tt/mla/`
- Preferred operator-side entry: `ttnn/cpp/ttnn/operations/transformer/sdpa/`
- Phase 0 scope frozen in `archive/phase-0-scope-note.md`
- Current paper-side mainline: `single-chip MLA decode`
- `prefill` remains as supporting evidence for forwarding / multicast / coupling analysis; `multi-chip` is deferred
- `autotuner` is currently positioned as offline/cached DSE, pending oracle-gap evidence
- Phase 1 code path map completed in `archive/phase-1-code-path-map.md`
- Experiment D execution order frozen in `experiment-D-ablation-plan.md`
- First target test fixed as `tests/ttnn/unit_tests/operations/sdpa/test_mla_prefill_v_embedding_space.py`
- **B0 (`NC-current-auto`)** baseline doc + result template: `experiment-D-b0-baseline.md`（在目标设备上补全性能数字）

## Paper Framing Update

Current default interpretation for all new writeups:

- The paper is **not** positioned as "a faster FlashMLA kernel".
- The paper is positioned as `operator characterization + dataflow design + cost model + DSE + architecture implication`.
- Old `prefill-first` notes remain useful as engineering history, but should no longer be read as the final paper thesis.
- See `sf-mla-paper-positioning.md` for the current unified wording.

## 重要警告

> **绝对不可以使用 `tt-smi -r` 重置设备！** 该命令会导致硬件卡死。如果设备处于脏状态（如出现 `Read unexpected run_mailbox value` 或 `dispatch kernels still running` 错误），应通过杀掉残留进程后等待设备自行恢复，而非使用 `tt-smi -r`。

## 环境配置备忘

由于 Anaconda Python 的 RPATH 会强制加载自身的旧版 `libstdc++.so.6`（仅到 GLIBCXX_3.4.26），而 `_ttnn.so` 需要 GLIBCXX_3.4.31，运行测试时必须使用 `LD_PRELOAD`：

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
pytest tests/ttnn/unit_tests/operations/sdpa/test_mla_prefill.py -v
```

## Experiment B0 Baseline 实测记录

### 实验日期：2026-03-23

### 元信息

| 字段 | 值 |
| --- | --- |
| 日期 | 2026-03-23 |
| Git commit | `b207563825` (ring causal tests #40199) |
| 设备架构 | `WORMHOLE_B0` |
| 计算网格 | 8×7 = 56 核 |
| 构建类型 | Release (pre-compiled firmware) |
| Python | 3.10.14 (Anaconda) |
| pytest | 9.0.2 |
| LD_PRELOAD | `/usr/lib/x86_64-linux-gnu/libstdc++.so.6` (必需) |
| 备注 | B0 正确性验证，含本地 SDPA 修改 |

### 本地修改状态

当前分支基于 `dev`，有两个 SDPA 核心文件的本地修改：
- `sdpa_program_factory.cpp`：56 行变动（chain 构建 / mcast 判定相关）
- `reader_interleaved.cpp`：19 行变动（KV forwarding 读路径相关）

### P0 测试结果：`test_mla_prefill_v_embedding_space.py`

**主 guardrail — latent-space 与 embedding-space MLA prefill 一致性**

| # | 参数 | 结果 | 耗时 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | B=1, S_q=1024, S_kv=1024, NH=16, NKV=16, lora=512, rope=64 | **PASS** | 5.11s | causal, chunk=32 |
| 2 | B=1, S_q=4096, S_kv=4096, NH=16, NKV=16, lora=512, rope=64 | **PASS** | 16.49s | causal, chunk=32 |
| 3 | B=1, S_q=1024, S_kv=1024, NH=32, NKV=32, lora=512, rope=64 | **PASS** | 3.60s | causal, chunk=32 |
| 4 | B=1, S_q=4096, S_kv=4096, NH=32, NKV=32, lora=512, rope=64 | **PASS** | 13.54s | causal, chunk=32 |
| 5 | B=1, S_q=512, S_kv=16384, NH=32, NKV=32, lora=512, rope=64 | **PASS** | 13.03s | non-causal, 32/32 chains mcast |
| 6 | B=1, S_q=1024, S_kv=32768, NH=32, NKV=32, lora=512, rope=64 | **PASS** | 12.45s | non-causal, 32/32 chains mcast |
| 7 | B=1, S_q=4096, S_kv=131072, NH=32, NKV=32, lora=512, rope=64 | **SKIP** | — | WH_B0 已知 PCC 问题，预期跳过 |

**总计：6 passed, 1 skipped, 总耗时 68.48s**

关键观察：
- non-causal 长序列 case (#5, #6) 触发了 `per-chain hybrid` multicast：`32/32 chains using mcast`
- WH_B0 上所有 chunk size 被强制设为 32（测试代码中有 WH_B0 分支处理）
- PCC 阈值 ≥ 0.99，所有 pass case 均满足

### P1 测试结果：`test_mla_prefill.py`

**prefill 基础回归 — 含 paged attention**

| # | 参数 | PCC | 结果 | 耗时 |
| --- | --- | --- | --- | --- |
| 1 | B=2, S=1024, NH=128, NKV=1, lora=512, rope=64, bf16/bf8, paged bs=128 | 0.9997 | **PASS** | 13.54s |
| 2 | B=2, S=4096, NH=64, NKV=1, lora=256, rope=0, bf16/bf8, paged bs=128 | 0.9994 | **PASS** | 10.41s |
| 3 | B=2, S=1024, NH=128, NKV=1, lora=512, rope=64, bf8/bf4, paged bs=32 | 0.9888 | **PASS** | 16.10s |
| 4 | B=2, S=4096, NH=64, NKV=1, lora=256, rope=0, bf8/bf4, paged bs=32 | 0.9931 | **PASS** | 15.35s |

**总计：4 passed, 总耗时 58.36s**

关键观察：
- bf8/bf4 低精度场景 PCC 略低（0.988-0.993），但仍在阈值 0.98 以上
- paged attention chunked_flash_mla_prefill 路径正常工作
- rope=0 场景（纯 LoRA，无 RoPE）也能正确处理

### P2 测试结果：`test_mla_decode.py`

**decode 回归 — 防止 SDPA 改动影响 decode 分支**

| # | 参数 | PCC (3轮) | 结果 | 耗时 |
| --- | --- | --- | --- | --- |
| 1 | B=4, S=1024, NH=128, NKV=1, lora=512, rope=64, 64核 Q shard, paged | 0.9999/0.9999/0.9998 | **PASS** | 5.71s |
| 2 | B=2, S=1024, NH=8, NKV=1, lora=128, rope=64, DRAM Q, paged | 0.9999/0.9998/0.9998 | **PASS** | 4.29s |

**总计：2 passed, 总耗时 12.31s**

关键观察：
- decode PCC 非常高（>0.999），说明 decode 路径精度优于 prefill
- program cache 验证通过（每次运行 3 轮迭代，仅生成 2 个 cache entry）
- reuse_k=True（V 复用 K 前 kv_lora_rank 列）模式正常

### B0 结论

1. **正确性全部达标**：三层测试（P0/P1/P2）共 12 个 case，12 passed + 1 expected skip
2. **MLA prefill 主路径健康**：latent-space 和 embedding-space 输出 PCC ≥ 0.99
3. **本地 SDPA 修改未引入回退**：所有 decode 和 prefill 路径正确
4. **Multicast 已在 non-causal 场景生效**：长序列 case 观察到 `32/32 chains using mcast (per-chain hybrid)`
5. **环境稳定**：LD_PRELOAD 解决了 GLIBCXX 兼容问题，设备可正常打开和关闭

### B0 退出检查清单更新

- [x] `B0` 含义与 `NC-current-auto` 在本文档中已固定，不再与 `B1` 混淆
- [x] P0 测试命令已记录，并在目标环境至少跑通一次
- [x] P1/P2 测试也已通过
- [ ] `W1–W8` 结果表已建立（可先空，但列齐全）— 待后续性能指标采集
- [x] `dev_log.md` 已记录本次 B0 结论

---

## DeepSeek 模型级 MLA 测试执行记录

### 实验日期：2026-03-23

### 测试清单与设备兼容性分析

DeepSeek 仓库中共有 5 套 MLA 相关测试，逐一分析后发现当前 **单设备 WH_B0 (8×7 = 56 核)** 无法运行任何 DeepSeek 模型级 MLA 测试：

| 测试文件 | 位置 | 设备要求 | 结果 | 原因 |
| --- | --- | --- | --- | --- |
| `test_mla.py` | `deepseek_v3/tests/` | TG/DUAL/QUAD 多设备 mesh + HF 权重 | **不可运行** | 需要 `MLA2D` + `mesh_device`，单芯片无法满足 |
| `test_flash_mla_deepseek.py` (trace mode) | `deepseek_v3/tests/fused_op_unit_tests/mla/` | 单设备，但需 64 核 (8×8) | **FAIL** | `num_cores=64 > available=56`，WH_B0 仅 8×7 |
| `test_flash_mla_deepseek.py` (aliasing) | 同上 | 1×2 mesh + FABRIC_1D | **不可运行** | 需要 2 设备 mesh |
| `test_flash_mla.py` | `deepseek_v3_b1/tests/unit_tests/` | Blackhole 架构 (≥11 列) | **FAIL** | `Device must have at least 11 columns, got 8` |
| `test_ring_joint_mla.py` | `deepseek_v3_d_p/tests/op_unit_tests/` | 4×2 或 2×2 mesh + FABRIC_1D | **不可运行** | 需要多设备 mesh |
| `mla_deepseek_perf_and_pcc_tests.py` | `deepseek_v3/tests/fused_op_unit_tests/` | TG 32 设备集群 | **不可运行** | 性能基准测试，需 TG 完整集群 |

### 实际执行的测试

#### 1. `test_deepseek_v3_mla_flash_mla_trace_mode` (MESH_DEVICE=N150)

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export MESH_DEVICE=N150
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_flash_mla_deepseek.py::test_deepseek_v3_mla_flash_mla_trace_mode -v
```

- **结果**：**FAIL** — `Target number of cores 64 is greater than total number of available cores 56`
- **原因**：测试硬编码 `num_cores=64`，对应 `min(users_per_device * num_heads, device_cores) = min(4 * 128, 64) = 64`，但 WH_B0 只有 56 核
- **耗时**：6.25s（在 core 分配时即失败）

#### 2. `test_flash_mla_decode` (deepseek_v3_b1, position=127)

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_flash_mla.py -k "test_flash_mla_decode[32768-128-127-1]" -v
```

- **结果**：**FAIL** — `Device must have at least 11 columns, got 8`
- **原因**：`FlashMLADecode.op()` 在 `op.py:390` 断言 `device_grid.x >= 11`，该算子为 Blackhole 架构（14×10 网格）设计
- **耗时**：3.33s（创建完 KV cache 和参考输出后，在 op 调用时失败）
- **观察**：设备成功完成了 DRAM bank → optimal worker core 映射（12 banks），KV cache ND sharding 配置也正确，仅在计算 grid 尺寸检查时失败

### 与 TTNN 单元测试的关系

**重要说明**：上一轮已成功运行的 TTNN 单元测试（P0/P1/P2）实际上测试的正是 DeepSeek MLA 使用的相同底层算子：

| TTNN 单元测试 | 对应 DeepSeek MLA 调用 |
| --- | --- |
| `flash_mla_prefill` | `mla1d.py → forward_prefill → ttnn.transformer.flash_mla_prefill` |
| `chunked_flash_mla_prefill` | `mla1d.py → forward_prefill → ttnn.transformer.chunked_flash_mla_prefill` (paged) |
| `flash_multi_latent_attention_decode` | `mla1d.py → _fwd_decode_flash_mla → ttnn.transformer.flash_multi_latent_attention_decode` |
| `paged_flash_multi_latent_attention_decode` | `mla1d.py → _fwd_decode_flash_mla → ttnn.transformer.paged_flash_multi_latent_attention_decode` (paged) |

因此，**P0/P1/P2 的 12 个通过 case 已经验证了 DeepSeek MLA 的核心算子正确性**，只是未覆盖模型层（权重加载、RoPE、投影矩阵、cache 更新等）。

### 结论

1. **DeepSeek 模型级 MLA 测试全部不兼容当前单 WH_B0 环境**：需要多设备集群（TG/DUAL/QUAD）或更大网格（Blackhole）
2. **核心算子正确性已通过 TTNN 单元测试验证**：`flash_mla_prefill`、`chunked_flash_mla_prefill`、`flash_multi_latent_attention_decode`、`paged_flash_multi_latent_attention_decode` 全部 PASS
3. **如需运行完整 DeepSeek MLA 模型测试**，需要：
   - TG 集群（32 设备）用于 `test_mla.py` 和性能测试
   - Blackhole 设备（≥11×10 网格）用于 `test_flash_mla.py` (b1)
   - 至少 2 设备用于 aliasing 验证测试
   - 至少 4×2 或 2×2 mesh 用于 ring joint MLA 测试

---

## TODO

- [x] 记录 WH 8-core S-block 当前状态（见 `flash-mla-wh-8core-status.md`，2026-04-12）
- [x] Clarify prefill vs decode scope
- [x] Identify the first target test
- [x] Track performance baselines（模板与流程已落地；表中实测数需在 TT 环境补全）
- [x] Build `model -> op -> kernel` code path map
- [x] Freeze Experiment D `B0 -> B4` execution order
- [x] Run B0 correctness baseline (P0/P1/P2 全部通过 2026-03-23)
- [ ] 采集 W1-W8 性能指标（latency / NoC / DRAM 等）
- [ ] 开始 B1 (per-chain hybrid) 实现
