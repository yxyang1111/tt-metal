# S-FMLA Implementation Map

Python API for the paper's Block-Lane Flash MLA dataflow. This package is the
canonical entry point for benchmarks and profiling; low-level kernels remain in
`ttnn/` and grid constants in `models/demos/deepseek_v3_b1/micro_ops/flash_mla/`.

## Paper ↔ Code

| Paper (§3) | Symbol | This package | Device kernel |
|------------|--------|--------------|---------------|
| S-block count | N_S | `SFMLAGrid.num_s_blocks` | `sdpa_decode_program_factory.cpp` work split |
| Lane width | C_S | `SFMLAGrid.cores_per_lane` | `max_cores_per_head_batch` in `SDPAProgramConfig` |
| Q shards per batch | B = H_q/τ | `SFMLAMapping.num_q_shards` | height-sharded Q grid |
| Heads per shard | τ | `SFMLAWorkloadConfig.num_q_heads_per_core` | Q shard height |
| K chunk | L_k | `SFMLAWorkloadConfig.k_chunk_size` | `Sk_chunk_t` in reader |
| K DRAM reader | sender(i) | first core in S-block i | `reader_decode_all.cpp` |
| K NoC multicast | Broadcast | `use_k_mcast` path | `dataflow_common.hpp` |
| Lane reduction | Reduce | tree reduction | `writer_decode_all.cpp` |

## Three implementation tracks

```
                    ┌─────────────────────────────────────────┐
  Baseline (naive)  │ unfused matmul + softmax (TTNN MLA)     │
                    └─────────────────────────────────────────┘
                    ┌─────────────────────────────────────────┐
  FlashMLA (TT)     │ paged_flash_multi_latent_attention_decode│  ← impl A
                    │ generic height-sharded Q, paged KV        │
                    └─────────────────────────────────────────┘
                    ┌─────────────────────────────────────────┐
  S-FMLA (ours)     │ flash_multi_latent_attention_decode     │  ← this package
                    │ Block-Lane grid, ND K, K multicast      │
                    └─────────────────────────────────────────┘
```

Experimental unified-kernel path (not used by benchmarks):
`models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py` → `FlashMLADecode.op`
(falls back to `flash_multi_latent_attention_decode` on WH).

## Module layout

```
sfmla/
├── config.py           Workload parameters (H_q, d_k, L_k, C_S, τ)
├── dataflow/
│   ├── grid.py         Virtual grid φ, N_S × C_S Wormhole layouts
│   └── mapping.py      B ≤ C_S, batch×B ≤ active cores
├── runtime/
│   └── decode.py       Q/K placement + decode op invocation
└── reference.py        PyTorch golden
```

## Usage

```python
from mla_flash_attention_dev.sfmla import (
    SFMLAWorkloadConfig,
    build_decode_inputs,
    run_decode,
)

config = SFMLAWorkloadConfig(batch=8, seq_len=4096, cores_per_block=8)
inputs = build_decode_inputs(device, q_torch, kv_torch, config)
out = run_decode(device, inputs)
```

## C++ source of truth (device)

| Stage | File |
|-------|------|
| Program factory / core assignment | `ttnn/.../sdpa_decode/device/sdpa_decode_program_factory.cpp` |
| K read + multicast | `ttnn/.../sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp` |
| Multicast helpers | `ttnn/.../sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` |
| Online softmax + PV | `ttnn/.../sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp` |
| Tree reduction + write | `ttnn/.../sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp` |
| Python binding | `ttnn/.../sdpa_decode/sdpa_decode.cpp` → `flash_multi_latent_attention_decode` |

## Related docs

- Design: `docs/02-sfmla-design/flash-mla-dataflow-paper-section.md`
- DSE params: `docs/02-sfmla-design/flash-mla-impl-b-dse-formalization.md`
- Two-path comparison: `docs/03-implementation/mla-two-implementations-deep-dive.md`
