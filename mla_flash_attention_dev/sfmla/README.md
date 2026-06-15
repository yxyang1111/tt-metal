# S-FMLA Implementation Map

Python API for the paper's Block-Lane Flash MLA dataflow. This package is the
canonical entry point for benchmarks and profiling; low-level kernels remain in
`ttnn/` and grid constants in `models/demos/deepseek_v3_b1/micro_ops/flash_mla/`.

## Paper ↔ Code

| Paper (§3) | Symbol | This package | Device kernel |
|------------|--------|--------------|---------------|
| S-block count | N_S | `SFMLAGrid.num_s_blocks` | `sdpa_decode_program_factory.cpp` work split |
| Lane width | C_S | `SFMLAGrid.cores_per_lane` / `SFMLAWorkloadConfig.cores_per_block` | `max_cores_per_head_batch` in `SDPAProgramConfig` (set to C_S, not batch×B) |
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

## Custom N_S × C_S grids

Three ways to go beyond catalog floorplans (`wh_6x4`, `wh_6x8`):

| Mode | Config fields | Use when |
|------|---------------|----------|
| Auto-place | `num_s_blocks`, `lane_cols`, `lane_rows` | Free N_S×C_S; cores placed near DRAM banks on device |
| Manual anchors | `export_sfmla_grid.py --anchors` → JSON | Rectangular lanes at chosen (x,y) per S-block |
| Full manual | `grid_layout_json` or `custom_core_coords` | Arbitrary core lists per S-block |

```python
from mla_flash_attention_dev.sfmla import SFMLAWorkloadConfig, auto_place_wormhole, build_grid_class

# Auto 4×2 lanes, 5 S-blocks → N_S=5, C_S=8
config = SFMLAWorkloadConfig(
    batch=4, seq_len=4096,
    num_s_blocks=5, lane_cols=4, lane_rows=2,
)
spec = auto_place_wormhole(device, num_s_blocks=5, lane_cols=4, lane_rows=2)
grid_cls = build_grid_class(spec)
```

Export a placement JSON (auto or manual):

```bash
python mla_flash_attention_dev/experiments/profiling/export_sfmla_grid.py \
  --num-s-blocks 4 --lane-cols 2 --lane-rows 2 --output /tmp/grid.json

python mla_flash_attention_dev/experiments/sweeps/run_wh_mla_batch_seq_sweep.py \
  --methods sfmla --modes decode --batches 4 \
  --sfmla-num-s-blocks 4 --sfmla-lane-cols 2 --sfmla-lane-rows 2
```

Constraints still apply: **B ≤ C_S**, **batch×B ≤ N_S×C_S**, cores must be unique and inside the device grid.

## Module layout

```
sfmla/
├── config.py           Workload parameters (H_q, d_k, L_k, C_S, τ, N_S)
├── dse.py              validate_config / list_valid_configs
├── dataflow/
│   ├── catalog.py      WH topology catalog (wh_6x4, wh_6x8, bh_8x8)
│   ├── custom.py       GridLayoutSpec, build_grid_class, tree reduction
│   ├── placement.py    auto_place_wormhole (DRAM-affinity search)
│   ├── subset.py       Truncate to fewer active S-blocks
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

config = SFMLAWorkloadConfig(
    batch=8,
    seq_len=4096,
    cores_per_block=8,
    num_q_heads_per_core=8,
    k_chunk_size=128,
    num_s_blocks_active=4,  # optional N_S sweep
)
inputs = build_decode_inputs(device, q_torch, kv_torch, config)
out = run_decode(device, inputs)
```

Sweep and profiling entry points accept the same knobs:

```bash
python mla_flash_attention_dev/experiments/sweeps/run_wh_mla_batch_seq_sweep.py \
  --methods sfmla --modes decode \
  --sfmla-cores-per-block 8 --sfmla-num-q-heads-per-core 8 \
  --sfmla-k-chunk-size 128 --sfmla-num-s-blocks-active 4

python mla_flash_attention_dev/experiments/profiling/profile_sfmla_wh.py \
  --batch 8 --seq-len 4096 --sfmla-cores-per-block 4
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
