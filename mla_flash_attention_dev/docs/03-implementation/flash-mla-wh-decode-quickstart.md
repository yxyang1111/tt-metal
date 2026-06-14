# FlashMLA WH Decode 使用说明

本文说明如何在 Wormhole (`WH`) 上使用当前可工作的 `FlashMLA` decode 路径。

## 1. 当前状态

当前 `WH` 上的 `FlashMLA decode` 已经验证可运行。

- 入口函数：`models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py` 中的 `FlashMLADecode.op()`
- 当前 `WH` 实现策略：
  - 用户仍然调用 `FlashMLADecode.op(...)`
  - 在 `WH` 上，代码会自动走稳定后端 `ttnn.transformer.flash_multi_latent_attention_decode`
  - 如果你显式设置 `allow_wh_fallback=True`，且设备后端失败，才会退回 PyTorch reference fallback
- 推荐真实使用时设置：`allow_wh_fallback=False`

已验证命令：

```bash
ARCH_NAME=wormhole_b0 pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_flash_mla_wh.py -q -s
```

验证结果：

- `15 passed`
- decode 主路径通过
- 单点设备实测 `PCC = 0.9996366831858049`

## 2. 使用前检查

先确认硬件空闲：

```bash
fuser /dev/tenstorrent/0 2>&1 || echo "Device free"
```

如果上一次异常退出留下了 1GB hugepage 残留文件，先检查是否真的无人占用，再删除：

```bash
fuser /dev/hugepages-1G/tenstorrent 2>&1 || true
fuser /dev/hugepages-1G/device_0_channel_1_tenstorrent 2>&1 || true
rm -f /dev/hugepages-1G/tenstorrent
rm -f /dev/hugepages-1G/device_0_channel_1_tenstorrent
```

注意：

- 不要使用 `tt-smi -r`
- 运行时建议带上 `ARCH_NAME=wormhole_b0`

## 3. 输入输出约定

当前 decode 路径的核心张量约定如下。

### Q

- 形状：`[1, batch, num_heads, kvpe_dim]`
- 当前验证配置：
  - `batch = 1`
  - `num_heads = 32`
  - `kvpe_dim = 576`
- 推荐数据类型：`ttnn.bfloat16`
- 推荐布局：`TILE_LAYOUT`
- 推荐放置：`L1 HEIGHT_SHARDED`

### KV Cache

- 形状：`[batch, 1, max_seq_len, kvpe_dim]`
- 当前验证配置：
  - `max_seq_len = 32 * 1024`
  - `kvpe_dim = 576`
- 推荐数据类型：`ttnn.bfloat8_b`
- 推荐放置：`DRAM ND_SHARDED`

### Position

- 含义：当前 decode 位置
- 数据类型：`int32`
- 推荐布局：`ROW_MAJOR_LAYOUT`

### Output

- 形状：`[1, batch, num_heads, head_dim_v]`
- 当前验证配置：
  - `head_dim_v = 512`
- 推荐数据类型：`ttnn.bfloat16`
- 推荐放置：`L1 HEIGHT_SHARDED`

## 4. 推荐配置

### Program Config

```python
program_config = FlashMLAProgramConfig(
    k_chunk_size=128,
    exp_approx_mode=False,
    grid=FlashMLAOptimalGridNOC0_WH,
    allow_wh_fallback=False,
)
```

说明：

- `k_chunk_size=128` 是当前验证通过的配置
- `allow_wh_fallback=False` 表示强制使用真实设备后端
- 如果只是调试链路，不关心是否走设备后端，可以临时设为 `True`

### Compute Kernel Config

```python
compute_kernel_config = ttnn.types.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
    dst_full_sync_en=True,
)
```

这是当前验证通过的推荐配置。

## 5. 最小可用示例

下面是一个最小 decode 示例，基本等价于当前通过的单测配置。

```python
import torch
import ttnn

from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import (
    FlashMLADecode,
    FlashMLAOptimalGridNOC0_WH,
    FlashMLAProgramConfig,
)


device = ttnn.open_device(device_id=0)

batch_size = 1
num_heads = 32
num_q_heads_per_core = 8
kv_lora_rank = 512
qk_rope_head_dim = 64
qk_nope_head_dim = 128
qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
kvpe_dim = kv_lora_rank + qk_rope_head_dim
decode_position = 127
max_seq_len = 32 * 1024
scale = qk_head_dim ** -0.5

grid = FlashMLAOptimalGridNOC0_WH
num_q_shards = num_heads // num_q_heads_per_core
tiny_tile = ttnn.Tile((num_q_heads_per_core, 32))

s1_cores, _ = grid.BLOCKS[0]
q_cores = s1_cores[:num_q_shards]
q_core_grid = ttnn.CoreRangeSet(
    [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in q_cores]
)

q_mem_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(q_core_grid, (num_q_heads_per_core, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR),
)

out_mem_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(q_core_grid, (num_q_heads_per_core, kv_lora_rank), ttnn.ShardOrientation.ROW_MAJOR),
)

torch_q = torch.randn((1, batch_size, num_heads, kvpe_dim), dtype=torch.bfloat16)
tt_q = ttnn.from_torch(
    torch_q,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=q_mem_config,
    tile=tiny_tile,
)

program_config = FlashMLAProgramConfig(
    k_chunk_size=128,
    exp_approx_mode=False,
    grid=grid,
    allow_wh_fallback=False,
)

torch_cache = torch.randn((batch_size, 1, max_seq_len, kvpe_dim), dtype=torch.bfloat16)
kv_nd_shard_spec = ttnn.NdShardSpec(
    shard_shape=[1, 1, 128, kvpe_dim],
    grid=grid.optimal_dram_grid(),
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
)
kv_mem_config = ttnn.MemoryConfig(
    buffer_type=ttnn.BufferType.DRAM,
    nd_shard_spec=kv_nd_shard_spec,
)
tt_cache = ttnn.from_torch(
    torch_cache,
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=kv_mem_config,
)

grid_size = device.compute_with_storage_grid_size()
position_ids = torch.ones(batch_size, dtype=torch.int32) * decode_position
position_replicated = position_ids.repeat(grid_size.x * grid_size.y, 1)
pos_core_grid = ttnn.CoreRangeSet(
    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
)
pos_mem_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(pos_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
)
tt_position_ids = ttnn.from_torch(
    position_replicated,
    dtype=ttnn.int32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=pos_mem_config,
)

tt_out_template = ttnn.from_torch(
    torch.zeros((1, batch_size, num_heads, kv_lora_rank), dtype=torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=out_mem_config,
    tile=tiny_tile,
)

compute_kernel_config = ttnn.types.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
    dst_full_sync_en=True,
)

tt_out = FlashMLADecode.op(
    q_tensor=tt_q,
    kv_cache_tensor=tt_cache,
    head_dim_v=kv_lora_rank,
    cur_pos_tensor=tt_position_ids,
    output_tensor=tt_out_template,
    scale=scale,
    program_config=program_config,
    compute_kernel_config=compute_kernel_config,
)

torch_out = ttnn.to_torch(tt_out)
print(torch_out.shape)  # [1, batch, num_heads, 512]

ttnn.close_device(device)
```

## 6. 一个重要注意点

`WH` 路径下的 `output_tensor` 目前更像“输出模板”：

- 它提供目标 `dtype / layout / memory_config / tile`
- `FlashMLADecode.op(...)` 返回的张量会匹配这个模板
- 但调用时应始终使用返回值：

```python
tt_out = FlashMLADecode.op(...)
```

不要假设传进去的 `output_tensor` 一定被原地写回。

## 7. 推荐调用方式

真实 decode 使用时，建议遵守下面三条：

1. 用 `allow_wh_fallback=False`，确保你跑的是设备真实路径。
2. 用返回值 `tt_out = FlashMLADecode.op(...)`，不要忽略返回值。
3. 初次集成时先固定到已验证配置：
   - `num_heads=32`
   - `head_dim_v=512`
   - `kvpe_dim=576`
   - `k_chunk_size=128`
   - `math_fidelity=HiFi4`

## 8. 常见问题

### Q1: 运行前为什么要检查 `fuser /dev/tenstorrent/0`？

因为如果设备正被别的进程占用，decode 运行很容易失败或卡住。

### Q2: hugepage 残留文件会有什么影响？

会导致后续 `open_device()` 失败，常见报错就是 hugepage 或 pin pages 相关错误。

### Q3: 当前文档覆盖 prefill 吗？

不覆盖。本文只针对 decode。

### Q4: 当前 `WH` decode 走的是 `flash_mla.hpp` 自定义 kernel 吗？

不是。当前 `WH` 实际可用路径是 `op.py` 中接入的稳定内建 MLA decode 后端。

