# FlashMLA：实现方式与具体形状示例

本文只保留两部分，并且只讨论和当前 `tt-metal` 代码路径里 `FlashMLADecode` 直接相关的内容。重点是两件事：

1. `FlashMLA` 这个 op 自己到底接收什么输入、产生什么输出、`KV` 发生了什么变化。
2. 给一个完整的具体例子，把进入 `FlashMLA` 前后以及运行过程中的关键 tensor shape 全部讲清楚。

## 1. FlashMLA 是如何实现的

当前仓库里的 `FlashMLA`，指的是 decode 场景下这条路径：

```text
FlashMLADecode.op(
    q_tensor,
    kv_cache_tensor,
    head_dim_v,
    cur_pos_tensor,
    output_tensor,
    scale,
    program_config,
    compute_kernel_config,
)
```

它不是传统意义上的“直接输入 `Q/K/V` 三个张量”的 attention。它接收的是：

- 已经被上游整理过的 `Q_attn`
- 一份统一的 `kv_cache`
- 当前 decode 位置 `cur_pos`

### 1.1 输入和输出是什么

从 op 语义上看，当前 FlashMLA 的核心输入输出是：

| 名称 | 逻辑形状 | 含义 |
|---|---|---|
| `q_tensor` | `[1, B, Hq, kvpe_dim]` | 已经整理好的 query |
| `kv_cache_tensor` | `[B, 1, S, kvpe_dim]` | 统一的 KV latent cache |
| `cur_pos_tensor` | `[B]` | 当前 token 的位置 |
| `head_dim_v` | 标量 | 输出 value 维度 |
| `scale` | 标量 | attention 缩放因子 |
| `output_tensor` | `[1, B, Hq, head_dim_v]` | FlashMLA 的输出 |

这里：

- `B` 是 batch
- `Hq` 是 query 头数
- `S` 是当前 cache 的最大序列长度
- `kvpe_dim` 是统一 KV latent 的宽度

在当前 DeepSeek-V3 这条实现里，常见取值是：

```text
kv_lora_rank = 512
qk_rope_head_dim = 64
kvpe_dim = 512 + 64 = 576
head_dim_v = 512
```

所以最常见的输入输出其实就是：

```text
q_tensor        : [1, B, Hq, 576]
kv_cache_tensor : [B, 1, S, 576]
output_tensor   : [1, B, Hq, 512]
```

### 1.2 正常 attention 的 Q / K / V 是什么样的

为了看清 FlashMLA 到底改了什么，先看普通 attention。忽略不同框架在维度顺序上的小差异，只看逻辑含义的话，标准 attention 通常是：

```text
Q      : [B, Sq, Hq, Dqk]
K      : [B, S,  Hkv, Dqk]
V      : [B, S,  Hkv, Dv]
Output : [B, Sq, Hq, Dv]
```

其中：

- `B` 是 batch
- `Sq` 是当前 query 的长度，decode 时通常 `Sq = 1`
- `S` 是历史 cache 长度
- `Hq` 是 query 头数
- `Hkv` 是 key/value 头数
- `Dqk` 是做 `QK^T` 时使用的 head 维度
- `Dv` 是 value 维度

如果站在 decode 的 cache 视角看，普通 attention 的本质就是：

1. 当前 token 生成一份 `Q`
2. 历史 token 已经各自存好一份 `K`
3. 历史 token 也已经各自存好一份 `V`
4. 运行时做

```text
scores = Q @ K^T
P = softmax(scores)
out = P @ V
```

所以普通 attention 的一个核心特点是：**`K` 和 `V` 是两份独立张量，也是两份独立 cache。**

如果写成 decode 时更接近当前代码风格的形状，可以理解为：

```text
Q      : [1, B, Hq, Dqk]
Kcache : [B, Hkv, S, Dqk]
Vcache : [B, Hkv, S, Dv]
```

这里再补一句：

- 在 `MHA` 里，通常 `Hkv = Hq`
- 在 `GQA/MQA` 里，通常 `Hkv < Hq`

但即使在 `GQA/MQA` 里，`K` 和 `V` 仍然还是两份分开的 cache；只是多个 Q 头共享同一组 KV 头而已。

### 1.3 FlashMLA 的 Q / K / V 是什么样的

FlashMLA 不是“没有 Q/K/V”，而是 **Q、K、V 的组织方式变了**。最核心的变化有两条：

1. `Q` 不再是原始 attention head 空间里的 query，而是先被变换到和 latent KV 对齐的空间。
2. `K` 和 `V` 不再分别存成两份 cache，而是合并成一份统一的 latent cache。

先看当前实现里 FlashMLA 真正接到的 `Q`。

上游不会直接把“原始 Q head”交给 FlashMLA，而是会先整理成：

```text
Q_attn = [ q_latent(512) | q_rope(64) ]   -> [1, B, Hq, 576]
```

这里的含义是：

- 原始 `q_nope` 会先投影到和 latent KV 对齐的 `512` 维空间
- `q_rope` 保留 `64` 维
- 两者拼接以后，形成 FlashMLA 实际使用的 query

再看 FlashMLA 的 `K` 和 `V`。

在当前实现里，单个 token 写入 cache 的不是分开的 `K_t` 和 `V_t`，而是一条统一的：

```text
kvpe_t = [ kv_nope_t(512) | kv_rope_t(64) ]   -> [1, 1, 1, 576]
```

把所有 token 累起来以后，就得到整份 cache：

```text
kv_cache_tensor : [B, 1, S, 576]
```

然后在 FlashMLA 内部：

- **整条 `kv_cache` 的 `576` 维都参与 `QK^T`，也就是它扮演 K**
- **同一条 `kv_cache` 的前 `head_dim_v` 列扮演 V**

也就是：

```text
K_chunk = kv_cache_chunk[..., :576]
V_chunk = kv_cache_chunk[..., :512]
```

因此，FlashMLA 的 Q/K/V 可以直接并排写成：

```text
Q : [1, B, Hq, 576]          = [q_latent(512) | q_rope(64)]
K : [B, 1, S, 576]           = 整条 kv_cache
V : [B, 1, S, 512]           = kv_cache[..., :512]
Output : [1, B, Hq, 512]
```

这里最重要的一点是：

> **FlashMLA 里的 `V` 不是一份独立存储的 tensor，而是 `K` 所在那份 unified KV cache 的前半部分视图。**

所以和正常 attention 对比，差别可以压缩成下面这张表：

| 项目 | 正常 attention | FlashMLA |
|---|---|---|
| `Q` | 原始 query head | 先投影成 `Q_attn = [q_latent | q_rope]` |
| `K` | 独立 `K cache` | 统一 `kv_cache` 整条 `576` 维 |
| `V` | 独立 `V cache` | `kv_cache[..., :head_dim_v]` 的视图 |
| cache 数量 | `K cache + V cache` 两份 | 一份 unified `kv_cache` |
| 计算 | `Q @ K^T`，再 `P @ V` | 形式一样，但 `K` 和 `V` 来自同一份 cache |

所以这里的变化可以总结成四句：

1. 普通 attention 把 `K` 和 `V` 当成两份独立状态保存。
2. FlashMLA 把它们折叠成一份 unified latent cache。
3. 普通 attention 的 `Q` 直接在原始 key space 里和 `K` 匹配。
4. FlashMLA 的 `Q` 会先被变换到 latent 对齐空间，再和 unified `kv_cache` 匹配。

### 1.4 Q 的变换发生在算子前，还是算子内

这个问题很容易混淆，因为“Q 的变化”其实有两层含义。

第一层是**数学上的表示变换**。也就是前面说的：

```text
原始 q_nope -> q_latent(512)
再与 q_rope(64) 拼接
-> Q_attn(576)
```

这一步发生在 **FlashMLA 算子进入之前**。也就是说，`FlashMLADecode.op` 接收到的 `q_tensor` 已经不是原始 query，而是已经整理好的 `Q_attn`。

第二层是**算子内部的数据搬运**。FlashMLA 内部当然仍然会“处理 Q”，但那不是再做一次数学变换，而是：

- 把 `Q_attn` 以 `HEIGHT_SHARDED` 的方式放在 L1
- 每个 output core 持有一个 `Q shard`
- 再把这份 `Q shard` 分发给需要参与计算的 worker

所以，结论可以一句话记住：

> **Q 的“表示变换”发生在 FlashMLA 算子之前；FlashMLA 算子内部对 Q 做的是分片、驻留和分发，而不是把原始 Q 再变换一遍。**

### 1.5 FlashMLA 和普通 attention 等价性

- **FlashMLA -> 普通 attention**：成立
- **普通 attention -> FlashMLA**：一般不成立

### 1.6 运行时是怎么计算的

把实现过程压缩一下，FlashMLA 实际上做的是下面这条数据流。

#### 第一步：Q 放在 L1，按 head shard

`q_tensor` 在 kernel 启动前就已经被放到 L1 的 sharded memory 里。它不是运行时从 DRAM 现读出来的，而是：

- 按 head 维做 `HEIGHT_SHARDED`
- 每个 output core 持有一个 `Q shard`
- 后续整次 op 生命周期里反复使用这份 `Q shard`

#### 第二步：KV cache 在 DRAM 里按 chunk 存

`kv_cache_tensor` 会沿序列轴按 `k_chunk_size` 切块。当前默认通常是：

```text
k_chunk_size = 128
```

所以：

```text
kv_cache_tensor [B, 1, S, 576]
-> 切成多个 chunk
每个 chunk : [B, 1, 128, 576]
```

这些 chunk 再按 round-robin 方式 ND-shard 到多个 DRAM bank。

#### 第三步：每个 S-block 只让一个 sender 去读 DRAM

每个 `S-block` 里只有一个 sender core 会回 DRAM 读它负责的 K chunk。

然后这个 sender 会把读上来的 chunk multicast 给同一个 `S-block` 里的其它 worker，所以：

- 同一个 K chunk 只从 DRAM 读一次
- 读上来以后在片上复用
- 其它 core 不需要重复回 DRAM 抢同一份数据

#### 第四步：TRISC 在本地做 online softmax

每个 worker 在自己手里已经有：

- 一份 `Q_shard`
- 当前负责的 `K_chunk`

然后它做：

```text
S = Q_shard @ K_chunk^T
S = S * scale
S = S + mask

m_new = max(m_old, rowmax(S))
alpha = exp(m_old - m_new)
P = exp(S - m_new)
l_new = alpha * l_old + rowsum(P)
o_new = alpha * o_old + P @ V_chunk
```

其中：

```text
V_chunk = K_chunk[..., :head_dim_v]
```

所以这里最重要的点是：

- `QK^T` 用的是整条 `576` 维 `K_chunk`
- `P @ V` 用的是同一个 chunk 的前 `512` 列
- 真正跨 chunk 保存下来的只有 running state `(m, l, o)`
- 不会显式保存整条序列的 logits 或概率矩阵

#### 第五步：不同 S-block 的 partial 结果做 tree reduction

对同一个 `Q shard` 来说，不同 `S-block` 会分别算出自己负责的那段序列的 partial result：

```text
(m_local, l_local, o_local)
```

这些 partial 会通过 tree reduction 合并成一个最终结果。合并公式和 online softmax 保持一致：

```text
m = max(m_a, m_b)
l = exp(m_a - m) * l_a + exp(m_b - m) * l_b
o = exp(m_a - m) * o_a + exp(m_b - m) * o_b
```

最终 root core 得到：

```text
o_final / l_final
```

并把它写到：

```text
output_tensor : [1, B, Hq, 512]
```

所以，如果只用一句话总结实现方式，可以记成：

> **FlashMLA 就是把 `Q` 固定在片上，把统一的 `kv_cache` 按序列分块从 DRAM 流进来；每块 `K` 同时承担 `QK^T` 和 `P @ V` 两个角色，其中 `V` 只是这块 `K` 的前几列视图；中间只保留 `(m, l, o)`，最后把不同块的 partial 做 tree reduction。**

## 2. 一个具体例子：把所有 tensor 形状都写清楚

下面用一个和 `test_flash_mla.py` 一致的例子来说明。这个例子适合讲 shape，因为它比较规整。

### 2.1 例子的配置

我们取下面这组参数：

| 参数 | 值 |
|---|---|
| `batch_size` | `1` |
| `num_heads` | `64` |
| `num_q_heads_per_core` | `8` |
| `num_kv_heads` | `1` |
| `kv_lora_rank` | `512` |
| `qk_nope_head_dim` | `128` |
| `qk_rope_head_dim` | `64` |
| `kvpe_dim` | `576` |
| `head_dim_v` | `512` |
| `k_chunk_size` | `128` |
| `max_seq_len` | `32768` |
| `decode_position` | `1023` |
| `S-block` 数 | `8` |

因为：

```text
decode_position = 1023
```

所以当前 token 实际可见的有效序列长度是：

```text
valid_seq_len = 1024
```

### 2.2 进入 FlashMLA 之前，Q 和 KV 已经被整理成什么样

这里先把和 FlashMLA 直接相关的上游张量列出来。

#### Query 侧

上游会先得到两部分 query：

```text
q_nope : [1, 1, 64, 128]
q_rope : [1, 1, 64, 64]
```

然后把 `q_nope` 投影到 latent 空间：

```text
q_latent : [1, 1, 64, 512]
```

再和 `q_rope` 拼接：

```text
Q_attn = concat(q_latent, q_rope, dim=-1)
       = [1, 1, 64, 576]
```

这就是 FlashMLA 真正看到的 `q_tensor`。

#### KV 侧

当前 token 写入 cache 的那一行是：

```text
kv_nope_t : [1, 1, 1, 512]
kv_rope_t : [1, 1, 1, 64]
kvpe_t    : [1, 1, 1, 576]
```

把所有 token 的 `kvpe_t` 依次写入以后，整份 cache 是：

```text
kv_cache_tensor : [1, 1, 32768, 576]
```

但在这一次 decode 中，真正会参与 attention 的前缀只有：

```text
kv_cache_used = kv_cache_tensor[:, :, :1024, :]
              = [1, 1, 1024, 576]
```

`cur_pos_tensor` 的逻辑含义是：

```text
cur_pos_tensor : [1]
```

实现上会把这个值复制到所有参与计算的 core 上，但它表达的逻辑仍然只是“当前位置是 1023”。

### 2.3 FlashMLA 的输入和输出 shape

于是，对这一次调用来说，FlashMLA 看到的主要输入输出就是：

```text
q_tensor        : [1, 1, 64, 576]
kv_cache_tensor : [1, 1, 32768, 576]
cur_pos_tensor  : [1]
output_tensor   : [1, 1, 64, 512]
```

注意这里：

- `kv_cache_tensor` 的物理分配长度是 `32768`
- 但这次实际只会用到前 `1024` 个位置
- 输出宽度是 `512`，因为 `head_dim_v = 512`

### 2.4 先按 Q head 切 shard

因为：

```text
num_heads = 64
num_q_heads_per_core = 8
```

所以 Q 会被切成 `8` 个 shard：

```text
Q_shard_0 : [1, 1, 8, 576]
Q_shard_1 : [1, 1, 8, 576]
...
Q_shard_7 : [1, 1, 8, 576]
```

如果把 batch 维和 `Sq=1` 这两个长度为 1 的维度先忽略掉，那么每个 shard 的核心形状就是：

```text
[8, 576]
```

这就是一个 output core 手里长期驻留的 `Q shard`。

### 2.5 再按序列把 KV cache 切成 chunk

因为：

```text
valid_seq_len = 1024
k_chunk_size = 128
```

所以一共会有：

```text
1024 / 128 = 8 个有效 chunk
```

也就是：

```text
K_chunk_0 : [1, 1, 128, 576]
K_chunk_1 : [1, 1, 128, 576]
...
K_chunk_7 : [1, 1, 128, 576]
```

同时，对每一个 chunk，`V` 不是单独的张量，而是它的前 `512` 列视图：

```text
V_chunk_0 = K_chunk_0[..., :512] : [1, 1, 128, 512]
V_chunk_1 = K_chunk_1[..., :512] : [1, 1, 128, 512]
...
V_chunk_7 = K_chunk_7[..., :512] : [1, 1, 128, 512]
```

所以到这里要非常明确：

- `K_chunk_i` 的 shape 是 `[1,1,128,576]`
- `V_chunk_i` 的 shape 是 `[1,1,128,512]`
- 但它们不是两份独立存储
- `V_chunk_i` 只是 `K_chunk_i` 的前半部分视图

### 2.6 这 8 个 chunk 怎么分给 8 个 S-block

因为这个例子里正好有：

```text
8 个有效 chunk
8 个 S-block
```

所以分配最整齐：

```text
S-block 0 -> K_chunk_0 : [1, 1, 128, 576]
S-block 1 -> K_chunk_1 : [1, 1, 128, 576]
S-block 2 -> K_chunk_2 : [1, 1, 128, 576]
S-block 3 -> K_chunk_3 : [1, 1, 128, 576]
S-block 4 -> K_chunk_4 : [1, 1, 128, 576]
S-block 5 -> K_chunk_5 : [1, 1, 128, 576]
S-block 6 -> K_chunk_6 : [1, 1, 128, 576]
S-block 7 -> K_chunk_7 : [1, 1, 128, 576]
```

每个 S-block 里的 sender 会从 DRAM 读自己这个 chunk，再 multicast 给同 block 的其它 worker。

### 2.7 单个 `Q shard` 和单个 `K chunk` 相遇时，局部 tensor shape 是什么

现在拿一个具体配对来讲，比如：

```text
Q_shard_3  对  K_chunk_5
```

忽略那些长度为 1 的维度以后，这时局部计算看到的 tensor shape 是：

```text
Q_local : [8, 576]
K_local : [128, 576]
V_local : [128, 512]
```

于是：

#### 1. 先做 `QK^T`

```text
scores_local = Q_local @ K_local^T
             = [8, 576] @ [576, 128]
             = [8, 128]
```

这表示：

- 当前这个 `Q shard` 里有 `8` 个 heads
- 当前这个 `K chunk` 里有 `128` 个 token 位置

所以这一小块 attention score 的形状就是：

```text
[8, 128]
```

#### 2. 再做 softmax 概率

```text
P_local : [8, 128]
```

#### 3. 再做 `P @ V`

```text
partial_o = P_local @ V_local
          = [8, 128] @ [128, 512]
          = [8, 512]
```

#### 4. 同时维护 online softmax 的状态

每个 worker 不会保存整条序列的完整概率矩阵，而是只维护：

```text
m_local : [8]
l_local : [8]
o_local : [8, 512]
```

这里：

- `m_local` 是每个 head 当前的 running row max
- `l_local` 是每个 head 当前的 running exp sum
- `o_local` 是每个 head 当前累积出来的 partial output

### 2.8 对一个 `Q shard` 来说，全部 8 个 chunk 合并后是什么 shape

对 `Q_shard_3` 来说，它会从 8 个 S-block 得到 8 份 partial result：

```text
(m_0, l_0, o_0) : [8], [8], [8, 512]
(m_1, l_1, o_1) : [8], [8], [8, 512]
...
(m_7, l_7, o_7) : [8], [8], [8, 512]
```

经过 tree reduction 之后，得到这个 shard 的最终结果：

```text
out_shard_3 : [1, 1, 8, 512]
```

同理，每个 Q shard 都会得到一份：

```text
out_shard_0 : [1, 1, 8, 512]
out_shard_1 : [1, 1, 8, 512]
...
out_shard_7 : [1, 1, 8, 512]
```

### 2.9 最终输出是怎么拼回去的

最后把 8 个 output shard 沿 head 维拼起来：

```text
output_tensor
= concat(
    out_shard_0,
    out_shard_1,
    ...,
    out_shard_7,
    dim=head
  )
= [1, 1, 64, 512]
```

这就是本次 FlashMLA op 的最终输出 shape。

### 2.10 把这个例子的所有关键 tensor shape 汇总成一张表

下面这张表可以当成这个例子的总索引。

| 张量 | shape |
|---|---|
| `q_nope` | `[1, 1, 64, 128]` |
| `q_rope` | `[1, 1, 64, 64]` |
| `q_latent` | `[1, 1, 64, 512]` |
| `Q_attn / q_tensor` | `[1, 1, 64, 576]` |
| `kv_nope_t` | `[1, 1, 1, 512]` |
| `kv_rope_t` | `[1, 1, 1, 64]` |
| `kvpe_t` | `[1, 1, 1, 576]` |
| `kv_cache_tensor` | `[1, 1, 32768, 576]` |
| `kv_cache_used` | `[1, 1, 1024, 576]` |
| `cur_pos_tensor` | `[1]` |
| `Q_shard_i` | `[1, 1, 8, 576]` |
| `K_chunk_i` | `[1, 1, 128, 576]` |
| `V_chunk_i` | `[1, 1, 128, 512]` |
| `scores_local` | `[8, 128]` |
| `P_local` | `[8, 128]` |
| `m_local` | `[8]` |
| `l_local` | `[8]` |
| `o_local / partial_o` | `[8, 512]` |
| `out_shard_i` | `[1, 1, 8, 512]` |
| `output_tensor` | `[1, 1, 64, 512]` |

如果把这张表再压成一句话，就是：

> **当前这个例子里，FlashMLA 做的事就是：拿 `Q_attn[1,1,64,576]` 去读 `kv_cache[1,1,1024,576]` 的前缀，把它切成 `8` 个 `K_chunk[1,1,128,576]`；每个 chunk 的前 `512` 列直接当 `V_chunk[1,1,128,512]`；最后输出 `output[1,1,64,512]`。**
