# Attention 方法与技术全景整理

更新时间：`2026-05-02`

## 1. 这份文档要回答什么

这份文档希望把今天常见的 attention 相关方法和技术放到同一张地图里，尽量回答四个问题：

1. 从最基础的 attention 到 Transformer 里的 `scaled dot-product attention`，主线到底是什么。
2. `MHA`、`MQA`、`GQA`、`MLA`、`CSA/HCA`、`Hamilton Attention` 这些名字分别属于哪一层。
3. `FlashAttention`、`PagedAttention`、`Ring Attention` 这类方法，到底是在改模型语义、改 kernel，还是改系统执行。
4. 2024-2026 这批长上下文 attention 新方法里，哪些方向最值得重点关注。

本文尽量做到全面，但要先强调一个很重要的事实：

> 很多 attention 名字并不在同一个抽象层上。
> `MHA/GQA/MLA/CSA/HCA` 更偏模型结构与 `Q/K/V` 表示；
> `FlashAttention/FlashMLA/Flash-Decoding` 更偏精确 kernel；
> `PagedAttention` 更偏 KV cache 管理；
> `Ring Attention/Hamilton Attention` 更偏多卡并行和通信。

如果不先把这些层次拆开，后面几乎一定会把不同问题混在一起。

## 2. 先给一张总图

| 层次 | 核心问题 | 代表方法/技术 | 它到底改了什么 |
|---|---|---|---|
| 基础语义层 | attention 从数学上在算什么 | Additive attention, Dot-product attention, Scaled dot-product attention | 定义匹配分数和加权求和 |
| 头与 KV 组织层 | `Q/K/V` 怎样分头、共享、压缩 | `MHA`, `MQA`, `GQA`, `MLA` | 改 head 结构、KV cache 形态和表达能力/成本权衡 |
| 位置与长度外推层 | 模型如何表示远距离位置信息 | Relative bias, `RoPE`, `ALiBi`, `YaRN`, `LongRoPE`, `Self-Extend` | 改位置编码或长度外推策略 |
| 稀疏/低秩/线性层 | 如何减少 attention 的时间/空间复杂度 | Longformer, BigBird, Reformer, Performer, Linformer, RetNet, `Kimi Linear`, `Native Sparse Attention` | 改可见性、核函数、近似方式或记忆机制 |
| 精确 kernel 层 | 不改变数学结果，怎么更快算出来 | `FlashAttention`, `FlashAttention-2`, `FlashAttention-3`, `FlashAttention-4`, `FlashMLA`, Flash-Decoding | 改执行顺序、tiling、online softmax、低精度管线 |
| KV cache / serving 层 | 长上下文推理时，如何更省显存/带宽 | `PagedAttention`, StreamingLLM, H2O, SnapKV, DuoAttention, `Quest`, `Kascade` | 改缓存管理、保留策略、页管理、头预算 |
| 分布式并行层 | 多卡如何分摊长序列 attention | Ulysses, Ring Attention, Striped Attention, Megatron CP, Hamilton Attention/TASP | 改通信拓扑、切分方式与 overlap |
| 混合长上下文架构层 | 如何把局部精确、稀疏检索、压缩摘要组合起来 | DeepSeek `CSA/HCA`, Infini-attention, `Native Hybrid Attention`, Star Attention | 多路径混合 attention，而不是单一配方 |

一句话记忆：

- `MHA/GQA/MLA`：更像“模型里把 attention 状态怎么表示”
- `FlashAttention`：更像“相同 attention 结果怎样更快算”
- `PagedAttention`：更像“KV cache 怎样像虚拟内存那样管理”
- `Hamilton Attention`：更像“多卡上 attention 通信怎样更高效”
- `CSA/HCA`：更像“长上下文模型如何把局部、稀疏、压缩三种路径混合起来”
- `Quest/MInference/Kascade`：更像“推理时按 query、tile 或 layer，只挑最相关的页、块或 token 去看”
- `DuoAttention/SnapKV/H2O`：更像“按 head 或 token 重要性，重新分配 KV cache 预算”
- `Native Sparse Attention (NSA)`：更像“把压缩、选择、滑窗三条 sparse path 原生合成一个可训练 attention”
- `Infini-attention/Native Hybrid Attention (NHA)`：更像“把局部精确 attention 和长期压缩记忆放进同一层”
- `Kimi Linear/RetNet/GLA`：更像“用线性/递归记忆替代大部分二次 attention，再保留必要的精确路径”

## 3. 从最基础的 attention 开始

### 3.1 经典 attention 的起点

Transformer 之前，attention 主要出现在 seq2seq 里，典型是：

- **Bahdanau / Additive Attention**：用一个小前馈网络给 query 和 key 打分。
- **Luong / Multiplicative Attention**：直接做点积或线性变换后的点积。

从抽象上讲，attention 的核心始终是两步：

1. 先算“当前 query 应该看哪些 key”
2. 再用这些权重去加权 value

### 3.2 Transformer 的标准形式

现代 Transformer 里最核心的对象是：

```text
Attn(Q, K, V) = softmax(QK^T / sqrt(d_k) + M) V
```

这里：

- `Q` 是 query
- `K` 是 key
- `V` 是 value
- `M` 往往是 mask，比如 causal mask
- `sqrt(d_k)` 是缩放项，用来稳定训练

这就是 **scaled dot-product attention**。

### 3.3 Self-Attention、Cross-Attention、Causal Attention

- **Self-Attention**：`Q/K/V` 都来自同一段序列。
- **Cross-Attention**：`Q` 来自当前序列，`K/V` 来自另一段序列，常见于 encoder-decoder 和多模态。
- **Causal / Masked Attention**：每个 token 只能看自己和过去，不能看未来，是自回归 LLM decode 的标准配置。

### 3.4 Multi-Head Attention, MHA

Transformer 最标志性的设计之一是 **Multi-Head Attention (MHA)**：

```text
head_i = Attn(Q W_i^Q, K W_i^K, V W_i^V)
MHA(Q, K, V) = Concat(head_1, ..., head_h) W^O
```

直觉上，多头的作用是：

- 不同 head 可以学不同关系
- 有的 head 更偏局部
- 有的 head 更偏检索
- 有的 head 更偏位置或格式模式

MHA 的优点是表达力强，但代价也直接体现在 KV cache 上：**每个 head 都要有自己的 K/V**，长上下文 decode 时非常吃显存和带宽。

## 4. 头与 KV 组织方式的演进

这一层回答的问题是：

> 在不改变“softmax attention”大框架的前提下，`K/V` 到底要按多少头存、怎么共享、怎么压缩？

### 4.1 MHA

- 每个 query head 对应独立的 key/value head
- 表达力最完整
- KV cache 最大
- decode 时最容易被 memory bandwidth 卡住

### 4.2 MQA

**Multi-Query Attention (MQA)** 的核心是：

- 保持很多 query heads
- 但所有 query heads 共享同一组 `K/V`

好处：

- KV cache 大幅下降
- decode 带宽压力显著减轻

代价：

- 共享过头时，表达能力可能下降
- 对某些模型，精度不如完整 MHA 稳

### 4.3 GQA

**Grouped-Query Attention (GQA)** 可以看成 MHA 和 MQA 之间的折中：

- query heads 仍然很多
- 但把它们分组
- 同组 query heads 共享一组 `K/V`

这也是今天很多 LLM 非常常见的选择，因为它通常能在：

- KV cache
- decode 吞吐
- 模型精度

之间给出更稳的折中点。

### 4.4 MLA

**Multi-head Latent Attention (MLA)** 是 DeepSeek 这条线最重要的结构创新之一。它和 `MQA/GQA` 的区别不是“再少几个 KV heads”，而是：

1. 不直接缓存语义级完整 `K/V`
2. 先把 token 的 KV 信息压成共享 latent
3. 再通过 query 侧吸收和输出侧后移，在 runtime 上尽量直接对 latent cache 计算

MLA 的关键点可以简化成：

- 历史 token 真正缓存的是更紧凑的 latent 状态，而不是完整 per-head `K/V`
- `RoPE` 通过 decoupled RoPE 拆出一条较小的位置分支
- 从语义上仍然是 exact softmax attention
- 但从工程上，cache 更小，decode 更省带宽

### 4.5 MHA / MQA / GQA / MLA 对比

| 方法 | KV 组织方式 | decode cache 成本 | 优点 | 典型问题 |
|---|---|---|---|---|
| `MHA` | 每个 head 独立 `K/V` | 最高 | 表达力强，最直接 | cache 大，decode 最吃带宽 |
| `MQA` | 所有 Q heads 共享 1 组 `K/V` | 最低 | 推理便宜，工程简单 | 容易有精度损失 |
| `GQA` | 多个 Q heads 分组共享 `K/V` | 中等偏低 | 精度/成本折中最好用 | 仍是“少头共享”，不是压缩表示 |
| `MLA` | 共享 latent + 小位置分支 | 很低 | 更激进压缩 KV，同时保留 exact attention 语义 | 前后处理更复杂，对 kernel 设计要求高 |

补充说明：

- `WGQA`、`MGQA` 之类工作，通常是在 `GQA` 的组策略或加权方式上继续优化。
- 今天很多工业 LLM 实际上是 `GQA + FlashAttention/PagedAttention` 的组合。
- DeepSeek 路线则更偏 `MLA + FlashMLA + 混合长上下文结构`。

## 5. 位置编码与长度外推

很多人把“长上下文能力”完全归因于 attention 结构，但实际上，**位置编码和长度外推** 同样关键。

### 5.1 基础位置编码类型

| 方法 | 核心想法 | 特点 |
|---|---|---|
| 绝对位置编码 | 每个位置一个 embedding | 简单，但超长外推弱 |
| 正弦位置编码 | 固定频率位置函数 | 泛化比 learned absolute 好一些 |
| 相对位置偏置 | 建模 token 间相对距离 | 更适合长距离关系 |
| `RoPE` | 把旋转位置结构编码到 `Q/K` | 现代 LLM 最主流 |
| `ALiBi` | 给远距离增加线性偏置 | 零样本长度外推常较稳 |

### 5.2 长上下文时代最常见的几类方案

- **Transformer-XL 相对位置**：把 segment recurrence 和相对位置结合。
- **T5 bias**：相对位置桶化偏置。
- **RoPE**：Llama、Qwen、DeepSeek 等现代 LLM 常见主流。
- **ALiBi**：训练简单，长度外推常表现稳定。
- **xPos / NTK-aware scaling / YaRN / LongRoPE / ReRoPE / Self-Extend**：更多是为了把已有模型从原 context window 拉到更长。

### 5.3 为什么它对 attention 很关键

即使 attention kernel 和 KV cache 完全够快，如果位置编码在超长范围严重 out-of-distribution，模型仍然可能：

- 远距离定位失败
- 出现 lost-in-the-middle
- 在 128K 以上迅速退化

因此长上下文系统里，通常要同时考虑：

1. attention 结构本身
2. KV cache 成本
3. 位置编码和长度外推

## 6. 高效 attention 的算法家族

这一节把真正“改 attention 计算方式”的方法分成几大类。

### 6.1 IO-aware exact attention：不改语义，只改算法

这一类方法最重要的共同点是：

> attention 的数学结果不变，变的是执行顺序、tile 组织、片上缓存和流水方式。

| 方法 | 核心思想 | 适用点 | 备注 |
|---|---|---|---|
| `FlashAttention` | tile 化 `QK/PV`，online softmax，避免物化大 logits 矩阵 | 训练与推理 | 经典里程碑方法 |
| `FlashAttention-2` | 更好的 work partition、warp 调度与并行利用 | 更高吞吐训练/推理 | 对 GPU 友好度更高 |
| `FlashAttention-3` | 针对 Hopper，引入更深异步流水和低精度支持 | H100/Hopper 类平台 | 重点是 asynchrony + low precision |
| `FlashAttention-4` | 针对 Blackwell 做算法与 kernel pipeline 协同设计，缓解 exp/smem 等非 MMA 瓶颈 | B200/GB200/Blackwell 类平台 | 更像硬件代际演进下的 exact attention co-design |
| Flash-Decoding | 针对 decode 场景优化 attention | 单 token / 小 batch decode | 常与 KV cache 技术搭配 |
| `FlashMLA` | 在 MLA 语义上做 Flash 风格 exact 执行 | DeepSeek MLA decode/prefill | 不显式物化语义级完整 `K/V` |

这类方法和 `GQA/MLA` 的关系可以理解成：

- `GQA/MLA` 回答“缓存什么”
- `FlashAttention/FlashMLA` 回答“怎样高效算出来”

### 6.2 固定稀疏模式：先规定谁能看谁

这类方法的核心是直接约束可见性模式，让 attention 不再是完整 `L x L`。

| 方法 | 稀疏模式 | 主要优点 | 主要问题 |
|---|---|---|---|
| Sparse Transformer | 局部 + 跨步/strided | 比全连接便宜 | 模式固定 |
| Longformer | 滑动窗口 + 全局 token | 文档任务常见 | 需要指定全局位 |
| BigBird | 局部 + 全局 + 随机 | 理论性质好，长文本经典 | 实现复杂 |
| ETC | 全局-局部结构 | 长文档与结构化任务 | 工程复杂 |
| LongNet | 扩张式/dilated attention | 极长序列扩展性强 | 训练和任务适配较特殊 |
| Sliding Window Attention | 只看最近窗口 | 简单高效 | 远距离检索弱 |

这一类方法的优点是结构清楚、硬件友好；缺点是**稀疏模式先验较强**，一旦真实依赖不符合预设结构，就容易掉效果。

### 6.3 动态稀疏 / 查询感知 / 分块检索

固定稀疏不够灵活，于是很多方法开始让模型或 runtime 动态决定“到底看哪些 token / blocks”。

#### 6.3.1 经典代表

- **Reformer**：用 `LSH attention` 把相似 token 放进桶里做近似注意力。
- **Routing Transformer**：用内容路由把 token 分配到不同簇。
- **Sinkhorn Transformer**：通过可学习排序得到更结构化的块稀疏模式。

#### 6.3.2 2024-2025 值得重点关注的方法

| 方法 | 核心想法 | 更准确的定位 |
|---|---|---|
| `Native Sparse Attention (NSA)` | 把 compression、selection、sliding window 三条路径合成一个 hardware-aligned、可训练的稀疏 attention | 原生 sparse attention 架构，不只是推理时后处理 |
| `Quest` | query-aware 地从 page/block 级 KV 中选 Top-K 关键页 | 推理时的动态稀疏选择 |
| `MInference 1.0` | 不改模型参数，离线识别每个 head 的稀疏模式并在 prefill 动态构造索引 | 预填充阶段的动态 sparse acceleration |
| `MoBA` | 把 MoE 思想引入 block attention，让模型学习路由到哪些 blocks | 训练期 block sparse attention |
| `Kascade` | 只在少数 anchor layers 精确求 Top-K，并在相邻 reuse layers 复用这些索引 | 利用跨层 Top-K 稳定性的 training-free sparse attention |
| Squeezed Attention | 对长 prompt 的 keys 做聚类/压缩，再动态检索相关聚类 | 长静态上下文推理加速 |
| Core Context Aware Attention | 把重要上下文合并成 core tokens，同时保留 locality | 长上下文结构化压缩 |
| DHSA | 分层动态稀疏，面向 on-device 长上下文 | 层级 sparse 设计 |
| Star Attention | 分布式/块稀疏长序列推理 | 偏系统和并行视角 |
| Recycled Attention | 复用前一步 attention 模式，减少长上下文数据移动 | 推理时模式重用 |

这些方法的共同趋势是：

- 不再假设所有历史 token 同等重要
- 更偏“先筛候选，再做主 attention”
- attention 真正的成本越来越取决于**读了多少 KV bytes**，而不只是逻辑序列长度

### 6.4 低秩 / 核技巧 / 线性 attention

这一类方法的目标是把原本二次复杂度的 attention 变成更接近线性或亚二次。

#### 6.4.1 低秩近似

| 方法 | 核心想法 |
|---|---|
| Linformer | 假设 attention 矩阵低秩，把序列维投影到更小空间 |
| Nyströmformer | 用 landmark / Nyström 近似 attention |
| LRQK | 对 `Q/K` 做更紧凑低秩分解，用代理分数做长上下文推理 |

#### 6.4.2 核技巧与显式线性化

| 方法 | 核心想法 |
|---|---|
| Linear Transformer | 用正特征映射把 softmax 改写成线性形式 |
| Performer | 用随机特征近似 softmax kernel |
| RFA | 随机特征 attention |
| cosFormer | 用 cosine 重加权构造线性 attention |

#### 6.4.3 递归记忆 / fast-weight / retention 系

| 方法 | 核心想法 | 备注 |
|---|---|---|
| RetNet | 用 retention / decay 取代标准 attention | 兼顾并行训练和线性推理 |
| Lightning Attention | 强调硬件友好线性 attention | 常和分块/块内块间分解结合 |
| GLA | gated linear attention，加入数据相关遗忘门 | 提高表达能力 |
| DeltaNet / Gated DeltaNet | 把线性 attention 看作在线更新的 memory | fast-weight 视角强 |
| Kimi Linear | 2025 代表性新方法，混合线性注意力与 MLA 风格模块 | 代表“hybrid efficient attention”趋势 |

这条线的优点是理论复杂度漂亮，但也要看到现实：

- 有些方法训练很强，但工业部署不一定完全替代 dense attention
- 很多“线性 attention”最后会以 hybrid 形式出现，而不是整网纯替换
- 当硬件和 kernel 很强时，exact dense path 未必一定输给近似 attention

### 6.5 记忆增强 / 压缩记忆 / hybrid memory

这类方法不是简单稀疏，也不是单纯 kernel 优化，而是给 attention 加上显式长期记忆或压缩记忆。

| 方法 | 核心想法 |
|---|---|
| Transformer-XL | segment recurrence + relative position，让模型跨 segment 记忆 |
| Compressive Transformer | 把旧记忆压缩后长期保留 |
| Memorizing Transformer | 引入外部 memory / retrieval |
| Infini-attention | 局部精确 attention + 长期压缩 memory 的混合体 |
| `Native Hybrid Attention (NHA)` | 把 sliding-window 中的精确短期 token 和线性 RNN 更新的长期 memory slots 拼到同一个 softmax 里统一分配注意力 |

这一类方法和后面的 `CSA/HCA` 很像的一点是：**远距离信息不一定还以“原始 token 逐个精确可见”的方式存在。**

## 7. KV cache、长上下文推理与 serving 技术

这一层很多时候不改训练好的模型，而是**在推理时减少 cache、减少内存碎片、减少带宽**。

### 7.1 PagedAttention 与页式 KV 管理

**PagedAttention** 最重要的贡献不是改 attention 数学，而是把 KV cache 管理成类似虚拟内存分页：

- 减少长上下文服务时的显存碎片
- 支持更灵活的请求调度
- 非常适合 serving 系统

它通常和下列技术组合出现：

- continuous batching
- chunked prefill
- prefix caching
- KV cache quantization

### 7.2 Streaming / sink / eviction 类方法

这类方法通常基于一个经验事实：

> 不是所有历史 token 都值得一直留在高精度、全量 KV cache 里。

| 方法 | 核心想法 | 定位 |
|---|---|---|
| StreamingLLM | 保留 attention sinks 和最近窗口，使窗口 attention 能延伸到更长流式输入 | 无需大改模型的流式推理 |
| H2O | 保留 heavy hitters | 动态 KV eviction |
| Scissorhands | 基于“重要性持续性”保留 token | 动态 KV 管理 |
| SnapKV | 用 attention 统计找重要 token 并压缩/聚合 | 推理时 KV 压缩 |
| PyramidInfer | 金字塔式 KV 压缩 | 长上下文吞吐优化 |
| FastGen | 根据若干 attention 模式自适应保留 KV | 推理时 KV 选择 |
| AdaKV / HeadKV | 按 head 分配不同 KV 预算 | head-aware cache budget |

### 7.3 头异质性被显式利用

最近一个很重要的趋势，是大家不再假设所有 attention heads 行为相同。

| 方法 | 核心思想 |
|---|---|
| RazorAttention | 区分 head 的不同重要性和行为模式 |
| `DuoAttention` | 把 heads 分成 retrieval heads 和 streaming heads，前者保全量 KV，后者只保 sink + recent |

`DuoAttention` 的价值很典型：它没有直接发明一种全新 attention 数学，而是利用了**不同头职责不同**这个现象，重新分配了 KV cache 预算。

### 7.4 这类技术的本质

这类方法大多回答的是：

1. 哪些 token 必须长期保留
2. 哪些 token 可以压缩、量化、丢弃
3. 哪些 head 真的需要完整历史
4. 物理内存怎样组织得更像一个高效的 KV 存储系统

因此它们常常和模型结构方法是正交的，可以出现：

- `GQA + FlashAttention + PagedAttention`
- `MLA + FlashMLA + chunked prefill`
- `GQA + DuoAttention + KV quantization`

## 8. 分布式 attention 与多卡长序列并行

当上下文长到单卡难以承受时，就必须把序列切到多卡上。这时很多“attention 名字”实际上是在解决**通信**而不是注意力分数本身。

### 8.1 Ulysses

Ulysses 的核心是：

- 沿序列切分
- 再通过 AlltoAll 让不同设备获得所需 head / token 视图

优点是思路直接，但它通常受限于 `KV heads` 数量。对于 `MQA`、`MLA` 这类 KV heads 很少的方法，扩展性会受到影响。

### 8.2 Ring Attention

**Ring Attention** 的核心是：

- 各卡固定本地 `Q`
- `KV` 分块沿 ring 旋转
- 每轮处理当前拿到的 `KV chunk`

它的优势是：

- 不要求每张卡一次持有全量 KV
- 能和 FlashAttention 风格计算结合

但瓶颈也明显：

- 通信是核心约束
- causal mask 下还会有负载均衡问题

### 8.3 Striped / Zigzag / Megatron Context Parallel

这些方法更像 Ring Attention 的负载均衡和分片策略增强版，目标通常是：

- 在 causal mask 下让不同卡工作量更均匀
- 更好 overlap compute 和 communication

### 8.4 Hamilton Attention / TASP

用户提到的 **Hamilton Attention**，根据其官方仓库与对应论文，更准确地说是：

> **Hamilton Attention = TASP (Topology-aware Sequence Parallelism)**
> 它是一种分布式 attention 执行方法，不是新的 attention 打分公式。

它的关键思想是：

- 观察现代加速器常有 All-to-All 或接近全连接拓扑
- Ring Attention 的 Ring AllGather 只能利用其中很小一部分链路
- 用 Hamiltonian decomposition 把拓扑分解成多条互不干扰的 ring datapaths
- 再把通信原语分解成 multi-ring 并发传输

因此它主要改进的是：

- 通信利用率
- 多卡长序列 attention 的端到端吞吐
- Ring Attention 在现代互联拓扑上的效率问题

如果只想记一句话，可以记成：

> `Hamilton Attention` 本质上是“让 sequence parallelism 更懂硬件拓扑”的方法。

## 9. DeepSeek 路线：MLA、FlashMLA、CSA/HCA

DeepSeek 这条线值得单独拎出来，因为它把“结构压缩”“kernel 高效化”“混合长上下文设计”几件事同时推进了。

### 9.1 MLA：压缩 KV 表示

`MLA` 的关键词是：

- shared KV latent
- decoupled RoPE
- query-side absorption
- output-side deferred up-projection

核心价值：

- 压缩 KV cache
- 尽量不在 runtime 物化完整语义级 `K/V`
- 仍保持 exact softmax attention 语义

### 9.2 FlashMLA：对 MLA 做 exact flash-style 执行

`FlashMLA` 可以理解为：

- `MLA` 负责“attention 应该怎样表达”
- `FlashMLA` 负责“这个表达怎样在硬件上高效执行”

它的工程重点是：

- 按 chunk 流式处理长历史
- online softmax
- 不物化完整大 attention matrix
- runtime 使用紧凑 cache 视图

### 9.3 DeepSeek V4 的 CSA / HCA

根据 DeepSeek V4 公开 blog/model card 线索，V4 最重要的 attention 升级是 **Hybrid Attention Architecture**，由：

- **CSA: Compressed Sparse Attention**
- **HCA: Heavily Compressed Attention**

组成。

#### CSA

公开信息显示，CSA 大致具有下面这些特征：

- 沿序列维做约 `4x` 压缩
- 有一条最近 token 的 sliding-window 精确分支
- 有一个 lightning indexer，从压缩块里选 `top-k` 候选
- 本质上是“先压缩，再稀疏选择，再做主 attention”

#### HCA

公开信息显示，HCA 大致具有下面这些特征：

- 更强的序列压缩，约 `128x`
- 不再强调稀疏选择
- 直接对高度压缩后的序列做 dense attention
- 由于压缩后的长度已经很短，所以 dense 也不贵

#### V4 的关键趋势

和 V3/FlashMLA 相比，V4 最重要的变化不是“把同一种 attention 再做快一点”，而是：

> **把长上下文 attention 拆成多条不同精度、不同压缩率、不同访问模式的路径。**

如果从本文前面的分类去看：

- `MLA`：属于 **KV 表示压缩**
- `FlashMLA`：属于 **exact kernel**
- `CSA/HCA`：属于 **混合长上下文 attention 架构**

### 9.4 对 CSA / HCA 的一个谨慎提醒

需要谨慎的是：

> 截至目前，`CSA/HCA` 的高层设计已经公开，但完整数学定义、全部 kernel 细节、训练细节并没有像 `FlashAttention` 或传统论文那样完全展开。

因此更稳妥的理解方式是：

- 公开确认：`CSA/HCA`、局部窗口、压缩、索引器、长上下文成本大幅下降
- 仍待公开：完整实现公式、全部 kernel 细节、所有层的精确 schedule 细节

## 10. 2024-2026 最值得关注的趋势

如果把这几年的 attention 演化压成几条主线，我认为最重要的是下面这些。

### 10.1 从“单一 attention 配方”走向“混合路径”

代表：

- Infini-attention
- `Native Hybrid Attention`
- DuoAttention
- DeepSeek `CSA/HCA`
- Kimi Linear 这类 hybrid efficient architectures

趋势是：

- 近处精确看
- 远处稀疏挑着看
- 更远处只看压缩摘要
- 不同 head / 不同 layer 还可能用不同策略

### 10.2 decode 主要矛盾越来越像“KV bytes 问题”

在长上下文推理里，瓶颈通常不是理论 FLOPs，而是：

- KV cache 占用
- 带宽
- page/gather 访问模式
- 多卡通信

所以 `GQA/MLA/PagedAttention/Quest/DuoAttention/Hamilton Attention` 这些名字虽然分属不同层，最终都在围绕同一个物理问题打转：**减少真实搬运的字节数和同步成本。**

### 10.3 exact dense attention 并没有被淘汰

很多人会把“高效 attention”理解成“一定要近似、一定要线性”。其实不是。

现实里非常强的一条路线是：

- 语义上仍然 exact
- 但通过 `FlashAttention`、`FlashMLA`、`PagedAttention`、多卡并行把工程成本压下来

所以今天真正有竞争力的系统，往往不是“只靠一个数学 trick”，而是：

- 模型结构
- KV 表示
- kernel
- cache 管理
- 通信拓扑

一起协同设计。

### 10.4 head 异质性越来越重要

从 `MHA` 到 `GQA/MLA`，再到 `DuoAttention`，一个越来越清楚的事实是：

- 不同 head 的职责不同
- 不同 head 对长上下文的重要性不同
- “所有 head 一刀切”越来越不是最优设计

### 10.5 分布式 attention 会越来越“拓扑感知”

`Hamilton Attention/TASP` 很能说明这一点：

- attention 不再只是矩阵乘法问题
- 也是一个通信拓扑问题
- 未来多卡长上下文系统，很多创新会更多来自拓扑、页管理、pipeline 和 precision co-design

## 11. 该如何选型

如果只是想快速建立直觉，可以按下面方式理解。

### 11.1 你想保留标准 attention 语义，但把工程跑快

优先看：

- `FlashAttention`
- `FlashAttention-2/3`
- Flash-Decoding
- `PagedAttention`
- Ring / Hamilton Attention

### 11.2 你想显著降低 decode 的 KV cache 成本

优先看：

- `MQA`
- `GQA`
- `MLA`
- KV quantization
- `DuoAttention`

### 11.3 你想让模型支持更长上下文，而且愿意改变结构

优先看：

- Longformer / BigBird / LongNet
- Performer / Linear Transformer / RetNet / Kimi Linear
- Infini-attention
- DeepSeek `MLA`
- DeepSeek `CSA/HCA`

### 11.4 你想在不重训大模型的前提下加速长上下文推理

优先看：

- `PagedAttention`
- `Quest`
- `MInference`
- StreamingLLM
- H2O / SnapKV / AdaKV / HeadKV
- `DuoAttention`

### 11.5 你主要关心多卡长序列

优先看：

- Ulysses
- Ring Attention
- Striped / Megatron CP
- Hamilton Attention / TASP

## 12. 容易混淆的几个点

### 12.1 FlashAttention 不是新的 attention 语义

它算的还是同一个 softmax attention，只是算得更高效。

### 12.2 PagedAttention 不是 sparse attention

它主要是 KV cache 的页式管理与 serving 技术，不是改注意力掩码。

### 12.3 Hamilton Attention 不是新的 query-key 打分公式

它主要是 sequence parallelism / distributed attention 的通信优化方法。

### 12.4 GQA 和 MLA 不是一回事

- `GQA`：还是 per-group `K/V`，只是减少 KV heads
- `MLA`：把 KV 先压成共享 latent，再通过代数改写高效计算

### 12.5 CSA/HCA 也不只是“再一个 FlashAttention 变种”

`CSA/HCA` 更像是新的长上下文混合 attention 架构，而不是单个 kernel 技巧。

## 13. 建议继续跟踪的补充名字

如果想把文献面继续铺得更宽，下面这些名字也值得继续跟踪：

- 固定/结构化 sparse：ETC, Sinkhorn Transformer, Zebra, Native Sparse Attention, SeerAttention
- 动态 sparse / retrieval：Routing Transformer, MagicPIG, ClusterKV, LONGHEADS, CORM
- KV eviction / compression：TOVA, Keyformer, MiKV, FastGen, KV-Compress, vTensor
- 线性/记忆系：AFT, RWKV, Eagle, H3, GateLoop, HGRN, Titans, TTT, Longhorn
- 长度外推：LM-Infinite, LongRoPE, YaRN, ReRoPE, Self-Extend
- 混合预训练模型：Jamba, MiniMax-01, Gemma 3, Llama 4, Falcon Mamba, MiniCPM 4

## 14. 推荐阅读顺序

如果你是第一次系统看 attention，建议顺序是：

1. 先掌握 `scaled dot-product attention` 与 `MHA`
2. 再看 `MQA/GQA`，理解 KV cache 为什么会成为 decode 瓶颈
3. 再看 `FlashAttention`，理解“exact 但更快”的思路
4. 再看 Longformer / BigBird / Reformer / Performer，理解 sparse 与 linear 两大方向
5. 再看 `MLA` 与 `FlashMLA`
6. 最后看 `PagedAttention`、`DuoAttention`、`Quest/MInference`、`CSA/HCA`、`Hamilton Attention`

如果你主要关心当前仓库的主线，建议继续看：

- `deepseek-flash-mla-introduction.md`
- `deepseek-v4-attention-and-tt-dataflow-notes.md`
- `flash-mla-dataflow-first-principles.md`

## 15. 精选参考资料

下面只列最值得作为入口的一批资料，按主题分组。

### 15.1 基础 attention

- Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*
  <https://arxiv.org/abs/1409.0473>
- Luong et al., *Effective Approaches to Attention-based Neural Machine Translation*
  <https://arxiv.org/abs/1508.04025>
- Vaswani et al., *Attention Is All You Need*
  <https://arxiv.org/abs/1706.03762>

### 15.2 MHA / MQA / GQA / MLA

- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (`MQA`)
  <https://arxiv.org/abs/1911.02150>
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*
  <https://arxiv.org/abs/2305.13245>
- DeepSeek-AI, *DeepSeek-V2* (`MLA`)
  <https://arxiv.org/abs/2405.04434>
- DeepSeek-AI, *FlashMLA* repository
  <https://github.com/deepseek-ai/FlashMLA>

### 15.3 Flash / exact kernel

- Dao et al., *FlashAttention*
  <https://arxiv.org/abs/2205.14135>
- Dao, *FlashAttention-2*
  <https://tridao.me/publications/flash2/flash2.pdf>
- Shah et al., *FlashAttention-3*
  <https://arxiv.org/abs/2407.08608>
- Zadouri et al., *FlashAttention-4*
  <https://arxiv.org/abs/2603.05451>

### 15.4 稀疏 / 长上下文 / 线性 attention

- Beltagy et al., *Longformer*
  <https://arxiv.org/abs/2004.05150>
- Zaheer et al., *BigBird*
  <https://arxiv.org/abs/2007.14062>
- Kitaev et al., *Reformer*
  <https://arxiv.org/abs/2001.04451>
- Wang et al., *Linformer*
  <https://arxiv.org/abs/2006.04768>
- Choromanski et al., *Performer*
  <https://arxiv.org/abs/2009.14794>
- Xiong et al., *Nyströmformer*
  <https://arxiv.org/abs/2102.03902>
- Yuan et al., *Native Sparse Attention*
  <https://arxiv.org/abs/2502.11089>
- Sun et al., *A Survey of Efficient Attention Mechanisms for Large Language Models*
  <https://arxiv.org/abs/2507.19595>

### 15.5 KV cache / 推理技术

- Kwon et al., *PagedAttention / vLLM*
  <https://arxiv.org/abs/2309.06180>
- Xiao et al., *StreamingLLM*
  <https://arxiv.org/abs/2309.17453>
- Tang et al., *Quest*
  <https://arxiv.org/abs/2406.10774>
- Jiang et al., *MInference 1.0*
  <https://arxiv.org/abs/2407.02490>
- Xiao et al., *DuoAttention*
  <https://arxiv.org/abs/2410.10819>
- Hooper et al., *Squeezed Attention*
  <https://arxiv.org/abs/2411.09688>
- Xu et al., *Recycled Attention / RefreshKV*
  <https://arxiv.org/abs/2411.05787>
- Microsoft, *Kascade*
  <https://arxiv.org/abs/2512.16391>

### 15.6 混合与近期方法

- Munkhdalai et al., *Infini-attention*
  <https://arxiv.org/abs/2404.07143>
- Du et al., *Native Hybrid Attention*
  <https://arxiv.org/abs/2510.07019>
- Moonshot AI, *Kimi Linear*
  <https://arxiv.org/abs/2510.26692>
- DeepSeek V4 overview
  <https://huggingface.co/blog/deepseekv4>
- DeepSeek V4 Pro model card
  <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro>
- DeepSeek V4 Flash model card
  <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>

### 15.7 分布式并行

- Liu et al., *Ring Attention with Blockwise Transformers for Near-Infinite Context*
  <https://arxiv.org/abs/2310.01889>
- Wang et al., *TASP: Topology-aware Sequence Parallelism* (`Hamilton Attention`)
  <https://arxiv.org/abs/2509.26541>

## 16. 一句话总结

今天 attention 的演化已经不是一条单线，而是几条线同时前进：

- 一条线在改 **表示**，比如 `GQA`、`MLA`
- 一条线在改 **可见性/复杂度**，比如 sparse、linear、memory attention
- 一条线在改 **kernel**，比如 `FlashAttention/FlashMLA`
- 一条线在改 **缓存与 serving**，比如 `PagedAttention`、`DuoAttention`
- 一条线在改 **分布式通信**，比如 Ring / Hamilton Attention

如果用 2026 的眼光看，最值得关注的不是某一个孤立技巧，而是：

> **局部精确 + 稀疏检索 + 压缩摘要 + 高效 cache + 拓扑感知并行**
> 这几件事正在逐渐合并成下一代长上下文 attention 系统的共同形态。
