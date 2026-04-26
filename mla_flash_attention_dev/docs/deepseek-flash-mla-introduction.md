# DeepSeek Flash MLA 简明介绍

更新时间：`2026-04-25`

这篇文档想回答一个问题：

> **DeepSeek Flash MLA 到底是什么，它相比普通 attention 改了什么，为什么它能更省 cache、更适合长上下文 decode？**

本文尽量少堆公式，优先把直觉讲清楚；但在关键地方会保留必要的术语，方便继续往实现细节里追。

## 1. 一句话先讲清楚

**DeepSeek Flash MLA = DeepSeek 的 MLA（Multi-head Latent Attention）表示方式 + FlashAttention 风格的分块执行方式。**

顺带说明一下：在公开社区里，`FlashMLA` 这个词有时也特指 DeepSeek 开源的高性能 kernel 库；本文更侧重解释它背后的**注意力思路**和**在当前仓库里的实现形态**。

它的目标不是改变 attention 的基本语义，而是：

1. **把 KV cache 变小**
2. **把 decode 时的访存和中间状态变少**
3. **让长上下文推理更快、更容易做高效 kernel**

如果再压缩成一句话，就是：

> **普通 attention 是“缓存很多 K/V，再一次次去读”；DeepSeek Flash MLA 是“只缓存更紧凑的 latent 状态，并按 chunk 流式算完 softmax”。**

## 2. 为什么要做它

### 2.1 普通 attention 的瓶颈

在大模型 decode 时，当前步的 `Q` 往往很小，通常只有一个新 token；但历史 `K/V cache` 会随着上下文长度 `S` 一直增长。

所以真正的瓶颈常常不是“算不动”，而是：

- 历史 `K/V` 太大，**显存/内存压力大**
- 每次 decode 都要反复读历史 `K/V`，**带宽压力大**
- 序列变长后，中间 attention 状态也会变大，**kernel 更难做高效**

这也是为什么推理阶段经常会感觉：

- batch 不大，但显存已经很紧
- FLOPs 看起来没满，速度却上不去
- 上下文一长，decode 吞吐明显掉下去

### 2.2 MQA / GQA 为什么还不够

业界常见的做法是 `MQA` 或 `GQA`，本质上都是让多个 Q 头共享更少的 KV 头，从而减小 cache。

这类方法确实能省 cache，但通常会面临一个经典权衡：

- **省内存**：是的
- **保持效果**：不一定总能和完整 MHA 一样好

DeepSeek 的 MLA 想做的是：

> **尽量保留多头 attention 的表达能力，同时把推理时真正要缓存的东西压缩到更小。**

## 3. MLA 到底改了什么

### 3.1 标准 attention 在缓存什么

先看普通多头注意力。对每个历史 token，它会缓存两份主要状态：

- `K`
- `V`

运行时做的是：

```text
scores = Q @ K^T
P = softmax(scores)
out = P @ V
```

所以，普通 attention 的直觉很简单：

> **历史里每个 token 都保留一份 K 和一份 V；decode 时，新 Q 去扫一遍这些历史状态。**

### 3.2 MLA 的核心：把 K 和 V 先压成 latent

MLA 的关键变化是：

1. 不再直接缓存“完整的 K”和“完整的 V”
2. 先把它们压成一个更小的 latent 向量
3. 推理时尽量在这个 latent 表示上完成 attention

如果只看概念，可以写成：

```text
c_t^KV = W_DKV * h_t
K_t = W_UK * c_t^KV
V_t = W_UV * c_t^KV
```

但这里有一个非常容易混淆的点：

> **这 3 行里的 `c_t^KV` 是“每个 token 一份、所有 head 共享”的 latent；而 `K_t` / `V_t` 不是“某一个 head 的 K/V”，而是“这个 token 对所有 K/V heads 展开后的整体结果”。**

换句话说，这里不是：

- 一个 head 对应一个 `c_t^KV`

而是：

- **一个 token 先压成一份共享的 `c_t^KV`**
- **再由不同 head 的上投影切片，把这同一份 latent 展开成各个 head 自己的 `K/V`**

如果按 head 写开，会更直观一些。设 `Hkv` 是 K/V head 数，那么可以把上面 3 行理解成：

```text
c_t^KV = W_DKV * h_t

k_{t,1} = W_UK^(1) * c_t^KV
k_{t,2} = W_UK^(2) * c_t^KV
...
k_{t,Hkv} = W_UK^(Hkv) * c_t^KV

v_{t,1} = W_UV^(1) * c_t^KV
v_{t,2} = W_UV^(2) * c_t^KV
...
v_{t,Hkv} = W_UV^(Hkv) * c_t^KV
```

然后把所有 head 的结果拼起来，才得到前面简写里的：

```text
K_t = concat(k_{t,1}, k_{t,2}, ..., k_{t,Hkv})
V_t = concat(v_{t,1}, v_{t,2}, ..., v_{t,Hkv})
```

所以从 head 的角度看，MLA 的关键不是“每个 head 单独存一份压缩 cache”，而是：

> **先对每个 token 只存一份共享 latent，再让不同 head 从这同一份 latent 中读出各自需要的 K/V 分量。**

这里：

- `h_t` 是当前 token 的 hidden state
- `c_t^KV` 是这个 token 的压缩 latent 表示，**它本身不是按 head 分开的**
- `W_DKV` 是下投影
- `W_UK / W_UV` 是把 latent 展开回 K/V 的上投影；可以把它们看成“按 head 切块的一大组上投影”

如果用更工程化的话来说，就是：

- `W_DKV`：把一个 token 压成一份共享的 KV latent
- `W_UK`：从这份共享 latent 里恢复所有 K heads 的内容
- `W_UV`：从这份共享 latent 里恢复所有 V heads 的内容

再提醒一句，前面这组式子是 **论文层面的概念写法**。在当前仓库的 `FlashMLA` 实现里，运行时通常不会真的把每个 head 的完整 `K_t / V_t` 全部显式展开出来，而是进一步把计算改写到统一 latent cache 上。也正因为这样，当前实现里你会看到：

- `q_tensor` 仍然有很多个 Q heads
- `kv_cache_tensor` 却可以只有一条共享的 latent KV 流

这部分在后面的“当前仓库这条实现里，它具体长什么样”一节会展开讲。

最重要的一点不是公式本身，而是：

> **推理阶段真正缓存的，不再是完整的 `K_t / V_t`，而是更小的 `c_t^KV`。**

这就是 MLA 的第一层节省来源。

### 3.3 为什么它不只是“压缩一下”

如果只是“先压，再每次都完整解压回 K/V”，那推理时还是会很重。

DeepSeek MLA 真正聪明的地方在于：**它尽量避免把完整 K/V 每次都显式恢复出来。**

论文里一个非常关键的点叫 **weight absorption（权重吸收）**。直觉上，它做的是：

- 把原本要在 K 侧做的一部分变换，提前吸收到 Q 侧
- 这样在推理时，attention 可以更多地直接在 latent 空间里完成

可以把它理解成：

> **不是“先把历史全展开，再算 attention”，而是“把计算改写成更适合对压缩表示直接操作的形式”。**

这会继续减少推理时的算子和访存负担。

### 3.4 RoPE 为什么会麻烦

如果没有位置编码，事情会比较整齐；但实际模型通常要用 `RoPE`。

问题在于：

- `RoPE` 是和位置相关的
- 一旦直接把 `RoPE` 混进被压缩/被吸收的那条 K 路径里
- 很多原本能“提前吸收”的矩阵关系就被打乱了

结果就是：

> **你本来想省掉的那部分 K 重建，可能又得回来。**

### 3.5 DeepSeek 的解法：Decoupled RoPE

DeepSeek MLA 为了解决这个问题，引入了 **decoupled RoPE（解耦 RoPE）**。

直觉上它做的是把 attention 里的信息拆成两部分：

- **内容部分**：尽量留在 latent/可吸收的路径里
- **位置部分**：单独留一条更小的 RoPE 分支来承载

这样做的好处是：

1. 内容路径还能继续享受 MLA 的压缩和吸收优势
2. 位置路径仍然保留 RoPE 的能力
3. 两者组合起来，既省 cache，又不丢掉位置信息

所以可以把 DeepSeek MLA 的思想简单记成：

> **内容走压缩路径，位置走单独路径。**

### 3.6 只看理论：MLA 的输入、计算和输出

前面已经分别讲了“KV joint compression”“weight absorption”“decoupled RoPE”。但如果第一次接触 MLA，最容易迷糊的地方其实是：

- 输入到底是什么样子
- 理论上的 Q/K/V 到底怎么定义
- 真正的 attention 分数和输出是怎么计算的
- `Q` 的吸收到底改了什么，没改什么

这一小节先**完全不看当前 `tt-metal` 实现**，只讲理论上的 MLA。

#### 3.6.A 一版不展开到 head 和 token 的写法

如果你暂时不想管“第几个 head”“第几个 token”，只想抓住 MLA 里 `Q/K/V` 的主变换关系，那么可以先看这一版。

下面我故意把记号压缩到最少，只保留最核心的对象：

- `h_q`：query 侧输入 hidden state
- `h_kv`：KV 侧输入 hidden state
- `c^Q`：query 路径的 latent
- `c^KV`：KV 路径的共享 latent
- `q^C / q^R`：query 的内容分支 / 位置分支
- `k^C / k^R`：key 的内容分支 / 位置分支
- `v`：value

这里要先提醒一句：

> **`h_q` 和 `h_kv` 不一定是同一个 hidden state。** 在真正做 attention 时，`h_q` 更接近“当前 query 这一侧的输入”，`h_kv` 更接近“历史 cache 那一侧的输入”。这里故意把两者分开写，就是为了不再混淆。

##### 第一步：理论上的 Q 是怎么来的

先从 query 侧 hidden state 生成 query latent，再生成 query 的内容分支和位置分支：

```text
c^Q = W_DQ h_q
q^C = W_UQ c^Q
q^R = RoPE(W_QR c^Q)
Q   = [q^C | q^R]
```

这几条式子的意思是：

- `W_DQ` 先把 query 侧 hidden state `h_q` 压到更小的 latent 空间，得到 `c^Q`
- `W_UQ` 再从 `c^Q` 恢复出 query 的**内容部分** `q^C`
- `W_QR` 再从 `c^Q` 生成 query 的**位置部分**，并施加 `RoPE`，得到 `q^R`
- 最后把内容部分和位置部分拼起来，得到理论上的 query：`Q = [q^C | q^R]`

##### 第二步：理论上的 K 和 V 是怎么来的

再看 KV 侧。MLA 的核心就是：先把 KV 压成一份共享 latent，再从这份 latent 中恢复出语义上的 `K` 和 `V`。

```text
c^KV = W_DKV h_kv
k^C  = W_UK c^KV
k^R  = RoPE(W_KR h_kv)
v    = W_UV c^KV
K    = [k^C | k^R]
V    = v
```

这几条式子的意思是：

- `W_DKV` 先把 KV 侧 hidden state `h_kv` 压成一份共享 latent `c^KV`
- `W_UK` 从 `c^KV` 恢复出 key 的**内容部分** `k^C`
- `W_KR` 从 `h_kv` 直接生成 key 的**位置部分**，再施加 `RoPE`，得到 `k^R`
- `W_UV` 从同一份 `c^KV` 恢复出 value `v`
- 最后拼成理论上的 key：`K = [k^C | k^R]`

所以 MLA 理论上最核心的一点就是：

> **`Q` 和 `K` 都由“内容分支 + 位置分支”组成，而 `V` 来自共享的 KV latent `c^KV`。**

##### 第三步：理论上的 attention 怎么算

有了上面的 `Q / K / V`，理论上的 attention 仍然是标准 softmax attention。但这里要先区分两种写法：

1. **单个 query 向量和单个 key 向量的点积写法**
2. **当前 query 对整段历史 key/value 的矩阵写法**

如果只看**单个 query 向量** `q = [q^C | q^R]` 和**单个 key 向量** `k = [k^C | k^R]` 的相似度，那么写成：

```text
score(q, k) = q^T k
```

这里的 `q^T k` 只是一个标量点积，所以这一步不是完整 attention，只是在定义“单个 query 和单个 key 之间的匹配分数”。

如果把当前 query 和整段历史 key/value 放在一起做真正的 attention，那么在**理论语义级**的粗粒度写法里，可以先用一组更短的记号：

- `K_s`：整段历史的语义级完整 key 集合
- `V_s`：整段历史的语义级完整 value 集合
- `C`：整段历史的 KV latent 集合
- `R`：整段历史的 RoPE key 集合
- `s`：分数向量
- `o`：语义级输出向量

```text
s = q^T K_s
p = softmax(s)
o = p V_s
```

这里：

- `K_s` 表示整段历史的**语义级完整 key 集合**
- `V_s` 表示整段历史的**语义级完整 value 集合**
- `s` 是当前 query 对所有历史位置的理论语义级分数向量
- `p` 是对这些分数做 softmax 后得到的权重向量
- `o` 是理论语义级输出向量

这里最关键的一点是：

> **`K_s / V_s` 只是为了表达“attention 从语义上等价于在什么 `K/V` 上计算”；它们不是 MLA 推理时真正缓存的 runtime 张量。**

如果改成更接近 MLA 实际缓存状态的粗粒度写法，那么应该记住：

- 历史侧真正长期保留的是 `C` 和 `R`
- 其中 `C` 是整段历史的 KV latent 集合
- `R` 是整段历史的 RoPE key 集合

于是，利用 Q 吸收之后，同一件事也可以等价地写成：

```text
s     = (q^abs)^T C + (q^R)^T R
o^lat = p C
o     = W_UV o^lat
```

这里：

- `q^abs` 是吸收了 `W_UK` 之后的 query 内容分支
- `C` 是整段历史的 KV latent 集合
- `R` 是整段历史的 key 位置分支集合
- `o_lat` 是在 latent 空间里的输出
- `o` 是恢复到语义级 value/output 空间后的输出

如果采用更常见的“每个 key/value 向量作为矩阵一行”的机器学习记法，那么同一件事通常会写成：

```text
s = q K_s^T
p = softmax(s)
o = p V_s
```

所以你问“为什么不是 `Q K^T`”，答案就是：

> **如果 `Q/K` 表示一批 query/key 组成的矩阵，常见写法确实是 `Q K^T`；如果 `q/k` 表示单个 query/key 向量，那么对应的点积写法就是 `q^T k`。而这里的 `K_s` 还要额外理解成“语义级完整 K 的集合”，不是 runtime cache 本身。**

下面为了说明 MLA 的打分结构，我先继续用“单个 query 向量 `q` 和单个 key 向量 `k` 的点积形式”来展开。把前面的定义代进去，`score(q, k)` 可以拆成两部分：

```text
score(q, k)
= [q^C | q^R]^T [k^C | k^R]
= (q^C)^T k^C + (q^R)^T k^R
= (q^C)^T (W_UK c^KV) + (q^R)^T k^R
```

这条式子的含义很重要：

- 第一项 `(q^C)^T (W_UK c^KV)` 是**内容匹配**
- 第二项 `(q^R)^T k^R` 是**位置匹配**

所以 MLA 不是把 attention 改成别的东西，而是把 attention 的打分拆成：

1. 一部分看内容
2. 一部分看位置

##### 第四步：Q 是如何吸收 K 侧变换的

这是 MLA 最关键的代数改写。

前面内容打分那一项是：

```text
(q^C)^T (W_UK c^KV)
```

它可以严格等价地改写成：

```text
(q^C)^T (W_UK c^KV)
= ((W_UK)^T q^C)^T c^KV
```

于是我们定义一个“吸收后的 query 内容部分”：

```text
q^abs = (W_UK)^T q^C
```

那么内容打分就变成：

```text
(q^abs)^T c^KV
```

于是总分数可以改写成：

```text
score = (q^abs)^T c^KV + (q^R)^T k^R
```

这一步的含义可以用一句话概括：

> **原本应该在 K 侧做的线性展开 `W_UK c^KV`，被等价地搬到了 Q 侧来做。**

所以“Q 吸收”不是说 K 没了，而是说：

- 原来你先把 `c^KV` 展开成 `k^C`
- 现在你改成先把 `q^C` 改写成 `q^abs`
- 然后直接让 `q^abs` 去和缓存里的 `c^KV` 做点积

这样做的好处是：**历史侧只需要缓存 `c^KV`，不需要把完整 `k^C` 每次都显式展开出来。**

##### 第五步：V 为什么也可以后移

输出这边也有同样的线性改写。

理论上：

```text
o = p V = p (W_UV c^KV)
```

由于 `W_UV` 是线性的，这可以改写成：

```text
o = W_UV (p c^KV)
```

如果定义一个 latent 输出：

```text
o^lat = p c^KV
```

那么：

```text
o = W_UV o^lat
```

这说明：

> **attention 主体可以先在 latent 空间里做完，再通过 `W_UV` 把结果恢复回 value/output 空间。**

##### 这一版只记三句话就够了

如果只想抓住最核心的主线，可以记下面三句话：

1. **理论上，MLA 的 `Q` 和 `K` 都是“内容分支 + 位置分支”，`V` 来自共享的 `c^KV`。**
2. **Q 吸收的本质是把 `W_UK` 从 K 侧搬到 Q 侧：`(q^C)^T (W_UK c^KV) = ((W_UK)^T q^C)^T c^KV`。**
3. **输出侧也可以后移：先在 latent 空间里得到 `o^lat`，再通过 `W_UV` 恢复成最终输出。**

如果你觉得这一版已经够用了，后面的 `3.6.0` 到 `3.6.8` 可以看成是这套逻辑的“细粒度展开版”。

#### 3.6.0 先约定本文这一小节用到的符号

这一小节后面会反复出现一些符号。为了避免你每次都要猜，我先把最常用的符号集中写清楚；后面每条公式里，我也会继续把当场出现的符号再解释一遍。

- `H`：整段序列的 hidden states 集合
- `h_t`：第 `t` 个 token 的 hidden state
- `h_cur`：当前 decode token 的 hidden state
- `t`：token 下标，一般表示“某个 token”
- `j`：历史 token 下标，一般表示“某个被 attend 的历史位置”
- `cur`：当前 query token 的位置
- `i`：attention head 下标
- `n_h`：query head 总数
- `d_model`：模型 hidden state 的宽度
- `W_*`：可学习的线性投影矩阵
- 上标 `Q`：query 路径上的量
- 上标 `KV`：key/value 共享 latent 路径上的量
- 上标 `C`：content，表示“内容分支”，不带 RoPE
- 上标 `R`：RoPE，表示“位置分支”
- 上标 `sem`：semantic，表示“理论语义级”的量
- 上标 `abs`：absorbed，表示“把 K 侧变换吸收到 Q 侧之后”的量
- 上标 `lat`：latent，表示“还处在 latent 空间里”的量
- `[a | b]`：把 `a` 和 `b` 在最后一维上拼接
- `α_{i,j}`：第 `i` 个 head 对历史位置 `j` 的 attention 权重
- `RoPE(·)`：rotary position embedding 操作

#### 3.6.1 理论 MLA 的输入是什么

把 MLA 当成一个 attention 层来看，它最原始的输入仍然是标准 Transformer 的 hidden states。

如果是一整段序列，可以写成：

```text
H = [h_1, h_2, ..., h_S],   h_t ∈ R^{d_model}
```

这里这条式子里每个符号的意思分别是：

- `H`：整段输入序列在这一层的 hidden states 集合
- `h_1, h_2, ..., h_S`：第 `1` 到第 `S` 个 token 的 hidden state
- `S`：序列长度，也就是这一段里一共有多少个 token
- `h_t`：第 `t` 个 token 的 hidden state
- `t`：token 下标
- `R^{d_model}`：表示 `h_t` 是一个 `d_model` 维实向量
- `d_model`：模型主干 hidden state 的宽度

如果是 decode 场景，则可以把输入理解成两部分：

- 当前 token 的 hidden state：`h_cur`
- 历史 token 已经缓存好的状态

这里：

- `h_cur`：当前正在生成的这个 token 的 hidden state
- `cur`：当前 token 的位置下标

也就是说，**MLA 的原始输入不是某种新的特殊张量，本质上还是每个 token 的 hidden state。**

只是和普通 MHA 不同，MLA 不会直接从 `h_t` 投出一整套完整 K/V cache，而是先把它们压成 latent。

#### 3.6.2 理论上的 Q/K/V 是怎样定义的

理论 MLA 可以拆成三条路径来看：

1. **Q 路径**
2. **KV 路径**
3. **RoPE 路径**

下面一条一条写。

这里先特别说明一下，避免后面混淆：

> **下面 Q 路径、KV 路径、RoPE 路径这几组式子，先写成“逐 token 的局部定义”。也就是说，这里 `h_t` 统一表示“同一个 token `t` 的 hidden state”，只是为了说明单个 token 的 Q/K/V 相关表示是怎么构造出来的。真正做 attention 时，当前 query token 和历史 key/value token 会分别写成 `h_cur` 和 `h_j`。**

##### Q 路径

先对 query 做低秩压缩，再恢复出每个 head 的 query 内容部分：

```text
c_t^Q        = W_DQ * h_t
q_{t,i}^C    = W_UQ^(i) * c_t^Q
```

这里这两条式子里出现的每个符号分别表示：

- `c_t^Q`：第 `t` 个 token 在 query 路径上的 latent 表示
- 上标 `Q`：说明这个 latent 来自 query 路径
- `W_DQ`：query 路径的下投影矩阵，把 `h_t` 压到 query latent 空间
- `h_t`：第 `t` 个 token 的 hidden state
- `q_{t,i}^C`：第 `t` 个 token、head `i` 的 query 内容分支
- 下标 `t`：这是第 `t` 个 token 的量
- 下标 `i`：这是第 `i` 个 attention head 的量
- 上标 `C`：content，表示这是 query 的内容部分，不带 RoPE
- `W_UQ^(i)`：query 路径恢复第 `i` 个 head 内容分支的上投影矩阵
- `*`：这里表示普通线性矩阵乘法

一句话说，这两条式子的意思是：

> **先把 `h_t` 压成 query latent `c_t^Q`，再从这份 query latent 中恢复第 `i` 个 head 的 query 内容部分 `q_{t,i}^C`。**

##### KV 路径

KV 路径是 MLA 的核心。它先把当前 token 的 hidden state 压成一份共享的 KV latent：

```text
c_t^KV       = W_DKV * h_t
```

这里这条式子里每个符号的意思是：

- `c_t^KV`：第 `t` 个 token 的 KV 共享 latent
- 上标 `KV`：说明这份 latent 是给 key/value 共用的
- `W_DKV`：KV 路径的下投影矩阵，把 `h_t` 压到 KV latent 空间
- `h_t`：第 `t` 个 token 的 hidden state

然后，再从这同一份 `c_t^KV` 中恢复每个 head 语义上需要的 key/value 内容部分：

```text
k_{t,i}^C    = W_UK^(i) * c_t^KV
v_{t,i}^C    = W_UV^(i) * c_t^KV
```

这里这两条式子里每个符号的意思分别是：

- `k_{t,i}^C`：第 `t` 个 token、head `i` 的 key 内容分支
- `v_{t,i}^C`：第 `t` 个 token、head `i` 的 value 内容分支
- 上标 `C`：content，表示内容分支，不带 RoPE
- `W_UK^(i)`：把 KV latent 恢复成第 `i` 个 head 的 key 内容分支的上投影矩阵
- `W_UV^(i)`：把 KV latent 恢复成第 `i` 个 head 的 value 内容分支的上投影矩阵
- `c_t^KV`：第 `t` 个 token 的 KV 共享 latent

这里最关键的一点可以再重复一次：

> **`c_t^KV` 是“每个 token 一份、所有 head 共享”的 latent；`k_{t,i}^C` 和 `v_{t,i}^C` 才是“第 `i` 个 head 自己的语义级 key/value 内容”。**

##### RoPE 路径

如果直接把 RoPE 用在 `k_{t,i}^C` 上，会破坏后面要讲的吸收结构，所以 DeepSeek 把位置编码单独拉出一条分支：

```text
q_{t,i}^R    = RoPE(W_QR^(i) * c_t^Q)
k_t^R        = RoPE(W_KR * h_t)
```

这里这两条式子里每个符号的意思分别是：

- `q_{t,i}^R`：第 `t` 个 token、head `i` 的 query 位置分支
- `k_t^R`：第 `t` 个 token 的共享 key 位置分支
- 上标 `R`：RoPE 分支，也就是位置分支
- `RoPE(·)`：对输入向量施加 rotary position embedding
- `W_QR^(i)`：为第 `i` 个 query head 生成位置分支的线性矩阵
- `W_KR`：为 key 生成共享位置分支的线性矩阵
- `c_t^Q`：第 `t` 个 token 的 query latent
- `h_t`：第 `t` 个 token 的 hidden state

注意这里的一个细节：

- `q_{t,i}^R` 是**按 head 分开**的
- `k_t^R` 是**共享**的

也就是说，`k_t^R` **不是每个 head 各有一份不同的内容**，而是一条共享的 key 位置分支。

##### 把局部定义实例化到真正做 attention 的 `cur / j`

上面那几组式子是在说：“给定任意一个 token `t`，它自己的 Q/K/V 相关表示怎么构造。”

但真正做 attention 时，我们通常讨论的是：

- **当前 query token**：记作 `cur`
- **某个历史 token**：记作 `j`

所以更贴近实际 attention 计算的写法应该是：

```text
c_cur^Q      = W_DQ * h_cur
q_{cur,i}^C  = W_UQ^(i) * c_cur^Q
q_{cur,i}^R  = RoPE(W_QR^(i) * c_cur^Q)

c_j^KV       = W_DKV * h_j
k_{j,i}^C    = W_UK^(i) * c_j^KV
v_{j,i}^C    = W_UV^(i) * c_j^KV
k_j^R        = RoPE(W_KR * h_j)
```

这里这组式子里每个新出现或最关键的符号意思是：

- `h_cur`：当前 query token 的 hidden state
- `cur`：当前 query token 的位置下标
- `c_cur^Q`：当前 query token 的 query latent
- `q_{cur,i}^C`：当前 query token 在第 `i` 个 head 上的 query 内容分支
- `q_{cur,i}^R`：当前 query token 在第 `i` 个 head 上的 query 位置分支
- `h_j`：历史 token `j` 的 hidden state
- `j`：某个历史 token 的位置下标
- `c_j^KV`：历史 token `j` 的 KV 共享 latent
- `k_{j,i}^C`：历史 token `j` 在第 `i` 个 head 上的 key 内容分支
- `v_{j,i}^C`：历史 token `j` 在第 `i` 个 head 上的 value 内容分支
- `k_j^R`：历史 token `j` 的共享 key 位置分支

这样一来就能明确区分：

- `h_t`：逐 token 局部定义时使用的通用记号
- `h_cur`：真正做 attention 时当前 query token 的 hidden state
- `h_j`：真正做 attention 时某个历史 token 的 hidden state

##### 合成理论语义级 Q/K/V

于是，对第 `i` 个 head 来说，真正参与当前这次 attention 的理论语义级 query、key、value 可以写成：

```text
q_{cur,i}^{sem} = [ q_{cur,i}^C | q_{cur,i}^R ]
k_{j,i}^{sem}   = [ k_{j,i}^C | k_j^R ]
v_{j,i}^{sem}   = v_{j,i}^C
```

这里这三条式子里每个符号的意思是：

- `q_{cur,i}^{sem}`：当前 query token 在第 `i` 个 head 上的理论语义级 query
- `k_{j,i}^{sem}`：历史 token `j` 在第 `i` 个 head 上的理论语义级 key
- `v_{j,i}^{sem}`：历史 token `j` 在第 `i` 个 head 上的理论语义级 value
- 上标 `sem`：semantic，表示这是“理论语义级”的量，不是在讨论实现细节
- `[ a | b ]`：把 `a` 和 `b` 在最后一维上拼接
- `q_{cur,i}^C`：当前 query token 的内容分支
- `q_{cur,i}^R`：当前 query token 的位置分支
- `k_{j,i}^C`：历史 token `j` 的 key 内容分支
- `k_j^R`：历史 token `j` 的共享 key 位置分支
- `v_{j,i}^C`：历史 token `j` 的 value 内容分支

这里我特意写成 `sem`，是为了强调：

> **这是一组“语义级 Q/K/V”定义，说明模型等价于在什么量上做 attention；它不代表推理实现一定会逐 head、逐 token 把这些量全部显式展开出来。**

#### 3.6.3 理论上的 attention 是怎么计算的

为了突出 MLA 的结构，下面先不把 causal mask 写进公式里，只看最核心的 attention 计算。

假设我们现在在算当前 query token `cur` 对历史 token `j` 的注意力。对第 `i` 个 head，有：

```text
score_sem(i,j)
  = (q_{cur,i}^{sem})^T k_{j,i}^{sem}
  = (q_{cur,i}^C)^T k_{j,i}^C + (q_{cur,i}^R)^T k_j^R
```

这里这条式子里每个符号的意思分别是：

- `score_sem(i,j)`：第 `i` 个 head 上，当前 query token `cur` 对历史 token `j` 的理论语义级打分
- `i`：head 下标
- `j`：历史 token 下标
- `cur`：当前 query token 的位置
- `(·)^T`：向量转置，因此这里表示点积
- `q_{cur,i}^{sem}`：当前 query token 在第 `i` 个 head 上的理论语义级 query
- `k_{j,i}^{sem}`：历史 token `j` 在第 `i` 个 head 上的理论语义级 key
- `q_{cur,i}^C`：当前 query token 在第 `i` 个 head 上的内容分支
- `k_{j,i}^C`：历史 token `j` 在第 `i` 个 head 上的 key 内容分支
- `q_{cur,i}^R`：当前 query token 在第 `i` 个 head 上的位置分支
- `k_j^R`：历史 token `j` 的共享 key 位置分支

把 `k_{j,i}^C = W_UK^(i) c_j^KV` 代进去，就得到：

```text
score_sem(i,j)
  = (q_{cur,i}^C)^T (W_UK^(i) c_j^KV) + (q_{cur,i}^R)^T k_j^R
```

这里新出现或需要重新强调的符号是：

- `W_UK^(i)`：把 KV latent 恢复成第 `i` 个 head 的 key 内容分支的上投影矩阵
- `c_j^KV`：历史 token `j` 的 KV 共享 latent
- 其余符号 `q_{cur,i}^C`、`q_{cur,i}^R`、`k_j^R` 的含义和上一条式子相同

然后对所有历史位置做 softmax，得到 attention 权重：

```text
α_{i,j}
  = exp(score_sem(i,j)) / Σ_m exp(score_sem(i,m))
```

这里这条式子里每个符号的意思分别是：

- `α_{i,j}`：第 `i` 个 head 对历史位置 `j` 的 attention 权重
- `exp(·)`：指数函数
- `score_sem(i,j)`：第 `i` 个 head 对位置 `j` 的理论语义级打分
- `Σ_m`：对所有历史位置 `m` 求和
- `m`：softmax 归一化时使用的历史位置下标；它只是求和用的临时下标

再用这些权重去加权 value，得到第 `i` 个 head 的输出：

```text
o_{cur,i}^{sem}
  = Σ_j α_{i,j} v_{j,i}^{sem}
  = Σ_j α_{i,j} (W_UV^(i) c_j^KV)
```

这里这条式子里每个符号的意思分别是：

- `o_{cur,i}^{sem}`：当前 query token 在第 `i` 个 head 上的理论语义级输出
- `Σ_j`：对所有历史位置 `j` 的贡献求和
- `α_{i,j}`：第 `i` 个 head 对位置 `j` 的 attention 权重
- `v_{j,i}^{sem}`：历史 token `j` 在第 `i` 个 head 上的理论语义级 value
- `W_UV^(i)`：把 KV latent 恢复成第 `i` 个 head 的 value 内容分支的上投影矩阵
- `c_j^KV`：历史 token `j` 的 KV 共享 latent

最后把所有 head 的输出拼起来，再过输出投影：

```text
u_cur = W_O [ o_{cur,1}^{sem}; o_{cur,2}^{sem}; ...; o_{cur,n_h}^{sem} ]
```

这里这条式子里每个符号的意思分别是：

- `u_cur`：当前 query token 经过这一层 MLA attention 后的层输出
- `W_O`：attention 的输出投影矩阵
- `[ o_{cur,1}^{sem}; ...; o_{cur,n_h}^{sem} ]`：把所有 head 的输出按 head 维拼起来
- `o_{cur,1}^{sem}, ..., o_{cur,n_h}^{sem}`：第 `1` 到第 `n_h` 个 head 的理论语义级输出
- `n_h`：query head 总数
- `;`：这里表示“沿 head 维把多个 head 输出拼接起来”

所以从语义上说，MLA 仍然是标准的 softmax attention，只是：

- Q 被拆成了内容部分和 RoPE 部分
- K 也被拆成了内容部分和 RoPE 部分
- V 不再单独缓存，而是从 `c_t^KV` 恢复

#### 3.6.4 理论 MLA 推理时真正缓存什么

到这里，MLA 最重要的收益就看出来了。

如果按照上面的语义级定义硬算，你会以为推理时要缓存：

- 每个 head 的 `k_{t,i}^C`
- 每个 head 的 `v_{t,i}^C`
- 再加 `k_t^R`

但其实没必要。因为：

```text
k_{t,i}^C = W_UK^(i) c_t^KV
v_{t,i}^C = W_UV^(i) c_t^KV
```

这里这两条式子里每个符号的意思分别是：

- `k_{t,i}^C`：第 `t` 个 token、head `i` 的 key 内容分支
- `v_{t,i}^C`：第 `t` 个 token、head `i` 的 value 内容分支
- `W_UK^(i)`：恢复第 `i` 个 head key 内容分支的上投影矩阵
- `W_UV^(i)`：恢复第 `i` 个 head value 内容分支的上投影矩阵
- `c_t^KV`：第 `t` 个 token 的 KV 共享 latent

这说明：`k_{t,i}^C` 和 `v_{t,i}^C` 都来自同一个 `c_t^KV`。

所以理论上，推理时只需要缓存：

```text
c_t^KV
k_t^R
```

这里：

- `c_t^KV`：第 `t` 个 token 的 KV 共享 latent
- `k_t^R`：第 `t` 个 token 的共享 key 位置分支

也就是说，每个历史 token 真正要留下的状态是：

```text
cache_t = [ c_t^KV | k_t^R ]
```

这里这条式子里每个符号的意思分别是：

- `cache_t`：第 `t` 个历史 token 在推理时真正被缓存下来的状态
- `[ c_t^KV | k_t^R ]`：把 KV 共享 latent 和共享 key 位置分支拼起来
- `c_t^KV`：KV 共享 latent
- `k_t^R`：共享 key 位置分支

这就是 MLA 为什么能显著减小 KV cache。

#### 3.6.5 Q 是如何“吸收” K 侧变换的

这一步是 MLA 最容易被问到、也最容易误解的地方。

前面理论上的分数写成：

```text
score_sem(i,j)
  = (q_{cur,i}^C)^T (W_UK^(i) c_j^KV) + (q_{cur,i}^R)^T k_j^R
```

这里这条式子里每个符号的意思分别是：

- `score_sem(i,j)`：第 `i` 个 head 上，当前 query token `cur` 对历史 token `j` 的理论语义级打分
- `q_{cur,i}^C`：当前 query token 在第 `i` 个 head 上的内容分支
- `W_UK^(i)`：恢复第 `i` 个 head key 内容分支的上投影矩阵
- `c_j^KV`：历史 token `j` 的 KV 共享 latent
- `q_{cur,i}^R`：当前 query token 在第 `i` 个 head 上的位置分支
- `k_j^R`：历史 token `j` 的共享 key 位置分支

看起来你似乎需要先把 `c_j^KV` 通过 `W_UK^(i)` 展开成 `k_{j,i}^C`，再去和 query 做点积。

但线性代数告诉我们：

```text
(q_{cur,i}^C)^T (W_UK^(i) c_j^KV)
= ((W_UK^(i))^T q_{cur,i}^C)^T c_j^KV
```

这里这条式子里每个符号的意思分别是：

- `(q_{cur,i}^C)^T`：当前 query token 在第 `i` 个 head 上的内容分支转置
- `W_UK^(i)`：恢复第 `i` 个 head key 内容分支的上投影矩阵
- `c_j^KV`：历史 token `j` 的 KV 共享 latent
- `(W_UK^(i))^T`：`W_UK^(i)` 的转置
- `((W_UK^(i))^T q_{cur,i}^C)`：把 K 侧上投影吸收到 Q 侧后的新 query 内容向量

于是我们可以定义一个“吸收后的 query 内容部分”：

```text
q_{cur,i}^{C,abs} = (W_UK^(i))^T q_{cur,i}^C
```

这里这条式子里每个符号的意思分别是：

- `q_{cur,i}^{C,abs}`：当前 query token 在第 `i` 个 head 上、吸收了 K 侧变换之后的 query 内容分支
- 上标 `abs`：absorbed，表示“吸收后的”
- `(W_UK^(i))^T`：第 `i` 个 head 的 K 侧上投影矩阵转置
- `q_{cur,i}^C`：当前 query token 在第 `i` 个 head 上的原始内容分支

那么分数就能改写成：

```text
score_sem(i,j)
  = (q_{cur,i}^{C,abs})^T c_j^KV + (q_{cur,i}^R)^T k_j^R
```

这里这条式子里每个符号的意思分别是：

- `score_sem(i,j)`：第 `i` 个 head 对位置 `j` 的理论语义级打分
- `q_{cur,i}^{C,abs}`：吸收后的 query 内容分支
- `c_j^KV`：历史 token `j` 的 KV 共享 latent
- `q_{cur,i}^R`：当前 query token 的位置分支
- `k_j^R`：历史 token `j` 的共享 key 位置分支

这一步的意义非常大：

- 原来：你得先把每个历史 token 的 `c_j^KV` 展开成 `k_{j,i}^C`
- 现在：你只需要把**当前 query**改写一次，就能直接和缓存里的 `c_j^KV` 做点积

所以这里的“Q 吸收”不是说模型把 K 删掉了，而是说：

> **K 侧那一步线性展开，被等价地搬到了 Q 侧来做。**

这是一种**严格等价的代数改写**，不是近似。

#### 3.6.6 输出侧是怎样处理的

value 侧也有一个类似的改写。

理论上：

```text
o_{cur,i}^{sem}
  = Σ_j α_{i,j} (W_UV^(i) c_j^KV)
```

这里这条式子里每个符号的意思分别是：

- `o_{cur,i}^{sem}`：当前 query token 在第 `i` 个 head 上的理论语义级输出
- `Σ_j`：对所有历史位置 `j` 求和
- `α_{i,j}`：第 `i` 个 head 对位置 `j` 的 attention 权重
- `W_UV^(i)`：第 `i` 个 head 的 value 上投影矩阵
- `c_j^KV`：历史 token `j` 的 KV 共享 latent

由于 `W_UV^(i)` 也是线性的，所以可以把它移到求和外面：

```text
o_{cur,i}^{sem}
  = W_UV^(i) (Σ_j α_{i,j} c_j^KV)
```

这里这条式子里每个符号的意思分别是：

- `o_{cur,i}^{sem}`：理论语义级 head 输出
- `W_UV^(i)`：第 `i` 个 head 的 value 上投影矩阵
- `Σ_j α_{i,j} c_j^KV`：在 latent 空间里对所有历史 token 的 KV latent 做加权求和
- `α_{i,j}`：attention 权重
- `c_j^KV`：历史 token `j` 的 KV 共享 latent

定义一个 latent output：

```text
o_{cur,i}^{lat} = Σ_j α_{i,j} c_j^KV
```

这里这条式子里每个符号的意思分别是：

- `o_{cur,i}^{lat}`：当前 query token 在第 `i` 个 head 上的 latent 空间输出
- 上标 `lat`：latent，表示“还在 latent 空间里”
- `Σ_j`：对所有历史 token 做加权求和
- `α_{i,j}`：attention 权重
- `c_j^KV`：历史 token `j` 的 KV 共享 latent

那么：

```text
o_{cur,i}^{sem} = W_UV^(i) o_{cur,i}^{lat}
```

这里这条式子里每个符号的意思分别是：

- `o_{cur,i}^{sem}`：理论语义级 head 输出
- `W_UV^(i)`：第 `i` 个 head 的 value 上投影矩阵
- `o_{cur,i}^{lat}`：第 `i` 个 head 的 latent 空间输出

也就是说，理论上你可以先在 latent 空间里完成 attention 累加，最后再把结果映射回 value/output 空间。

所以 MLA 在理论上可以分成两步理解：

1. **attention 主体**先在 latent 空间里完成
2. **最终 value 语义**再通过 `W_UV` 恢复

#### 3.6.7 理论 MLA 最终得到的结果是什么

如果只看一个 head，第 `i` 个 head 最终得到的是：

```text
o_{cur,i}^{sem}
```

这里：

- `o_{cur,i}^{sem}`：当前 query token 在第 `i` 个 head 上、对全部历史 token 做完 softmax attention 后得到的理论语义级 head 输出
- `cur`：当前 query token 的位置
- `i`：head 下标

如果看整层的最终输出，则是：

```text
u_cur = W_O [ o_{cur,1}^{sem}; ...; o_{cur,n_h}^{sem} ]
```

这里这条式子里每个符号的意思分别是：

- `u_cur`：当前 query token 通过这一整层 attention 后的层输出
- `W_O`：输出投影矩阵
- `o_{cur,1}^{sem}, ..., o_{cur,n_h}^{sem}`：第 `1` 到第 `n_h` 个 head 的理论语义级输出
- `n_h`：query head 总数
- `[ · ; · ]`：沿 head 维把多个 head 的结果拼起来

所以从“输入 -> 输出”的角度，可以把理论 MLA 总结成：

```text
输入:
    当前/整段序列 hidden states h_t

中间状态:
    c_cur^Q, c_j^KV, q_{cur,i}^R, k_j^R

理论语义级 Q/K/V:
    q_{cur,i}^{sem}, k_{j,i}^{sem}, v_{j,i}^{sem}

attention 结果:
    o_{cur,i}^{sem}

层输出:
    u_cur
```

这里这段总结里每个符号的意思是：

- `h_t`：第 `t` 个 token 的 hidden state
- `c_cur^Q`：当前 query token 的 query latent
- `c_j^KV`：历史 token `j` 的 KV 共享 latent
- `q_{cur,i}^R`：当前 query token、head `i` 的 query 位置分支
- `k_j^R`：历史 token `j` 的共享 key 位置分支
- `q_{cur,i}^{sem}`：当前 query token、head `i` 的理论语义级 query
- `k_{j,i}^{sem}`：历史 token `j`、head `i` 的理论语义级 key
- `v_{j,i}^{sem}`：历史 token `j`、head `i` 的理论语义级 value
- `o_{cur,i}^{sem}`：当前 query token 在第 `i` 个 head 上的理论语义级输出
- `u_cur`：当前 query token 的整层输出

#### 3.6.8 一句话总结理论 MLA

如果只想记住一句最核心的话，可以记这个：

> **理论上的 MLA 本质上仍然是 softmax attention，只是把每个 token 的 K/V 先压成共享 latent `c_t^KV`，把位置编码单独走 `k_t^R / q_t^R` 分支，再利用线性代数把 K 侧展开吸收到 Q 侧、把 V 侧展开推迟到输出侧。**

## 4. “Flash” 又是什么意思

这里的 `Flash` 不是指 flash memory，而是指 **FlashAttention 风格的内存高效执行方式**。

它解决的是另一个问题：

- 即使 MLA 把 cache 变小了
- 如果你还按传统方式一次性具象化大块 logits / probability
- 中间状态和访存仍然会很重

Flash 风格的核心做法是：

1. 把长序列沿 `K` 轴切成很多小 chunk
2. 一块一块地处理
3. 不在片上/显存里完整存下整个 logits 矩阵
4. 用 **online softmax** 逐块累加最终结果

也就是说，它并不是改了 softmax attention 的结果，而是改了**算它的方法**。

### 4.1 online softmax 的直觉

普通写法看起来像是：

```text
一次性算完整 QK^T
一次性做完整 softmax
一次性再乘 V
```

Flash 风格则是：

```text
读一小块 K
更新一次局部 softmax 状态
继续读下一块 K
继续更新
直到所有块处理完
```

因此它长期保留的不是完整 logits，而只是少量 running state。直觉上可以理解成：

- 当前见过的最大值
- 当前 softmax 归一化所需的累计量
- 当前输出的累计量

这就是 Flash 的第二层节省来源。


### 8.1 当前实现里的关键维度

当前文档和实现里经常出现的典型参数是：

```text
kv_lora_rank = 512
qk_rope_head_dim = 64
kvpe_dim = 576
head_dim_v = 512
```

这里可以把它读成：

- `512` 维：主要的 latent 内容部分
- `64` 维：RoPE 相关部分
- `576 = 512 + 64`：attention 真正使用的总 K/Q 维度
- `512`：最终 V / 输出使用的 value 维度

### 8.2 当前实现里的 Q 长什么样

在当前实现里，FlashMLA 接到的并不是“原始 query”，而是已经被上游整理好的：

```text
Q_attn = [ q_latent(512) | q_rope(64) ]
```

因此逻辑形状通常是：

```text
q_tensor : [1, B, Hq, 576]
```

也就是说：

- `q_latent` 对应内容路径
- `q_rope` 对应位置路径
- 两者拼起来以后，才是 FlashMLA 实际消费的 `Q`

### 8.3 当前实现里的 KV cache 长什么样

历史 cache 在当前实现里通常表现为一条统一的：

```text
kv_cache_tensor : [B, 1, S, 576]
```

其中单个 token 写入 cache 的内容，可以理解成：

```text
kvpe_t = [ kv_nope_t(512) | kv_rope_t(64) ]
```

这里需要特别提醒：

> **这一节里的 `K` / `V`，已经不是前面论文概念写法里的 `K_t = W_UK * c_t^KV`、`V_t = W_UV * c_t^KV` 那组“语义级 K/V”了，而是当前 `FlashMLA` kernel 在运行时真正消费的“实现级操作数”。**

两者之所以能对上，是因为当前实现已经做了两步代数改写：

1. **K 侧解压被吸收到 Q 里了**
2. **V 侧解压被推迟到 attention 之后了**

所以更准确地说：

- **论文语义级**

```text
K_sem,t = W_UK * c_t^KV
V_sem,t = W_UV * c_t^KV
```

- **当前 kernel 运行级**

```text
K_runtime,t = [ c_t^KV | k_t^R ]
V_runtime,t = c_t^KV
```

其中：

- `c_t^KV` 对应当前实现里的 `kv_nope_t(512)`
- `k_t^R` 对应当前实现里的 `kv_rope_t(64)`

也就是说，当前 kernel 实际拿来做 attention 的，并不是“已经完整展开后的 `K_sem,t / V_sem,t`”，而是：

- 一条共享内容 latent `c_t^KV`
- 外加一条 RoPE key `k_t^R`

最关键的一点是：

> **当前这条实现里，`K` 和 `V` 不再是两份完全独立的 cache。**

更具体地说：

- `K_runtime` 使用整条 `576` 维 cache
- `V_runtime` 使用同一条 cache 的前 `512` 维视图

也就是：

```text
K_runtime = kv_cache[..., :576]
V_runtime = kv_cache[..., :512]
```

为什么这在数学上是合理的？因为：

- 对 `QK^T` 来说，`W_UK` 已经吸收到 `Q` 侧，所以当前 kernel 只需要拿 `c_t^KV` 来和改写后的 `q_latent` 做点积
- 对 `PV` 来说，当前 kernel 先累加 `c_t^KV` 的加权和，之后再通过后续的 `W_UV` 路径把它映射到真正的 value/output 空间

所以：

> **前面那组式子是在说“这个模型从语义上等价于什么 attention”；这里这组式子是在说“当前实现运行时真正把什么东西放进 cache，并喂给 kernel”。**

这也是当前实现里“统一 latent cache”最直观的地方。

### 8.3.1 把理论符号和当前实现一一对上

如果把论文里的符号和当前 `tt-metal` 实现强行放在同一个平面上看，最容易乱掉。更稳妥的方式是先把“谁是语义级量，谁是运行级量”分开。

| 层次 | 理论/实现符号 | 当前实现里的名字 | 维度 | 含义 |
|---|---|---|---|---|
| 输入 hidden state | `h_t` | 上游当前 token hidden state | `d_model` | 生成 Q/KV 分支的源输入 |
| KV 内容 latent | `c_t^KV` | `kv_nope_t` | `512` | KV 的共享内容压缩表示 |
| KV 位置分支 | `k_t^R` | `kv_rope_t` | `64` | decoupled RoPE key |
| 运行时 cache entry | `[c_t^KV \| k_t^R]` | `kvpe_t` / `kv_cache[..., t, :]` | `576` | 当前 kernel 真正读到的单 token cache |
| Q 内容分支（吸收前） | `q_{t,i}^{C,orig}` | `q_nope` | `128` / head | 原始 query 内容部分 |
| Q 内容分支（吸收后） | `q_{t,i}^{C,lat}` | `q_latent` | `512` / head | 把 `W_UK` 吸收到 Q 侧后的 query 内容部分 |
| Q 位置分支 | `q_{t,i}^R` | `q_rope` | `64` / head | query 的 RoPE 分支 |
| kernel 真正消费的 query | `[q_{t,i}^{C,lat} \| q_{t,i}^R]` | `Q_attn` / `q_tensor` | `576` / head | FlashMLA 实际输入的 Q |
| kernel 输出的 latent 结果 | `o_{i}^{lat}` | `output_tensor` / `attn_out` | `512` / head | attention 后但还没过 `W_UV` 的 latent 输出 |
| 最终 value/output 空间结果 | `o_i` | `wkv_b2` 之后的结果 | `v_head_dim` | 真正恢复到 value 空间的输出 |

最容易混淆的一点是：

> **`c_t^KV` 不是 `576` 维；`576` 维的是 `kvpe_t = [c_t^KV | k_t^R]`。**

在当前实现里，这三个量的关系应该记成：

```text
c_t^KV        <-> kv_nope_t      (512)
k_t^R         <-> kv_rope_t      (64)
kvpe_t        =  [c_t^KV | k_t^R] = [kv_nope_t | kv_rope_t]   (576)
```

同理，Q 侧也不是直接把原始 `q_nope` 拿来和 `c_t^KV` 做点积，而是先做一次“吸收后的改写”：

```text
q_{t,i}^{C,lat} = (W_UK^(i))^T q_{t,i}^{C,orig}
Q_attn,i        = [ q_{t,i}^{C,lat} | q_{t,i}^R ]
```

这样一来，理论上的注意力打分和当前 kernel 的运行时打分就能严格对上。对单个 head `i`、单个历史 token `t` 来说：

```text
score_sem(i,t)
  = (q_{i}^{C,orig})^T (W_UK^(i) c_t^KV) + (q_i^R)^T k_t^R
  = ((W_UK^(i))^T q_i^{C,orig})^T c_t^KV + (q_i^R)^T k_t^R
  = (q_i^{C,lat})^T c_t^KV + (q_i^R)^T k_t^R
  = [q_i^{C,lat} | q_i^R]^T [c_t^KV | k_t^R]
```

也就是说：

- **理论上**，你可以理解成 `q_nope` 去和 `W_UK c_t^KV` 做内容匹配
- **实现上**，等价地改写成 `q_latent` 去和 `c_t^KV` 做内容匹配，再把 RoPE 分支单独拼上

输出侧也是同一个道理。理论上每个 token 对应的是：

```text
v_{t,i}^{sem} = W_UV^(i) c_t^KV
```

但当前实现不会在 attention 前把这一步显式做出来，而是先在 latent 空间里累加：

```text
o_i^{lat} = Σ_t α_{i,t} c_t^KV
```

然后再在 attention 之后做：

```text
o_i = W_UV^(i) o_i^{lat}
```

这正对应当前实现里的：

- `FlashMLA` kernel 输出 `512` 维 latent `output_tensor`
- 后续 `wkv_b2` 再把这 `512` 维映射回真正的 value/output 空间

如果只想记一句最不容易错的话，可以记这个：

> **论文里缓存的是 `c_t^KV + k_t^R` 这组“可重建语义”的状态；当前实现里真正写进 cache、喂给 kernel 的，就是它们拼起来形成的 `kvpe_t`，而不是完整展开后的 `K_sem / V_sem`。**

### 8.3.2 按时间顺序看：理论 QKV 是怎样变成当前实现 QKV 的

如果把整条链路按“什么时候做了什么”展开，那么最清楚的方式不是只盯着某一个符号，而是分成下面四层：

1. **理论语义级 Q/K/V**
2. **进入 FlashMLA kernel 之前的前处理**
3. **FlashMLA kernel 运行时真正消费的 Q/K/V**
4. **FlashMLA kernel 结束之后的后处理**

下面按这个顺序讲。

#### 第 1 层：理论语义级 Q/K/V 是什么

先只看论文层面的 attention 语义，不管 kernel 怎么写。

对当前 decode token 的第 `i` 个 query head，可以把理论上的 query 写成：

```text
c^Q            = W_DQ * h_cur
q_i^{C,orig}   = W_UQ^(i) * c^Q
q_i^R          = RoPE(W_QR^(i) * c^Q)
q_i^{sem}      = [ q_i^{C,orig} | q_i^R ]
```

对历史第 `t` 个 token，可以把理论上的 key/value 写成：

```text
c_t^KV         = W_DKV * h_t
k_t^R          = RoPE(W_KR * h_t)
k_{t,i}^{sem}  = [ W_UK^(i) * c_t^KV | k_t^R ]
v_{t,i}^{sem}  = W_UV^(i) * c_t^KV
```

于是理论上的 attention 语义就是：

```text
score_sem(i,t) = (q_i^{sem})^T k_{t,i}^{sem}
               = (q_i^{C,orig})^T (W_UK^(i) c_t^KV) + (q_i^R)^T k_t^R

o_i^{sem}      = Σ_t α_{i,t} v_{t,i}^{sem}
               = Σ_t α_{i,t} (W_UV^(i) c_t^KV)
```

这里要注意两点：

- `q_i^{sem}`、`k_{t,i}^{sem}`、`v_{t,i}^{sem}` 是**理论上等价的语义级 Q/K/V**
- 它们并不意味着当前实现真的会在 runtime 把这些量全部逐 head 显式展开出来

#### 第 2 层：进入 FlashMLA kernel 之前做了哪些前处理

当前 `tt-metal` 实现不会把“理论语义级 K/V”原样交给 FlashMLA kernel，而是先做一轮前处理，把它们改写成更适合 kernel 的形式。

##### 2.1 Q 侧前处理

Q 侧在进入 FlashMLA kernel 之前，会先完成下面几步：

```text
tt_q
 -> q_norm
 -> wq_b
 -> reshape/slice
 -> q_nope(128), q_rope(64)
 -> wkv_b1 对 q_nope 做 W_UK^T
 -> q_latent(512)
 -> 对 q_rope 做 RoPE
 -> concat
 -> Q_attn = [ q_latent | q_rope ]   (576)
```

这一步的本质是：

> **把理论上应该放在 K 侧的 `W_UK`，提前吸收到 Q 侧。**

对应的代数关系就是：

```text
(q_i^{C,orig})^T (W_UK^(i) c_t^KV)
= ((W_UK^(i))^T q_i^{C,orig})^T c_t^KV
```

所以进入 kernel 前，Q 已经从“理论语义级的 `q_i^{sem}`”变成了“运行时更适合做点积的 `Q_attn_i`”：

```text
q_i^{C,lat} = (W_UK^(i))^T q_i^{C,orig}
Q_runtime,i = Q_attn_i = [ q_i^{C,lat} | q_i^R ]
```

换句话说，**Q 的这部分变化属于前处理，发生在 FlashMLA kernel 之前。**

##### 2.2 KV 侧前处理

KV 侧在进入 FlashMLA kernel 之前，也会先完成一轮整理：

```text
tt_kv_nope
 -> kv_norm
 -> kv_nope_t          (对应 c_t^KV 这条内容 latent)

tt_kv_rope
 -> RoPE
 -> kv_rope_t          (对应 k_t^R 这条位置分支)

concat
 -> kvpe_t = [ kv_nope_t | kv_rope_t ]   (576)
 -> 写入 kv_cache
```

这一步的本质是：

- **内容侧 latent** 直接保存在 cache 里
- **RoPE key 分支** 也直接保存在 cache 里
- 但**理论语义级的 `k_{t,i}^{sem} = [W_UK^(i)c_t^KV | k_t^R]` 和 `v_{t,i}^{sem} = W_UV^(i)c_t^KV` 并不会在这里显式展开**

所以，KV 进入 cache 时，存进去的是：

```text
K_runtime,t = [ c_t^KV | k_t^R ]
V_runtime,t = c_t^KV
```

而不是：

```text
K_sem,t = [ W_UK^(i)c_t^KV | k_t^R ]
V_sem,t = W_UV^(i)c_t^KV
```

换句话说，**KV 的“压缩并打包成 cache entry”也是前处理，发生在 FlashMLA kernel 之前。**

#### 第 3 层：FlashMLA kernel 运行时真正消费的 Q/K/V 是什么

当 FlashMLA kernel 真正开始跑时，它拿到的不是理论语义级 Q/K/V，而是：

```text
Q_runtime,i = [ q_i^{C,lat} | q_i^R ]        = Q_attn_i
K_runtime,t = [ c_t^KV | k_t^R ]             = kvpe_t
V_runtime,t = c_t^KV                         = kvpe_t[..., :512]
```

因此 kernel 内部真正做的打分是：

```text
score_runtime(i,t)
  = (Q_runtime,i)^T K_runtime,t
  = [q_i^{C,lat} | q_i^R]^T [c_t^KV | k_t^R]
  = (q_i^{C,lat})^T c_t^KV + (q_i^R)^T k_t^R
```

这和前面的理论语义级打分是严格等价的，因为：

```text
q_i^{C,lat} = (W_UK^(i))^T q_i^{C,orig}
```

所以 kernel 内部**没有再做一次 `W_UK` 的显式 K 侧展开**。这件事已经在 Q 侧前处理里通过 `wkv_b1` 等价完成了。

同理，kernel 内部对输出做的是：

```text
o_i^{lat} = Σ_t α_{i,t} c_t^KV
```

也就是说，FlashMLA kernel 输出的是一个 **latent output**，还不是最终语义级的 `o_i^{sem}`。

#### 第 4 层：FlashMLA kernel 之后做了哪些后处理

FlashMLA kernel 输出 `attn_out` 之后，当前实现还要再做一次后处理：

```text
attn_out / output_tensor   (512 latent dim)
 -> wkv_b2
 -> 恢复到 value/output 空间
 -> 后续再接 output projection
```

这一步对应的代数关系是：

```text
o_i^{sem}
  = Σ_t α_{i,t} (W_UV^(i) c_t^KV)
  = W_UV^(i) (Σ_t α_{i,t} c_t^KV)
  = W_UV^(i) o_i^{lat}
```

所以：

> **理论上属于 `V` 侧的 `W_UV`，在当前实现里不是 attention 前做的，而是 attention 后作为后处理通过 `wkv_b2` 做的。**

这也是为什么当前实现里：

- kernel 输出是 `512` 维 latent output
- 之后还需要再过一次 `wkv_b2`

#### 一句话总结这四层的关系

如果只想记一句最不容易错的话，可以记这个：

> **理论上，attention 是在 `q_i^{sem}`、`k_{t,i}^{sem}`、`v_{t,i}^{sem}` 上定义的；实现上，`W_UK` 被前移成 Q 侧前处理，`W_UV` 被后移成输出侧后处理，而 FlashMLA kernel 中间真正消费的是 `Q_attn = [q_latent|q_rope]` 和 `kvpe_t = [c_t^KV|k_t^R]`。**

### 8.4 当前实现的输入输出

把它压缩成最常见的输入输出形式，就是：

```text
q_tensor        : [1, B, Hq, 576]
kv_cache_tensor : [B, 1, S, 576]
output_tensor   : [1, B, Hq, 512]
```

如果只记一件事，可以记这个：

> **在当前实现里，进入 FlashMLA kernel 的 `Q_runtime` 是“吸收后的内容 + 位置”的拼接，`K_runtime` 是整条统一 cache，而 `V_runtime` 是这条 cache 的前半部分 latent 视图。**

## 9. 一次 decode 大致是怎么跑的

不看底层 kernel 细节，只看逻辑流程，Flash MLA 的一次 decode 可以分成下面几步。

### 第 1 步：准备当前 token 的 Query

上游先从当前 token 的 hidden state 里生成：

- 内容侧的 `q_latent`
- 位置侧的 `q_rope`

然后拼成：

```text
Q_attn = [q_latent | q_rope]
```

### 第 2 步：历史 token 已经躺在统一 KV cache 里

历史每个 token 都已经有一条压缩后的 `kvpe_t` 存在 cache 中，所以这一步不会像普通 attention 那样去找两大份独立的 `K cache` 和 `V cache`。

### 第 3 步：沿序列轴分块读取历史 cache

Flash 风格不会一次把整条历史都摊开来算，而是把历史 `K` 按 chunk 切开，逐块处理。

每处理一块，就做一次：

```text
局部 QK^T
局部 mask
局部 softmax 更新
局部 PV 更新
```

### 第 4 步：逐块累加，而不是保存整张大矩阵

这里不会长期保存完整的 logits 矩阵，也不会长期保存完整的概率矩阵。

它保留的是 online softmax 的累计状态，并在每个 chunk 上继续更新。

### 第 5 步：得到输出

当所有 chunk 都处理完以后，累计状态就可以还原出最终输出：

```text
output_tensor : [1, B, Hq, head_dim_v]
```

在当前实现里通常就是：

```text
output_tensor : [1, B, Hq, 512]
```

## 10. 为什么它通常特别适合 decode

Flash MLA 的收益在 **decode** 场景里最明显，因为 decode 有几个很特殊的特点：

1. 当前 `Q` 很小，通常只有一个 token
2. 历史 `K/V` 很长，cache 是主要成本
3. 每一步都要重新扫历史，所以带宽特别重要

MLA 正好解决第 2 点，Flash 正好解决第 3 点，所以二者叠加后，decode 往往是最受益的场景。

换句话说：

> **prefill 更像“大批量算一次”；decode 更像“每次只来一点 Q，但要不停回看很长历史”。Flash MLA 正是为这种场景量身定做的。**

## 11. 它的收益主要来自哪里

把上面的内容再汇总一下，Flash MLA 的收益主要来自四个方面。

### 11.1 cache 更小

不再缓存完整 K/V，而是缓存更紧凑的 latent 状态，加上一小部分位置相关信息。

### 11.2 历史读取更少

因为历史状态本身更小，所以每次 decode 要搬运的字节数也更少。

### 11.3 中间状态更少

Flash 风格不会长期具象化完整 logits / probability 矩阵，而是只维护少量 running state。

### 11.4 更容易做硬件友好的数据流

一旦历史被切成 chunk，kernel 就更容易围绕：

- chunk 大小
- 片上 buffer
- 多核分工
- 流水重叠

来做高效实现。

这也是为什么 Flash MLA 不只是“数学上更省”，而且“工程上更值得做”。

## 12. 容易误解的几个点

### 12.1 它不是“没有 K/V”

MLA 不是把 K/V 取消了，而是：

- **语义上**仍然有 K 和 V
- **存储上**不一定把它们当作两份完整独立张量来缓存
- **执行上**尽量在更紧凑的表示里完成计算

### 12.2 它也不是通常意义上的“线性 attention”

虽然有些二手资料会把它说得很像“线性 attention”，但 DeepSeek 的 MLA 重点不是把 softmax 换掉，而是：

- 压缩 KV
- 重写 Q/K/V 的表示
- 用更高效的 kernel 去执行原本的 attention 语义

所以更准确的理解仍然是：

> **它是 memory-efficient / cache-efficient 的 softmax attention 变体。**

### 12.3 “MLA”和“FlashMLA”不是同一层概念

- `MLA` 更偏 **模型表示/注意力形式**
- `FlashMLA` 更偏 **高效执行/kernel 实现**

可以类比成：

- MLA 解决“应该缓存什么、attention 应该怎样表达”
- FlashMLA 解决“这些东西应该怎样高效地算出来”

### 12.4 不同实现的张量长相可能不同

DeepSeek 论文、GPU kernel、当前 `tt-metal` 路径在张量布局上可能不完全一样。

但只要抓住下面三点，就不会迷路：

1. **历史状态比普通 K/V cache 更紧凑**
2. **Q 会被整理到和 latent 路径兼容的空间**
3. **执行时按 chunk 流式完成 softmax 和输出累加**

## 13. 如果只想记住最核心的图景

可以把 DeepSeek Flash MLA 想成下面这张“脑内流程图”：

```text
普通 attention:
    缓存完整 K + 完整 V
    每次 decode 反复读取大 cache
    中间容易出现很大的 attention 状态

DeepSeek MLA:
    先把历史 K/V 压成更小的 latent cache
    尽量把计算改写到 latent 空间里做
    RoPE 单独走一条小分支

Flash MLA:
    在 MLA 的基础上
    再把长历史按 chunk 流式处理
    不具象化完整 logits
    只维护 online softmax 的累计状态
```

如果再压成最后一句：

> **DeepSeek Flash MLA 的本质，就是“压缩历史状态 + 流式计算长注意力”。**

## 14. 继续往下读可以看什么

如果你想继续深挖当前仓库里的具体实现，建议接着看：

- `flash-mla-implementation-walkthrough.md`
- `flash-mla-dataflow-first-principles.md`
- `flash-mla-wh-decode-quickstart.md`

如果你想区分“公开确认的 DeepSeek 信息”和“当前仓库里的实现取舍”，可以再看：

- `deepseek-v4-attention-and-tt-dataflow-notes.md`

## 15. 参考线索

本文主要基于以下两类材料整理：

1. **公开资料**
   - DeepSeek-V2 论文：<https://arxiv.org/abs/2405.04434>
   - DeepSeek FlashMLA 仓库：<https://github.com/deepseek-ai/FlashMLA>
2. **当前仓库文档**
   - `flash-mla-implementation-walkthrough.md`
   - `flash-mla-dataflow-first-principles.md`
   - `mla-performance-analysis.md`
