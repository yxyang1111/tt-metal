# DeepSeek MLA 理论计算顺序说明

更新时间：`2026-04-26`

## 0. 先看总公式

如果先不展开细节，只把理论 MLA 的整条计算链压成 3 行，可以写成：

$$
\begin{aligned}
C &= W_{DKV} H_{kv}, \qquad R = \mathrm{RoPE}(W_{KR} H_{kv}) \\
q_a &= (W_{UK})^T W_{UQ} W_{DQ} h_q, \qquad q_r = \mathrm{RoPE}(W_{QR} W_{DQ} h_q) \\
o &= W_{UV}\left(\mathrm{softmax}\left(q_a^T C + q_r^T R\right) C\right)
\end{aligned}
$$

其中，`h_q` 是当前 query 的 hidden state，`H_kv` 是整段历史 hidden state 的集合；`C` 是历史侧共享的 KV latent 集合，`R` 是历史侧 key 的 RoPE 分支集合；`q_a` 是把 `W_UK` 吸收到 Q 侧之后得到的 query 内容分支，`q_r` 是 query 的 RoPE 分支。中间的 `q_a^T C + q_r^T R` 给出当前 query 对整段历史的打分，`softmax` 把它变成注意力权重，最后再由 `W_UV` 把 latent 空间中的加权结果恢复成最终输出 `o`。

这篇文档只讲 **理论上的 MLA**，不讲 `tt-metal` 的具体实现，也不讲 Flash kernel。

目标只有一个：

> **按真正的计算顺序，把 MLA 讲清楚：输入 -> `QK`（含 Q 吸收）-> softmax -> `PV`。**

为了尽量少被符号干扰，下面只看：

- 一个当前 query
- 一条历史序列
- 一个 attention head

多头情况只是把同样的过程并行重复很多次。

## 1. 输入

我们先定义三类输入/中间集合：

- `h_q`：当前 query 这一侧的 hidden state
- `H_kv`：整段历史侧 hidden states 的集合
- `h_kv`：历史侧任意一个位置的 hidden state

这里：

- `h_q` 用来生成当前 query
- `H_kv` 用来表示整段历史
- `h_kv` 只是为了说明“单个历史元素是怎么构造出 key/value 的”

MLA 用到的主要线性矩阵有 6 个：

- `W_DQ`：把 query 侧 hidden state 压成 query latent
- `W_UQ`：从 query latent 恢复 query 的内容分支
- `W_DKV`：把历史侧 hidden state 压成 KV latent
- `W_UK`：从 KV latent 恢复 key 的内容分支
- `W_UV`：从 KV latent 恢复 value
- `W_QR` / `W_KR`：生成 RoPE 分支

如果只记一句话，就是：

> **MLA 的原始输入仍然是 hidden state，只是它不会直接生成完整 K/V，而是先压成 latent。**

## 2. `QK`：先看 Q 和 K 怎么来

### 2.1 Query 怎么来

先从当前 query 的 hidden state `h_q` 出发：

```text
c_q = W_DQ h_q
q_c = W_UQ c_q
q_r = RoPE(W_QR c_q)
q   = [q_c | q_r]
```

这四步的含义分别是：

1. `c_q = W_DQ h_q`
   把当前 query 的 hidden state 压成一份 query latent `c_q`

2. `q_c = W_UQ c_q`
   从 `c_q` 恢复 query 的内容分支 `q_c`

3. `q_r = RoPE(W_QR c_q)`
   从 `c_q` 生成 query 的位置分支，并施加 RoPE，得到 `q_r`

4. `q = [q_c | q_r]`
   把内容分支和位置分支拼起来，得到理论上的 query

所以：

- `q_c` 负责“内容匹配”
- `q_r` 负责“位置匹配”

### 2.2 历史侧的 K / V 怎么来

如果只看历史侧任意一个元素 `h_kv`，它的构造方式和 query 类似，只是路径不同：

```text
c   = W_DKV h_kv
k_c = W_UK c
r   = RoPE(W_KR h_kv)
k   = [k_c | r]
v   = W_UV c
```

这几步的含义分别是：

1. `c = W_DKV h_kv`
   把历史侧 hidden state 压成共享的 KV latent `c`

2. `k_c = W_UK c`
   从 `c` 恢复 key 的内容分支

3. `r = RoPE(W_KR h_kv)`
   从历史侧 hidden state 生成 key 的位置分支

4. `k = [k_c | r]`
   拼成理论上的 key

5. `v = W_UV c`
   从同一份 latent `c` 恢复理论上的 value

这里最重要的一点是：

> **历史侧真正的“压缩表示”是 `c`。`k_c` 和 `v` 都是从 `c` 恢复出来的语义级量。**

### 2.3 把单个历史元素扩展成整段历史

如果把整段历史上的每个位置都做同样的变换，就得到下面这些集合：

```text
C = 历史侧所有 c 组成的集合
R = 历史侧所有 r 组成的集合
K = [W_UK C | R]
V = W_UV C
```

这里：

- `C` 是整段历史的 KV latent 集合
- `R` 是整段历史的 key 位置分支集合
- `K` 是整段历史的语义级完整 key 集合
- `V` 是整段历史的语义级完整 value 集合

### 2.4 理论上的 `QK` 打分

有了 `q` 和整段历史的 `K` 之后，理论上的打分可以写成：

```text
s = q^T K
  = [q_c | q_r]^T [W_UK C | R]
  = q_c^T (W_UK C) + q_r^T R
```

这条式子的含义非常重要：

- `s` 是当前 query 对整段历史的分数向量
- `q_c^T (W_UK C)` 是**内容部分的匹配**
- `q_r^T R` 是**位置部分的匹配**

所以 MLA 的 `QK` 不是简单的一项，而是两部分相加：

```text
s = 内容匹配 + 位置匹配
```

## 3. Q 是如何吸收的

上面内容匹配那一项是：

```text
q_c^T (W_UK C)
```

这一步看起来像是要先在 K 侧把 `C` 展开成完整的内容 key，再和 `q_c` 做点积。

但它可以严格等价地改写成：

```text
q_c^T (W_UK C)
= ((W_UK)^T q_c)^T C
```

于是我们定义一个吸收后的 query 内容分支：

```text
q_a = (W_UK)^T q_c
```

这里：

- `q_a` 里的 `a` 表示 absorbed
- 它的意思是：已经把原本属于 K 侧的 `W_UK` 吸收到 Q 侧之后的新 query 内容分支

这样一来，打分就能改写成：

```text
s = q_a^T C + q_r^T R
```

这一步的意义可以直接说成：

> **原本在 K 侧做的线性展开 `W_UK C`，被等价地搬到了 Q 侧来做。**

所以“Q 吸收”不是说 K 消失了，而是说：

- 原来：先把 `C` 展开成完整内容 key
- 现在：先把 `q_c` 改写成 `q_a`
- 然后直接用 `q_a` 去和 `C` 做内容点积

这样做的直接好处是：

> **历史侧不需要每次都显式构造完整的内容 key，只需要保留 `C`。**

## 4. softmax

有了整段历史上的分数向量 `s` 之后，下一步就是做 softmax：

```text
p = softmax(s)
```

这里：

- `p` 是整段历史上的注意力权重向量
- `softmax` 会把分数向量 `s` 归一化成概率权重

这一步和普通 attention 没有本质区别：

> **MLA 没有改 softmax 本身，它改的是分数 `s` 是怎么构造出来的。**

你也可以把它理解成：

- 前面 `QK` 负责产生“整段历史里每个位置该看多少”
- softmax 负责把这些分数归一化成概率权重

## 5. `PV`：value 是怎么来的，输出怎么得到

### 5.1 理论上的 value

前面已经有：

```text
V = W_UV C
```

也就是说：

- `V` 不是直接缓存下来的
- 它是从历史侧 latent 集合 `C` 恢复出来的语义级 value 集合

### 5.2 理论上的 `PV`

如果按最直接的语义级写法，输出是：

```text
o = p V
  = p (W_UV C)
```

这就是普通 attention 里的 `PV`：

- `p` 是 softmax 权重
- `V` 是 value 集合
- `o` 是最终输出向量

### 5.3 为什么 `PV` 也可以后移

因为 `W_UV` 是线性的，所以：

```text
o = p (W_UV C)
  = W_UV (p C)
```

于是我们定义一个 latent 空间中的输出：

```text
o_lat = p C
```

那么最终输出就可以写成：

```text
o = W_UV o_lat
```

这一步的含义是：

> **attention 主体可以先在 latent 空间里完成，再通过 `W_UV` 把结果恢复到最终输出空间。**

所以从 `PV` 的角度看，MLA 理论上可以分成两步：

1. 先用注意力权重去加权历史 latent 集合，得到 `o_lat`
2. 再通过 `W_UV` 把 `o_lat` 映射成最终输出 `o`

## 6. 按顺序串起来看一遍

如果把整条 MLA 理论计算链压成最简的顺序，就是：

### 第一步：输入

```text
当前 query 输入: h_q
历史侧输入集合: H_kv
```

### 第二步：构造 query

```text
c_q = W_DQ h_q
q_c = W_UQ c_q
q_r = RoPE(W_QR c_q)
```

### 第三步：把历史侧压成 latent 和 RoPE 集合

```text
C = W_DKV H_kv
R = RoPE(W_KR H_kv)
```

### 第四步：做 `QK`，并把 K 侧变换吸收到 Q 侧

```text
q_a = (W_UK)^T q_c
s   = q_a^T C + q_r^T R
```

### 第五步：softmax

```text
p = softmax(s)
```

### 第六步：做 `PV`

```text
o_lat = p C
o     = W_UV o_lat
```

这就是理论 MLA 最核心的顺序。

## 7. 一句话总结

如果只记一句话，可以记这个：

> **MLA 的理论本质是：先把历史侧 K/V 压成共享 latent 集合 `C`，在 `QK` 阶段把 `W_UK` 吸收到 Q 侧，用 `q_a^T C + q_r^T R` 计算分数；softmax 之后，不直接对完整 `V` 做加权，而是先得到 latent 输出 `o_lat`，最后再通过 `W_UV` 恢复成最终输出。**
