# FlashMLA WH Profile 泳道图

## 1. 使用说明

这份文档把当前 Wormhole 上 FlashMLA 的关键 profile 结果，整理成几张**代表性泳道图**。

读图时请注意：

- 横向时间轴是**概念时序**，不是严格按真实时间比例绘制。
- 泳道分别对应：
  - `Reader / NCRISC`
  - `Compute / TRISC`
  - `Writer / BRISC`
- 图中的百分比来自 profile 的平均 stage share 或 bubble share。
- `wait-front / reserve-back` 仍按 **accumulated stall density** 解读，不直接等价于 wall-time 占比。

这几张图分别对应：

- `decode_1k`：短序列，仍接近 compute-critical
- `decode_4k`：关键迁移点，reader 开始转向 `K reserve`
- `decode_16k`：长序列形态已经基本固定
- `prefill_4k`：direct PM util 可读，适合画完整饱和流水线图

## 2. Decode 短序列：`decode_1k`

```mermaid
sequenceDiagram
    participant R as Reader / NCRISC
    participant C as Compute / TRISC
    participant W as Writer / BRISC

    Note over R,W: decode_1k | compute 仍是最后退休线程，reader 以 K/V issue 为主

    rect rgb(239, 246, 255)
        loop steady-state tile
            Note over R: issue 77.27%<br/>reserve 0.70%<br/>wait 16.49%<br/>K total 72.51% / V total 27.49%
            R->>C: 连续发 K/V tile

            Note over C: max TRISC share 99.64%<br/>wait-front 187.5 us<br/>bubble 内占比 86.12%<br/>reserve-back 30.2 us
            C->>W: 产出 partial output / reduction fragment

            Note over W: cb_wait 94.15%<br/>sender 48.58%<br/>tree 51.42%<br/>root/output-gather = 0%
            W-->>C: 已有回压，但 output-side 仍不是主导矛盾
        end
    end
```

### 读图要点

- reader 主要还是在主动发 `K/V` 请求，`reserve` 还很小。
- compute bubble 主要是 `wait-front`，说明 compute 主要在等输入。
- writer 已经明显不轻，但其主等待仍然是内部 `sender/tree` 链路，不是 final gather。

## 3. Decode 迁移点：`decode_4k`

```mermaid
sequenceDiagram
    participant R as Reader / NCRISC
    participant C as Compute / TRISC
    participant W as Writer / BRISC

    Note over R,W: decode_4k | 从 issue-dominant 向 K reserve-dominant 迁移的关键转折点

    rect rgb(255, 248, 235)
        loop steady-state tile
            Note over R: issue 47.94%<br/>reserve 39.78%<br/>wait 10.65%<br/>k_reserve 31.64%
            R->>C: K 仍在持续前送，但 turnover 已被下游压住

            Note over C: wait-front 332.1 us (78.12%)<br/>reserve-back 93.0 us (21.88%)<br/>output/backpressure 开始抬升
            C->>W: partial output 进入 reduction / output path

            Note over W: cb_wait 98.10%<br/>sender 49.59%<br/>tree 50.41%
            W-->>R: 回压开始明确传回 K path
            W-->>C: compute 已开始明显感受到 output-side pressure
        end
    end
```

### 读图要点

- `decode_4k` 是最值得关注的迁移点。
- reader 的主状态从 `issue` 快速切到 `issue + reserve` 混合，而且关键新增项是 `k_reserve`。
- compute bubble 中 `reserve-back` 占比开始变大，说明 output/backpressure 成分已经不可忽略。
- writer 已经几乎整段都在 `cb_wait`。

## 4. Decode 长序列：`decode_16k`

```mermaid
sequenceDiagram
    participant R as Reader / NCRISC
    participant C as Compute / TRISC
    participant W as Writer / BRISC

    Note over R,W: decode_16k | 长序列形态已基本固定，reader / writer / compute 三侧强耦合

    rect rgb(255, 240, 244)
        loop steady-state tile
            Note over R: reserve 59.71%<br/>issue 34.90%<br/>k_reserve 52.79%<br/>v_reserve 2.77%
            R->>C: 输入供给越来越由 K reserve / downstream turnover 决定

            Note over C: wait-front 442.9 us (71.96%)<br/>reserve-back 172.6 us (28.04%)<br/>max TRISC share 99.95%
            C->>W: partial output 持续进入 sender / tree reduction

            Note over W: cb_wait 99.56%<br/>sender 49.89%<br/>tree 50.11%<br/>issue/barrier/pop 接近 0
            W-->>R: sender/tree 链路形成稳定回压
            W-->>C: output/backpressure 已成为 bubble 的重要组成
        end
    end
```

### 读图要点

- 到 `decode_16k`，reader 已明确由 `K reserve` 主导，而不是 `V reserve`。
- writer 的主等待仍然不在 final gather，而是在 `sender` 和 `tree child arrival`。
- compute 虽然仍然是满窗线程，但 bubble 里 `reserve-back` 已经接近 `28%`，说明长序列下 output-side pressure 明显增强。

## 5. Decode 极长序列：`decode_32k`

```mermaid
sequenceDiagram
    participant R as Reader / NCRISC
    participant C as Compute / TRISC
    participant W as Writer / BRISC

    Note over R,W: decode_32k | reader_writer_saturated，16k 形态继续固化

    rect rgb(246, 239, 255)
        loop steady-state tile
            Note over R: reserve 61.54%<br/>issue 33.38%<br/>k_reserve 54.82%
            R->>C: reader 几乎一直受 K path reserve 限制

            Note over C: wait-front 829.7 us (70.91%)<br/>reserve-back 340.3 us (29.09%)
            C->>W: 输出推进到 sender / tree reduction

            Note over W: cb_wait 99.77%<br/>sender 49.95%<br/>tree 50.05%
            W-->>R: 回压继续锁定 reader turnover
            W-->>C: reserve-back 继续抬升
        end
    end
```

### 读图要点

- `decode_32k` 不是新模式，而是 `decode_16k` 的进一步固化。
- reader / writer / compute 三侧都在贴近 kernel window。
- 如果只说“compute-bound”已经不准确，更准确的是强耦合饱和流水线。

## 6. Prefill 饱和流水线：`prefill_4k`

```mermaid
sequenceDiagram
    participant R as Reader / NCRISC
    participant C as Compute / TRISC
    participant W as Writer / BRISC

    Note over R,W: prefill_4k | direct PM util 可读，但主 wall-time 仍由流水线等待决定

    rect rgb(238, 252, 241)
        loop steady-state chunk
            Note over R: wait 66.10%<br/>issue 33.15%<br/>reserve 0.33%<br/>NCRISC share 99.98%
            R->>C: tile 到达前存在长 in-flight read wait

            Note over C: PM FPU util 19.427%<br/>PM compute 10.785 ms / kernel 55.516 ms<br/>wait-front 92.6% of bubble
            C->>W: compute / reduction output 前送

            Note over W: cb_wait 99.62%<br/>issue 0.19%<br/>barrier 0.19%<br/>BRISC share 100.00%
            W-->>C: 持续等待 compute/reduction 结果 ready
        end
    end
```

### 读图要点

- prefill 从一开始就是强耦合流水线，不存在 decode 那种明显的阶段迁移。
- reader 主要在 `wait`，不是在 `reserve`。
- writer 几乎整个窗口都在 `cb_wait`。
- compute thread 虽然几乎满窗，但 direct PM 说明真正算术利用率只有 `19.4%`，所以主矛盾不是“算力不够”，而是流水线等待。

## 7. Prefill PM 曲线对应的泳道解读

`prefill_256 -> prefill_4k` 的 direct PM util 是：

- `10.196% -> 13.904% -> 16.762% -> 18.475% -> 19.427%`

对应的泳道含义是：

- compute 确实越来越“热”，所以 `PM FPU util` 在上升。
- 但 reader `wait` 仍在 `70.30% -> 66.10%`，writer `cb_wait` 仍在 `91.26% -> 99.62%`。
- 这说明 prefill 的性能提升，并不是把流水线等待彻底消掉了，而是在等待仍然占主导的前提下，算术活动比例有所上升。

## 8. 一页总览

### 8.1 Decode 的泳道迁移

- `decode_1k`：reader 以 `K/V issue` 为主，compute 主要看到 `wait-front`
- `decode_4k`：`k_reserve` 第一次大幅抬升，writer `cb_wait` 几乎贴满
- `decode_16k/32k`：reader 由 `K reserve` 主导，writer 稳定卡在 `sender/tree` 两段，compute 的 `reserve-back` 明显增大

### 8.2 Prefill 的泳道形态

- 从 `prefill_256` 开始就已经是 reader / writer / compute 强耦合
- reader 主导项始终是 `wait`
- writer 主导项始终是 `cb_wait`
- compute 的 direct PM util 可读，但仍不足以把 prefill 归类成“纯算力瓶颈”

## 9. 配套文档

更详细的数字分析见：

- `mla_flash_attention_dev/docs/flash-mla-wh-component-utilization-and-bubble-analysis.md`

如果需要看包含：

- 同一个 `S block` 内部
- 不同 `S block` 之间
- 多个 tensix core
- `Host CPU` / `DRAM` / `CB` / `semaphore` 信号

的详细版，请看：

- `mla_flash_attention_dev/docs/flash-mla-wh-detailed-swimlane-diagrams.md`
- `mla_flash_attention_dev/docs/assets/flash-mla-wh-profile-swimlanes-detailed/`
