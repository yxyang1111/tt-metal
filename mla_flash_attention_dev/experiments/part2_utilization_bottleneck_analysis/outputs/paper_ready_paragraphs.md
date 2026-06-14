# 正文段落草稿（保守版）

## 1. 可直接写入正文

### 1.1 利用率与 Bubble 结果

现有 `Part II` 的利用率结果需要按“critical-path attachment”而不是“算术峰值”来解读。在 `decode_1k / 4k / 16k / 32k` 上，`compute share` 已达到 `99.68% / 99.86% / 99.96% / 99.98%`，不过，修正 runtime 后的 representative seq direct rerun（`1k/4k/8k/16k/32k`）已经给出 decode arithmetic-utilization curve：`PM FPU util (TT / DeepSeek-4c)` = `1k` 12.17% / 12.63%；`4k` 20.57% / 20.84%；`8k` 23.11% / 23.34%；`16k` 24.73% / 24.85%；`32k` 25.61% / 25.67%，`Avg FPU util on full grid` = `1k` 10.36% / 10.75%；`4k` 17.13% / 17.35%；`8k` 19.18% / 19.37%；`16k` 20.48% / 20.58%；`32k` 21.19% / 21.24%。因此现在可以把 decode 代表点写成“算术利用率会从 low-teens 迁移到 mid-20s，但仍远未接近峰值”，而不能把 full-sweep 的 `compute share` 直接写成“算术利用率已经很高”。更准确的说法是：compute thread 始终贴着 kernel critical path，任何前后级 stall 最终都会投影成 compute bubble。同时，bubble 组成也在迁移：`wait-front share` 从 `86.1%` 下降到 `70.9%`，`reserve-back share` 从 `13.9%` 升到 `29.1%`；`decode_4k` 已出现 `reader reserve=32.0%` 与 `writer cb_wait=98.0%` 的同步抬升。这说明短序列更接近 front-side starvation，而长序列已经演化成前端供给不足与后端回压并存的 stalled pipeline。

### 1.2 供给为什么不好

从 source-level 看，当前 decode 的供给瓶颈几乎全部集中在 `K` 路，而不是 `V` 路。`K issue` 从 `decode_1k` 的 `56.87%` 下降到 `decode_32k` 的 `26.10%`，`K reserve` 从 `0.37%` 升到 `54.82%`；相比之下，`V reserve` 到 `decode_32k` 仍只有 `2.80%`。与此同时，`effective single-pass K read bandwidth` 从 `39.52` 下降到 `18.00`。因此 long-seq 的关键矛盾不是“有没有读到 KV”，而是 `K` 路能否持续、稳定地把下一批数据喂给后续流水线；reader 越来越多地停在 reserve / slot 可用性等待上，而不是忙于真正 issue 新请求。

### 1.3 回压为什么强

回压的主矛盾也很清楚。`writer cb_wait` 从 `decode_1k` 的 `93.87%` 升到 `decode_32k` 的 `99.78%`；writer source-level 上 `sender/tree` 在代表点几乎稳定五五开，`decode_32k` 为 `49.95%/50.05%`，`root/output` 近乎为 `0`。这说明回压不是 final gather 的尾部偶发问题，而是跨 S-block sender/tree reduction 的结构性同步代价。更重要的是，`reserve-back share` 从 `13.9%` 升到 `29.1%`。
`D2` 的 `DeepSeek-8c` 在跨过 q-core 容量墙之后 `reserve-back / kernel` 仍有 `276.3%`，且 `sender/tree/root/output = 49.41%/49.72%/0.88%/0.00%`。这说明就算 admission 问题被 `8c` 解决，后端 reduction / output 链路仍然排不空。

### 1.4 为什么当前 TT-MLA 会慢

把供给与回压合在一起看，当前 `TT-MLA` 的 slowdown 可以概括为双向挤压：前端 `K` 供给跟不上，compute 更容易出现 `wait-front`；后端 writer/reduction 排不空，compute 又承担越来越重的 `reserve-back`。因此表面上看到的是 compute share 一直很高，但本质上这不是算术优势，而是 compute 被迫暴露在前后级 stall 的总和上。这也解释了为什么阶段迁移会从 `compute_on_critical_path` 经 `writer_close_to_critical_path`、`reader_close_to_critical_path`，最终走到 `reader_writer_saturated`：序列越长，供给与回压越同时显性化，kernel wall-time 就越难下降。

### 1.5 现阶段能如何写 SF-MLA 的优势

现阶段关于 `SF-MLA` 优势来源的写法仍应保持克制，但现在已经不必再完全停留在 proxy 边界上。`A1/A2/A3` 的 paired direct device-only probe 显示，`DeepSeek-4c` 的 device kernel duration 分别为 `70.36 / 171.26 / 1110.67 us`，对应 `TT` 为 `73.86 / 174.67 / 1114.36 us`；`wait-front / kernel` 也从 `TT` 的 `108.1% / 87.2% / 74.5%` 下降到 `DeepSeek-4c` 的 `94.9% / 81.3% / 73.2%`。`A2/A3` 的 raw log 还显示 `DeepSeek-4c` 的 `K reserve` 为 `28.59% / 51.06%`，对应 `TT` 为 `31.48% / 51.40%`。与此同时，`B/C` 六个 `4c vs 8c` direct 点的 `8c / 4c kernel` 只落在 `0.995x~1.009x`，`B/C` 的 reader `K reserve` 与 writer `sender/tree` 也仍停留在近乎相同的结构：`4c` 为 `K reserve=0.11%~0.13%`、`sender/tree=41.73%~43.83% / 55.96%~57.56%`，`8c` 为 `K reserve=0.11%~0.13%`、`sender/tree=41.71%~43.82% / 55.97%~57.58%`。而这些点都满足 `batch * q_shards = 24`，即正好 fit 在 `4c` 的 q-core 容量里；`D1/D2` 则进一步表明，一旦需求升到 `32 / 48`，`4c` 会直接 unsupported，而 `8c` 才重新变得可运行。同一条 `seq=8k, H=32, q_shards=4` 轴上的 boundary corrected-runtime `perf-fpu` rerun 也已经补齐：`TT PM FPU util @ 24q/28q/32q/40q/48q = 24q 21.86%; 28q 21.60%; 32q 21.27%; 40q 20.36%; 48q 18.21%`，`DeepSeek-8c PM FPU util @ 24q/28q/32q/40q/48q = 24q 12.08%; 28q 13.78%; 32q 13.74%; 40q 16.40%; 48q 18.53%；`DeepSeek-4c` 则只在 `<= 24q` 仍可运行，`28q` 起直接停在 unsupported_precheck。另外，修正 runtime 后的 `perf-fpu` rerun（`1k/4k/8k/16k/32k`）还表明 `PM FPU util (TT / DeepSeek-4c)` = `1k` 12.17% / 12.63%；`4k` 20.57% / 20.84%；`8k` 23.11% / 23.34%；`16k` 24.73% / 24.85%；`32k` 25.61% / 25.67%，`Avg FPU util on full grid` = `1k` 10.36% / 10.75%；`4k` 17.13% / 17.35%；`8k` 19.18% / 19.37%；`16k` 20.48% / 20.58%；`32k` 21.19% / 21.24%，且两条实现的 `PM FPU util` 差值始终不超过 0.46 pp。这说明 `SF-MLA` 的领先并不是把 decode 算术利用率抬到一个截然不同的饱和区间，而更像是通过 mapping / dataflow 缓解了 front-side starvation；而“更多核心”主要在跨越容量墙时才是必要条件，而不是所有点上的普适速度来源。

### 1.6 为什么 SF-MLA 仍有改进空间

即使接受上述改进，当前 direct 结果也已经说明 `SF-MLA` 仍有明显优化空间。`DeepSeek-4c` 在 `A2/A3` 上的 `reserve-back / kernel` 仍有 `26.8% / 30.1%`，`DeepSeek-8c` 在 `D2` 上的 `reserve-back / kernel` 也仍有 `276.3%`，`D2` 的 writer source-level 还显示 `sender/tree = 49.41%/49.72%`，而 proxy 侧的 `decode_32k` 仍显示 `K reserve=54.82%`、`sender/tree = 49.95%/50.05%`。这说明更优 mapping 可以降低 front-side starvation，但不会自动消除 `K` 供给、reduction 同步和输出回压问题；这些仍然是 long-seq 场景下的主要优化靶点。

### 1.7 Prefill 控制对照

`prefill` 结果可作为一个有用的控制组：`PM FPU util` 从 `prefill_256` 的 `10.20%` 升到 `prefill_4k` 的 `19.43%`，但 `writer cb_wait` 也从 `91.50%` 升到 `99.62%`。这说明 `prefill` 从很早开始就是强耦合的饱和流水线，因此它更适合作为控制对照，而不是当前论文主矛盾的展开重点。

## 2. 不建议现在这样写

- 不建议写“`DeepSeek FlashMLA` 已被当前 `Part II` fully direct 地证明是因为某个特定 stage/source 更优所以更快”，因为当前 direct 虽然已经补到 reader / writer / reduction 三侧，也补到了 `1k/4k/8k/16k/32k` 的 PM/perf-counter representative points，但它还不是贯穿 `decode_1k~32k` dense sweep 的 direct 算术利用率曲线。
- 不建议写“`8c` 总是比 `4c` 快”，因为 `B/C` 的 direct 结果已经显示 `8c / 4c` 只在 `0.995x~1.009x`；更准确的写法是：`8c` 主要用来抬高 q-core 容量上限，而不是已 fit 点上的普适加速。
- 不建议把当前 representative `1k/4k/8k/16k/32k` 的 `PM FPU util` 直接当成 `decode_1k~32k` 的 dense 算术利用率曲线，因为它目前仍只覆盖少量 representative points，而不是完整 sweep。

## 3. 下一步最小补证建议

- `reader / writer / reduction` 三侧 direct，加上 corrected-runtime `1k/4k/8k/16k/32k` PM/perf-counter representative points，已经足够支撑“为什么现在快/慢”的正文写法。
- 如果正文必须给出更强的 decode 算术利用率随序列变化趋势，不要再重复当前 `1k/4k/8k/16k/32k` representative rerun，而应直接在当前已修通的 runtime 基础上继续补容量边界点，或进一步向 dense sweep 靠拢。

