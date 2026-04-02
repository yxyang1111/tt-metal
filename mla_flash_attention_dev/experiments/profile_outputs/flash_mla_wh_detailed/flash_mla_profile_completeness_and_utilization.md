# FlashMLA Profile 完整性与利用率说明

## 最后建议阅读顺序

- 全量结果表：`flash_mla_wh_detailed_profile_report.md`
- 完整性与利用率说明：`flash_mla_profile_completeness_and_utilization.md`
- 读图版摘要：`flash_mla_visual_summary.md`
- 可视化总入口：`flash_mla_visual_dashboard.html`

## 这轮结果完整到哪一层

- `base measured profile`：`13/13`，包含 `kernel / BRISC / NCRISC / TRISC* / classification`。
- `reader stage breakdown`：`13/13`，覆盖 `page_table / reserve / issue / wait / push`。
- `writer stage breakdown`：`13/13`，覆盖 `cb_wait / issue / barrier / pop`。
- `decode reader source breakdown`：`0/8`，当前结果里还没有真实 `K/V` source 数据。
- `decode writer source breakdown`：`0/8`，当前结果里还没有真实 `sender/root/tree/output` source 数据。
- `PM 利用率计数`：`0/13`，当前结果里 `PM FPU/NOC/DRAM` 都还是空。
- `compute 空泡计数`：`0/13`，当前结果里 `DEVICE COMPUTE CB WAIT FRONT/RESERVE BACK` 都还是空。

结论：这轮数据已经 **完整到 stage-level**，但还没有完整到 **PM utilization / compute-side bubble / source-level attribution** 这三层。

## Case 完整性矩阵

| case | base profile | reader stages | writer stages | reader source | writer source | PM util | compute bubble |
|---|---|---|---|---|---|---|---|
| prefill_256 | ok | ok | ok | n/a | n/a | missing | missing |
| prefill_512 | ok | ok | ok | n/a | n/a | missing | missing |
| prefill_1k | ok | ok | ok | n/a | n/a | missing | missing |
| prefill_2k | ok | ok | ok | n/a | n/a | missing | missing |
| prefill_4k | ok | ok | ok | n/a | n/a | missing | missing |
| decode_256 | ok | ok | ok | missing | missing | missing | missing |
| decode_512 | ok | ok | ok | missing | missing | missing | missing |
| decode_1k | ok | ok | ok | missing | missing | missing | missing |
| decode_2k | ok | ok | ok | missing | missing | missing | missing |
| decode_4k | ok | ok | ok | missing | missing | missing | missing |
| decode_8k | ok | ok | ok | missing | missing | missing | missing |
| decode_16k | ok | ok | ok | missing | missing | missing | missing |
| decode_32k | ok | ok | ok | missing | missing | missing | missing |

## 能不能用 profile 看利用率和空泡？

可以，但要分成两层来看。

### 现在已经能直接看的

- `线程窗口占比`：`BRISC/NCRISC/TRISC` 相对 `kernel window` 的占比，能看谁在逼近 critical path，谁还有 slack。
- `reader 空泡/反压代理量`：`reserve share`。它表示 reader 在等下游 CB 空间，本质上就是 backpressure stall。
- `writer 空泡/等待代理量`：`cb_wait share`。它表示 writer 在等 compute/reduction 把结果推到输出 CB。
- `hot-path coverage`：详细报告里的 `coverage vs thread` 能告诉你 marker 已经解释了多少线程时间；剩余部分是未跟踪控制路径，不能直接等同于空泡。

### 当前还不能直接看的

- `精确硬件利用率百分比`：当前结果里的 `PM FPU UTIL (%)`、`NOC UTIL (%)`、`MULTICAST NOC UTIL (%)`、`DRAM BW UTIL (%)` 都没有值，所以现在不能直接回答“FPU 利用率是 63% 还是 81%”这种问题。
- `compute-side 空泡`：当前结果里的 `DEVICE COMPUTE CB WAIT FRONT` 和 `DEVICE COMPUTE CB RESERVE BACK` 也没有值，所以还不能直接量化 compute 自己在等输入还是等输出。
- `source-level stall attribution`：虽然内核里已经接好了 `K/V reserve` 和 `sender/root/tree/output wait` 的 marker，但这批 JSON 还没有下一次重跑后的真实值。

### 这轮最可靠的空泡结论

- `decode_32k`：reader 的 `reserve share=61.5%`，writer 的 `cb_wait share=99.8%`。这说明长序列 decode 的主要空泡不是纯 compute idle，而是 **reader downstream backpressure + writer output-availability wait**。
- `prefill_4k`：reader 的 `wait share=66.1%`，writer 的 `cb_wait share=99.6%`。这说明 prefill 从一开始就更像强耦合饱和流水线，而不是短 decode 那种 compute-dominant 形态。

## 如果你下一步要真正看“利用率百分比”和“compute 空泡”

- 继续保留现有 detailed stage-level profile，因为这层已经完整。
- 重新跑一次能稳定产出 `PM FPU/NOC/DRAM` 的 profile，确认这些列不再是空值。
- 用下一次 detailed rerun 把新的 `source-level marker` 数据真正灌进 JSON，这样就能把 `reader reserve` 和 `writer cb_wait` 拆到更直接的来源。
