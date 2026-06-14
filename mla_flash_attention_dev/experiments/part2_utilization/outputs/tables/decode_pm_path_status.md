# Decode PM / Perf-Counter 路径状态（A1 smoke + corrected-runtime representative seq rerun）

说明：

- 这张表汇总为解决 decode 侧 `PM / perf-counter` 直采而额外发起的六条 `A1` 最小 smoke 链路，覆盖 `DeepSeek-4c` 与 `TT` 两个 decode 变体。
- 目标不是比较性能，而是判断哪条链路能为 decode 提供**可解释的** `PM IDEAL / PM COMPUTE / PM BANDWIDTH / PM FPU UTIL (%)`。
- 初始 six-smoke 的结论是：当时活跃 runtime 下，`DeepSeek-4c` 的 `tracy_report + sum / no-runtime-analysis / perf-fpu + sync` 与 `TT` 的 `tracy_report + sum / perf-fpu + sync` 都只返回 placeholder，`DeepSeek-4c` 的 `device_only + perf-fpu` 则不给 PM/FPU 列；这一步帮助把问题定位到 decode PM 路径而不是 case 级缺失。

| 方法 | case | path | capture | trigger | sync | PM cols | bubble cols | PM IDEAL | PM COMPUTE | PM BANDWIDTH | PM FPU util | extra FPU cols | 结论 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-4c | A1__deepseek4c | tracy_report + sum + sync | tracy_report | sum | yes | yes | yes | 1.000 | 1.000 | 1.000 | 0.001% | n/a | PM 列存在但仍是 placeholder；bubble 列可用 |
| DeepSeek-4c | A1__deepseek4c | tracy_report + no-runtime-analysis + sync | tracy_report | no-runtime-analysis | yes | yes | yes | 1.000 | 1.000 | 1.000 | 0.001% | n/a | PM 列存在但仍是 placeholder；bubble 列可用 |
| DeepSeek-4c | A1__deepseek4c | device_only + perf-fpu | device_only | perf-fpu | no | no | no | n/a | n/a | n/a | n/a | n/a | 无可解释的 PM/FPU 列 |
| DeepSeek-4c | A1__deepseek4c | tracy_report + perf-fpu + sync | tracy_report | perf-fpu | yes | yes | yes | 1.000 | 1.000 | 1.000 | 0.001% | n/a | PM 列仍是 placeholder；无额外硬件 FPU counter 列；bubble 列可用 |
| TT | A1__tt | tracy_report + sum + sync | tracy_report | sum | yes | yes | yes | 1.000 | 1.000 | 1.000 | 0.001% | n/a | PM 列存在但仍是 placeholder；bubble 列可用 |
| TT | A1__tt | tracy_report + perf-fpu + sync | tracy_report | perf-fpu | yes | yes | yes | 1.000 | 1.000 | 1.000 | 0.001% | n/a | PM 列仍是 placeholder；无额外硬件 FPU counter 列；bubble 列可用 |


补充修正（corrected-runtime rerun）：

- 后续补齐了 `SdpaDecodeDeviceOperation::create_op_performance_model`，并确认 Python runtime 实际仍在加载旧的 `ttnn/ttnn/_ttnn.so -> build_Release/lib/_ttnncpp.so`。
- 在执行 `cmake --install build_Release --component tt_pybinds` 与 `cmake --install build_Release --component tar` 之后，修正 runtime 的 `A1` rerun 已恢复为非-placeholder PM；同一条 `perf-fpu + sync` 链路现也已扩到 `1k/4k/8k/16k/32k`，并能在同一份 artifact 中直接产出硬件 `FPU / SFPU / MATH` 列。

| 方法 | case | path | PM IDEAL | PM COMPUTE | PM BANDWIDTH | PM FPU util | extra FPU cols | 结论 |
|---|---|---|---|---|---|---|---|---|
| TT | A1__tt | corrected-runtime `tracy_report + sum + sync` | 8704 | 8704 | 2129 | 11.801% | n/a | PM 已恢复为非-placeholder |
| DeepSeek-4c | A1__deepseek4c | corrected-runtime `tracy_report + sum + sync` | 8704 | 8704 | 2129 | 12.328% | n/a | PM 已恢复为非-placeholder |
| TT | A1__tt | corrected-runtime `tracy_report + perf-fpu + sync` | 8704 | 8704 | 2129 | 12.169% | SFPU/FPU/MATH | 同一份 artifact 同时包含 PM 与 perf-counter 列 |
| DeepSeek-4c | A1__deepseek4c | corrected-runtime `tracy_report + perf-fpu + sync` | 8704 | 8704 | 2129 | 12.628% | SFPU/FPU/MATH | 同一份 artifact 同时包含 PM 与 perf-counter 列 |


代表点 corrected-runtime `perf-fpu` seq trend（1k/4k/8k/16k/32k）：

- 同一条修正后的 `tracy_report + perf-fpu + sync` 链路现已覆盖 `1k/4k/8k/16k/32k`；`PM FPU util (TT / DeepSeek-4c)` = `1k` 12.17% / 12.63%；`4k` 20.57% / 20.84%；`8k` 23.11% / 23.34%；`16k` 24.73% / 24.85%；`32k` 25.61% / 25.67%，`Avg FPU util on full grid` = `1k` 10.36% / 10.75%；`4k` 17.13% / 17.35%；`8k` 19.18% / 19.37%；`16k` 20.48% / 20.58%；`32k` 21.19% / 21.24%。这说明 decode arithmetic-utilization 会随序列拉长单调抬升，但 `TT` 与 `DeepSeek-4c` 之间的差值始终很小。

| point | case | seq_len | PM IDEAL | PM BANDWIDTH | TT PM FPU util | DeepSeek PM FPU util | gap | TT Avg FPU util | DeepSeek Avg FPU util | TT Avg MATH util | DeepSeek Avg MATH util |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1k | A1 | 1024 | 8704 | 2129 | 12.17% | 12.63% | 0.46 pp | 10.36% | 10.75% | 11.70% | 12.14% |
| 4k | A2 | 4096 | 34816 | 8516 | 20.57% | 20.84% | 0.26 pp | 17.13% | 17.35% | 18.61% | 18.84% |
| 8k | A2_8K | 8192 | 69632 | 17033 | 23.11% | 23.34% | 0.23 pp | 19.18% | 19.37% | 20.68% | 20.89% |
| 16k | A2_16K | 16384 | 139264 | 34066 | 24.73% | 24.85% | 0.11 pp | 20.48% | 20.58% | 22.01% | 22.12% |
| 32k | A3 | 32768 | 278528 | 68132 | 25.61% | 25.67% | 0.06 pp | 21.19% | 21.24% | 22.73% | 22.79% |
