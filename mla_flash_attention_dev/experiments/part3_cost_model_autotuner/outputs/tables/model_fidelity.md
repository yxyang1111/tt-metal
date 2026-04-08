# Cost Model 误差汇总

说明：`first-order` 指 profile bridge 绑定到 reference-current-A candidate 后的原始 analytical latency；`second-order fit` 为全样本拟合；`second-order LOO` 为 leave-one-out。

| metric | first-order | second-order fit | second-order LOO |
|---|---:|---:|---:|
| MAPE | 38.62% | 19.58% | 24.65% |
| MAE | 4.822 ms | 0.030 ms | 0.069 ms |
| RMSE | 13.592 ms | 0.043 ms | 0.109 ms |
| max abs error | 47.607 ms | 0.131 ms | 0.321 ms |
