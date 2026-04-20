# Part I 代理对齐检查

说明：Part I 已把 `DeepSeek FlashMLA` 固定为 `decode` 主方法；此表只用来判断当前 `TT 主线 FlashMLA` 的 dense profile 是否还能作为 Part II 长序列机理 proxy。

| case | seq_len | TT mainline ms | DeepSeek ms | DeepSeek / TT | delta vs TT |
|---|---|---:|---:|---|---|
| decode_1k | 1024 | 0.203 | 0.181 | 0.894x | -10.6% |
| decode_4k | 4096 | 0.312 | 0.298 | 0.956x | -4.4% |
| decode_16k | 16384 | 0.699 | 0.681 | 0.973x | -2.7% |
| decode_32k | 32768 | 1.224 | 1.216 | 0.994x | -0.6% |
