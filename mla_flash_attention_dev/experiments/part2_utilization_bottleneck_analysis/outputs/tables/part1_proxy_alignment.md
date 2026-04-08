# Part I 代理对齐检查

说明：Part I 已把 `DeepSeek FlashMLA` 固定为 `decode` 主方法；此表只用来判断当前 `TT 主线 FlashMLA` 的 dense profile 是否还能作为 Part II 长序列机理 proxy。

| case | seq_len | TT mainline ms | DeepSeek ms | DeepSeek / TT | delta vs TT |
|---|---|---:|---:|---|---|
| decode_1k | 1024 | 0.209 | 0.180 | 0.864x | -13.6% |
| decode_4k | 4096 | 0.291 | 0.298 | 1.025x | 2.5% |
| decode_16k | 16384 | 0.708 | 0.690 | 0.976x | -2.4% |
| decode_32k | 32768 | 1.240 | 1.237 | 0.997x | -0.3% |
