# L40S Same-Latency-Protocol Remeasurement Request

## 背景

当前 `flashinfer_mla_decode_prefill_sweep_l40s.md` 里的 L40S latency 和 Wormhole/TT 的 latency 不是严格同一种测量口径：

- L40S 当前报告更接近 GPU kernel/device timing。报告中写的是 CUDA event latency：decode 使用 `warmup=10, iters=50`；prefill 使用 chunked total CUDA event latency，长 shape 为 `warmup=0, iters=1`。
- TT/Wormhole sweep 更接近一次 `ttnn` op 调用的 host-observed synchronized latency。代码在每次测量前后调用 `ttnn.synchronize_device(device)`，并用 Python `time.perf_counter()` 包住一次 `ttnn` 调用。

因此，当前 `L40S/3` 可以作为 bandwidth-normalized GPU reference，但不能说和 TT latency 是严格 apples-to-apples 的同口径测量。

## 目标

请在 L40S 服务器上重新测 FlashInfer MLA latency，采用与 TT sweep 更一致的 host-observed synchronized timing 口径。

重新测出的 raw L40S latency 后续仍会乘以 `3.0` 得到 `L40S/3` bandwidth-normalized reference，但 raw latency 本身需要先改成同口径 host-synchronized latency。

## 必须使用的测量口径

不要使用 CUDA event 作为最终 latency。请使用 Python wall-clock timing，并在 timed region 前后显式同步 GPU：

```python
torch.cuda.synchronize()
t0 = time.perf_counter()
out = run_once()
torch.cuda.synchronize()
t1 = time.perf_counter()
latency_ms = (t1 - t0) * 1000.0
del out
```

这对应 TT sweep 里的模式：

```python
ttnn.synchronize_device(device)
t0 = time.perf_counter()
out = run_once()
ttnn.synchronize_device(device)
t1 = time.perf_counter()
latency_ms = (t1 - t0) * 1000.0
```

## Timed Region

Timed region 只包含一次 FlashInfer MLA operator execution：

- 输入 tensor 创建不计入 timed region。
- paged index/cache 构造不计入 timed region。
- FlashInfer wrapper/workspace 创建不计入 timed region。
- `wrapper.plan(...)` 不计入 timed region。
- 只计 `wrapper.run(...)` 加上 host dispatch、runtime launch、device execution，以及最终同步等待。

这和 TT 当前数据类似：TT 的 tensor transfer/layout conversion/page-table setup 也在 timed region 外，timed region 包住一次 `ttnn` op 调用。

## Workload

保持当前 L40S 报告中的 DeepSeek-V3 MLA shape：

- `Hq = 32`
- `Hkv = 1`
- `d_c = 512`
- `d_r = 64`
- page/block size = `64`
- K chunk size = `128` 如脚本中适用
- dtype = BF16
- `seq_len = {256, 512, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K}`
- `batch = {1, 2, 4, 8, 16, 32, 64}`

## Repeats

为了和 TT sweep 更接近，请使用下面的 repeat 策略：

- Decode: `warmup=1, iters=3`
- Prefill:
  - 如果 `batch * seq_len >= 16384`: `warmup=0, iters=1`
  - 否则如果 `batch * seq_len >= 8192`: `warmup=1, iters=2`
  - 否则: `warmup=1, iters=3`

请在输出里记录每个点实际使用的 `warmup` 和 `iters`。

## Decode 测量

Decode 使用 FlashInfer MLA paged attention，`q_len=1`，`kv_len=seq_len`。

每个 `(batch, seq_len)`：

1. 在 GPU 上创建输入 tensor、paged metadata、workspace。
2. 创建 `BatchMLAPagedAttentionWrapper`。
3. 调用 `wrapper.plan(...)`。
4. warmup 若干次，每次执行 `wrapper.run(...)` 后 `torch.cuda.synchronize()`。
5. 用 host-synchronized `time.perf_counter()` 测 `iters` 次。
6. 输出所有 samples、mean/min/max/std。

## Prefill 测量

优先测两种 prefill 输出，方便后续选择使用：

### A. TT 当前覆盖可比口径

这个口径用于和当前 consolidated TT prefill 数据直接对比。

- 只需要对 TT 有效覆盖区域做严格对比：`seq_len <= 4096` 且 `batch * seq_len <= 16384`。
- 使用 `q_len=seq_len`, `kv_len=seq_len`, `causal=True`。
- `wrapper.plan(...)` 在 timed region 外。
- timed region 只包一次 `wrapper.run(...)`。

### B. 长序列 chunked total 口径

这个口径用于保留当前 L40S 长序列 prefill reference 的能力。

- 使用 `chunk_size=4096`。
- 对 `start = 0, 4096, ...` 逐 chunk 测量。
- 每个 chunk 使用 `q_len = min(chunk_size, seq_len - start)`，`kv_len = start + q_len`。
- 每个 chunk 的 `wrapper.plan(...)` 不计入 timed region。
- 每个 chunk 的 timed region 用 host-synchronized `time.perf_counter()` 包住一次 `wrapper.run(...)`。
- total latency = 所有 chunk latency 之和。

请在报告中明确区分：

- `prefill_single_host_sync_ms`
- `prefill_chunked_total_host_sync_ms`

如果时间有限，优先完成 A，因为它和当前 TT consolidated prefill 数据最可比。

## 输出格式

请生成新的 Markdown/CSV/JSON，文件名建议：

- `flashinfer_mla_decode_prefill_sweep_l40s_host_sync.md`
- `outputs/flashinfer_mla_decode_sweep_l40s_host_sync.csv`
- `outputs/flashinfer_mla_decode_sweep_l40s_host_sync.json`
- `outputs/flashinfer_mla_prefill_sweep_l40s_host_sync.csv`
- `outputs/flashinfer_mla_prefill_sweep_l40s_host_sync.json`

Markdown 至少包含：

1. GPU/stack 信息：GPU name、SM、memory、PyTorch、CUDA、FlashInfer version。
2. 明确说明：latency 使用 host-synchronized `time.perf_counter()`，不是 CUDA event。
3. Decode mean latency table。
4. Prefill single host-sync mean latency table，如果有。
5. Prefill chunked total host-sync latency table，如果有。
6. 每类测试的 warmup/iters 策略。
7. 对每个点记录 raw L40S latency，以及 `latency_l40s_div3_ms = raw_latency_ms * 3.0`。

JSON/CSV 每行建议字段：

- `mode`
- `batch`
- `seq_len`
- `seq_label`
- `status`
- `latency_ms`
- `latency_l40s_div3_ms`
- `min_ms`
- `max_ms`
- `std_ms`
- `samples`
- `warmup`
- `iters`
- `timing_method = "host_sync_perf_counter"`
- `note`

## 验收标准

- 报告中不能再把最终 latency 称为 CUDA event latency。
- Decode 和 prefill 的 timing method 必须明确是 `host_sync_perf_counter`。
- `wrapper.plan(...)` 不应计入 timed region。
- 每个 measured point 都要保存 samples，不只保存 mean。
- 如果某些 long prefill 点只能做 chunked total，请在 note 中标明 `chunked_total` 和 `chunk_size=4096`。

