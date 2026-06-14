# Consolidated MLA Results

## Files

- `all_methods_long.csv` / `all_methods_long.json`: one row per method, mode, batch, and sequence length.
- `plot_ready_decode.csv`: one row per `(batch, seq_len)` with L40S, MLA, FlashMLA, and S-FMLA columns.
- `plot_ready_prefill.csv`: same layout for prefill.

## Inputs

- L40S source: `flashinfer_mla_decode_prefill_sweep_l40s.md`.
- Wormhole source: `outputs/wh_batch_seq_sweep/wh_mla_flash_sfmla_batch_seq_results.json`.

## Coverage

- Total rows: 560; ok=383, error=9, skipped=168.
- L40S rows: 140; ok=140.
- Wormhole rows: 420; ok=243, error=9, skipped=168.

## Method Status

- `l40s_flashmla` decode: ok=70, total=70; prefill: ok=70, total=70.
- `mla` decode: ok=42, error=1, skipped=27, total=70; prefill: ok=27, skipped=43, total=70.
- `flash_mla` decode: ok=59, error=4, skipped=7, total=70; prefill: ok=28, skipped=42, total=70.
- `sfmla` decode: ok=59, error=4, skipped=7, total=70; prefill: ok=28, skipped=42, total=70.

## Error Summary

- Other runtime error. Cases: flash_mla decode B=64 L=256, sfmla decode B=64 L=256, flash_mla decode B=64 L=512, sfmla decode B=64 L=512, flash_mla decode B=64 L=1K, sfmla decode B=64 L=1K, mla decode B=64 L=2K, flash_mla decode B=32 L=128K, sfmla decode B=32 L=128K.

## Notes

- L40S rows are raw FlashInfer MLA measurements from the provided report.
- The long table also includes `latency_l40s_div3_ms = 3x raw L40S latency` for the bandwidth-normalized L40S/3 reference.
- WH prefill uses conservative safety caps after the prior server crash: high-risk prefill points are recorded as `skipped` instead of being executed.
- `sfmla_speedup_summary.csv` includes ratios against S-FMLA wherever both sides have valid latency.
