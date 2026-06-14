# Prefill/Decode Plot Data: L=256..32K, B=1/2/4/8

This folder contains the requested subset for MLA, FlashMLA, and S-FMLA.

Required grid:

- modes: `prefill`, `decode`
- sequence lengths: `256`, `512`, `1K`, `2K`, `4K`, `8K`, `16K`, `32K`
- batches: `1`, `2`, `4`, `8`
- methods: `MLA`, `FlashMLA`, `S-FMLA`

## Files

- `prefill_decode_l256_32k_b1_2_4_8_full_long.csv`: complete required grid, one row per `(mode,batch,seq_len,method)`; includes skipped/OOM rows.
- `prefill_decode_l256_32k_b1_2_4_8_ok_long.csv`: only rows with valid latency; use this for plotting measured latency curves.
- `prefill_decode_l256_32k_b1_2_4_8_wide.csv`: one row per `(mode,batch,seq_len)` with per-method status/latency columns.
- `prefill_decode_l256_32k_b1_2_4_8_speedups.csv`: S-FMLA latency plus `FlashMLA/S-FMLA` and `MLA/S-FMLA` ratios where available.
- `prefill_decode_l256_32k_b1_2_4_8_missing_latency.csv`: rows in the requested grid that do not have a plottable latency value.
- `prefill_decode_l256_32k_b1_2_4_8_mla_oom.csv`: all MLA rows relabeled as OOM by the memory estimate.
- `prefill_decode_l256_32k_b1_2_4_8_prefill_likely_oom.csv`: prefill-only subset of the MLA OOM report.

## Completeness Check

- Expected full-long rows: 192; actual: 192.
- Expected wide rows: 64; actual: 64.
- Missing required records: 0.
- Rows with valid latency: 150.
- Rows without valid latency: 42.

## Status By Mode

- prefill: ok=57, skipped=27, OOM=12, error=0, missing_record=0, total=96
- decode: ok=93, skipped=0, OOM=3, error=0, missing_record=0, total=96

## Valid Latency By Method

- prefill MLA: 19/32 rows have valid latency
- prefill FlashMLA: 19/32 rows have valid latency
- prefill S-FMLA: 19/32 rows have valid latency
- decode MLA: 29/32 rows have valid latency
- decode FlashMLA: 32/32 rows have valid latency
- decode S-FMLA: 32/32 rows have valid latency

## OOM Labeling

- OOM labels are applied only to `MLA` rows whose naive MLA memory estimate exceeds the 8 GiB MLA budget.
- FlashMLA and S-FMLA rows are not labeled OOM by this post-processing pass; their no-latency prefill points remain `skipped` safety-cap rows.
- MLA prefill OOM rows: 12.
- MLA decode OOM rows: 3.

## Notes

- The required grid is complete: every requested `(mode,batch,seq_len,method)` row exists in `full_long.csv`.
- Some rows are not plottable because they are marked as `skipped`, `OOM`, or `error`; see `missing_latency.csv` for exact points and reasons.

## Prefill Completion Estimates

- Completed prefill plotting table: `prefill_decode_l256_32k_b1_2_4_8_prefill_completed_long.csv`.
- Completed prefill wide table: `prefill_decode_l256_32k_b1_2_4_8_prefill_completed_wide.csv`.
- Combined decode + completed prefill table: `prefill_decode_l256_32k_b1_2_4_8_with_prefill_estimates_long.csv`.
- Estimate details: `prefill_decode_l256_32k_b1_2_4_8_prefill_completion_README.md` and `prefill_decode_l256_32k_b1_2_4_8_prefill_estimation_fits.csv`.
- Prefill completed rows: 96; measured=57, estimated=27, estimated_oom=12.
- Estimates use per-method/per-batch quadratic prefill scaling from measured `L>=1K` points and should be labeled as estimated in figures.
