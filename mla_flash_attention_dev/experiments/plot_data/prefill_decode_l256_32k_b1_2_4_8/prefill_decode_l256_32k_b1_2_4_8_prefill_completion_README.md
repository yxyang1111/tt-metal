# Prefill Completion Estimates

These files complete the requested prefill grid for plotting. They do not replace the original measured/status tables.

## Method

- Prefill attention is asymptotically quadratic in sequence length for fixed batch and method, so missing values are estimated with `latency_ms = alpha * seq_len^2`.
- `alpha` is fitted independently for each `(method, batch)` from measured prefill points with `L >= 1K`.
- Measured values are copied unchanged. Estimated values are tagged in `value_kind` as `estimated`; estimates for original MLA `OOM` rows are tagged as `estimated_oom`.
- FlashMLA and S-FMLA are not relabeled OOM; their missing prefill values are estimated from their own measured scaling.

## Files

- `prefill_decode_l256_32k_b1_2_4_8_prefill_completed_long.csv`: completed prefill long table; use `latency_ms` for plotting and `value_kind` for style/annotation.
- `prefill_decode_l256_32k_b1_2_4_8_prefill_completed_wide.csv`: completed prefill wide table.
- `prefill_decode_l256_32k_b1_2_4_8_prefill_estimated_only.csv`: only imputed prefill rows.
- `prefill_decode_l256_32k_b1_2_4_8_prefill_estimation_fits.csv`: fitted alpha and in-sample MAPE per `(method,batch)`.
- `prefill_decode_l256_32k_b1_2_4_8_with_prefill_estimates_long.csv`: original decode rows plus completed prefill rows in long form.

## Counts

- Completed prefill rows: 96.
- Measured rows: 57.
- Estimated non-OOM rows: 27.
- Estimated MLA OOM rows: 12.

## Per-Method Counts

- MLA: measured=19, estimated=1, estimated_oom=12
- FlashMLA: measured=19, estimated=13, estimated_oom=0
- S-FMLA: measured=19, estimated=13, estimated_oom=0

## Caveat

These are model-based estimates for visualization and trend analysis. They should be labeled as estimated in paper figures or excluded from claims that require measured hardware latency.
