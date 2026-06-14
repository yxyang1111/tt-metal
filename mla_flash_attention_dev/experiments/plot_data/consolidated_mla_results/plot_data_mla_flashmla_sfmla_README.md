# MLA / FlashMLA / S-FMLA Plot Data

Generated from `all_methods_long.csv` and `sfmla_speedup_summary.csv`.

## Files

- `plot_data_mla_flashmla_sfmla_long.csv`: full long-form table, one row per method/mode/batch/sequence point; includes `status`, `latency_ms`, `source`, and `note`.
- `plot_data_mla_flashmla_sfmla_ok_long.csv`: same long-form schema but only rows where `status == ok`; use this directly for latency line/scatter plots.
- `plot_data_mla_flashmla_sfmla_wide.csv`: one row per `(mode, batch, seq_len)` with per-method latency/status columns plus speedup ratios.
- `plot_data_mla_flashmla_sfmla_speedups.csv`: S-FMLA latency and `FlashMLA/S-FMLA`, `MLA/S-FMLA` ratios for heatmaps or bar charts.

## Coverage

- prefill: ok=83, error=0, skipped=127, total=210
- decode: ok=160, error=9, skipped=41, total=210

## Per-Method Status

- prefill MLA: ok=27, error=0, skipped=43, total=70
- prefill FlashMLA: ok=28, error=0, skipped=42, total=70
- prefill S-FMLA: ok=28, error=0, skipped=42, total=70
- decode MLA: ok=42, error=1, skipped=27, total=70
- decode FlashMLA: ok=59, error=4, skipped=7, total=70
- decode S-FMLA: ok=59, error=4, skipped=7, total=70

## Plotting Notes

- Prefer `plot_data_mla_flashmla_sfmla_ok_long.csv` for regular latency curves: x=`seq_len`, y=`latency_ms`, hue=`method_label`, facet by `mode` and/or `batch`.
- Use log scale for `seq_len`; latency also benefits from log scale when plotting prefill and decode together.
- Use the wide table when you need missing-point annotations from `*_status` and `*_note`.
