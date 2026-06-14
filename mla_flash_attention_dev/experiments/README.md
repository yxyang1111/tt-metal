# SF-MLA Experiments

Evaluation pipelines for the S-FMLA paper and WH FlashMLA characterization.

## Paper Pipeline (Parts I–IV)

| Part | Directory | Build command | Primary output |
|------|-----------|---------------|----------------|
| I — Baselines | [part1_baselines/](part1_baselines/) | `python3 mla_flash_attention_dev/experiments/part1_baselines/run_part1_benchmarks.py` | `part1_baselines/outputs/report.md` |
| II — Utilization | [part2_utilization/](part2_utilization/) | `python3 mla_flash_attention_dev/experiments/part2_utilization/build_part2_results.py` | `part2_utilization/outputs/report.md` |
| III — Autotuner | [part3_autotuner/](part3_autotuner/) | `python3 mla_flash_attention_dev/experiments/part3_autotuner/build_part3_results.py` | `part3_autotuner/outputs/report.md` |
| IV — Architecture | [part4_architecture/](part4_architecture/) | `python3 mla_flash_attention_dev/experiments/part4_architecture/build_part4_results.py` | `part4_architecture/outputs/report.md` |

Part I supplemental sweeps (4c/8c, batch/head/seq) are archived under `part1_baselines/runs/`.
The canonical Part I result set is in `part1_baselines/outputs/`.

## Supporting Tracks

### Characterization
[characterization/](characterization/) — DRAM traffic verification, roofline, batch×seq sweeps for baseline vs SF-MLA.

### Baselines
[baselines/](baselines/) — External comparison points:
- `gpu_metal/` — simple GPU MLA benchmark
- `gpu_flashinfer/` — FlashInfer MLA on L40S / RTX 3080 Ti
- `naive_ttnn/` — unfused TTNN MLA (used by sweeps for memory estimation)

### Profiling
[profiling/](profiling/) — Tracy/PM profiling scripts and raw outputs:
- `profile_flash_mla_wh_detailed.py` — full-sweep WH detailed profiling
- `run_deepseek_flash_mla_part2_probe.py` — Part II direct device probes
- `outputs/` — all profile run artifacts (50+ probe directories)

### Sweeps
[sweeps/](sweeps/) — Consolidated WH evaluation:
- `run_wh_mla_batch_seq_sweep.py` — mla / flash_mla / sfmla batch×seq sweep
- `outputs/consolidated_mla_results/` — merged cross-method results + viz
- `plot_data/` — CSV tables ready for figures

### Perf Model
[perf_model/](perf_model/) — Cost-model ablation sweeps (mcast, SRAM capacity, architecture what-if).

## Experiment Docs

Methodology and protocol notes: [docs/](docs/)
- `sf_mla_paper_experiment_plan.md` — master experiment plan
- `flash_mla_wh_profile_and_bh_simulation.md` — profiling protocol
- `l40s_same_latency_protocol.md` — GPU comparison protocol

## Dependency Graph

```
profiling/outputs/flash_mla_wh_detailed/
        ↓
part2_utilization/  →  part3_autotuner/  →  part4_architecture/
        ↑
part1_baselines/outputs/
        ↓
sweeps/outputs/consolidated_mla_results/
```

Autotuner code: `../autotuner/`
