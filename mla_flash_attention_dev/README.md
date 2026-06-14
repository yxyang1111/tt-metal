# SF-MLA Development Workspace

Local development, documentation, and experiments for **S-FMLA** (Sequence-parallel Flash Multi-head Latent Attention) on Tenstorrent Wormhole.

Production kernel code lives in `ttnn/` and `models/`; this folder holds design docs, evaluation pipelines, cost-model tooling, and profiling assets.

## Directory Layout

```
mla_flash_attention_dev/
├── autotuner/          Part III: analytical cost model + design-space search
├── docs/               SF-MLA design docs, diagrams, and background notes
├── experiments/        Part I–IV evaluation pipelines + sweeps + profiling
├── archive/            Superseded root-level notes and early prototypes
└── latex/              Local paper draft (gitignored, not uploaded)
```

## Quick Start by Task

| Goal | Start here |
|------|------------|
| Understand the method & paper framing | `docs/01-paper/sf-mla-paper-positioning.md` |
| Read dataflow / tiling design | `docs/02-sfmla-design/` |
| Run Part I baseline benchmarks | `experiments/part1_baselines/run_part1_benchmarks.py` |
| Analyze utilization & bubbles (Part II) | `experiments/part2_utilization/build_part2_results.py` |
| Run cost-model autotuner (Part III) | `experiments/part3_autotuner/build_part3_results.py` |
| Architecture implications (Part IV) | `experiments/part4_architecture/build_part4_results.py` |
| WH batch×seq sweep (mla / flash_mla / sfmla) | `experiments/sweeps/run_wh_mla_batch_seq_sweep.py` |
| Detailed FlashMLA profiling | `experiments/profiling/profile_flash_mla_wh_detailed.py` |

## Paper Experiment Pipeline (Parts I–IV)

```
Part I  part1_baselines/     Four/five-method fair comparison (Reference, FA, FlashMLA, DeepSeek)
Part II part2_utilization/   Decode phase transition, bubble attribution, direct probes
Part III part3_autotuner/    Calibrated cost model fidelity + autotuner search
Part IV part4_architecture/  WH vs BH implications, SRAM/NOC sensitivity
```

Supporting tracks:

- `experiments/characterization/` — DRAM traffic, roofline, batch-seq characterization
- `experiments/baselines/` — GPU FlashInfer benchmarks + naive TTNN MLA
- `experiments/sweeps/` — Consolidated WH sweep results and plot-ready CSVs
- `experiments/profiling/` — Tracy/PM profiling scripts and raw outputs
- `experiments/perf_model/` — Cost-model ablation sweeps and architecture what-if

## Documentation Index

See `docs/README.md` for the curated reading order.

## Engineering Log

Day-to-day commands and environment notes: `docs/dev_log.md`
