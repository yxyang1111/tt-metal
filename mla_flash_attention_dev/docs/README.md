# SF-MLA Documentation Index

Updated: 2026-06-14 (reorganized into topic folders)

Use this index to navigate SF-MLA design and evaluation docs. Legacy material is in `archive/`.

## 01 — Paper & Positioning

| Doc | Purpose |
|-----|---------|
| [sf-mla-paper-positioning.md](01-paper/sf-mla-paper-positioning.md) | Paper framing, contributions, method structure |
| [sf-mla-research-statement.md](01-paper/sf-mla-research-statement.md) | Research statement draft |

## 02 — SF-MLA Design & Dataflow

Core technical entry points for the method:

| Doc | Purpose |
|-----|---------|
| [flash-mla-impl-b-dse-formalization.md](02-sfmla-design/flash-mla-impl-b-dse-formalization.md) | Implementation-B formalization + DSE |
| [flash-mla-dataflow-first-principles.md](02-sfmla-design/flash-mla-dataflow-first-principles.md) | First-principles dataflow derivation |
| [flash-mla-dataflow-paper-section.md](02-sfmla-design/flash-mla-dataflow-paper-section.md) | Paper-ready dataflow section |
| [flash-mla-dual-flow-formalization.md](02-sfmla-design/flash-mla-dual-flow-formalization.md) | Dual-flow (prefill/decode) formalization |
| [flash-mla-current-dataflow-analysis.md](02-sfmla-design/flash-mla-current-dataflow-analysis.md) | Current tt-metal FlashMLA dataflow |
| [experimental-flash-mla-dataflow-analysis.md](02-sfmla-design/experimental-flash-mla-dataflow-analysis.md) | Experimental SF-MLA dataflow variant |
| [deepseek-flash-mla-introduction.md](02-sfmla-design/deepseek-flash-mla-introduction.md) | DeepSeek FlashMLA background |

Diagram generator: [assets/gen_wh_sfmla_diagram.py](assets/gen_wh_sfmla_diagram.py) → [assets/images/](assets/images/)

## 03 — Implementation & Status

| Doc | Purpose |
|-----|---------|
| [flash-mla-implementation-walkthrough.md](03-implementation/flash-mla-implementation-walkthrough.md) | Code-path walkthrough |
| [flash-mla-wh-migration-analysis.md](03-implementation/flash-mla-wh-migration-analysis.md) | WH migration notes |
| [flash-mla-wh-decode-quickstart.md](03-implementation/flash-mla-wh-decode-quickstart.md) | Decode quickstart |
| [flash-mla-wh-8core-phase-summary.md](03-implementation/flash-mla-wh-8core-phase-summary.md) | 4c vs 8c phase summary |
| [flash-mla-wh-8core-status.md](03-implementation/flash-mla-wh-8core-status.md) | 8-core mapping status |
| [mla-two-implementations-deep-dive.md](03-implementation/mla-two-implementations-deep-dive.md) | Two implementation paths compared |

## 04 — Characterization & Profiling

| Doc | Purpose |
|-----|---------|
| [mla-performance-analysis.md](04-characterization/mla-performance-analysis.md) | Performance analysis overview |
| [flash-mla-wh-component-utilization-and-bubble-analysis.md](04-characterization/flash-mla-wh-component-utilization-and-bubble-analysis.md) | Component utilization & bubbles |
| [flash-mla-wh-profile-swimlane-diagrams.md](04-characterization/flash-mla-wh-profile-swimlane-diagrams.md) | Profile swimlane diagrams |
| [flash-mla-wh-detailed-swimlane-diagrams.md](04-characterization/flash-mla-wh-detailed-swimlane-diagrams.md) | Detailed swimlane diagrams |
| [../experiments/docs/flash_mla_wh_profile_and_bh_simulation.md](../experiments/docs/flash_mla_wh_profile_and_bh_simulation.md) | WH profiling + BH simulation protocol |

Swimlane assets: [assets/swimlanes/](assets/swimlanes/)

## 05 — Background

| Doc | Purpose |
|-----|---------|
| [attention-methods-landscape.md](05-background/attention-methods-landscape.md) | Attention method survey |
| [deepseek-mla-theory-by-computation-order.md](05-background/deepseek-mla-theory-by-computation-order.md) | DeepSeek MLA theory |
| [deepseek-v4-attention-and-tt-dataflow-notes.md](05-background/deepseek-v4-attention-and-tt-dataflow-notes.md) | DeepSeek V4 → TT dataflow mapping |
| [tenstorrent-wormhole-blackhole-architecture.md](05-background/tenstorrent-wormhole-blackhole-architecture.md) | WH/BH architecture notes |

## Engineering

- [dev_log.md](dev_log.md) — commands, environment, daily notes
- [archive-index.md](archive-index.md) — legacy doc catalog

## Archived

Historical prefill-first / ICCAD narratives and old roadmaps: `archive/`
