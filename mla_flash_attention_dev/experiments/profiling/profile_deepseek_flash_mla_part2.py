#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from loguru import logger

import ttnn
import mla_flash_attention_dev.experiments.part1_baselines.run_part1_benchmarks as part1
from mla_flash_attention_dev.experiments.part1_baselines.experiment_config import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_DEEPSEEK_NUM_Q_HEADS_PER_CORE,
    DEFAULT_K_CHUNK_SIZE,
    DEFAULT_MAX_CORES_PER_HEAD_BATCH,
    ExperimentConfig,
    format_seq_len,
    make_strict_four_way_config,
)


SUPPORTED_BASELINES = ("flash_mla", "deepseek_flash_mla")


@dataclass(frozen=True)
class ProbePreset:
    label: str
    group: str
    batch: int
    num_heads: int
    seq_len: int
    num_kv_heads: int = 1
    value_dim: int = 512
    rope_dim: int = 64
    block_size: int = DEFAULT_BLOCK_SIZE
    k_chunk_size: int = DEFAULT_K_CHUNK_SIZE
    max_cores_per_head_batch: int = DEFAULT_MAX_CORES_PER_HEAD_BATCH
    deepseek_num_q_heads_per_core: int = DEFAULT_DEEPSEEK_NUM_Q_HEADS_PER_CORE
    warmup_iterations: int = 2
    iterations: int = 4

    def make_config(self) -> ExperimentConfig:
        return make_strict_four_way_config(
            batch=self.batch,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            value_dim=self.value_dim,
            rope_dim=self.rope_dim,
            block_size=self.block_size,
            k_chunk_size=self.k_chunk_size,
            max_cores_per_head_batch=self.max_cores_per_head_batch,
            deepseek_num_q_heads_per_core=self.deepseek_num_q_heads_per_core,
        )

    def make_workload(self) -> part1.Workload:
        config = self.make_config()
        case_group = f"decode_{format_seq_len(self.seq_len)}"
        workload_name = f"{case_group}_{self.label.lower()}_{config.config_signature}"
        return part1.Workload(
            name=workload_name,
            case_group=case_group,
            mode="decode",
            seq_len=self.seq_len,
            config=config,
        )


PRESET_CASES: dict[str, ProbePreset] = {
    "A1": ProbePreset("A1", "A", batch=1, num_heads=32, seq_len=1024),
    "A2": ProbePreset("A2", "A", batch=1, num_heads=32, seq_len=4096),
    "A2_8K": ProbePreset("A2_8K", "A", batch=1, num_heads=32, seq_len=8192),
    "A2_16K": ProbePreset("A2_16K", "A", batch=1, num_heads=32, seq_len=16384),
    "A3": ProbePreset("A3", "A", batch=1, num_heads=32, seq_len=32768),
    "A4": ProbePreset("A4", "A", batch=1, num_heads=32, seq_len=131072, iterations=3, warmup_iterations=1),
    "B1": ProbePreset("B1", "B", batch=6, num_heads=32, seq_len=4096),
    "B2": ProbePreset("B2", "B", batch=6, num_heads=32, seq_len=8192),
    "B3": ProbePreset("B3", "B", batch=6, num_heads=32, seq_len=32768),
    "C1": ProbePreset("C1", "C", batch=8, num_heads=24, seq_len=4096),
    "C2": ProbePreset("C2", "C", batch=8, num_heads=24, seq_len=8192),
    "C3": ProbePreset("C3", "C", batch=8, num_heads=24, seq_len=32768),
    "D28": ProbePreset("D28", "D", batch=7, num_heads=32, seq_len=8192),
    "D1": ProbePreset("D1", "D", batch=8, num_heads=32, seq_len=8192),
    "D40": ProbePreset("D40", "D", batch=10, num_heads=32, seq_len=8192),
    "D2": ProbePreset("D2", "D", batch=12, num_heads=32, seq_len=8192),
}

PRESET_GROUPS: dict[str, list[str]] = {
    "A": ["A1", "A2", "A2_8K", "A2_16K", "A3", "A4"],
    "B": ["B1", "B2", "B3"],
    "C": ["C1", "C2", "C3"],
    "D": ["D28", "D1", "D40", "D2"],
}


def baseline_from_key(key: str) -> part1.BaselineConfig:
    for baseline in getattr(part1, "BASELINES", []):
        if baseline.key == key:
            return baseline
    raise KeyError(f"Unknown baseline: {key}")


def list_preset_cases() -> None:
    for label, preset in PRESET_CASES.items():
        config = preset.make_config()
        print(
            f"{label:>2}  group={preset.group}  seq={format_seq_len(preset.seq_len):<5}  "
            f"B={preset.batch:<2} H={preset.num_heads:<2} H_kv={preset.num_kv_heads:<2}  "
            f"dqhpc={preset.deepseek_num_q_heads_per_core:<2}  q_shards={config.deepseek_num_q_shards}  "
            f"iters={preset.iterations} warmup={preset.warmup_iterations}"
        )


def build_run_callable(
    device: Any,
    workload: part1.Workload,
    baseline_key: str,
    *,
    deepseek_wh_cores_per_block: int,
) -> tuple[callable, tuple[Any, ...]]:
    inputs = part1.make_torch_inputs(workload)

    if baseline_key == "flash_mla":
        tt_inputs = part1.build_decode_tt_inputs(device, inputs, workload, baseline_key)
        scale = inputs["scale_mla"]

        def run_one() -> Any:
            return part1.run_flash_mla_decode(device, tt_inputs, scale)

        dealloc = (
            tt_inputs["tt_q"],
            tt_inputs["tt_k"],
            tt_inputs["tt_v"],
            tt_inputs["tt_page_table"],
            tt_inputs["tt_cur_pos"],
        )
        return run_one, dealloc

    if baseline_key == "deepseek_flash_mla":
        tt_inputs = part1.build_deepseek_decode_tt_inputs(
            device,
            inputs,
            workload,
            wh_cores_per_block=deepseek_wh_cores_per_block,
        )
        scale = inputs["scale_deepseek"]

        def run_one() -> Any:
            return part1.run_deepseek_flash_mla_decode(device, tt_inputs, scale)

        dealloc = (
            tt_inputs["backend_q"],
            tt_inputs["backend_k"],
        )
        return run_one, dealloc

    raise ValueError(f"Unsupported baseline for Part II direct profile: {baseline_key}")


def execute_child_run(
    preset_case: str,
    baseline_key: str,
    *,
    deepseek_wh_cores_per_block: int,
    warmup_iterations: int | None,
    iterations: int | None,
    device_id: int,
) -> None:
    from tracy import signpost

    preset = PRESET_CASES[preset_case]
    workload = preset.make_workload()
    baseline = baseline_from_key(baseline_key)
    support_reasons = part1.validate_workload_baseline_support(
        workload,
        baseline,
        None,
        deepseek_wh_cores_per_block=deepseek_wh_cores_per_block,
    )
    if support_reasons:
        raise ValueError("; ".join(support_reasons))

    effective_warmup = preset.warmup_iterations if warmup_iterations is None else warmup_iterations
    effective_iters = preset.iterations if iterations is None else iterations

    device = ttnn.open_device(device_id=device_id)
    tt_out = None
    dealloc_tensors: tuple[Any, ...] = ()
    logger.info(
        f"Opened device for child run: preset_case={preset_case} "
        f"baseline={baseline_key} wh_cores_per_block={deepseek_wh_cores_per_block} "
        f"device_id={device_id}"
    )

    try:
        support_reasons = part1.validate_workload_baseline_support(
            workload,
            baseline,
            device,
            deepseek_wh_cores_per_block=deepseek_wh_cores_per_block,
        )
        if support_reasons:
            raise ValueError("; ".join(support_reasons))

        run_one, dealloc_tensors = build_run_callable(
            device,
            workload,
            baseline_key,
            deepseek_wh_cores_per_block=deepseek_wh_cores_per_block,
        )

        for _ in range(effective_warmup):
            tt_out = run_one()
            ttnn.synchronize_device(device)
            part1.safe_deallocate(tt_out)
            tt_out = None

        signpost("start")
        for _ in range(effective_iters):
            tt_out = run_one()
            ttnn.synchronize_device(device)
            part1.safe_deallocate(tt_out)
            tt_out = None
        signpost("stop")
    finally:
        part1.safe_deallocate(tt_out)
        part1.safe_deallocate(*dealloc_tensors)
        ttnn.close_device(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Child-run executor for Part II DeepSeek/TT representative decode profiling."
    )
    parser.add_argument("--child-run", action="store_true", help="Execute the selected preset case for tracy.")
    parser.add_argument(
        "--preset-case",
        choices=sorted(PRESET_CASES.keys()),
        help="Preset representative case label from the Part II experiment plan.",
    )
    parser.add_argument(
        "--baseline",
        choices=SUPPORTED_BASELINES,
        help="Baseline to execute in child-run mode.",
    )
    parser.add_argument(
        "--deepseek-wh-cores-per-block",
        type=int,
        choices=(4, 8),
        default=4,
        help="Wormhole S-block width for DeepSeek FlashMLA. Ignored for TT mainline baseline.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=None,
        help="Override preset warmup iterations.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override preset measured iterations.",
    )
    parser.add_argument("--device-id", type=int, default=0, help="TT device id.")
    parser.add_argument("--list-preset-cases", action="store_true", help="List preset representative cases and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_preset_cases:
        list_preset_cases()
        return

    if args.child_run:
        if not args.preset_case:
            raise ValueError("--preset-case is required with --child-run")
        if not args.baseline:
            raise ValueError("--baseline is required with --child-run")
        execute_child_run(
            args.preset_case,
            args.baseline,
            deepseek_wh_cores_per_block=args.deepseek_wh_cores_per_block,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            device_id=args.device_id,
        )
        return

    raise SystemExit("Use --list-preset-cases or --child-run.")


if __name__ == "__main__":
    main()
