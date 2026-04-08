#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import statistics
import shutil
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import (
    FlashMLADecode,
    FlashMLAOptimalGridNOC0_WH,
    FlashMLAProgramConfig,
)
from models.common.utility_functions import nearest_y
from models.tt_transformers.tt.common import PagedAttentionConfig
from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import (
    nearest_n,
    nearest_pow_2,
    page_table_setup,
    scaled_dot_product_attention_reference,
    scaled_dot_product_attention_reference_prefill,
    to_paged_cache,
)

import mla_flash_attention_dev.experiments.profile_flash_mla_wh_detailed as flash_mla_detailed_profile
from mla_flash_attention_dev.experiments.part1_three_baselines.experiment_config import (
    DEFAULT_BATCHES,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_DECODE_SEQ_LENS,
    DEFAULT_DEEPSEEK_NUM_Q_HEADS_PER_CORE,
    DEFAULT_DEVICE_ID,
    DEFAULT_K_CHUNK_SIZE,
    DEFAULT_MAX_CORES_PER_HEAD_BATCH,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_KV_HEADS,
    DEFAULT_PREFILL_SEQ_LENS,
    DEFAULT_PROBE_BATCHES,
    DEFAULT_PROBE_DECODE_SEQ_LENS,
    DEFAULT_PROBE_NUM_HEADS,
    DEFAULT_PROBE_NUM_KV_HEADS,
    DEFAULT_PROBE_ROPE_DIMS,
    DEFAULT_PROBE_VALUE_DIMS,
    DEFAULT_ROPE_DIMS,
    DEFAULT_TORCH_INPUT_DTYPE_STR,
    DEFAULT_VALUE_DIMS,
    ExperimentConfig,
    format_seq_len,
    get_sweep_preset,
    make_strict_four_way_config,
    sweep_preset_names,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
RAW_DIR = OUTPUT_DIR / "raw"
FLASHMLA_DETAIL_DIR = OUTPUT_DIR / "flashmla_detailed"
JSON_PATH = RAW_DIR / "part1_four_method_results.json"
CSV_PATH = RAW_DIR / "part1_four_method_results.csv"
CAPABILITY_PROBE_JSON_PATH = RAW_DIR / "capability_probe_results.json"
CAPABILITY_PROBE_CHECKPOINT_PATH = RAW_DIR / "capability_probe_checkpoint.json"
EXISTING_FLASHMLA_DETAIL_SOURCE = SCRIPT_DIR.parent / "profile_outputs" / "flash_mla_wh_detailed"

DEVICE_ID = DEFAULT_DEVICE_ID
TORCH_INPUT_DTYPE = torch.bfloat16
DETAIL_SEQ_LENS = [1024, 4096, 8192, 16384, 32768]


def legacy_case_group(mode: str, seq_len: int) -> str:
    return f"{mode}_{format_seq_len(seq_len)}"


@dataclass(frozen=True)
class Workload:
    name: str
    case_group: str
    mode: str
    seq_len: int
    config: ExperimentConfig

    @property
    def q_seq_len(self) -> int:
        return self.seq_len if self.mode == "prefill" else 1

    @property
    def normalized_items(self) -> int:
        return self.batch * self.seq_len if self.mode == "prefill" else self.batch

    @property
    def batch(self) -> int:
        return self.config.batch

    @property
    def config_signature(self) -> str:
        return self.config.config_signature


@dataclass(frozen=True)
class BaselineConfig:
    key: str
    label: str
    backend: str


BASELINES = [
    BaselineConfig("reference_attention", "Reference Attention", "torch_reference"),
    BaselineConfig("flash_attention", "Flash Attention", "tt_device"),
    BaselineConfig("flash_mla", "FlashMLA (TT Mainline)", "tt_device"),
    BaselineConfig("deepseek_flash_mla", "DeepSeek FlashMLA", "tt_device"),
]


def active_baselines_for_workload(workload: Workload) -> list[BaselineConfig]:
    if workload.mode == "prefill":
        return [baseline for baseline in BASELINES if baseline.key != "deepseek_flash_mla"]
    return BASELINES


def parse_int_list(values: list[int]) -> list[int]:
    return sorted(dict.fromkeys(values))


def apply_sweep_preset_to_args(args: argparse.Namespace, preset_name: str, *, probe: bool) -> None:
    if preset_name == "manual":
        return
    preset = get_sweep_preset(preset_name)
    if probe:
        args.probe_batches = list(preset.batches)
        args.probe_num_heads = list(preset.num_heads)
        args.probe_num_kv_heads = list(preset.num_kv_heads)
        args.probe_value_dims = list(preset.value_dims)
        args.probe_rope_dims = list(preset.rope_dims)
        args.probe_decode_seq_lens = list(preset.decode_seq_lens)
        return

    args.batches = list(preset.batches)
    args.num_heads_list = list(preset.num_heads)
    args.num_kv_heads_list = list(preset.num_kv_heads)
    args.value_dims = list(preset.value_dims)
    args.rope_dims = list(preset.rope_dims)
    args.decode_seq_lens = list(preset.decode_seq_lens)
    args.prefill_seq_lens = list(preset.prefill_seq_lens)


def print_sweep_presets() -> None:
    print("Available sweep presets:")
    for name in sweep_preset_names():
        preset = get_sweep_preset(name)
        print(f"- {name}: {preset.description}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Part I four-method decode benchmarks, optional strict four-way capability probe, "
            "and three-method prefill control with multi-dimensional B/H/dim sweep support."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Explicit workload case names to run after workload expansion. Defaults to all generated cases.",
    )
    parser.add_argument(
        "--detail-cases",
        nargs="+",
        default=[legacy_case_group("decode", seq_len) for seq_len in DETAIL_SEQ_LENS],
        help="FlashMLA detailed profile decode cases to run when --run-flashmla-detailed is enabled.",
    )
    parser.add_argument(
        "--list-sweep-presets",
        action="store_true",
        help="Print available main/probe sweep presets and exit.",
    )
    parser.add_argument(
        "--sweep-preset",
        default="manual",
        choices=("manual", *sweep_preset_names()),
        help="Apply a named main-sweep preset. Overrides the main sweep axis flags.",
    )
    parser.add_argument(
        "--probe-preset",
        default="manual",
        choices=("manual", *sweep_preset_names()),
        help="Apply a named capability-probe preset. Overrides the probe axis flags.",
    )
    parser.add_argument("--comparison-mode", default="strict_four_way", choices=["strict_four_way"], help="Comparison mode.")
    parser.add_argument("--batches", nargs="+", type=int, default=list(DEFAULT_BATCHES), help="Main sweep batch values.")
    parser.add_argument("--num-heads-list", nargs="+", type=int, default=list(DEFAULT_NUM_HEADS), help="Main sweep num_heads values.")
    parser.add_argument(
        "--num-kv-heads-list",
        nargs="+",
        type=int,
        default=list(DEFAULT_NUM_KV_HEADS),
        help="Main sweep num_kv_heads values.",
    )
    parser.add_argument("--value-dims", nargs="+", type=int, default=list(DEFAULT_VALUE_DIMS), help="Main sweep common value dimensions.")
    parser.add_argument("--rope-dims", nargs="+", type=int, default=list(DEFAULT_ROPE_DIMS), help="Main sweep rope dimensions.")
    parser.add_argument(
        "--decode-seq-lens",
        nargs="+",
        type=int,
        default=list(DEFAULT_DECODE_SEQ_LENS),
        help="Decode sequence lengths to benchmark.",
    )
    parser.add_argument(
        "--prefill-seq-lens",
        nargs="+",
        type=int,
        default=list(DEFAULT_PREFILL_SEQ_LENS),
        help="Prefill sequence lengths to benchmark.",
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE, help="Paged attention block size.")
    parser.add_argument("--k-chunk-size", type=int, default=DEFAULT_K_CHUNK_SIZE, help="Decode K chunk size.")
    parser.add_argument(
        "--max-cores-per-head-batch",
        type=int,
        default=DEFAULT_MAX_CORES_PER_HEAD_BATCH,
        help="TT mainline decode max_cores_per_head_batch.",
    )
    parser.add_argument(
        "--deepseek-num-q-heads-per-core",
        type=int,
        default=DEFAULT_DEEPSEEK_NUM_Q_HEADS_PER_CORE,
        help="DeepSeek FlashMLA Q heads per core.",
    )
    parser.add_argument(
        "--run-capability-probe",
        action="store_true",
        help="Run a strict four-way decode capability probe before the main benchmark.",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Run capability probe only; skip the main benchmark sweep.",
    )
    parser.add_argument(
        "--probe-resume",
        action="store_true",
        help="Resume capability probe from the latest checkpoint/final JSON if available.",
    )
    parser.add_argument(
        "--probe-reset-checkpoint",
        action="store_true",
        help="Delete existing capability probe checkpoint/final JSON before starting fresh.",
    )
    parser.add_argument(
        "--probe-workload-batch-size",
        type=int,
        default=0,
        help="Process at most this many probe workloads per invocation; 0 means all remaining workloads.",
    )
    parser.add_argument(
        "--probe-checkpoint-every-workloads",
        type=int,
        default=5,
        help="Write a capability probe checkpoint after every N completed workloads.",
    )
    parser.add_argument(
        "--probe-batches",
        nargs="+",
        type=int,
        default=list(DEFAULT_PROBE_BATCHES),
        help="Capability probe batch values.",
    )
    parser.add_argument(
        "--probe-num-heads",
        nargs="+",
        type=int,
        default=list(DEFAULT_PROBE_NUM_HEADS),
        help="Capability probe num_heads values.",
    )
    parser.add_argument(
        "--probe-num-kv-heads",
        nargs="+",
        type=int,
        default=list(DEFAULT_PROBE_NUM_KV_HEADS),
        help="Capability probe num_kv_heads values.",
    )
    parser.add_argument(
        "--probe-value-dims",
        nargs="+",
        type=int,
        default=list(DEFAULT_PROBE_VALUE_DIMS),
        help="Capability probe common value dimensions.",
    )
    parser.add_argument(
        "--probe-rope-dims",
        nargs="+",
        type=int,
        default=list(DEFAULT_PROBE_ROPE_DIMS),
        help="Capability probe rope dimensions.",
    )
    parser.add_argument(
        "--probe-decode-seq-lens",
        nargs="+",
        type=int,
        default=list(DEFAULT_PROBE_DECODE_SEQ_LENS),
        help="Capability probe decode sequence lengths.",
    )
    parser.add_argument("--warmup-device", type=int, default=2, help="Warmup iterations for TT device baselines.")
    parser.add_argument("--iters-device", type=int, default=10, help="Measured iterations for TT device baselines.")
    parser.add_argument("--warmup-reference", type=int, default=1, help="Warmup iterations for the torch reference baseline.")
    parser.add_argument("--iters-reference", type=int, default=10, help="Measured iterations for the torch reference baseline.")
    parser.add_argument("--run-flashmla-detailed", action="store_true", help="Also run FlashMLA detailed tracy profiling into this folder.")
    parser.add_argument(
        "--reuse-existing-flashmla-detailed",
        action="store_true",
        help="Reuse existing stable FlashMLA detailed C++ device reports instead of rerunning tracy.",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Only run benchmarks/profile; skip the report + figure rendering step.",
    )
    args = parser.parse_args()
    if args.list_sweep_presets:
        print_sweep_presets()
        raise SystemExit(0)
    if args.probe_only and not args.run_capability_probe:
        parser.error("--probe-only requires --run-capability-probe")
    if args.probe_workload_batch_size < 0:
        parser.error("--probe-workload-batch-size must be >= 0")
    if args.probe_checkpoint_every_workloads <= 0:
        parser.error("--probe-checkpoint-every-workloads must be > 0")
    apply_sweep_preset_to_args(args, args.sweep_preset, probe=False)
    apply_sweep_preset_to_args(args, args.probe_preset, probe=True)
    args.probe_workload_batch_size = args.probe_workload_batch_size or None
    args.batches = parse_int_list(args.batches)
    args.num_heads_list = parse_int_list(args.num_heads_list)
    args.num_kv_heads_list = parse_int_list(args.num_kv_heads_list)
    args.value_dims = parse_int_list(args.value_dims)
    args.rope_dims = parse_int_list(args.rope_dims)
    args.decode_seq_lens = parse_int_list(args.decode_seq_lens)
    args.prefill_seq_lens = parse_int_list(args.prefill_seq_lens)
    args.probe_batches = parse_int_list(args.probe_batches)
    args.probe_num_heads = parse_int_list(args.probe_num_heads)
    args.probe_num_kv_heads = parse_int_list(args.probe_num_kv_heads)
    args.probe_value_dims = parse_int_list(args.probe_value_dims)
    args.probe_rope_dims = parse_int_list(args.probe_rope_dims)
    args.probe_decode_seq_lens = parse_int_list(args.probe_decode_seq_lens)
    return args


def build_experiment_configs(args: argparse.Namespace, *, probe: bool = False) -> list[ExperimentConfig]:
    batches = args.probe_batches if probe else args.batches
    num_heads_values = args.probe_num_heads if probe else args.num_heads_list
    num_kv_heads_values = args.probe_num_kv_heads if probe else args.num_kv_heads_list
    value_dims = args.probe_value_dims if probe else args.value_dims
    rope_dims = args.probe_rope_dims if probe else args.rope_dims

    configs = [
        make_strict_four_way_config(
            batch=batch,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            value_dim=value_dim,
            rope_dim=rope_dim,
            block_size=args.block_size,
            k_chunk_size=args.k_chunk_size,
            max_cores_per_head_batch=args.max_cores_per_head_batch,
            deepseek_num_q_heads_per_core=args.deepseek_num_q_heads_per_core,
            torch_input_dtype=DEFAULT_TORCH_INPUT_DTYPE_STR,
        )
        for batch, num_heads, num_kv_heads, value_dim, rope_dim in product(
            batches, num_heads_values, num_kv_heads_values, value_dims, rope_dims
        )
    ]
    configs.sort(key=lambda config: (config.batch, config.num_heads, config.num_kv_heads, config.common_value_dim, config.mla_d_rope))
    return configs


def default_config_from_list(configs: list[ExperimentConfig]) -> ExperimentConfig:
    for config in configs:
        if (
            config.batch == 1
            and config.num_heads == 32
            and config.num_kv_heads == 1
            and config.common_value_dim == 512
            and config.mla_d_rope == 64
            and config.block_size == DEFAULT_BLOCK_SIZE
            and config.k_chunk_size == DEFAULT_K_CHUNK_SIZE
        ):
            return config
    return configs[0]


def build_workload_catalog(
    configs: list[ExperimentConfig],
    *,
    decode_seq_lens: list[int],
    prefill_seq_lens: list[int],
    default_config: ExperimentConfig,
) -> dict[str, Workload]:
    catalog: dict[str, Workload] = {}
    for config in configs:
        is_default_named = config == default_config
        for seq_len in decode_seq_lens:
            case_group = legacy_case_group("decode", seq_len)
            case_name = case_group if is_default_named else f"{case_group}_{config.config_signature}"
            catalog[case_name] = Workload(case_name, case_group, "decode", seq_len, config)
        for seq_len in prefill_seq_lens:
            case_group = legacy_case_group("prefill", seq_len)
            case_name = case_group if is_default_named else f"{case_group}_{config.config_signature}"
            catalog[case_name] = Workload(case_name, case_group, "prefill", seq_len, config)
    return catalog


def select_workloads(catalog: dict[str, Workload], case_names: list[str] | None) -> list[Workload]:
    if not case_names:
        return list(catalog.values())
    missing = [name for name in case_names if name not in catalog]
    if missing:
        raise ValueError(f"Unknown case names: {missing}")
    return [catalog[name] for name in case_names]


def ensure_output_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FLASHMLA_DETAIL_DIR.mkdir(parents=True, exist_ok=True)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp_path.replace(path)


def delete_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def torch_dtype_from_config(config: ExperimentConfig) -> torch.dtype:
    if config.torch_input_dtype != DEFAULT_TORCH_INPUT_DTYPE_STR:
        raise ValueError(f"Unsupported torch input dtype: {config.torch_input_dtype}")
    return TORCH_INPUT_DTYPE


def seed_for_case(workload: Workload) -> int:
    values = [
        workload.seq_len,
        workload.batch,
        workload.config.num_heads,
        workload.config.num_kv_heads,
        workload.config.std_head_dim,
        workload.config.mla_head_dim_v,
        workload.config.mla_d_rope,
        workload.config.deepseek_qk_nope_head_dim,
        workload.config.deepseek_qk_rope_head_dim,
        workload.config.deepseek_kv_lora_rank,
        workload.config.block_size,
        workload.config.k_chunk_size,
    ]
    seed = 20260407 + (0 if workload.mode == "decode" else 100000)
    for idx, value in enumerate(values, start=1):
        seed += idx * 9973 * value
    return seed


def make_torch_inputs(workload: Workload) -> dict[str, Any]:
    config = workload.config
    input_dtype = torch_dtype_from_config(config)
    generator = torch.Generator()
    generator.manual_seed(seed_for_case(workload))

    q_std = torch.randn(
        workload.batch,
        config.num_heads,
        workload.q_seq_len,
        config.std_head_dim,
        generator=generator,
        dtype=input_dtype,
    )
    k_std = torch.randn(
        workload.batch,
        config.num_kv_heads,
        workload.seq_len,
        config.std_head_dim,
        generator=generator,
        dtype=input_dtype,
    )
    v_std = torch.randn(
        workload.batch,
        config.num_kv_heads,
        workload.seq_len,
        config.std_head_dim,
        generator=generator,
        dtype=input_dtype,
    )
    q_mla = torch.randn(
        workload.batch,
        config.num_heads,
        workload.q_seq_len,
        config.mla_head_dim_qk,
        generator=generator,
        dtype=input_dtype,
    )
    latent_mla = torch.randn(
        workload.batch,
        config.num_kv_heads,
        workload.seq_len,
        config.mla_head_dim_v,
        generator=generator,
        dtype=input_dtype,
    )
    rope_mla = torch.randn(
        workload.batch,
        config.num_kv_heads,
        workload.seq_len,
        config.mla_d_rope,
        generator=generator,
        dtype=input_dtype,
    )
    k_mla = torch.cat([latent_mla, rope_mla], dim=-1)
    q_deepseek = torch.randn(
        workload.batch,
        config.num_heads,
        workload.q_seq_len,
        config.deepseek_kvpe_dim,
        generator=generator,
        dtype=input_dtype,
    )
    latent_deepseek = torch.randn(
        workload.batch,
        config.num_kv_heads,
        workload.seq_len,
        config.deepseek_kv_lora_rank,
        generator=generator,
        dtype=input_dtype,
    )
    rope_deepseek = torch.randn(
        workload.batch,
        config.num_kv_heads,
        workload.seq_len,
        config.deepseek_qk_rope_head_dim,
        generator=generator,
        dtype=input_dtype,
    )
    k_deepseek = torch.cat([latent_deepseek, rope_deepseek], dim=-1)
    return {
        "config": config,
        "q_std": q_std,
        "k_std": k_std,
        "v_std": v_std,
        "q_mla": q_mla,
        "latent_mla": latent_mla,
        "rope_mla": rope_mla,
        "k_mla": k_mla,
        "q_deepseek": q_deepseek,
        "latent_deepseek": latent_deepseek,
        "rope_deepseek": rope_deepseek,
        "k_deepseek": k_deepseek,
        "scale_std": config.std_head_dim ** -0.5,
        "scale_mla": config.mla_head_dim_qk ** -0.5,
        "scale_deepseek": config.deepseek_qk_head_dim ** -0.5,
    }


def safe_deallocate(*tensors: Any) -> None:
    for tensor in tensors:
        if tensor is None:
            continue
        try:
            ttnn.deallocate(tensor)
        except Exception:
            pass


def summarize_latencies(latencies_ms: list[float], normalized_items: int) -> dict[str, float]:
    mean_ms = statistics.mean(latencies_ms)
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)
    std_ms = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    throughput_tokens_per_s = 1000.0 * max(normalized_items, 1) / max(mean_ms, 1e-12)
    return {
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
        "us_per_item": (mean_ms * 1000.0) / max(normalized_items, 1),
        "throughput_tokens_per_s": throughput_tokens_per_s,
    }


def reference_decode(inputs: dict[str, Any], workload: Workload) -> torch.Tensor:
    q_std = inputs["q_std"].to(torch.float32)
    k_std = inputs["k_std"].to(torch.float32)
    v_std = inputs["v_std"].to(torch.float32)
    return scaled_dot_product_attention_reference(
        q_std,
        k_std,
        v_std,
        start_indices=[workload.seq_len - 1] * workload.batch,
        padded_layer_len=nearest_y(workload.seq_len, workload.config.k_chunk_size),
        scale=inputs["scale_std"],
        is_causal=True,
    )


def reference_prefill(inputs: dict[str, Any], _: Workload) -> torch.Tensor:
    q_std = inputs["q_std"].to(torch.float32)
    k_std = inputs["k_std"].to(torch.float32)
    v_std = inputs["v_std"].to(torch.float32)
    return scaled_dot_product_attention_reference_prefill(
        q_std,
        k_std,
        v_std,
        scale=inputs["scale_std"],
        is_causal=True,
    )


def benchmark_reference(
    callable_: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            out = callable_()
            del out

        latencies_ms = []
        for _ in range(iters):
            start = time.perf_counter()
            out = callable_()
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)
            del out
    return latencies_ms


def build_prefill_tt_inputs(device: Any, inputs: dict[str, Any], baseline_key: str) -> dict[str, Any]:
    config = inputs["config"]
    padded_num_heads = nearest_pow_2(nearest_n(config.num_heads, 32))
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=padded_num_heads,
        k_chunk_size=config.k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    if baseline_key == "flash_mla":
        q_torch = inputs["q_mla"]
        k_torch = inputs["k_mla"]
        v_torch = None
    else:
        q_torch = inputs["q_std"]
        k_torch = inputs["k_std"]
        v_torch = inputs["v_std"]
    tt_q = ttnn.from_torch(
        q_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_k = ttnn.from_torch(
        k_torch,
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_v = None
    if v_torch is not None:
        tt_v = ttnn.from_torch(
            v_torch,
            device=device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return {
        "program_config": program_config,
        "compute_kernel_config": compute_kernel_config,
        "head_dim_v": config.mla_head_dim_v if baseline_key == "flash_mla" else config.std_head_dim,
        "tt_q": tt_q,
        "tt_k": tt_k,
        "tt_v": tt_v,
    }


def build_decode_tt_inputs(device: Any, inputs: dict[str, Any], workload: Workload, baseline_key: str) -> dict[str, Any]:
    config = workload.config
    if baseline_key == "flash_mla":
        q_torch = inputs["q_mla"]
        k_torch = inputs["k_mla"]
        v_torch = None
        q_width = config.mla_head_dim_qk
    else:
        q_torch = inputs["q_std"]
        k_torch = inputs["k_std"]
        v_torch = inputs["v_std"]
        q_width = config.std_head_dim

    q_for_tt = q_torch.permute(2, 0, 1, 3)
    max_num_blocks = workload.seq_len // config.block_size * workload.batch
    paged_cfg = PagedAttentionConfig(block_size=config.block_size, max_num_blocks=max_num_blocks)
    page_table = page_table_setup(workload.batch, paged_cfg)
    tt_k_torch = to_paged_cache(k_torch, page_table, paged_cfg)
    tt_v_torch = to_paged_cache(v_torch, page_table, paged_cfg) if v_torch is not None else None

    grid_size = device.compute_with_storage_grid_size()
    q_num_cores = min(workload.batch * config.num_heads, grid_size.x * grid_size.y)
    block_height = nearest_y(int(np.prod(q_torch.shape[:-1])) // q_num_cores, ttnn.TILE_SIZE)
    q_core_grid = ttnn.num_cores_to_corerangeset(q_num_cores, grid_size, row_wise=True)
    q_mem_config = ttnn.create_sharded_memory_config(
        shape=(block_height, q_width),
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=0,
        k_chunk_size=config.k_chunk_size,
        exp_approx_mode=False,
        max_cores_per_head_batch=config.max_cores_per_head_batch,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    tt_q = ttnn.from_torch(
        q_for_tt,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=q_mem_config,
    )
    tt_k = ttnn.from_torch(
        tt_k_torch,
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_v = None
    if tt_v_torch is not None:
        tt_v = ttnn.from_torch(
            tt_v_torch,
            device=device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_cur_pos = ttnn.from_torch(
        torch.tensor([workload.seq_len - 1] * workload.batch, dtype=torch.int32),
        device=device,
        dtype=ttnn.int32,
    )
    return {
        "program_config": program_config,
        "compute_kernel_config": compute_kernel_config,
        "head_dim_v": config.mla_head_dim_v if baseline_key == "flash_mla" else config.std_head_dim,
        "out_mem_config": ttnn.DRAM_MEMORY_CONFIG,
        "tt_q": tt_q,
        "tt_k": tt_k,
        "tt_v": tt_v,
        "tt_page_table": tt_page_table,
        "tt_cur_pos": tt_cur_pos,
    }


def build_deepseek_decode_tt_inputs(device: Any, inputs: dict[str, Any], workload: Workload) -> dict[str, Any]:
    config = workload.config
    grid = FlashMLAOptimalGridNOC0_WH
    num_q_shards = config.deepseek_num_q_shards
    if num_q_shards is None:
        raise RuntimeError(
            "DeepSeek FlashMLA requires num_heads divisible by deepseek_num_q_heads_per_core "
            f"({config.num_heads} vs {config.deepseek_num_q_heads_per_core})"
        )
    required_q_cores = workload.batch * num_q_shards
    all_active_cores = [core for block_cores, _ in grid.BLOCKS for core in block_cores]
    if required_q_cores > len(all_active_cores):
        raise RuntimeError(
            f"DeepSeek FlashMLA requires batch * num_q_shards <= {len(all_active_cores)}, "
            f"got {workload.batch} * {num_q_shards} = {required_q_cores}"
        )

    tiny_tile = ttnn.Tile((config.deepseek_num_q_heads_per_core, 32))
    q_cores = all_active_cores[:required_q_cores]
    q_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in q_cores]
    )
    q_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            q_core_grid,
            (config.deepseek_num_q_heads_per_core, config.deepseek_kvpe_dim),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    out_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            q_core_grid,
            (config.deepseek_num_q_heads_per_core, config.deepseek_kv_lora_rank),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    q_for_tt = inputs["q_deepseek"].permute(2, 0, 1, 3).contiguous()
    kv_cache_torch = inputs["k_deepseek"]

    tt_q = ttnn.from_torch(
        q_for_tt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=q_mem_config,
        tile=tiny_tile,
    )

    program_config = FlashMLAProgramConfig(
        k_chunk_size=config.k_chunk_size,
        exp_approx_mode=False,
        grid=grid,
        allow_wh_fallback=False,
    )
    kv_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, config.num_kv_heads, program_config.k_chunk_size, config.deepseek_kvpe_dim],
        grid=grid.optimal_dram_grid(),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    kv_mem_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=kv_nd_shard_spec,
    )
    tt_cache = ttnn.from_torch(
        kv_cache_torch,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=kv_mem_config,
    )

    compute_kernel_config = ttnn.types.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=True,
    )
    backend_program_config = FlashMLADecode._build_wh_sdpa_program_config(tt_q, program_config)
    backend_q = FlashMLADecode._to_wh_backend_tensor(tt_q, dtype=ttnn.bfloat16)
    backend_k = FlashMLADecode._to_wh_backend_tensor(tt_cache, dtype=tt_cache.dtype)
    safe_deallocate(tt_q, tt_cache)
    return {
        "program_config": backend_program_config,
        "compute_kernel_config": compute_kernel_config,
        "head_dim_v": config.deepseek_kv_lora_rank,
        "backend_q": backend_q,
        "backend_k": backend_k,
        "cur_pos": [workload.seq_len - 1] * workload.batch,
        "measurement_note": (
            "timed only the backend device op after one-time DeepSeek->builtin tensor adaptation; "
            "excludes current Python wrapper to_torch/from_torch materialization overhead"
        ),
    }


def benchmark_device(
    device: Any,
    callable_: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
) -> list[float]:
    for _ in range(warmup):
        out = callable_()
        ttnn.synchronize_device(device)
        safe_deallocate(out)

    latencies_ms = []
    for _ in range(iters):
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        out = callable_()
        ttnn.synchronize_device(device)
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)
        safe_deallocate(out)
    return latencies_ms


def run_flash_attention_prefill(device: Any, tt_inputs: dict[str, Any], scale: float) -> ttnn.Tensor:
    return ttnn.transformer.scaled_dot_product_attention(
        tt_inputs["tt_q"],
        tt_inputs["tt_k"],
        tt_inputs["tt_v"],
        is_causal=True,
        scale=scale,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=tt_inputs["program_config"],
        compute_kernel_config=tt_inputs["compute_kernel_config"],
    )


def run_flash_mla_prefill(device: Any, tt_inputs: dict[str, Any], scale: float) -> ttnn.Tensor:
    return ttnn.transformer.flash_mla_prefill(
        tt_inputs["tt_q"],
        tt_inputs["tt_k"],
        head_dim_v=tt_inputs["head_dim_v"],
        scale=scale,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=tt_inputs["program_config"],
        compute_kernel_config=tt_inputs["compute_kernel_config"],
        is_causal=True,
    )


def run_flash_attention_decode(device: Any, tt_inputs: dict[str, Any], scale: float) -> ttnn.Tensor:
    return ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_inputs["tt_q"],
        tt_inputs["tt_k"],
        tt_inputs["tt_v"],
        tt_inputs["tt_page_table"],
        cur_pos_tensor=tt_inputs["tt_cur_pos"],
        is_causal=True,
        scale=scale,
        memory_config=tt_inputs["out_mem_config"],
        program_config=tt_inputs["program_config"],
        compute_kernel_config=tt_inputs["compute_kernel_config"],
    )


def run_flash_mla_decode(device: Any, tt_inputs: dict[str, Any], scale: float) -> ttnn.Tensor:
    return ttnn.transformer.paged_flash_multi_latent_attention_decode(
        tt_inputs["tt_q"],
        tt_inputs["tt_k"],
        head_dim_v=tt_inputs["head_dim_v"],
        page_table_tensor=tt_inputs["tt_page_table"],
        cur_pos_tensor=tt_inputs["tt_cur_pos"],
        is_causal=True,
        scale=scale,
        memory_config=tt_inputs["out_mem_config"],
        program_config=tt_inputs["program_config"],
        compute_kernel_config=tt_inputs["compute_kernel_config"],
    )


def run_deepseek_flash_mla_decode(device: Any, tt_inputs: dict[str, Any], scale: float) -> ttnn.Tensor:
    return ttnn.transformer.flash_multi_latent_attention_decode(
        tt_inputs["backend_q"],
        tt_inputs["backend_k"],
        None,
        head_dim_v=tt_inputs["head_dim_v"],
        cur_pos=tt_inputs["cur_pos"],
        scale=scale,
        program_config=tt_inputs["program_config"],
        compute_kernel_config=tt_inputs["compute_kernel_config"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def logical_kv_bytes(workload: Workload, baseline_key: str) -> int:
    config = workload.config
    if baseline_key in {"flash_mla", "deepseek_flash_mla"}:
        width = config.mla_head_dim_qk if baseline_key == "flash_mla" else config.deepseek_kvpe_dim
        return workload.batch * config.num_kv_heads * workload.seq_len * width * 2
    return workload.batch * config.num_kv_heads * workload.seq_len * (config.std_head_dim + config.std_head_dim) * 2


def baseline_head_dims(workload: Workload, baseline_key: str) -> tuple[int, int]:
    config = workload.config
    if baseline_key == "flash_mla":
        return config.mla_head_dim_qk, config.mla_head_dim_v
    if baseline_key == "deepseek_flash_mla":
        return config.deepseek_kvpe_dim, config.deepseek_kv_lora_rank
    return config.std_head_dim, config.std_head_dim


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def validate_workload_baseline_support(
    workload: Workload,
    baseline: BaselineConfig,
    device: Any | None,
) -> list[str]:
    config = workload.config
    reasons: list[str] = []

    if config.batch <= 0:
        reasons.append("batch must be positive")
    if config.num_heads <= 0:
        reasons.append("num_heads must be positive")
    if config.num_kv_heads <= 0:
        reasons.append("num_kv_heads must be positive")
    if config.num_heads % config.num_kv_heads != 0:
        reasons.append("num_heads must be divisible by num_kv_heads")
    if workload.seq_len % config.block_size != 0:
        reasons.append("seq_len must be divisible by block_size")
    if not is_power_of_two(config.k_chunk_size) or config.k_chunk_size % 32 != 0:
        reasons.append("k_chunk_size must be a power of two and divisible by 32")

    if baseline.key in {"flash_attention", "flash_mla", "deepseek_flash_mla"}:
        for name, value in (
            ("std_head_dim", config.std_head_dim),
            ("mla_head_dim_v", config.mla_head_dim_v),
            ("mla_d_rope", config.mla_d_rope),
            ("mla_head_dim_qk", config.mla_head_dim_qk),
            ("deepseek_kv_lora_rank", config.deepseek_kv_lora_rank),
            ("deepseek_kvpe_dim", config.deepseek_kvpe_dim),
        ):
            if value % 32 != 0:
                reasons.append(f"{name} must be divisible by 32")

    if baseline.key == "deepseek_flash_mla":
        if workload.mode != "decode":
            reasons.append("DeepSeek FlashMLA currently supports decode only")
        if config.num_kv_heads != 1:
            reasons.append("DeepSeek FlashMLA currently requires num_kv_heads=1")
        if config.deepseek_num_q_heads_per_core <= 0 or config.deepseek_num_q_heads_per_core >= 32:
            reasons.append("deepseek_num_q_heads_per_core must be in range (0, 32)")
        if config.deepseek_num_q_shards is None:
            reasons.append("num_heads must be divisible by deepseek_num_q_heads_per_core")
        else:
            available_q_cores = len([core for block_cores, _ in FlashMLAOptimalGridNOC0_WH.BLOCKS for core in block_cores])
            required_q_cores = workload.batch * config.deepseek_num_q_shards
            if required_q_cores > available_q_cores:
                reasons.append(
                    f"batch * deepseek_num_q_shards must be <= {available_q_cores}, got {required_q_cores}"
                )
        if device is not None:
            grid_size = device.compute_with_storage_grid_size()
            if grid_size.x < 8 or grid_size.y < 7:
                reasons.append("DeepSeek FlashMLA on Wormhole requires compute grid >= 8x7")

    return reasons


def run_tt_baseline(
    device: Any,
    workload: Workload,
    baseline: BaselineConfig,
    inputs: dict[str, Any],
    *,
    warmup: int,
    iters: int,
) -> list[float]:
    if workload.mode == "prefill":
        tt_inputs = build_prefill_tt_inputs(device, inputs, baseline.key)
        try:
            if baseline.key == "flash_attention":
                return benchmark_device(
                    device,
                    lambda: run_flash_attention_prefill(device, tt_inputs, inputs["scale_std"]),
                    warmup=warmup,
                    iters=iters,
                )
            if baseline.key == "flash_mla":
                return benchmark_device(
                    device,
                    lambda: run_flash_mla_prefill(device, tt_inputs, inputs["scale_mla"]),
                    warmup=warmup,
                    iters=iters,
                )
        finally:
            safe_deallocate(tt_inputs["tt_q"], tt_inputs["tt_k"], tt_inputs["tt_v"])

    if baseline.key == "deepseek_flash_mla":
        tt_inputs = build_deepseek_decode_tt_inputs(device, inputs, workload)
        try:
            return benchmark_device(
                device,
                lambda: run_deepseek_flash_mla_decode(device, tt_inputs, inputs["scale_deepseek"]),
                warmup=warmup,
                iters=iters,
            )
        finally:
            safe_deallocate(
                tt_inputs["backend_q"],
                tt_inputs["backend_k"],
            )

    tt_inputs = build_decode_tt_inputs(device, inputs, workload, baseline.key)
    try:
        if baseline.key == "flash_attention":
            return benchmark_device(
                device,
                lambda: run_flash_attention_decode(device, tt_inputs, inputs["scale_std"]),
                warmup=warmup,
                iters=iters,
            )
        if baseline.key == "flash_mla":
            return benchmark_device(
                device,
                lambda: run_flash_mla_decode(device, tt_inputs, inputs["scale_mla"]),
                warmup=warmup,
                iters=iters,
            )
    finally:
        safe_deallocate(
            tt_inputs["tt_q"],
            tt_inputs["tt_k"],
            tt_inputs["tt_v"],
            tt_inputs["tt_page_table"],
            tt_inputs["tt_cur_pos"],
        )
    raise ValueError(f"Unsupported TT baseline: {baseline.key}")


def run_single_benchmark(
    device: Any | None,
    workload: Workload,
    baseline: BaselineConfig,
    *,
    warmup_device: int,
    iters_device: int,
    warmup_reference: int,
    iters_reference: int,
) -> dict[str, Any]:
    head_dim_qk, head_dim_v = baseline_head_dims(workload, baseline.key)
    record: dict[str, Any] = {
        "case": workload.name,
        "case_group": workload.case_group,
        "mode": workload.mode,
        "seq_len": workload.seq_len,
        "q_seq_len": workload.q_seq_len,
        "batch": workload.batch,
        "normalized_items": workload.normalized_items,
        "baseline": baseline.key,
        "baseline_label": baseline.label,
        "backend": baseline.backend,
        **workload.config.to_dict(),
        "head_dim_qk": head_dim_qk,
        "head_dim_v": head_dim_v,
        "logical_kv_bytes": logical_kv_bytes(workload, baseline.key),
    }
    if baseline.key == "deepseek_flash_mla":
        record["measurement_scope"] = "device_core_only_after_one_time_adapter_conversion"
    else:
        record["measurement_scope"] = "host_wall_clock_with_inputs_prebuilt"

    support_reasons = validate_workload_baseline_support(workload, baseline, device)
    if support_reasons:
        record["status"] = "unsupported"
        record["unsupported_reasons"] = support_reasons
        record["error"] = "; ".join(support_reasons)
        return record

    inputs = make_torch_inputs(workload)

    try:
        if baseline.key == "reference_attention":
            if workload.mode == "decode":
                latencies_ms = benchmark_reference(
                    lambda: reference_decode(inputs, workload),
                    warmup=warmup_reference,
                    iters=iters_reference,
                )
            else:
                latencies_ms = benchmark_reference(
                    lambda: reference_prefill(inputs, workload),
                    warmup=warmup_reference,
                    iters=iters_reference,
                )
        else:
            if device is None:
                raise RuntimeError("TT device is not available for device baselines.")
            latencies_ms = run_tt_baseline(
                device,
                workload,
                baseline,
                inputs,
                warmup=warmup_device,
                iters=iters_device,
            )

        record["latencies_ms"] = [round(value, 6) for value in latencies_ms]
        record["num_samples"] = len(latencies_ms)
        record.update({key: round(value, 6) for key, value in summarize_latencies(latencies_ms, workload.normalized_items).items()})
        record["status"] = "ok"
    except Exception as exc:
        record["status"] = "error"
        record["error"] = str(exc)
        logger.exception(f"Failed on {baseline.key} / {workload.name}")

    return record


def run_all_benchmarks(args: argparse.Namespace) -> dict[str, Any]:
    ensure_output_dirs()
    configs = build_experiment_configs(args, probe=False)
    default_config = default_config_from_list(configs)
    workload_catalog = build_workload_catalog(
        configs,
        decode_seq_lens=args.decode_seq_lens,
        prefill_seq_lens=args.prefill_seq_lens,
        default_config=default_config,
    )
    workloads = select_workloads(workload_catalog, args.cases)
    results: list[dict[str, Any]] = []
    probe_payload: dict[str, Any] | None = None

    device = None
    try:
        device = ttnn.open_device(device_id=DEVICE_ID)
        grid = device.compute_with_storage_grid_size()
        device_info = {"device_id": DEVICE_ID, "grid_x": grid.x, "grid_y": grid.y, "arch": "wormhole_b0"}
    except Exception as exc:
        logger.warning(f"TT device open failed, device baselines will error: {exc}")
        device_info = {"device_id": DEVICE_ID, "error": str(exc), "arch": "unknown"}

    try:
        if args.run_capability_probe:
            probe_payload = run_capability_probe(
                args,
                device,
                device_info,
                resume=args.probe_resume,
                reset_checkpoint=args.probe_reset_checkpoint,
                workload_batch_size=args.probe_workload_batch_size,
                checkpoint_every_workloads=args.probe_checkpoint_every_workloads,
            )
        for workload in workloads:
            logger.info(f"Running workload {workload.name}")
            for baseline in active_baselines_for_workload(workload):
                logger.info(f"  baseline={baseline.key}")
                results.append(
                    run_single_benchmark(
                        device,
                        workload,
                        baseline,
                        warmup_device=args.warmup_device,
                        iters_device=args.iters_device,
                        warmup_reference=args.warmup_reference,
                        iters_reference=args.iters_reference,
                    )
                )
    finally:
        if device is not None:
            try:
                ttnn.close_device(device)
            except Exception:
                pass

    payload = {
        "metadata": {
            "device": device_info,
            "comparison_mode": args.comparison_mode,
            "sweep_preset": args.sweep_preset,
            "probe_preset": args.probe_preset,
            "generated_cases": list(workload_catalog.keys()),
            "cases": [workload.name for workload in workloads],
            "decode_seq_lens": args.decode_seq_lens,
            "prefill_seq_lens": args.prefill_seq_lens,
            "sweep_axes": {
                "batches": args.batches,
                "num_heads": args.num_heads_list,
                "num_kv_heads": args.num_kv_heads_list,
                "value_dims": args.value_dims,
                "rope_dims": args.rope_dims,
                "decode_seq_lens": args.decode_seq_lens,
                "prefill_seq_lens": args.prefill_seq_lens,
            },
            "configs": [config.to_dict() for config in configs],
            "default_config": default_config.to_dict(),
            "decode_baselines": [baseline.key for baseline in BASELINES],
            "prefill_baselines": [baseline.key for baseline in BASELINES if baseline.key != "deepseek_flash_mla"],
            "warmup_device": args.warmup_device,
            "iters_device": args.iters_device,
            "warmup_reference": args.warmup_reference,
            "iters_reference": args.iters_reference,
            "block_size": args.block_size,
            "k_chunk_size": args.k_chunk_size,
            "max_cores_per_head_batch": args.max_cores_per_head_batch,
            "deepseek_num_q_heads_per_core": args.deepseek_num_q_heads_per_core,
            "run_capability_probe": args.run_capability_probe,
            "capability_probe_json": str(CAPABILITY_PROBE_JSON_PATH) if args.run_capability_probe else None,
            "reference_backend_note": "reference attention uses torch reference SDPA on host as the non-TT control baseline",
            "statistics_note": "mean_ms=min/avg/max statistics are computed over repeated measured iterations per method",
            "deepseek_measurement_note": (
                "DeepSeek FlashMLA decode is timed on the backend device op after a one-time tensor adaptation step "
                "so the main four-method comparison is not dominated by the current Python wrapper's to_torch/from_torch overhead"
            ),
        },
        "results": results,
    }
    if probe_payload is not None:
        payload["capability_probe"] = {
            "summary_path": str(CAPABILITY_PROBE_JSON_PATH),
            "checkpoint_path": str(CAPABILITY_PROBE_CHECKPOINT_PATH),
            "status": probe_payload.get("metadata", {}).get("status"),
            "total_cases": len(probe_payload.get("cases", [])),
            "processed_workloads": probe_payload.get("metadata", {}).get("processed_workloads"),
            "remaining_workloads": probe_payload.get("metadata", {}).get("remaining_workloads"),
            "supported_all_cases": sum(1 for case in probe_payload.get("cases", []) if case.get("supported_by_all")),
        }
    with JSON_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
    write_csv(results)
    return payload


def write_csv(results: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in results:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in fieldnames})


def summarize_probe_case(workload: Workload, case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_status = {row["baseline"]: row["status"] for row in case_rows}
    baseline_errors = {
        row["baseline"]: row.get("error", "")
        for row in case_rows
        if row.get("status") != "ok"
    }
    supported_by_all = all(baseline_status.get(baseline.key) == "ok" for baseline in BASELINES)
    return {
        "case": workload.name,
        "case_group": workload.case_group,
        "seq_len": workload.seq_len,
        "mode": workload.mode,
        "config": workload.config.to_dict(),
        "config_signature": workload.config_signature,
        "baseline_status": baseline_status,
        "baseline_errors": baseline_errors,
        "supported_by_all": supported_by_all,
        "unsupported_baselines": [key for key, status in baseline_status.items() if status != "ok"],
    }


def build_probe_payload(
    args: argparse.Namespace,
    device_info: dict[str, Any],
    probe_configs: list[ExperimentConfig],
    probe_default_config: ExperimentConfig,
    probe_workloads: list[Workload],
    case_summaries: list[dict[str, Any]],
    probe_results: list[dict[str, Any]],
    *,
    status: str,
    resumed_from_checkpoint: bool,
) -> dict[str, Any]:
    total_workloads = len(probe_workloads)
    processed_workloads = len(case_summaries)
    case_order = [workload.name for workload in probe_workloads]
    return {
        "metadata": {
            "device": device_info,
            "comparison_mode": args.comparison_mode,
            "probe_preset": args.probe_preset,
            "probe_axes": {
                "batches": args.probe_batches,
                "num_heads": args.probe_num_heads,
                "num_kv_heads": args.probe_num_kv_heads,
                "value_dims": args.probe_value_dims,
                "rope_dims": args.probe_rope_dims,
                "decode_seq_lens": args.probe_decode_seq_lens,
            },
            "probe_baselines": [baseline.key for baseline in BASELINES],
            "warmup_device": 0,
            "iters_device": 1,
            "warmup_reference": 0,
            "iters_reference": 1,
            "configs": [config.to_dict() for config in probe_configs],
            "default_config": probe_default_config.to_dict(),
            "status": status,
            "checkpoint_path": str(CAPABILITY_PROBE_CHECKPOINT_PATH),
            "final_json_path": str(CAPABILITY_PROBE_JSON_PATH),
            "checkpoint_every_workloads": args.probe_checkpoint_every_workloads,
            "workload_batch_size": args.probe_workload_batch_size,
            "total_workloads": total_workloads,
            "processed_workloads": processed_workloads,
            "remaining_workloads": max(total_workloads - processed_workloads, 0),
            "last_completed_case": case_summaries[-1]["case"] if case_summaries else None,
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "case_order": case_order,
        },
        "cases": case_summaries,
        "results": probe_results,
    }


def load_probe_checkpoint(
    args: argparse.Namespace,
    probe_workloads: list[Workload],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    if args.probe_reset_checkpoint:
        delete_if_exists(CAPABILITY_PROBE_CHECKPOINT_PATH)
        delete_if_exists(CAPABILITY_PROBE_JSON_PATH)

    if not args.probe_resume:
        delete_if_exists(CAPABILITY_PROBE_CHECKPOINT_PATH)
        delete_if_exists(CAPABILITY_PROBE_JSON_PATH)
        return [], [], False

    source_path: Path | None = None
    if CAPABILITY_PROBE_CHECKPOINT_PATH.exists():
        source_path = CAPABILITY_PROBE_CHECKPOINT_PATH
    elif CAPABILITY_PROBE_JSON_PATH.exists():
        source_path = CAPABILITY_PROBE_JSON_PATH
    if source_path is None:
        return [], [], False

    payload = json.loads(source_path.read_text())
    metadata = payload.get("metadata", {})
    expected_case_order = [workload.name for workload in probe_workloads]
    actual_case_order = metadata.get("case_order")
    if actual_case_order is not None and actual_case_order != expected_case_order:
        raise RuntimeError(
            f"Capability probe checkpoint {source_path} does not match the current planned case order. "
            "Use --probe-reset-checkpoint to start fresh."
        )

    expected_probe_axes = {
        "batches": args.probe_batches,
        "num_heads": args.probe_num_heads,
        "num_kv_heads": args.probe_num_kv_heads,
        "value_dims": args.probe_value_dims,
        "rope_dims": args.probe_rope_dims,
        "decode_seq_lens": args.probe_decode_seq_lens,
    }
    actual_probe_axes = metadata.get("probe_axes")
    if actual_probe_axes is not None and actual_probe_axes != expected_probe_axes:
        raise RuntimeError(
            f"Capability probe checkpoint {source_path} does not match the current probe axes. "
            "Use --probe-reset-checkpoint to start fresh."
        )

    case_summaries = list(payload.get("cases", []))
    probe_results = list(payload.get("results", []))
    completed_cases = [case["case"] for case in case_summaries]
    if len(completed_cases) != len(set(completed_cases)):
        raise RuntimeError(f"Capability probe checkpoint {source_path} contains duplicate completed cases.")
    if completed_cases != expected_case_order[: len(completed_cases)]:
        raise RuntimeError(
            f"Capability probe checkpoint {source_path} completed cases are not a prefix of the current workload order. "
            "Use --probe-reset-checkpoint to start fresh."
        )
    logger.info(
        "Resuming capability probe from checkpoint: "
        f"{len(completed_cases)}/{len(expected_case_order)} workloads already completed"
    )
    return case_summaries, probe_results, True


def run_capability_probe(
    args: argparse.Namespace,
    device: Any | None,
    device_info: dict[str, Any],
    *,
    resume: bool = False,
    reset_checkpoint: bool = False,
    workload_batch_size: int | None = None,
    checkpoint_every_workloads: int | None = None,
) -> dict[str, Any]:
    probe_configs = build_experiment_configs(args, probe=True)
    probe_default_config = default_config_from_list(probe_configs)
    probe_catalog = build_workload_catalog(
        probe_configs,
        decode_seq_lens=args.probe_decode_seq_lens,
        prefill_seq_lens=[],
        default_config=probe_default_config,
    )
    probe_workloads = list(probe_catalog.values())
    original_probe_resume = args.probe_resume
    original_probe_reset_checkpoint = args.probe_reset_checkpoint
    original_batch_size = args.probe_workload_batch_size
    original_checkpoint_every = args.probe_checkpoint_every_workloads
    args.probe_resume = resume
    args.probe_reset_checkpoint = reset_checkpoint
    args.probe_workload_batch_size = workload_batch_size
    args.probe_checkpoint_every_workloads = checkpoint_every_workloads or original_checkpoint_every

    try:
        case_summaries, probe_results, resumed_from_checkpoint = load_probe_checkpoint(args, probe_workloads)
        completed_case_names = {case["case"] for case in case_summaries}
        pending_workloads = [workload for workload in probe_workloads if workload.name not in completed_case_names]
        total_workloads = len(probe_workloads)

        if args.probe_workload_batch_size is not None:
            pending_workloads = pending_workloads[: args.probe_workload_batch_size]

        logger.info(
            "Capability probe starting batch: "
            f"{len(case_summaries)}/{total_workloads} workloads completed, "
            f"{len(pending_workloads)} workloads selected this run"
        )

        processed_since_checkpoint = 0
        for workload in pending_workloads:
            logger.info(f"Capability probe workload {workload.name}")
            case_rows = []
            for baseline in BASELINES:
                logger.info(f"  probe baseline={baseline.key}")
                row = run_single_benchmark(
                    device,
                    workload,
                    baseline,
                    warmup_device=0,
                    iters_device=1,
                    warmup_reference=0,
                    iters_reference=1,
                )
                probe_results.append(row)
                case_rows.append(row)
            case_summaries.append(summarize_probe_case(workload, case_rows))
            processed_since_checkpoint += 1

            if processed_since_checkpoint >= args.probe_checkpoint_every_workloads:
                checkpoint_payload = build_probe_payload(
                    args,
                    device_info,
                    probe_configs,
                    probe_default_config,
                    probe_workloads,
                    case_summaries,
                    probe_results,
                    status="running" if len(case_summaries) < total_workloads else "completed",
                    resumed_from_checkpoint=resumed_from_checkpoint,
                )
                write_json_atomic(CAPABILITY_PROBE_CHECKPOINT_PATH, checkpoint_payload)
                logger.info(
                    "Capability probe checkpoint saved: "
                    f"{checkpoint_payload['metadata']['processed_workloads']}/{total_workloads} workloads complete"
                )
                processed_since_checkpoint = 0

        final_status = "completed" if len(case_summaries) == total_workloads else "running"
        probe_payload = build_probe_payload(
            args,
            device_info,
            probe_configs,
            probe_default_config,
            probe_workloads,
            case_summaries,
            probe_results,
            status=final_status,
            resumed_from_checkpoint=resumed_from_checkpoint,
        )
        write_json_atomic(CAPABILITY_PROBE_CHECKPOINT_PATH, probe_payload)
        logger.info(
            "Capability probe checkpoint saved: "
            f"{probe_payload['metadata']['processed_workloads']}/{total_workloads} workloads complete"
        )
        if final_status == "completed":
            write_json_atomic(CAPABILITY_PROBE_JSON_PATH, probe_payload)
            logger.info(f"Capability probe final JSON written to {CAPABILITY_PROBE_JSON_PATH}")
        return probe_payload
    finally:
        args.probe_resume = original_probe_resume
        args.probe_reset_checkpoint = original_probe_reset_checkpoint
        args.probe_workload_batch_size = original_batch_size
        args.probe_checkpoint_every_workloads = original_checkpoint_every


def run_probe_only(args: argparse.Namespace) -> dict[str, Any]:
    ensure_output_dirs()
    device = None
    try:
        device = ttnn.open_device(device_id=DEVICE_ID)
        grid = device.compute_with_storage_grid_size()
        device_info = {"device_id": DEVICE_ID, "grid_x": grid.x, "grid_y": grid.y, "arch": "wormhole_b0"}
    except Exception as exc:
        logger.warning(f"TT device open failed, device baselines will error: {exc}")
        device_info = {"device_id": DEVICE_ID, "error": str(exc), "arch": "unknown"}

    try:
        return run_capability_probe(
            args,
            device,
            device_info,
            resume=args.probe_resume,
            reset_checkpoint=args.probe_reset_checkpoint,
            workload_batch_size=args.probe_workload_batch_size,
            checkpoint_every_workloads=args.probe_checkpoint_every_workloads,
        )
    finally:
        if device is not None:
            try:
                ttnn.close_device(device)
            except Exception:
                pass


def run_flashmla_detailed_profile(detail_cases: list[str]) -> dict[str, Any]:
    old_output_root = flash_mla_detailed_profile.OUTPUT_ROOT
    flash_mla_detailed_profile.OUTPUT_ROOT = FLASHMLA_DETAIL_DIR
    try:
        profiles = []
        for case_name in detail_cases:
            logger.info(f"Running FlashMLA detailed tracy profile for {case_name}")
            csv_path = flash_mla_detailed_profile.run_tracy_profile(case_name)
            profiles.append(flash_mla_detailed_profile.parse_report(case_name, csv_path))

        results = {
            "metadata": {
                "arch": "wormhole_b0",
                "aiclk_mhz": flash_mla_detailed_profile.WORMHOLE_AICLK_MHZ,
                "selected_cases": detail_cases,
            },
            "profiles": profiles,
        }
        flash_mla_detailed_profile.write_summary(results)
        return results
    finally:
        flash_mla_detailed_profile.OUTPUT_ROOT = old_output_root


def copy_flashmla_case_artifacts(source_root: Path, target_root: Path, case_names: list[str]) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    for case_name in case_names:
        src_case_dir = source_root / case_name
        if not src_case_dir.exists():
            continue
        dst_case_dir = target_root / case_name
        if dst_case_dir.exists():
            shutil.rmtree(dst_case_dir)
        shutil.copytree(src_case_dir, dst_case_dir)


def reuse_existing_flashmla_detailed_profile(detail_cases: list[str], source_root: Path) -> dict[str, Any]:
    if not source_root.exists():
        raise FileNotFoundError(f"Existing FlashMLA detail source does not exist: {source_root}")

    old_output_root = flash_mla_detailed_profile.OUTPUT_ROOT
    try:
        flash_mla_detailed_profile.OUTPUT_ROOT = source_root
        profiles = []
        selected_cases = []
        for case_name in detail_cases:
            case_dir = source_root / case_name
            if not case_dir.exists():
                logger.warning(f"Skip missing existing FlashMLA detail case: {case_name}")
                continue
            csv_path = flash_mla_detailed_profile.latest_cpp_device_report(case_dir)
            profiles.append(flash_mla_detailed_profile.parse_report(case_name, csv_path))
            selected_cases.append(case_name)

        if not profiles:
            raise RuntimeError(f"No reusable FlashMLA detailed cases found under {source_root}")

        copy_flashmla_case_artifacts(source_root, FLASHMLA_DETAIL_DIR, selected_cases)

        results = {
            "metadata": {
                "arch": "wormhole_b0",
                "aiclk_mhz": flash_mla_detailed_profile.WORMHOLE_AICLK_MHZ,
                "selected_cases": selected_cases,
                "reused_existing": True,
                "reused_existing_source": str(source_root),
                "detail_note": (
                    "reused existing stable cpp_device_perf_report.csv/profile_log_device.csv assets because "
                    "local tracy post-processing is known to hang on this machine"
                ),
                "batch_note": (
                    "decode_1k/decode_4k/decode_8k come from historical detailed runs with batch=2; "
                    "decode_16k/decode_32k use batch=1"
                ),
            },
            "profiles": profiles,
        }

        flash_mla_detailed_profile.OUTPUT_ROOT = FLASHMLA_DETAIL_DIR
        flash_mla_detailed_profile.write_summary(results)
        return results
    finally:
        flash_mla_detailed_profile.OUTPUT_ROOT = old_output_root


def maybe_render_results() -> None:
    import subprocess

    cmd = ["python3", str(SCRIPT_DIR / "render_part1_results.py")]
    subprocess.run(cmd, cwd=SCRIPT_DIR.parents[2], check=True)


def main() -> None:
    args = parse_args()
    if args.probe_only:
        probe_payload = run_probe_only(args)
        probe_meta = probe_payload.get("metadata", {})
        logger.info(
            "Capability probe status: "
            f"{probe_meta.get('status')} "
            f"({probe_meta.get('processed_workloads')}/{probe_meta.get('total_workloads')} workloads complete)"
        )
        if not args.skip_render:
            maybe_render_results()
        return

    payload = run_all_benchmarks(args)
    if JSON_PATH.exists():
        logger.info(f"Wrote benchmark JSON to {JSON_PATH}")

    if args.reuse_existing_flashmla_detailed:
        reuse_existing_flashmla_detailed_profile(args.detail_cases, EXISTING_FLASHMLA_DETAIL_SOURCE)
        logger.info(f"Reused existing FlashMLA detailed profile into {FLASHMLA_DETAIL_DIR}")
    elif args.run_flashmla_detailed:
        run_flashmla_detailed_profile(args.detail_cases)
        logger.info(f"Wrote FlashMLA detailed profile into {FLASHMLA_DETAIL_DIR}")

    if not args.skip_render:
        maybe_render_results()

    default_signature = payload["metadata"].get("default_config", {}).get("config_signature")
    summary_rows = [
        row
        for row in payload["results"]
        if row.get("status") == "ok"
        and row.get("mode") == "decode"
        and row.get("case_group") == "decode_32k"
        and (default_signature is None or row.get("config_signature") == default_signature)
    ]
    if summary_rows:
        summary_rows.sort(key=lambda row: row["mean_ms"])
        logger.info(
            "decode_32k fastest baseline: "
            f"{summary_rows[0]['baseline_label']} ({summary_rows[0]['mean_ms']:.3f} ms)"
        )


if __name__ == "__main__":
    main()
