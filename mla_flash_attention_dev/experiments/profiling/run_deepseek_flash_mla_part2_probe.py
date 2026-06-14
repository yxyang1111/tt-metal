#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mla_flash_attention_dev.experiments.part1_baselines.run_part1_benchmarks as part1
from mla_flash_attention_dev.experiments.profiling.profile_deepseek_flash_mla_part2 import (
    PRESET_CASES,
    PRESET_GROUPS,
    baseline_from_key,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE_SCRIPT = REPO_ROOT / "mla_flash_attention_dev" / "experiments" / "profile_deepseek_flash_mla_part2.py"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "mla_flash_attention_dev" / "experiments" / "profiling" / "outputs" / "deepseek_flash_mla_part2_probe"
)
DEFAULT_PRESET_CASES = ["A1", "A2", "A3"]
DEFAULT_VARIANTS = ["tt", "deepseek4c"]
DEVICE_ANALYSIS_TYPES = [
    "device_kernel_duration",
    "device_kernel_first_to_last_start",
    "device_brisc_kernel_duration",
    "device_ncrisc_kernel_duration",
    "device_trisc0_kernel_duration",
    "device_trisc1_kernel_duration",
    "device_trisc2_kernel_duration",
    "device_compute_cb_wait_front",
    "device_compute_cb_reserve_back",
]
REQUIRED_TRACY_REPORT_COLUMNS = [
    "PM IDEAL [ns]",
    "PM COMPUTE [ns]",
    "PM BANDWIDTH [ns]",
    "PM FPU UTIL (%)",
    "DEVICE COMPUTE CB WAIT FRONT [ns]",
    "DEVICE COMPUTE CB RESERVE BACK [ns]",
]
REQUIRED_DEVICE_ONLY_COLUMNS = [
    "DEVICE KERNEL DURATION [ns]",
    "DEVICE COMPUTE CB WAIT FRONT [ns]",
    "DEVICE COMPUTE CB RESERVE BACK [ns]",
]
OPTIONAL_NOC_COLUMNS = [
    "NOC UTIL (%)",
    "MULTICAST NOC UTIL (%)",
    "DRAM BW UTIL (%)",
]
LEGACY_TRIGGER_CHOICES = ["sum", "no-runtime-analysis", "perf-fpu"]
CAPTURE_MODE_CHOICES = ["device_only", "tracy_report"]


@dataclass(frozen=True)
class VariantSpec:
    key: str
    baseline_key: str
    deepseek_wh_cores_per_block: int
    label: str


VARIANTS: dict[str, VariantSpec] = {
    "tt": VariantSpec("tt", "flash_mla", 4, "FlashMLA (TT Mainline)"),
    "deepseek4c": VariantSpec("deepseek4c", "deepseek_flash_mla", 4, "DeepSeek FlashMLA 4c"),
    "deepseek8c": VariantSpec("deepseek8c", "deepseek_flash_mla", 8, "DeepSeek FlashMLA 8c"),
}


@dataclass(frozen=True)
class ProbeInvocation:
    case: str
    preset_case: str
    group: str
    variant: str
    baseline: str
    variant_label: str
    deepseek_wh_cores_per_block: int
    seq_len: int
    batch: int
    num_heads: int
    num_kv_heads: int
    value_dim: int
    rope_dim: int
    block_size: int
    k_chunk_size: int
    max_cores_per_head_batch: int
    deepseek_num_q_heads_per_core: int
    deepseek_num_q_shards: int | None
    config_signature: str
    warmup_iterations: int
    iterations: int


@dataclass
class CaseResult:
    case: str
    preset_case: str
    group: str
    variant: str
    baseline: str
    variant_label: str
    deepseek_wh_cores_per_block: int
    seq_len: int
    batch: int
    num_heads: int
    num_kv_heads: int
    value_dim: int
    rope_dim: int
    config_signature: str
    deepseek_num_q_heads_per_core: int
    deepseek_num_q_shards: int | None
    warmup_iterations: int
    iterations: int
    status: str
    output_dir: str
    report_dir: str | None = None
    ops_csv: str | None = None
    available_required_columns: list[str] | None = None
    missing_required_columns: list[str] | None = None
    available_optional_noc_columns: list[str] | None = None
    unsupported_reasons: list[str] | None = None
    command: list[str] | None = None
    postprocess_command: list[str] | None = None
    error: str | None = None


@dataclass
class PreflightResult:
    case: str
    preset_case: str
    variant: str
    status: str
    command: list[str]
    error: str | None = None
    attempts: int = 1


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def resolve_preset_cases(args: argparse.Namespace) -> list[str]:
    if args.preset_cases:
        return ordered_unique(args.preset_cases)
    if args.groups:
        expanded: list[str] = []
        for group in args.groups:
            expanded.extend(PRESET_GROUPS[group])
        return ordered_unique(expanded)
    if args.smoke:
        return ["A1"]
    if args.minimal:
        return list(DEFAULT_PRESET_CASES)
    return list(DEFAULT_PRESET_CASES)


def make_invocation(preset_case: str, variant_key: str, args: argparse.Namespace) -> ProbeInvocation:
    preset = PRESET_CASES[preset_case]
    variant = VARIANTS[variant_key]
    config = preset.make_config()
    warmup_iterations = preset.warmup_iterations if args.warmup_iterations is None else args.warmup_iterations
    iterations = preset.iterations if args.iterations is None else args.iterations
    return ProbeInvocation(
        case=f"{preset_case}__{variant_key}",
        preset_case=preset_case,
        group=preset.group,
        variant=variant.key,
        baseline=variant.baseline_key,
        variant_label=variant.label,
        deepseek_wh_cores_per_block=variant.deepseek_wh_cores_per_block,
        seq_len=preset.seq_len,
        batch=preset.batch,
        num_heads=preset.num_heads,
        num_kv_heads=preset.num_kv_heads,
        value_dim=preset.value_dim,
        rope_dim=preset.rope_dim,
        block_size=preset.block_size,
        k_chunk_size=preset.k_chunk_size,
        max_cores_per_head_batch=preset.max_cores_per_head_batch,
        deepseek_num_q_heads_per_core=preset.deepseek_num_q_heads_per_core,
        deepseek_num_q_shards=config.deepseek_num_q_shards,
        config_signature=config.config_signature,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )


def latest_ops_csv(case_output_dir: Path, *, capture_mode: str) -> Path | None:
    reports_dir = case_output_dir / "reports"
    flat_ops_csv = reports_dir / "ops_perf_results.csv"
    if flat_ops_csv.exists():
        return flat_ops_csv
    if capture_mode == "device_only":
        return None
    if not reports_dir.exists():
        return None
    report_subdirs = sorted([path for path in reports_dir.iterdir() if path.is_dir()])
    if not report_subdirs:
        return None
    latest_report = report_subdirs[-1]
    ops_csv = latest_report / f"ops_perf_results_{latest_report.name}.csv"
    return ops_csv if ops_csv.exists() else None


def csv_header(path: Path) -> list[str]:
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def required_columns_for_capture_mode(capture_mode: str) -> list[str]:
    if capture_mode == "device_only":
        return REQUIRED_DEVICE_ONLY_COLUMNS
    return REQUIRED_TRACY_REPORT_COLUMNS


def verify_columns(path: Path, *, capture_mode: str) -> tuple[list[str], list[str], list[str]]:
    header = csv_header(path)
    required_columns = required_columns_for_capture_mode(capture_mode)
    available_required = [column for column in required_columns if column in header]
    missing_required = [column for column in required_columns if column not in header]
    available_optional = [column for column in OPTIONAL_NOC_COLUMNS if column in header]
    return available_required, missing_required, available_optional


def build_child_command(invocation: ProbeInvocation, args: argparse.Namespace) -> list[str]:
    command = [
        "python3",
        str(PROFILE_SCRIPT.relative_to(REPO_ROOT)),
        "--child-run",
        "--preset-case",
        invocation.preset_case,
        "--baseline",
        invocation.baseline,
        "--deepseek-wh-cores-per-block",
        str(invocation.deepseek_wh_cores_per_block),
        "--device-id",
        str(args.device_id),
    ]
    if args.warmup_iterations is not None:
        command.extend(["--warmup-iterations", str(args.warmup_iterations)])
    if args.iterations is not None:
        command.extend(["--iterations", str(args.iterations)])
    return command


def build_tracy_target_args(invocation: ProbeInvocation, args: argparse.Namespace) -> list[str]:
    command = [
        str(PROFILE_SCRIPT.relative_to(REPO_ROOT)),
        "--child-run",
        "--preset-case",
        invocation.preset_case,
        "--baseline",
        invocation.baseline,
        "--deepseek-wh-cores-per-block",
        str(invocation.deepseek_wh_cores_per_block),
        "--device-id",
        str(args.device_id),
    ]
    if args.warmup_iterations is not None:
        command.extend(["--warmup-iterations", str(args.warmup_iterations)])
    if args.iterations is not None:
        command.extend(["--iterations", str(args.iterations)])
    return command


def build_command(invocation: ProbeInvocation, output_dir: Path, args: argparse.Namespace) -> list[str]:
    command = [
        "python3",
        "-m",
        "tracy",
        "-v",
        "-r",
        "-p",
        "-o",
        str(output_dir),
        "--check-exit-code",
        "--op-support-count",
        str(args.op_support_count),
    ]
    if args.sync_host_device:
        command.append("--sync-host-device")
    if args.port is not None:
        command.extend(["-t", str(args.port)])
    if args.collect_noc_traces:
        command.append("--collect-noc-traces")
    if args.legacy_trigger == "sum":
        command.append("--enable-sum-profiling")
    elif args.legacy_trigger == "no-runtime-analysis":
        command.append("--no-runtime-analysis")
    elif args.legacy_trigger == "perf-fpu":
        perf_counter_groups = args.perf_counter_groups or "fpu"
        command.extend(["--profiler-capture-perf-counters", perf_counter_groups])
    for analysis in DEVICE_ANALYSIS_TYPES:
        command.extend(["-a", analysis])
    command.extend(build_tracy_target_args(invocation, args))
    return command


def build_capture_command(invocation: ProbeInvocation, output_dir: Path, args: argparse.Namespace) -> list[str]:
    if args.capture_mode == "device_only":
        return build_child_command(invocation, args)
    return build_command(invocation, output_dir, args)


def build_postprocess_command(output_dir: Path, args: argparse.Namespace) -> list[str] | None:
    if args.capture_mode != "device_only":
        return None
    command = [
        "python3",
        "tools/tracy/process_ops_logs.py",
        "-o",
        str(output_dir),
        "--device-only",
        "--force-legacy-device-logs",
    ]
    if args.collect_noc_traces:
        command.append("--analyze-noc-traces")
    for analysis in DEVICE_ANALYSIS_TYPES:
        command.extend(["-a", analysis])
    return command


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def terminate_process_group(process: subprocess.Popen, grace_seconds: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.2)

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def run_command_with_timeout(command: list[str], timeout_seconds: int | None, env: dict[str, str] | None = None) -> int:
    process = subprocess.Popen(command, cwd=REPO_ROOT, start_new_session=True, env=env)
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        terminate_process_group(process)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise subprocess.TimeoutExpired(command, timeout_seconds) from exc


def precheck_support(invocation: ProbeInvocation) -> list[str]:
    workload = PRESET_CASES[invocation.preset_case].make_workload()
    baseline = baseline_from_key(invocation.baseline)
    return part1.validate_workload_baseline_support(
        workload,
        baseline,
        None,
        deepseek_wh_cores_per_block=invocation.deepseek_wh_cores_per_block,
    )


def resolve_perf_counter_groups(args: argparse.Namespace) -> str | None:
    if args.legacy_trigger == "perf-fpu":
        return args.perf_counter_groups or "fpu"
    return args.perf_counter_groups


def perf_counter_bitfield(groups: str | None) -> str | None:
    if not groups:
        return None
    counter_group_bits = {
        "fpu": 0,
        "pack": 1,
        "unpack": 2,
        "l1_0": 3,
        "l1_1": 4,
        "instrn": 5,
    }
    bitfield = 0
    for group in [token.strip().lower() for token in groups.split(",") if token.strip()]:
        if group == "all":
            bitfield = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 5)
            break
        if group not in counter_group_bits:
            raise ValueError(f"Unknown perf counter group: {group}")
        bitfield |= 1 << counter_group_bits[group]
    if (bitfield & (1 << 3)) and (bitfield & (1 << 4)):
        raise ValueError("l1_0 and l1_1 cannot be enabled simultaneously")
    return str(bitfield) if bitfield > 0 else None


def build_device_only_env(output_dir: Path, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    env["TTNN_OP_PROFILER"] = "1"
    env["TT_METAL_PROFILER_TRACE_TRACKING"] = "1"
    env["TT_METAL_PROFILER_DIR"] = str(output_dir)
    if args.sync_host_device:
        env["TT_METAL_PROFILER_SYNC"] = "1"
    if args.legacy_trigger == "sum":
        env["TT_METAL_PROFILER_SUM"] = "1"
    perf_counter_groups = resolve_perf_counter_groups(args)
    perf_counter_bitmask = perf_counter_bitfield(perf_counter_groups)
    if perf_counter_bitmask is not None:
        env["TT_METAL_PROFILE_PERF_COUNTERS"] = perf_counter_bitmask
    if args.collect_noc_traces:
        env["TT_METAL_DEVICE_PROFILER_NOC_EVENTS"] = "1"
        env["TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH"] = str((output_dir / ".logs").resolve())
    return env


def write_manifest(
    output_root: Path,
    case_results: list[CaseResult],
    preflight: PreflightResult | None,
    args: argparse.Namespace,
    invocations: list[ProbeInvocation],
) -> Path:
    manifest_path = output_root / "run_manifest.json"
    manifest = {
        "output_root": str(output_root),
        "profile_script": str(PROFILE_SCRIPT),
        "selected_preset_cases": sorted({inv.preset_case for inv in invocations}),
        "selected_groups": sorted({inv.group for inv in invocations}),
        "selected_variants": sorted({inv.variant for inv in invocations}),
        "preflight": asdict(preflight) if preflight is not None else None,
        "runner_args": {
            "device_id": args.device_id,
            "capture_mode": args.capture_mode,
            "op_support_count": args.op_support_count,
            "sync_host_device": args.sync_host_device,
            "legacy_trigger": args.legacy_trigger,
            "perf_counter_groups": args.perf_counter_groups,
            "collect_noc_traces": args.collect_noc_traces,
            "warmup_iterations": args.warmup_iterations,
            "iterations": args.iterations,
        },
        "cases": [asdict(result) for result in case_results],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Part II DeepSeek/TT representative-point tracy probe sweeps."
    )
    parser.add_argument(
        "--preset-cases",
        nargs="+",
        choices=sorted(PRESET_CASES.keys()),
        help="Explicit Part II preset case labels to run.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(PRESET_GROUPS.keys()),
        help="Expand one or more Part II groups (A/B/C/D) into preset cases.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run only A1.")
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Run the default minimal direct-evidence anchor set (A1/A2/A3).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS.keys()),
        default=list(DEFAULT_VARIANTS),
        help="Method variants to run for each preset case.",
    )
    parser.add_argument(
        "--capture-mode",
        choices=CAPTURE_MODE_CHOICES,
        default="device_only",
        help="Capture flow to use. device_only avoids local tracy report hangs by post-processing device logs offline.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for profiler artifacts and reports.",
    )
    parser.add_argument("--device-id", type=int, default=0, help="TT device id.")
    parser.add_argument("--warmup-iterations", type=int, default=None, help="Override preset warmup iterations.")
    parser.add_argument("--iterations", type=int, default=None, help="Override preset measured iterations.")
    parser.add_argument("--op-support-count", type=int, default=4000, help="Profiler op support count.")
    parser.add_argument("--port", type=int, default=None, help="Optional Tracy port override.")
    parser.add_argument(
        "--sync-host-device",
        action="store_true",
        default=False,
        help="Pass --sync-host-device to tracy to reduce host/device capture races.",
    )
    parser.add_argument(
        "--legacy-trigger",
        choices=LEGACY_TRIGGER_CHOICES,
        default="sum",
        help="How to force legacy Python post-processing. Default: sum.",
    )
    parser.add_argument(
        "--perf-counter-groups",
        default=None,
        help="Comma-separated perf counter groups. Only used with --legacy-trigger perf-fpu.",
    )
    parser.add_argument(
        "--collect-noc-traces",
        dest="collect_noc_traces",
        action="store_true",
        default=False,
        help="Collect NoC traces for NOC/DRAM utilization analysis.",
    )
    parser.add_argument(
        "--no-collect-noc-traces",
        dest="collect_noc_traces",
        action="store_false",
        help="Disable NoC trace collection.",
    )
    parser.add_argument("--skip-preflight", action="store_true", help="Skip the plain child-run hardware health check.")
    parser.add_argument(
        "--preflight-timeout-seconds",
        type=int,
        default=180,
        help="Timeout for the plain child-run preflight check.",
    )
    parser.add_argument(
        "--preflight-retries",
        type=int,
        default=2,
        help="How many preflight attempts to make before aborting.",
    )
    parser.add_argument(
        "--preflight-wait-seconds",
        type=int,
        default=20,
        help="How long to wait between failed preflight attempts.",
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=int,
        default=1800,
        help="Timeout for each profiled case. On timeout the full process group is terminated.",
    )
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip cases whose latest ops_perf_results csv already exists.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Do not skip existing outputs.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run only the plain child-run health check and then exit without starting Tracy profiling.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running later cases even if one case fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    preset_cases = resolve_preset_cases(args)
    invocations = [make_invocation(preset_case, variant, args) for preset_case in preset_cases for variant in args.variants]

    print("Selected invocations:")
    for invocation in invocations:
        print(
            f"  - {invocation.case}: group={invocation.group} baseline={invocation.baseline} "
            f"seq={invocation.seq_len} B={invocation.batch} H={invocation.num_heads} "
            f"dqhpc={invocation.deepseek_num_q_heads_per_core} wh_cores={invocation.deepseek_wh_cores_per_block}"
        )

    case_results: list[CaseResult] = []
    preflight_result: PreflightResult | None = None

    support_cache = {invocation.case: precheck_support(invocation) for invocation in invocations}
    runnable_invocations = [inv for inv in invocations if not support_cache[inv.case]]

    if not runnable_invocations:
        for invocation in invocations:
            case_results.append(
                CaseResult(
                    case=invocation.case,
                    preset_case=invocation.preset_case,
                    group=invocation.group,
                    variant=invocation.variant,
                    baseline=invocation.baseline,
                    variant_label=invocation.variant_label,
                    deepseek_wh_cores_per_block=invocation.deepseek_wh_cores_per_block,
                    seq_len=invocation.seq_len,
                    batch=invocation.batch,
                    num_heads=invocation.num_heads,
                    num_kv_heads=invocation.num_kv_heads,
                    value_dim=invocation.value_dim,
                    rope_dim=invocation.rope_dim,
                    config_signature=invocation.config_signature,
                    deepseek_num_q_heads_per_core=invocation.deepseek_num_q_heads_per_core,
                    deepseek_num_q_shards=invocation.deepseek_num_q_shards,
                    warmup_iterations=invocation.warmup_iterations,
                    iterations=invocation.iterations,
                    status="unsupported_precheck",
                    output_dir=str(output_root / invocation.case),
                    unsupported_reasons=support_cache[invocation.case],
                )
            )
        manifest_path = write_manifest(output_root, case_results, preflight_result, args, invocations)
        print("[abort] no runnable invocations after precheck")
        print(f"[manifest] {manifest_path}")
        return 2

    preflight_invocation = runnable_invocations[0]
    preflight_command = build_child_command(preflight_invocation, args)
    if args.dry_run:
        print(f"[preflight] {shell_join(preflight_command)}")
        preflight_result = PreflightResult(
            case=preflight_invocation.case,
            preset_case=preflight_invocation.preset_case,
            variant=preflight_invocation.variant,
            status="dry_run",
            command=preflight_command,
        )
    elif not args.skip_preflight:
        print(f"[preflight] {preflight_invocation.case}")
        print(f"            {shell_join(preflight_command)}")
        attempts = max(1, args.preflight_retries)
        preflight_error = None
        for attempt in range(1, attempts + 1):
            print(f"[preflight] attempt {attempt}/{attempts}")
            try:
                returncode = run_command_with_timeout(preflight_command, args.preflight_timeout_seconds)
            except subprocess.TimeoutExpired:
                preflight_error = f"preflight timed out after {args.preflight_timeout_seconds}s"
            else:
                if returncode == 0:
                    preflight_result = PreflightResult(
                        case=preflight_invocation.case,
                        preset_case=preflight_invocation.preset_case,
                        variant=preflight_invocation.variant,
                        status="passed",
                        command=preflight_command,
                        attempts=attempt,
                    )
                    print("[preflight] passed")
                    break
                preflight_error = f"preflight exited with return code {returncode}"

            if attempt < attempts:
                print(f"[preflight] retrying after {args.preflight_wait_seconds}s")
                time.sleep(args.preflight_wait_seconds)

        if preflight_result is None:
            preflight_result = PreflightResult(
                case=preflight_invocation.case,
                preset_case=preflight_invocation.preset_case,
                variant=preflight_invocation.variant,
                status="failed",
                command=preflight_command,
                error=preflight_error,
                attempts=attempts,
            )
            manifest_path = write_manifest(output_root, case_results, preflight_result, args, invocations)
            print(f"[abort] preflight failed on {preflight_invocation.case}")
            print(f"[manifest] {manifest_path}")
            return 2
    else:
        preflight_result = PreflightResult(
            case=preflight_invocation.case,
            preset_case=preflight_invocation.preset_case,
            variant=preflight_invocation.variant,
            status="skipped",
            command=preflight_command,
        )

    if args.preflight_only:
        manifest_path = write_manifest(output_root, case_results, preflight_result, args, invocations)
        print("[preflight-only] skipping Tracy profiling")
        print(f"[manifest] {manifest_path}")
        return 0

    for invocation in invocations:
        case_output_dir = output_root / invocation.case
        command = build_capture_command(invocation, case_output_dir, args)
        postprocess_command = build_postprocess_command(case_output_dir, args)
        precheck_reasons = support_cache[invocation.case]

        if precheck_reasons:
            case_results.append(
                CaseResult(
                    case=invocation.case,
                    preset_case=invocation.preset_case,
                    group=invocation.group,
                    variant=invocation.variant,
                    baseline=invocation.baseline,
                    variant_label=invocation.variant_label,
                    deepseek_wh_cores_per_block=invocation.deepseek_wh_cores_per_block,
                    seq_len=invocation.seq_len,
                    batch=invocation.batch,
                    num_heads=invocation.num_heads,
                    num_kv_heads=invocation.num_kv_heads,
                    value_dim=invocation.value_dim,
                    rope_dim=invocation.rope_dim,
                    config_signature=invocation.config_signature,
                    deepseek_num_q_heads_per_core=invocation.deepseek_num_q_heads_per_core,
                    deepseek_num_q_shards=invocation.deepseek_num_q_shards,
                    warmup_iterations=invocation.warmup_iterations,
                    iterations=invocation.iterations,
                    status="unsupported_precheck",
                    output_dir=str(case_output_dir),
                    unsupported_reasons=precheck_reasons,
                    command=command,
                    postprocess_command=postprocess_command,
                    error="; ".join(precheck_reasons),
                )
            )
            print(f"[unsupported] {invocation.case}: {'; '.join(precheck_reasons)}")
            continue

        existing_ops_csv = latest_ops_csv(case_output_dir, capture_mode=args.capture_mode)
        if existing_ops_csv is not None and args.skip_existing and not args.force:
            available_required, missing_required, available_optional = verify_columns(
                existing_ops_csv, capture_mode=args.capture_mode
            )
            case_results.append(
                CaseResult(
                    case=invocation.case,
                    preset_case=invocation.preset_case,
                    group=invocation.group,
                    variant=invocation.variant,
                    baseline=invocation.baseline,
                    variant_label=invocation.variant_label,
                    deepseek_wh_cores_per_block=invocation.deepseek_wh_cores_per_block,
                    seq_len=invocation.seq_len,
                    batch=invocation.batch,
                    num_heads=invocation.num_heads,
                    num_kv_heads=invocation.num_kv_heads,
                    value_dim=invocation.value_dim,
                    rope_dim=invocation.rope_dim,
                    config_signature=invocation.config_signature,
                    deepseek_num_q_heads_per_core=invocation.deepseek_num_q_heads_per_core,
                    deepseek_num_q_shards=invocation.deepseek_num_q_shards,
                    warmup_iterations=invocation.warmup_iterations,
                    iterations=invocation.iterations,
                    status="skipped_existing",
                    output_dir=str(case_output_dir),
                    report_dir=str(existing_ops_csv.parent),
                    ops_csv=str(existing_ops_csv),
                    available_required_columns=available_required,
                    missing_required_columns=missing_required,
                    available_optional_noc_columns=available_optional,
                    command=command,
                    postprocess_command=postprocess_command,
                )
            )
            print(f"[skip] {invocation.case}: {existing_ops_csv}")
            continue

        print(f"[run] {invocation.case}")
        print(f"       {shell_join(command)}")
        if postprocess_command is not None:
            print(f"       postprocess: {shell_join(postprocess_command)}")

        if args.dry_run:
            case_results.append(
                CaseResult(
                    case=invocation.case,
                    preset_case=invocation.preset_case,
                    group=invocation.group,
                    variant=invocation.variant,
                    baseline=invocation.baseline,
                    variant_label=invocation.variant_label,
                    deepseek_wh_cores_per_block=invocation.deepseek_wh_cores_per_block,
                    seq_len=invocation.seq_len,
                    batch=invocation.batch,
                    num_heads=invocation.num_heads,
                    num_kv_heads=invocation.num_kv_heads,
                    value_dim=invocation.value_dim,
                    rope_dim=invocation.rope_dim,
                    config_signature=invocation.config_signature,
                    deepseek_num_q_heads_per_core=invocation.deepseek_num_q_heads_per_core,
                    deepseek_num_q_shards=invocation.deepseek_num_q_shards,
                    warmup_iterations=invocation.warmup_iterations,
                    iterations=invocation.iterations,
                    status="dry_run",
                    output_dir=str(case_output_dir),
                    command=command,
                    postprocess_command=postprocess_command,
                )
            )
            continue

        case_output_dir.mkdir(parents=True, exist_ok=True)
        run_env = build_device_only_env(case_output_dir, args) if args.capture_mode == "device_only" else None
        try:
            returncode = run_command_with_timeout(command, args.case_timeout_seconds, env=run_env)
        except subprocess.TimeoutExpired:
            case_results.append(
                CaseResult(
                    case=invocation.case,
                    preset_case=invocation.preset_case,
                    group=invocation.group,
                    variant=invocation.variant,
                    baseline=invocation.baseline,
                    variant_label=invocation.variant_label,
                    deepseek_wh_cores_per_block=invocation.deepseek_wh_cores_per_block,
                    seq_len=invocation.seq_len,
                    batch=invocation.batch,
                    num_heads=invocation.num_heads,
                    num_kv_heads=invocation.num_kv_heads,
                    value_dim=invocation.value_dim,
                    rope_dim=invocation.rope_dim,
                    config_signature=invocation.config_signature,
                    deepseek_num_q_heads_per_core=invocation.deepseek_num_q_heads_per_core,
                    deepseek_num_q_shards=invocation.deepseek_num_q_shards,
                    warmup_iterations=invocation.warmup_iterations,
                    iterations=invocation.iterations,
                    status="timed_out",
                    output_dir=str(case_output_dir),
                    command=command,
                    postprocess_command=postprocess_command,
                    error=f"case timed out after {args.case_timeout_seconds}s",
                )
            )
            if not args.keep_going:
                manifest_path = write_manifest(output_root, case_results, preflight_result, args, invocations)
                print(f"[manifest] {manifest_path}")
                return 124
            continue

        if returncode != 0:
            case_results.append(
                CaseResult(
                    case=invocation.case,
                    preset_case=invocation.preset_case,
                    group=invocation.group,
                    variant=invocation.variant,
                    baseline=invocation.baseline,
                    variant_label=invocation.variant_label,
                    deepseek_wh_cores_per_block=invocation.deepseek_wh_cores_per_block,
                    seq_len=invocation.seq_len,
                    batch=invocation.batch,
                    num_heads=invocation.num_heads,
                    num_kv_heads=invocation.num_kv_heads,
                    value_dim=invocation.value_dim,
                    rope_dim=invocation.rope_dim,
                    config_signature=invocation.config_signature,
                    deepseek_num_q_heads_per_core=invocation.deepseek_num_q_heads_per_core,
                    deepseek_num_q_shards=invocation.deepseek_num_q_shards,
                    warmup_iterations=invocation.warmup_iterations,
                    iterations=invocation.iterations,
                    status="failed",
                    output_dir=str(case_output_dir),
                    command=command,
                    postprocess_command=postprocess_command,
                    error=f"case exited with return code {returncode}",
                )
            )
            if not args.keep_going:
                manifest_path = write_manifest(output_root, case_results, preflight_result, args, invocations)
                print(f"[manifest] {manifest_path}")
                return returncode or 1
            continue

        if postprocess_command is not None:
            postprocess_returncode = run_command_with_timeout(postprocess_command, args.case_timeout_seconds)
            if postprocess_returncode != 0:
                case_results.append(
                    CaseResult(
                        case=invocation.case,
                        preset_case=invocation.preset_case,
                        group=invocation.group,
                        variant=invocation.variant,
                        baseline=invocation.baseline,
                        variant_label=invocation.variant_label,
                        deepseek_wh_cores_per_block=invocation.deepseek_wh_cores_per_block,
                        seq_len=invocation.seq_len,
                        batch=invocation.batch,
                        num_heads=invocation.num_heads,
                        num_kv_heads=invocation.num_kv_heads,
                        value_dim=invocation.value_dim,
                        rope_dim=invocation.rope_dim,
                        config_signature=invocation.config_signature,
                        deepseek_num_q_heads_per_core=invocation.deepseek_num_q_heads_per_core,
                        deepseek_num_q_shards=invocation.deepseek_num_q_shards,
                        warmup_iterations=invocation.warmup_iterations,
                        iterations=invocation.iterations,
                        status="postprocess_failed",
                        output_dir=str(case_output_dir),
                        command=command,
                        postprocess_command=postprocess_command,
                        error=f"postprocess exited with return code {postprocess_returncode}",
                    )
                )
                if not args.keep_going:
                    manifest_path = write_manifest(output_root, case_results, preflight_result, args, invocations)
                    print(f"[manifest] {manifest_path}")
                    return postprocess_returncode or 1
                continue

        ops_csv = latest_ops_csv(case_output_dir, capture_mode=args.capture_mode)
        if ops_csv is None:
            case_results.append(
                CaseResult(
                    case=invocation.case,
                    preset_case=invocation.preset_case,
                    group=invocation.group,
                    variant=invocation.variant,
                    baseline=invocation.baseline,
                    variant_label=invocation.variant_label,
                    deepseek_wh_cores_per_block=invocation.deepseek_wh_cores_per_block,
                    seq_len=invocation.seq_len,
                    batch=invocation.batch,
                    num_heads=invocation.num_heads,
                    num_kv_heads=invocation.num_kv_heads,
                    value_dim=invocation.value_dim,
                    rope_dim=invocation.rope_dim,
                    config_signature=invocation.config_signature,
                    deepseek_num_q_heads_per_core=invocation.deepseek_num_q_heads_per_core,
                    deepseek_num_q_shards=invocation.deepseek_num_q_shards,
                    warmup_iterations=invocation.warmup_iterations,
                    iterations=invocation.iterations,
                    status="missing_ops_csv",
                    output_dir=str(case_output_dir),
                    command=command,
                    postprocess_command=postprocess_command,
                    error="ops_perf_results csv was not found under reports/",
                )
            )
            if not args.keep_going:
                manifest_path = write_manifest(output_root, case_results, preflight_result, args, invocations)
                print(f"[manifest] {manifest_path}")
                return 1
            continue

        available_required, missing_required, available_optional = verify_columns(
            ops_csv, capture_mode=args.capture_mode
        )
        case_results.append(
            CaseResult(
                case=invocation.case,
                preset_case=invocation.preset_case,
                group=invocation.group,
                variant=invocation.variant,
                baseline=invocation.baseline,
                variant_label=invocation.variant_label,
                deepseek_wh_cores_per_block=invocation.deepseek_wh_cores_per_block,
                seq_len=invocation.seq_len,
                batch=invocation.batch,
                num_heads=invocation.num_heads,
                num_kv_heads=invocation.num_kv_heads,
                value_dim=invocation.value_dim,
                rope_dim=invocation.rope_dim,
                config_signature=invocation.config_signature,
                deepseek_num_q_heads_per_core=invocation.deepseek_num_q_heads_per_core,
                deepseek_num_q_shards=invocation.deepseek_num_q_shards,
                warmup_iterations=invocation.warmup_iterations,
                iterations=invocation.iterations,
                status="completed",
                output_dir=str(case_output_dir),
                report_dir=str(ops_csv.parent),
                ops_csv=str(ops_csv),
                available_required_columns=available_required,
                missing_required_columns=missing_required,
                available_optional_noc_columns=available_optional,
                command=command,
                postprocess_command=postprocess_command,
            )
        )
        print(f"[done] {invocation.case}: {ops_csv}")
        if missing_required:
            print(f"       missing required columns: {', '.join(missing_required)}")
        else:
            print(f"       all required {args.capture_mode} columns are present")

    manifest_path = write_manifest(output_root, case_results, preflight_result, args, invocations)
    print(f"[manifest] {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
