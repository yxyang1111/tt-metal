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


REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE_SCRIPT = REPO_ROOT / "mla_flash_attention_dev" / "experiments" / "profile_flash_mla_wh_detailed.py"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "mla_flash_attention_dev" / "experiments" / "profiling" / "outputs" / "flash_mla_pm_bubble_probe"
)
DEFAULT_DECODE_CASES = [
    "decode_256",
    "decode_512",
    "decode_1k",
    "decode_2k",
    "decode_4k",
    "decode_8k",
    "decode_16k",
    "decode_32k",
]
MINIMAL_CASES = ["decode_1k", "decode_4k", "decode_32k", "prefill_4k"]
SUPPORTED_PREFILL_CASES = [
    "prefill_256",
    "prefill_512",
    "prefill_1k",
    "prefill_2k",
    "prefill_4k",
    "prefill_8k",
    "prefill_16k",
    "prefill_32k",
]
SUPPORTED_CASES = DEFAULT_DECODE_CASES + SUPPORTED_PREFILL_CASES
SEQ_LEN_TO_CASE = {
    "256": "decode_256",
    "512": "decode_512",
    "1024": "decode_1k",
    "1k": "decode_1k",
    "2048": "decode_2k",
    "2k": "decode_2k",
    "4096": "decode_4k",
    "4k": "decode_4k",
    "8192": "decode_8k",
    "8k": "decode_8k",
    "16384": "decode_16k",
    "16k": "decode_16k",
    "32768": "decode_32k",
    "32k": "decode_32k",
}
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
REQUIRED_COLUMNS = [
    "PM IDEAL [ns]",
    "PM COMPUTE [ns]",
    "PM BANDWIDTH [ns]",
    "PM FPU UTIL (%)",
    "DEVICE COMPUTE CB WAIT FRONT [ns]",
    "DEVICE COMPUTE CB RESERVE BACK [ns]",
]
OPTIONAL_NOC_COLUMNS = [
    "NOC UTIL (%)",
    "MULTICAST NOC UTIL (%)",
    "DRAM BW UTIL (%)",
]
LEGACY_TRIGGER_CHOICES = ["sum", "no-runtime-analysis", "perf-fpu"]


@dataclass
class CaseResult:
    case: str
    status: str
    output_dir: str
    report_dir: str | None = None
    ops_csv: str | None = None
    available_required_columns: list[str] | None = None
    missing_required_columns: list[str] | None = None
    available_optional_noc_columns: list[str] | None = None
    command: list[str] | None = None
    error: str | None = None


@dataclass
class PreflightResult:
    case: str
    status: str
    command: list[str]
    error: str | None = None
    attempts: int = 1


def normalize_seq_token(token: str) -> str:
    normalized = token.strip().lower()
    if normalized not in SEQ_LEN_TO_CASE:
        raise ValueError(f"Unsupported sequence length token: {token}")
    return SEQ_LEN_TO_CASE[normalized]


def resolve_cases(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return args.cases
    if args.seq_lens:
        return [normalize_seq_token(token) for token in args.seq_lens]
    if args.smoke:
        return ["decode_1k"]
    if args.minimal:
        return MINIMAL_CASES
    return DEFAULT_DECODE_CASES


def latest_ops_csv(case_output_dir: Path) -> Path | None:
    reports_dir = case_output_dir / "reports"
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


def verify_columns(path: Path) -> tuple[list[str], list[str], list[str]]:
    header = csv_header(path)
    available_required = [column for column in REQUIRED_COLUMNS if column in header]
    missing_required = [column for column in REQUIRED_COLUMNS if column not in header]
    available_optional = [column for column in OPTIONAL_NOC_COLUMNS if column in header]
    return available_required, missing_required, available_optional


def build_command(case: str, output_dir: Path, args: argparse.Namespace) -> list[str]:
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
    command.extend([str(PROFILE_SCRIPT.relative_to(REPO_ROOT)), "--child-run", "--case", case])
    return command


def build_preflight_command(case: str) -> list[str]:
    return [
        "python3",
        str(PROFILE_SCRIPT.relative_to(REPO_ROOT)),
        "--child-run",
        "--case",
        case,
    ]


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def terminate_process_group(process: subprocess.Popen[bytes], grace_seconds: float = 5.0) -> None:
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


def run_command_with_timeout(command: list[str], timeout_seconds: int | None) -> int:
    process = subprocess.Popen(command, cwd=REPO_ROOT, start_new_session=True)
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        terminate_process_group(process)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise subprocess.TimeoutExpired(command, timeout_seconds) from exc


def write_manifest(output_root: Path, case_results: list[CaseResult], preflight: PreflightResult | None = None) -> Path:
    manifest_path = output_root / "run_manifest.json"
    manifest = {
        "output_root": str(output_root),
        "preflight": asdict(preflight) if preflight is not None else None,
        "cases": [asdict(result) for result in case_results],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FlashMLA PM util / compute bubble probe sweeps.")
    parser.add_argument("--cases", nargs="+", choices=SUPPORTED_CASES, help="Explicit case names to run.")
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        help="Decode sequence lengths to run. Accepts 256 512 1k 2k 4k 8k 16k 32k or numeric equivalents.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run only decode_1k.")
    parser.add_argument("--minimal", action="store_true", help="Run decode_1k/decode_4k/decode_32k/prefill_4k.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for profiler artifacts and reports.",
    )
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
        help="Collect NoC traces for NOC/DRAM utilization analysis. Disabled by default because it is unstable here.",
    )
    parser.add_argument(
        "--no-collect-noc-traces",
        dest="collect_noc_traces",
        action="store_false",
        help="Disable NoC trace collection.",
    )
    parser.add_argument(
        "--preflight-case",
        choices=SUPPORTED_CASES,
        default=None,
        help="Run a plain child-run on this case before starting the sweep. Defaults to the first selected case.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the plain child-run hardware health check.",
    )
    parser.add_argument(
        "--preflight-timeout-seconds",
        type=int,
        default=120,
        help="Timeout for the plain child-run preflight check.",
    )
    parser.add_argument(
        "--preflight-retries",
        type=int,
        default=3,
        help="How many preflight attempts to make before aborting.",
    )
    parser.add_argument(
        "--preflight-wait-seconds",
        type=int,
        default=30,
        help="How long to wait between failed preflight attempts.",
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=int,
        default=900,
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
    cases = resolve_cases(args)

    case_results: list[CaseResult] = []
    preflight_result: PreflightResult | None = None
    print("Selected cases:")
    for case in cases:
        print(f"  - {case}")

    preflight_case = args.preflight_case or cases[0]
    preflight_command = build_preflight_command(preflight_case)
    if args.dry_run:
        print(f"[preflight] {shell_join(preflight_command)}")
        preflight_result = PreflightResult(case=preflight_case, status="dry_run", command=preflight_command)
    elif not args.skip_preflight:
        print(f"[preflight] {preflight_case}")
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
                        case=preflight_case,
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
                case=preflight_case,
                status="failed",
                command=preflight_command,
                error=preflight_error,
                attempts=attempts,
            )
            manifest_path = write_manifest(output_root, case_results, preflight_result)
            print(f"[abort] preflight failed on {preflight_case}")
            print(f"[manifest] {manifest_path}")
            return 2
    else:
        preflight_result = PreflightResult(case=preflight_case, status="skipped", command=preflight_command)

    if args.preflight_only:
        manifest_path = write_manifest(output_root, case_results, preflight_result)
        print("[preflight-only] skipping Tracy profiling")
        print(f"[manifest] {manifest_path}")
        return 0

    for case in cases:
        case_output_dir = output_root / case
        existing_ops_csv = latest_ops_csv(case_output_dir)
        command = build_command(case, case_output_dir, args)

        if existing_ops_csv is not None and args.skip_existing and not args.force:
            available_required, missing_required, available_optional = verify_columns(existing_ops_csv)
            result = CaseResult(
                case=case,
                status="skipped_existing",
                output_dir=str(case_output_dir),
                report_dir=str(existing_ops_csv.parent),
                ops_csv=str(existing_ops_csv),
                available_required_columns=available_required,
                missing_required_columns=missing_required,
                available_optional_noc_columns=available_optional,
                command=command,
            )
            case_results.append(result)
            print(f"[skip] {case}: {existing_ops_csv}")
            continue

        print(f"[run] {case}")
        print(f"       {shell_join(command)}")

        if args.dry_run:
            case_results.append(
                CaseResult(
                    case=case,
                    status="dry_run",
                    output_dir=str(case_output_dir),
                    command=command,
                )
            )
            continue

        case_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            returncode = run_command_with_timeout(command, args.case_timeout_seconds)
        except subprocess.TimeoutExpired:
            result = CaseResult(
                case=case,
                status="timed_out",
                output_dir=str(case_output_dir),
                command=command,
                error=f"case timed out after {args.case_timeout_seconds}s",
            )
            case_results.append(result)
            if not args.keep_going:
                manifest_path = write_manifest(output_root, case_results, preflight_result)
                print(f"[manifest] {manifest_path}")
                return 124
            continue
        if returncode != 0:
            result = CaseResult(
                case=case,
                status="failed",
                output_dir=str(case_output_dir),
                command=command,
                error=f"case exited with return code {returncode}",
            )
            case_results.append(result)
            if not args.keep_going:
                manifest_path = write_manifest(output_root, case_results, preflight_result)
                print(f"[manifest] {manifest_path}")
                return returncode or 1
            continue

        ops_csv = latest_ops_csv(case_output_dir)
        if ops_csv is None:
            result = CaseResult(
                case=case,
                status="missing_ops_csv",
                output_dir=str(case_output_dir),
                command=command,
                error="ops_perf_results csv was not found under reports/",
            )
            case_results.append(result)
            if not args.keep_going:
                manifest_path = write_manifest(output_root, case_results, preflight_result)
                print(f"[manifest] {manifest_path}")
                return 1
            continue

        available_required, missing_required, available_optional = verify_columns(ops_csv)
        result = CaseResult(
            case=case,
            status="completed",
            output_dir=str(case_output_dir),
            report_dir=str(ops_csv.parent),
            ops_csv=str(ops_csv),
            available_required_columns=available_required,
            missing_required_columns=missing_required,
            available_optional_noc_columns=available_optional,
            command=command,
        )
        case_results.append(result)
        print(f"[done] {case}: {ops_csv}")
        if missing_required:
            print(f"       missing required columns: {', '.join(missing_required)}")
        else:
            print("       all required PM/bubble columns are present")

    manifest_path = write_manifest(output_root, case_results, preflight_result)
    print(f"[manifest] {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
