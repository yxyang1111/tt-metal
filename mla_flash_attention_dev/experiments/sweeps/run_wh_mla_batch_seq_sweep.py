#!/usr/bin/env python3
"""Run WH MLA/FlashMLA/S-FMLA batch x sequence sweeps.

The parent process launches one child process per case so device failures or
OOMs are isolated and the partial result file stays usable.
"""

from __future__ import annotations

import argparse
import csv
import ast
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "wh_batch_seq_sweep"
RESULT_JSON = OUTPUT_DIR / "wh_mla_flash_sfmla_batch_seq_results.json"
RESULT_CSV = OUTPUT_DIR / "wh_mla_flash_sfmla_batch_seq_results.csv"

SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
BATCHES = [1, 2, 4, 8, 16, 32, 64]
METHODS = ["mla", "flash_mla", "sfmla"]
MODES = ["decode", "prefill"]

DRAM_CAPACITY_BYTES = 12 * 1024**3
DEVICE_BUDGET_BYTES = int(DRAM_CAPACITY_BYTES * 0.82)
MLA_BUDGET_BYTES = 8 * 1024**3
PREFILL_SAFE_BUDGET_BYTES = 6 * 1024**3
PREFILL_SAFE_MAX_SEQ_LEN = 4096
PREFILL_SAFE_MAX_TOKENS = 16_384


def import_baseline_module():
    from mla_flash_attention_dev.experiments.baselines.naive_ttnn import run_naive_mla_benchmark

    return run_naive_mla_benchmark


def format_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MiB"
    return f"{n / 1024:.2f} KiB"


def summarize(latencies: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "std_ms": statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
    }


def result_record(
    *,
    method: str,
    mode: str,
    batch: int,
    seq_len: int,
    status: str,
    latencies_ms: list[float] | None = None,
    error: str = "",
    skipped_reason: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "method": method,
        "mode": mode,
        "batch": batch,
        "seq_len": seq_len,
        "status": status,
    }
    if latencies_ms:
        record["latencies_ms"] = [round(v, 6) for v in latencies_ms]
        record["num_samples"] = len(latencies_ms)
        record.update({k: round(v, 6) for k, v in summarize(latencies_ms).items()})
    if error:
        record["error"] = error
    if skipped_reason:
        record["skipped_reason"] = skipped_reason
    if extra:
        record.update(extra)
    return record


def estimate_prefill_device_bytes(batch: int, seq_len: int, *, d_qk: int, d_v: int, h: int) -> int:
    q_bytes = batch * h * seq_len * d_qk * 2
    k_bytes = batch * seq_len * d_qk  # BF8_B device cache.
    out_bytes = batch * h * seq_len * d_v * 2
    page_table_bytes = batch * max(1, seq_len // 64) * 4
    return q_bytes + k_bytes + out_bytes + page_table_bytes


def should_skip_case(method: str, mode: str, batch: int, seq_len: int) -> tuple[bool, str]:
    baseline = import_baseline_module()
    if mode == "decode" and method in {"flash_mla", "sfmla"} and batch > 56:
        return True, "current WH SDPA decode program requires available cores (56) >= batch size"
    if mode == "decode" and method in {"flash_mla", "sfmla"} and batch * seq_len > 2_097_152:
        return True, "current host BF8 cache construction becomes unreliable beyond 2M batch-tokens"

    if method == "mla":
        est = baseline.estimate_naive_dram_bytes(batch, seq_len, mode)
        if mode == "prefill":
            if seq_len > PREFILL_SAFE_MAX_SEQ_LEN:
                return True, f"safe prefill cap: seq_len {seq_len} exceeds {PREFILL_SAFE_MAX_SEQ_LEN}"
            if batch * seq_len > PREFILL_SAFE_MAX_TOKENS:
                return True, f"safe prefill cap: batch*seq_len {batch * seq_len} exceeds {PREFILL_SAFE_MAX_TOKENS}"
            if est > PREFILL_SAFE_BUDGET_BYTES:
                return True, f"estimated MLA prefill memory {format_bytes(est)} exceeds safe budget {format_bytes(PREFILL_SAFE_BUDGET_BYTES)}"
        if est > MLA_BUDGET_BYTES:
            return True, f"estimated MLA device memory {format_bytes(est)} exceeds budget {format_bytes(MLA_BUDGET_BYTES)}"
        return False, ""

    if mode == "prefill":
        if seq_len > PREFILL_SAFE_MAX_SEQ_LEN:
            return True, f"safe prefill cap: seq_len {seq_len} exceeds {PREFILL_SAFE_MAX_SEQ_LEN}"
        if batch * seq_len > PREFILL_SAFE_MAX_TOKENS:
            return True, f"safe prefill cap: batch*seq_len {batch * seq_len} exceeds {PREFILL_SAFE_MAX_TOKENS}"
        est = estimate_prefill_device_bytes(
            batch,
            seq_len,
            d_qk=baseline.D_QK,
            d_v=baseline.KV_LORA_RANK,
            h=baseline.NUM_HEADS,
        )
        if est > PREFILL_SAFE_BUDGET_BYTES:
            return True, f"estimated prefill device memory {format_bytes(est)} exceeds safe budget {format_bytes(PREFILL_SAFE_BUDGET_BYTES)}"
        if est > DEVICE_BUDGET_BYTES:
            return True, f"estimated prefill device memory {format_bytes(est)} exceeds budget {format_bytes(DEVICE_BUDGET_BYTES)}"
    return False, ""


def run_sfmla_decode(device: Any, batch: int, seq_len: int, warmup: int, iters: int) -> dict[str, Any]:
    from mla_flash_attention_dev.sfmla import SFMLAWorkloadConfig, build_decode_inputs, run_decode
    from mla_flash_attention_dev.sfmla.dataflow import derive_mapping, get_sfmla_grid
    from mla_flash_attention_dev.sfmla.runtime.decode import deallocate_decode_inputs

    baseline = import_baseline_module()
    q = torch.randn((1, batch, baseline.NUM_HEADS, baseline.D_QK), dtype=torch.bfloat16)
    k = torch.randn((batch, baseline.NUM_KV_HEADS, seq_len, baseline.D_QK), dtype=torch.bfloat16)

    num_q_heads_per_core = 8
    config = SFMLAWorkloadConfig(
        batch=batch,
        seq_len=seq_len,
        num_heads=baseline.NUM_HEADS,
        num_kv_heads=baseline.NUM_KV_HEADS,
        kv_lora_rank=baseline.KV_LORA_RANK,
        qk_rope_head_dim=baseline.D_ROPE,
        num_q_heads_per_core=num_q_heads_per_core,
        cores_per_block=baseline.MAX_CORES_PER_HEAD_BATCH,
        k_chunk_size=baseline.K_CHUNK_SIZE,
    )

    decode_inputs = None
    try:
        mapping = derive_mapping(config)
        decode_inputs = build_decode_inputs(device, q, k, config)

        for _ in range(warmup):
            out = run_decode(device, decode_inputs)
            ttnn.synchronize_device(device)
            ttnn.deallocate(out)

        latencies: list[float] = []
        for _ in range(iters):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            out = run_decode(device, decode_inputs)
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            ttnn.deallocate(out)

        grid = get_sfmla_grid(cores_per_block=config.cores_per_block)
        return result_record(
            method="sfmla",
            mode="decode",
            batch=batch,
            seq_len=seq_len,
            status="ok",
            latencies_ms=latencies,
            extra={
                "program": "flash_multi_latent_attention_decode",
                "dataflow": "block_lane",
                "N_S": grid.num_s_blocks,
                "C_S": grid.cores_per_lane,
                "B": mapping.num_q_shards,
                "k_chunk_size": baseline.K_CHUNK_SIZE,
            },
        )
    except Exception as e:
        return result_record(
            method="sfmla",
            mode="decode",
            batch=batch,
            seq_len=seq_len,
            status="error",
            error=str(e),
        )
    finally:
        if decode_inputs is not None:
            deallocate_decode_inputs(decode_inputs)


def run_sfmla_prefill(device: Any, batch: int, seq_len: int, warmup: int, iters: int) -> dict[str, Any]:
    import ttnn
    from models.tt_transformers.tt.common import PagedAttentionConfig
    from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import page_table_setup, to_paged_cache

    baseline = import_baseline_module()
    q = torch.randn((batch, baseline.NUM_HEADS, seq_len, baseline.D_QK), dtype=torch.bfloat16)
    k = torch.randn((batch, baseline.NUM_KV_HEADS, seq_len, baseline.D_QK), dtype=torch.bfloat16)

    max_num_blocks = max(1, seq_len // baseline.BLOCK_SIZE) * batch
    paged_cfg = PagedAttentionConfig(block_size=baseline.BLOCK_SIZE, max_num_blocks=max_num_blocks)
    page_table = page_table_setup(batch, paged_cfg)
    paged_cache_torch = to_paged_cache(k, page_table, paged_cfg)

    tt_q = None
    tt_k = None
    tt_page_table = None
    try:
        tt_q = ttnn.from_torch(
            q,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_k = ttnn.from_torch(
            paged_cache_torch,
            device=device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        q_chunk_size = 128
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=q_chunk_size,
            k_chunk_size=baseline.K_CHUNK_SIZE,
            exp_approx_mode=False,
        )
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        def run_once():
            return ttnn.transformer.chunked_flash_mla_prefill(
                tt_q,
                tt_k,
                baseline.KV_LORA_RANK,
                tt_page_table,
                chunk_start_idx=0,
                scale=baseline.SCALE,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        for _ in range(warmup):
            out = run_once()
            ttnn.synchronize_device(device)
            ttnn.deallocate(out)

        latencies: list[float] = []
        for _ in range(iters):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            out = run_once()
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            ttnn.deallocate(out)

        return result_record(
            method="sfmla",
            mode="prefill",
            batch=batch,
            seq_len=seq_len,
            status="ok",
            latencies_ms=latencies,
            extra={"program": "chunked_flash_mla_prefill", "q_chunk_size": q_chunk_size, "k_chunk_size": baseline.K_CHUNK_SIZE},
        )
    finally:
        baseline.safe_dealloc(tt_q, tt_k, tt_page_table)


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    skip, reason = should_skip_case(args.method, args.mode, args.batch, args.seq_len)
    if skip:
        return result_record(
            method=args.method,
            mode=args.mode,
            batch=args.batch,
            seq_len=args.seq_len,
            status="skipped",
            skipped_reason=reason,
        )

    import ttnn

    baseline = import_baseline_module()
    device = None
    try:
        device = ttnn.open_device(device_id=args.device_id)
        if args.method == "mla":
            if args.mode == "decode":
                result = baseline.run_naive_mla_decode(device, args.batch, args.seq_len, args.warmup, args.iters).to_dict()
            else:
                result = baseline.run_naive_mla_prefill(device, args.batch, args.seq_len, args.warmup, args.iters).to_dict()
            result["method"] = "mla"
            return result
        if args.method == "flash_mla":
            if args.mode == "decode":
                result = baseline.run_flash_mla_decode(device, args.batch, args.seq_len, args.warmup, args.iters).to_dict()
            else:
                result = baseline.run_flash_mla_prefill(device, args.batch, args.seq_len, args.warmup, args.iters).to_dict()
            return result
        if args.method == "sfmla":
            if args.mode == "decode":
                return run_sfmla_decode(device, args.batch, args.seq_len, args.warmup, args.iters)
            return run_sfmla_prefill(device, args.batch, args.seq_len, args.warmup, args.iters)
        raise ValueError(f"unknown method: {args.method}")
    except Exception as exc:
        return result_record(
            method=args.method,
            mode=args.mode,
            batch=args.batch,
            seq_len=args.seq_len,
            status="error",
            error=str(exc),
        )
    finally:
        if device is not None:
            try:
                ttnn.close_device(device)
            except Exception:
                pass


def case_key(record: dict[str, Any]) -> tuple[str, str, int, int]:
    return (record["method"], record["mode"], int(record["batch"]), int(record["seq_len"]))


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists() and not RESULT_CSV.exists():
        return []
    if path.exists() and path.stat().st_size > 0:
        try:
            with path.open() as f:
                payload = json.load(f)
            return list(payload.get("results", []))
        except json.JSONDecodeError:
            pass

    if not RESULT_CSV.exists():
        return []
    results: list[dict[str, Any]] = []
    with RESULT_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {key: value for key, value in row.items() if value not in (None, "")}
            for key in ("batch", "seq_len", "num_samples", "q_chunk_size", "k_chunk_size"):
                if key in record:
                    record[key] = int(float(record[key]))
            for key in ("mean_ms", "min_ms", "max_ms", "std_ms"):
                if key in record:
                    record[key] = float(record[key])
            if "latencies_ms" in record:
                try:
                    record["latencies_ms"] = ast.literal_eval(record["latencies_ms"])
                except (SyntaxError, ValueError):
                    pass
            results.append(record)
    return results


def write_outputs(results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = sorted(results, key=lambda r: (r["mode"], r["seq_len"], r["batch"], r["method"]))
    payload = {
        "metadata": {
            "seq_lens": args.seq_lens,
            "batches": args.batches,
            "methods": args.methods,
            "modes": args.modes,
            "warmup": args.warmup,
            "iters": args.iters,
            "device_id": args.device_id,
            "child_timeout_sec": args.child_timeout_sec,
            "device_memory_budget_bytes": DEVICE_BUDGET_BYTES,
            "mla_memory_budget_bytes": MLA_BUDGET_BYTES,
            "prefill_safe_budget_bytes": PREFILL_SAFE_BUDGET_BYTES,
            "prefill_safe_max_seq_len": PREFILL_SAFE_MAX_SEQ_LEN,
            "prefill_safe_max_tokens": PREFILL_SAFE_MAX_TOKENS,
            "notes": [
                "L40S/3 is intentionally excluded; run it on the GPU machine and merge later.",
                "mla is the unfused TT-NN matmul/softmax/matmul baseline.",
                "flash_mla is the TT-NN production FlashMLA path.",
                "sfmla decode uses flash_multi_latent_attention_decode; sfmla prefill uses chunked_flash_mla_prefill with q_chunk_size=128.",
                "Cases estimated to exceed the single-chip memory budget are recorded as skipped.",
                "Long prefill cases may use fewer iterations than the command-line default to keep the sweep tractable.",
                "Conservative prefill safety caps are enabled to avoid server instability after prior crash.",
            ],
        },
        "results": results,
    }
    tmp_json = RESULT_JSON.with_suffix(".json.tmp")
    with tmp_json.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp_json.replace(RESULT_JSON)

    fieldnames: list[str] = []
    for row in results:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    tmp_csv = RESULT_CSV.with_suffix(".csv.tmp")
    with tmp_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in fieldnames})
    tmp_csv.replace(RESULT_CSV)


def run_parent(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_existing(RESULT_JSON) if args.resume else []
    result_by_key = {case_key(r): r for r in existing}

    cases = [
        (method, mode, batch, seq_len)
        for mode in args.modes
        for seq_len in args.seq_lens
        for batch in args.batches
        for method in args.methods
    ]
    total = len(cases)

    for idx, (method, mode, batch, seq_len) in enumerate(cases, start=1):
        key = (method, mode, batch, seq_len)
        if key in result_by_key and not args.force:
            print(f"[{idx}/{total}] skip existing {method} {mode} B={batch} L={seq_len}")
            continue

        print(f"[{idx}/{total}] run {method} {mode} B={batch} L={seq_len}", flush=True)
        warmup, iters = effective_repeats(args, mode, batch, seq_len)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-run",
            "--method",
            method,
            "--mode",
            mode,
            "--batch",
            str(batch),
            "--seq-len",
            str(seq_len),
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--device-id",
            str(args.device_id),
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
                timeout=args.child_timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            record = result_record(
                method=method,
                mode=mode,
                batch=batch,
                seq_len=seq_len,
                status="error",
                error=(
                    f"child timed out after {args.child_timeout_sec}s; "
                    f"stdout={(exc.stdout or '')[-1000:]}; stderr={(exc.stderr or '')[-1000:]}"
                ),
            )
            result_by_key[key] = record
            write_outputs(list(result_by_key.values()), args)
            print("    -> error", flush=True)
            continue

        if completed.returncode != 0:
            record = result_record(
                method=method,
                mode=mode,
                batch=batch,
                seq_len=seq_len,
                status="error",
                error=(completed.stderr or completed.stdout)[-4000:],
            )
        else:
            record = None
            for line in completed.stdout.splitlines():
                line = line.strip()
                if not (line.startswith("{") and line.endswith("}")):
                    continue
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if candidate.get("method") == method and candidate.get("mode") == mode:
                    record = candidate
            if record is None:
                record = result_record(
                    method=method,
                    mode=mode,
                    batch=batch,
                    seq_len=seq_len,
                    status="error",
                    error=f"failed to find child JSON result; stdout={completed.stdout[-2000:]}; stderr={completed.stderr[-2000:]}",
                )
        result_by_key[key] = record
        write_outputs(list(result_by_key.values()), args)
        status = record.get("status")
        mean = record.get("mean_ms")
        suffix = f", mean={mean:.3f} ms" if isinstance(mean, (int, float)) else ""
        print(f"    -> {status}{suffix}", flush=True)

    write_outputs(list(result_by_key.values()), args)
    print(f"Saved JSON: {RESULT_JSON}")
    print(f"Saved CSV:  {RESULT_CSV}")


def effective_repeats(args: argparse.Namespace, mode: str, batch: int, seq_len: int) -> tuple[int, int]:
    if mode != "prefill":
        return args.warmup, args.iters

    tokens = batch * seq_len
    if seq_len >= 8192 or tokens >= 32768:
        return 0, 1
    if tokens >= 8192:
        return min(args.warmup, 1), min(args.iters, 2)
    return args.warmup, args.iters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--batches", nargs="+", type=int, default=BATCHES)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=SEQ_LENS)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--child-timeout-sec", type=int, default=600)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.child_run:
        missing = [name for name in ("method", "mode", "batch", "seq_len") if getattr(args, name.replace("-", "_"), None) is None]
        if missing:
            parser.error(f"--child-run missing required args: {missing}")
    return args


def main() -> None:
    args = parse_args()
    if args.child_run:
        print(json.dumps(run_child(args), ensure_ascii=False))
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
