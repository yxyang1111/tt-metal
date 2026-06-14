#!/usr/bin/env python3
"""
GPU FlashMLA benchmark — 与 Wormhole 论文 Section 7 对齐的 MLA decode/prefill 基准测试。

运行方式:
    python3 run_benchmark.py                      # 默认全部跑
    python3 run_benchmark.py --skip-prefill        # 只跑 decode
    python3 run_benchmark.py --skip-sdpa           # 不跑 PyTorch SDPA 对照
    python3 run_benchmark.py --batch 1 --iters 20  # 自定义参数
    python3 run_benchmark.py --flash-backend triton --skip-sdpa  # L40/L40S

DeepSeek-V3 MLA shapes (与论文一致):
    H_q=32, H_kv=1, d_qk=576 (d_c=512 + d_r=64), d_v=512, BF16
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

DECODE_SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
PREFILL_SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
LOCAL_DECODE_SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
LOCAL_PREFILL_SEQ_LENS = [256, 512, 1024]

WARMUP = 10
BENCH_ITERS = 50

# DeepSeek-V3 MLA dimensions
D_C = 512       # latent / value dimension
D_R = 64        # RoPE dimension
D_QK = D_C + D_R  # 576
D_V = D_C         # 512
H_Q_DEFAULT = 32
H_KV_DEFAULT = 1
BATCH_DEFAULT = 6
BLOCK_SIZE = 64    # paged KV cache block size


@dataclass(frozen=True)
class BenchConfig:
    batch: int
    h_q: int
    h_kv: int
    d_qk: int
    d_v: int
    block_size: int

    @property
    def label(self) -> str:
        return f"B={self.batch},Hq={self.h_q},Hkv={self.h_kv},dqk={self.d_qk},dv={self.d_v}"


# ═══════════════════════════════════════════════════════════════
# GPU info
# ═══════════════════════════════════════════════════════════════

def gpu_info() -> dict:
    props = torch.cuda.get_device_properties(0)
    cc = torch.cuda.get_device_capability(0)
    return {
        "name": props.name,
        "mem_gb": round(props.total_memory / 1024**3, 2),
        "sm": f"sm{cc[0]}{cc[1]}",
        "sm_count": props.multi_processor_count,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
    }


# ═══════════════════════════════════════════════════════════════
# FlashMLA decode benchmark
# ═══════════════════════════════════════════════════════════════

def _try_import_flash_mla(backend: str = "auto"):
    errors = []
    cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)

    if backend in ("auto", "triton"):
        try:
            from mla_flash_l40 import get_mla_metadata, flash_mla_with_kvcache
            if backend == "triton" or cc[0] < 9:
                return "triton", get_mla_metadata, flash_mla_with_kvcache, None
        except ImportError as e:
            errors.append(f"triton backend import failed: {e}")

    if backend in ("auto", "official"):
        try:
            from flash_mla import get_mla_metadata, flash_mla_with_kvcache
            return "official", get_mla_metadata, flash_mla_with_kvcache, None
        except ImportError as e:
            errors.append(f"official flash_mla import failed: {e}")

    if backend == "auto":
        try:
            from mla_flash_l40 import get_mla_metadata, flash_mla_with_kvcache
            return "triton", get_mla_metadata, flash_mla_with_kvcache, None
        except ImportError as e:
            errors.append(f"triton backend import failed: {e}")

    return None, None, None, "; ".join(errors) if errors else "no FlashMLA backend available"


def build_paged_kv_cache(cfg: BenchConfig, seq_len: int, device: torch.device):
    """Build paged KV cache matching FlashMLA's layout."""
    num_blocks_per_seq = math.ceil(seq_len / cfg.block_size)
    total_blocks = cfg.batch * num_blocks_per_seq

    k_cache = torch.randn(
        total_blocks, cfg.block_size, cfg.h_kv, cfg.d_qk,
        dtype=torch.bfloat16, device=device,
    )
    block_table = torch.arange(
        total_blocks, dtype=torch.int32, device=device,
    ).view(cfg.batch, num_blocks_per_seq)
    cache_seqlens = torch.full(
        (cfg.batch,), seq_len, dtype=torch.int32, device=device,
    )
    return k_cache, block_table, cache_seqlens


def bench_flash_mla_decode(
    cfg: BenchConfig,
    seq_len: int,
    device: torch.device,
    warmup: int,
    iters: int,
    backend: str,
) -> dict:
    """Benchmark FlashMLA dense decode (Q_len=1)."""
    backend_name, get_mla_metadata, flash_mla_with_kvcache, import_error = _try_import_flash_mla(backend)
    if get_mla_metadata is None:
        return {"status": "skip", "error": import_error or "flash_mla not installed"}

    q = torch.randn(cfg.batch, 1, cfg.h_q, cfg.d_qk, dtype=torch.bfloat16, device=device)
    k_cache, block_table, cache_seqlens = build_paged_kv_cache(cfg, seq_len, device)

    sched_meta, _ = get_mla_metadata()
    scale = cfg.d_qk ** -0.5

    def run():
        return flash_mla_with_kvcache(
            q, k_cache, block_table, cache_seqlens, cfg.d_v,
            sched_meta, None, softmax_scale=scale, causal=True,
        )

    try:
        with torch.inference_mode():
            for _ in range(warmup):
                out, lse = run()
                del out, lse
                # metadata 初始化后不能复用到不同形状，所以 warmup 只做一次 init
            torch.cuda.synchronize()

            lats = []
            for _ in range(iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out, lse = run()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                lats.append((t1 - t0) * 1000.0)
                del out, lse

        result = _summarize("flashmla_decode", seq_len, lats)
        result["backend"] = backend_name
        return result
    except Exception as e:
        torch.cuda.empty_cache()
        return {"status": "error", "error": str(e)[:200], "backend": backend_name}
    finally:
        del q, k_cache, block_table, cache_seqlens
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════
# PyTorch SDPA baseline (decode + prefill)
# ═══════════════════════════════════════════════════════════════

def _has_enable_gqa() -> bool:
    try:
        q = torch.zeros(1, 2, 1, 16, device="cpu")
        k = torch.zeros(1, 1, 1, 16, device="cpu")
        v = torch.zeros(1, 1, 1, 16, device="cpu")
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        return True
    except TypeError:
        return False


_GQA_NATIVE = _has_enable_gqa()


def bench_sdpa(
    cfg: BenchConfig,
    seq_len: int,
    mode: str,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict:
    """Benchmark PyTorch SDPA (decode or prefill)."""
    q_len = 1 if mode == "decode" else seq_len
    is_causal = (mode == "prefill")
    scale = cfg.d_qk ** -0.5
    dt = torch.bfloat16

    try:
        q = torch.randn(cfg.batch, cfg.h_q, q_len, cfg.d_qk, dtype=dt, device=device)
        k = torch.randn(cfg.batch, cfg.h_kv, seq_len, cfg.d_qk, dtype=dt, device=device)
        v = torch.randn(cfg.batch, cfg.h_kv, seq_len, cfg.d_v, dtype=dt, device=device)

        if not _GQA_NATIVE and cfg.h_kv != cfg.h_q:
            k = k.expand(cfg.batch, cfg.h_q, seq_len, cfg.d_qk)
            v = v.expand(cfg.batch, cfg.h_q, seq_len, cfg.d_v)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"status": "OOM", "error": str(e)[:120]}

    def run():
        kwargs = dict(scale=scale, is_causal=is_causal)
        if _GQA_NATIVE and cfg.h_kv != cfg.h_q:
            kwargs["enable_gqa"] = True
        return F.scaled_dot_product_attention(q, k, v, **kwargs)

    try:
        with torch.inference_mode():
            for _ in range(warmup):
                out = run(); del out
            torch.cuda.synchronize()

            lats = []
            for _ in range(iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = run()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                lats.append((t1 - t0) * 1000.0)
                del out

        tag = f"sdpa_{mode}"
        return _summarize(tag, seq_len, lats)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"status": "OOM", "error": str(e)[:120]}
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}
    finally:
        del q, k, v
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _summarize(tag: str, seq_len: int, lats: list[float]) -> dict:
    return {
        "kernel": tag,
        "seq_len": seq_len,
        "mean_ms": round(statistics.mean(lats), 4),
        "min_ms": round(min(lats), 4),
        "max_ms": round(max(lats), 4),
        "std_ms": round(statistics.pstdev(lats), 4),
        "samples": len(lats),
        "status": "ok",
    }


def fmt_seq(s: int) -> str:
    return f"{s // 1024}K" if s >= 1024 and s % 1024 == 0 else str(s)


def apply_local_auto(args: argparse.Namespace, info: dict) -> list[str]:
    """Use conservative defaults for desktop GPUs so a local smoke run finishes."""
    notes = []
    mem_gb = float(info.get("mem_gb", 0))

    if args.batch == BATCH_DEFAULT and mem_gb and mem_gb < 24:
        args.batch = 1
        notes.append(f"batch: {BATCH_DEFAULT} -> 1 (GPU memory {mem_gb:.2f} GB)")
    if args.warmup == WARMUP:
        args.warmup = 2
        notes.append(f"warmup: {WARMUP} -> 2")
    if args.iters == BENCH_ITERS:
        args.iters = 5
        notes.append(f"iters: {BENCH_ITERS} -> 5")
    if list(args.decode_seq_lens) == DECODE_SEQ_LENS:
        args.decode_seq_lens = LOCAL_DECODE_SEQ_LENS
        notes.append("decode seq lens: local smoke set up to 32K")
    if list(args.prefill_seq_lens) == PREFILL_SEQ_LENS:
        args.prefill_seq_lens = LOCAL_PREFILL_SEQ_LENS
        notes.append("prefill seq lens: local smoke set up to 1K")

    return notes


def print_table(title: str, rows: list[dict]):
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}")
    print(f"{'Seq Len':>10s}  {'Mean (ms)':>10s}  {'Min (ms)':>10s}  {'Max (ms)':>10s}  {'Std (ms)':>10s}  Status")
    print("-" * 78)
    for r in rows:
        sl = fmt_seq(r["seq_len"])
        if r["status"] == "ok":
            print(f"{sl:>10s}  {r['mean_ms']:10.4f}  {r['min_ms']:10.4f}  {r['max_ms']:10.4f}  {r['std_ms']:10.4f}  ok")
        else:
            err = r.get("error", "")[:50]
            print(f"{sl:>10s}  {'---':>10s}  {'---':>10s}  {'---':>10s}  {'---':>10s}  {r['status']}: {err}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GPU FlashMLA + SDPA benchmark")
    parser.add_argument("--batch", type=int, default=BATCH_DEFAULT)
    parser.add_argument("--h-q", type=int, default=H_Q_DEFAULT)
    parser.add_argument("--h-kv", type=int, default=H_KV_DEFAULT)
    parser.add_argument("--d-c", type=int, default=D_C)
    parser.add_argument("--d-r", type=int, default=D_R)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=BENCH_ITERS)
    parser.add_argument("--decode-seq-lens", nargs="+", type=int, default=DECODE_SEQ_LENS)
    parser.add_argument("--prefill-seq-lens", nargs="+", type=int, default=PREFILL_SEQ_LENS)
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--skip-sdpa", action="store_true", help="跳过 PyTorch SDPA 对照")
    parser.add_argument("--skip-flashmla", action="store_true", help="跳过 FlashMLA")
    parser.add_argument(
        "--local-auto",
        action="store_true",
        help="根据本机 GPU 自动收缩 batch、序列长度和迭代次数，用于先在桌面 GPU 上跑通",
    )
    parser.add_argument(
        "--flash-backend",
        choices=("auto", "triton", "official"),
        default="auto",
        help="FlashMLA 后端；auto 会在 L40/L40S(sm89) 上使用 Triton",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: 未检测到可用 CUDA GPU，无法运行 GPU benchmark")

    device = torch.device("cuda:0")
    info = gpu_info()
    local_notes = apply_local_auto(args, info) if args.local_auto else []

    cfg = BenchConfig(
        batch=args.batch,
        h_q=args.h_q,
        h_kv=args.h_kv,
        d_qk=args.d_c + args.d_r,
        d_v=args.d_c,
        block_size=args.block_size,
    )

    print("=" * 78)
    print("  GPU FlashMLA Benchmark")
    print("=" * 78)
    print(f"  GPU:     {info['name']} ({info['mem_gb']} GB, {info['sm']}, {info['sm_count']} SMs)")
    print(f"  Stack:   PyTorch {info['pytorch']}, CUDA {info['cuda']}")
    print(f"  Config:  {cfg.label}, block_size={cfg.block_size}")
    print(f"  Bench:   warmup={args.warmup}, iters={args.iters}")
    if local_notes:
        print("  Local:   " + "; ".join(local_notes))

    backend_name, get_mla_metadata, _, import_error = _try_import_flash_mla(args.flash_backend)
    flashmla_ok = get_mla_metadata is not None
    if flashmla_ok:
        print(f"  FlashMLA: {backend_name} backend")
    else:
        print(f"  FlashMLA: NOT AVAILABLE — {import_error}")
    print()

    all_results: list[dict] = []

    # ── Decode ────────────────────────────────────────────────
    if not args.skip_decode:
        # FlashMLA decode
        if flashmla_ok and not args.skip_flashmla:
            rows = []
            for sl in sorted(args.decode_seq_lens):
                r = bench_flash_mla_decode(cfg, sl, device, args.warmup, args.iters, args.flash_backend)
                r["seq_len"] = sl
                r["kernel"] = "flashmla_decode"
                rows.append(r)
                all_results.append(r)
            print_table("FlashMLA Decode (Q_len=1)", rows)

        # SDPA decode
        if not args.skip_sdpa:
            rows = []
            for sl in sorted(args.decode_seq_lens):
                r = bench_sdpa(cfg, sl, "decode", device, args.warmup, args.iters)
                r["seq_len"] = sl
                rows.append(r)
                all_results.append(r)
            print_table("PyTorch SDPA Decode (Q_len=1)", rows)

    # ── Prefill ───────────────────────────────────────────────
    if not args.skip_prefill:
        # SDPA prefill (FlashMLA dense prefill 仅 SM100)
        if not args.skip_sdpa:
            rows = []
            for sl in sorted(args.prefill_seq_lens):
                r = bench_sdpa(cfg, sl, "prefill", device, args.warmup, args.iters)
                r["seq_len"] = sl
                rows.append(r)
                all_results.append(r)
            print_table("PyTorch SDPA Prefill (Q_len=L)", rows)

    # ── 保存结果 ──────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "gpu": info,
        "config": asdict(cfg),
        "warmup": args.warmup,
        "iters": args.iters,
        "results": all_results,
    }
    out_path = OUTPUT_DIR / "gpu_flashmla_benchmark.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")

    # ── 生成 LaTeX 片段 ───────────────────────────────────────
    latex_path = OUTPUT_DIR / "gpu_flashmla_latex_table.tex"
    generate_latex(all_results, info, cfg, latex_path)
    print(f"LaTeX 表格: {latex_path}")


# ═══════════════════════════════════════════════════════════════
# LaTeX 输出
# ═══════════════════════════════════════════════════════════════

# Wormhole 论文 Table 1 的参考数据 (B=6)
WH_SYSNAME_DECODE = {
    256: 0.173, 512: 0.181, 1024: 0.207, 2048: 0.235,
    4096: 0.277, 8192: 0.410, 16384: 0.554, 32768: 0.788,
    65536: 1.364, 131072: 2.549,
}
WH_FLASHMLA_DECODE = {
    256: 0.178, 512: 0.188, 1024: 0.217, 2048: 0.251,
    4096: 0.302, 8192: 0.453, 16384: 0.727, 32768: 1.236,
    65536: 2.320, 131072: 4.429,
}
WH_SYSNAME_PREFILL = {
    256: 1.093, 512: 2.617, 1024: 8.491, 2048: 33.30,
    4096: 129.4, 8192: 510.0, 16384: 2028, 32768: 8050,
}


def generate_latex(results: list[dict], info: dict, cfg: BenchConfig, path: Path):
    fmla_decode = {r["seq_len"]: r for r in results if r.get("kernel") == "flashmla_decode" and r["status"] == "ok"}
    sdpa_decode = {r["seq_len"]: r for r in results if r.get("kernel") == "sdpa_decode" and r["status"] == "ok"}
    sdpa_prefill = {r["seq_len"]: r for r in results if r.get("kernel") == "sdpa_prefill" and r["status"] == "ok"}

    all_decode_lens = sorted(set(list(fmla_decode.keys()) + list(sdpa_decode.keys()) + list(WH_SYSNAME_DECODE.keys())))
    all_prefill_lens = sorted(set(list(sdpa_prefill.keys()) + list(WH_SYSNAME_PREFILL.keys())))

    lines = []
    lines.append(r"% Auto-generated by run_benchmark.py")
    lines.append(r"% GPU: " + info["name"] + f" ({info['sm']})")
    lines.append("")

    # ── Decode table ──
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Cross-platform decode latency (ms), $Q_{\mathrm{len}}{=}1$, $B{=}" + str(cfg.batch) + r"$.}")
    lines.append(r"  \label{tab:eval-gpu-decode}")
    lines.append(r"  \begin{tabular}{l rrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Seq Len}")
    lines.append(r"    & \textbf{GPU FlashMLA}")
    lines.append(r"    & \textbf{GPU SDPA}")
    lines.append(r"    & \textbf{\sysname{} (WH)}")
    lines.append(r"    & \textbf{FlashMLA (WH)} \\")
    lines.append(r"    \midrule")
    for sl in all_decode_lens:
        sl_str = fmt_seq(sl)
        cols = []
        vals = []
        # GPU FlashMLA
        if sl in fmla_decode:
            v = fmla_decode[sl]["mean_ms"]
            cols.append(f"{v:.3f}")
            vals.append(v)
        else:
            cols.append("---")
            vals.append(None)
        # GPU SDPA
        if sl in sdpa_decode:
            v = sdpa_decode[sl]["mean_ms"]
            cols.append(f"{v:.3f}")
            vals.append(v)
        else:
            cols.append("---")
            vals.append(None)
        # WH sysname
        if sl in WH_SYSNAME_DECODE:
            v = WH_SYSNAME_DECODE[sl]
            cols.append(f"{v:.3f}")
            vals.append(v)
        else:
            cols.append("---")
            vals.append(None)
        # WH FlashMLA
        if sl in WH_FLASHMLA_DECODE:
            v = WH_FLASHMLA_DECODE[sl]
            cols.append(f"{v:.3f}")
            vals.append(v)
        else:
            cols.append("---")
            vals.append(None)

        # Bold the minimum
        valid_vals = [v for v in vals if v is not None]
        if valid_vals:
            min_v = min(valid_vals)
            cols = [r"\textbf{" + c + "}" if (vals[i] is not None and vals[i] == min_v) else c
                    for i, c in enumerate(cols)]

        lines.append(f"    {sl_str:6s} & " + " & ".join(cols) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # ── Prefill table ──
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Cross-platform prefill latency (ms), $Q_{\mathrm{len}}{=}L$, $B{=}" + str(cfg.batch) + r"$.}")
    lines.append(r"  \label{tab:eval-gpu-prefill}")
    lines.append(r"  \begin{tabular}{l rrr}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Seq Len}")
    lines.append(r"    & \textbf{GPU SDPA}")
    lines.append(r"    & \textbf{\sysname{} (WH)}")
    lines.append(r"    & \textbf{FlashMLA (WH)} \\")
    lines.append(r"    \midrule")
    for sl in all_prefill_lens:
        sl_str = fmt_seq(sl)
        cols = []
        vals = []
        if sl in sdpa_prefill:
            v = sdpa_prefill[sl]["mean_ms"]
            cols.append(f"{v:.1f}")
            vals.append(v)
        else:
            cols.append(r"\emph{OOM}")
            vals.append(None)
        if sl in WH_SYSNAME_PREFILL:
            v = WH_SYSNAME_PREFILL[sl]
            cols.append(f"{v:.1f}")
            vals.append(v)
        else:
            cols.append("---")
            vals.append(None)
        # WH FlashMLA prefill 数据从论文补充 (如有)
        cols.append("---")
        vals.append(None)

        valid_vals = [v for v in vals if v is not None]
        if valid_vals:
            min_v = min(valid_vals)
            cols = [r"\textbf{" + c + "}" if (vals[i] is not None and vals[i] == min_v) else c
                    for i, c in enumerate(cols)]

        lines.append(f"    {sl_str:6s} & " + " & ".join(cols) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
