"""Refined offline analytical autotuner for MLA planning.

This version extends the initial skeleton with:
- explicit S-Block topology objects for method-B style decode
- compile-time vs runtime plan split
- bucketed policy cache keys
- optional top-K measured reranking
- workload-aware heuristic pruning for redundant search axes
- reserve/backpressure and sender-hotspot latency penalties
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Sequence

Mode = Literal["prefill", "decode"]
TopologyMode = Literal["independent", "sblock_multicast", "tree"]
LayoutPolicy = Literal["default", "row_packed_by_head", "bandwidth_balanced"]
MathFidelity = Literal["LoFi", "HiFi2", "HiFi4"]
PageSizeStrategy = Literal["auto_max_divisor", "auto_half", "auto_quarter"]
CacheKeyMode = Literal["exact", "sequence_pow2"]


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _soft_knee_penalty(value: float, *, knee: float, span: float, scale: float) -> float:
    if value <= knee:
        return 0.0
    normalized = (value - knee) / max(span, 1e-9)
    return scale * normalized * normalized


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return _ceil_div(value, multiple) * multiple


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _semantic_workload_payload(workload: MLAWorkloadSpec) -> dict[str, Any]:
    payload = asdict(workload)
    payload.pop("name", None)
    return payload


def _semantic_hardware_payload(hardware: HardwareTopologySpec) -> dict[str, Any]:
    payload = asdict(hardware)
    payload.pop("name", None)
    return payload


def _slack_ratio(total: int, parallel_factor: int) -> float:
    if total <= 0 or parallel_factor <= 0:
        return 1.0
    padded = _ceil_div(total, parallel_factor) * parallel_factor
    return (padded - total) / total


def _fidelity_flops_per_cycle(math_fidelity: MathFidelity) -> float:
    mapping = {
        "LoFi": 4096.0,
        "HiFi2": 2048.0,
        "HiFi4": 1024.0,
    }
    return mapping[math_fidelity]


def _largest_aligned_divisor(total_bytes: int, upper_limit: int, alignment: int = 32) -> int:
    candidate = upper_limit - (upper_limit % alignment)
    while candidate >= alignment:
        if total_bytes % candidate == 0:
            return candidate
        candidate -= alignment
    return alignment


def _select_page_size(total_bytes: int, max_page_size: int, strategy: PageSizeStrategy) -> tuple[int, int]:
    max_divisor = _largest_aligned_divisor(total_bytes, max_page_size)
    if strategy == "auto_max_divisor":
        page_size = max_divisor
    elif strategy == "auto_half":
        page_size = _largest_aligned_divisor(total_bytes, max(32, max_divisor // 2))
    else:
        page_size = _largest_aligned_divisor(total_bytes, max(32, max_divisor // 4))
    return page_size, total_bytes // page_size


def _tupleize_tree_order(tree_order: Sequence[Sequence[Sequence[int]]]) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(tuple((int(dst), int(src)) for dst, src in step) for step in tree_order)


@dataclass(frozen=True)
class MLAWorkloadSpec:
    name: str = "anonymous"
    mode: Mode = "decode"
    causal: bool = False
    paged: bool = False
    batch_size: int = 1
    seq_len_q: int = 1
    seq_len_kv: int = 4096
    num_q_heads: int = 32
    num_kv_heads: int = 1
    kv_lora_rank: int = 512
    d_rope: int = 64
    head_dim_qk: int | None = None
    head_dim_v: int | None = None
    q_dtype_bytes: int = 2
    kv_dtype_bytes: int = 1
    output_dtype_bytes: int = 2
    block_size: int = 64

    def __post_init__(self) -> None:
        if self.mode not in {"prefill", "decode"}:
            raise ValueError(f"unsupported mode: {self.mode}")
        if self.batch_size <= 0 or self.seq_len_q <= 0 or self.seq_len_kv <= 0:
            raise ValueError("batch_size, seq_len_q and seq_len_kv must be positive")
        if self.num_q_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("num_q_heads and num_kv_heads must be positive")
        if self.kv_lora_rank <= 0 or self.d_rope < 0:
            raise ValueError("kv_lora_rank must be positive and d_rope must be non-negative")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.head_dim_qk is None:
            object.__setattr__(self, "head_dim_qk", self.kv_lora_rank + self.d_rope)
        if self.head_dim_v is None:
            object.__setattr__(self, "head_dim_v", self.kv_lora_rank)
        if self.head_dim_qk <= 0 or self.head_dim_v <= 0:
            raise ValueError("resolved head dimensions must be positive")

    def q_num_chunks(self, q_chunk_size: int) -> int:
        return _ceil_div(self.seq_len_q, q_chunk_size)

    def kv_num_chunks(self, k_chunk_size: int) -> int:
        return _ceil_div(self.seq_len_kv, k_chunk_size)


@dataclass(frozen=True)
class HardwareTopologySpec:
    name: str
    arch: str
    compute_grid_x: int
    compute_grid_y: int
    dram_bank_endpoints: int
    dram_bandwidth_GBs: float
    noc_bandwidth_GBs: float
    l1_bytes_per_core: int
    clock_GHz: float
    max_noc_page_size: int
    max_trid_window: int
    noc_count: int = 2
    num_devices: int = 1
    supports_multicast: bool = True
    supports_tree: bool = True
    supports_dual_noc: bool = True

    def __post_init__(self) -> None:
        if self.compute_grid_x <= 0 or self.compute_grid_y <= 0:
            raise ValueError("compute grid dimensions must be positive")
        if self.dram_bank_endpoints <= 0 or self.noc_count <= 0 or self.num_devices <= 0:
            raise ValueError("dram_bank_endpoints, noc_count, and num_devices must be positive")
        if self.dram_bandwidth_GBs <= 0 or self.noc_bandwidth_GBs <= 0 or self.clock_GHz <= 0:
            raise ValueError("bandwidths and clock must be positive")
        if self.l1_bytes_per_core <= 0 or self.max_noc_page_size <= 0 or self.max_trid_window <= 0:
            raise ValueError("L1/page size/TRID values must be positive")

    @property
    def total_cores_per_device(self) -> int:
        return self.compute_grid_x * self.compute_grid_y

    @property
    def total_cores(self) -> int:
        return self.total_cores_per_device * self.num_devices


@dataclass(frozen=True)
class SBlockTopologySpec:
    name: str
    arch: str
    num_s_blocks: int
    cores_per_s_block: int
    block_shape: tuple[int, int]
    bank_map: tuple[int, ...]
    tree_order: tuple[tuple[tuple[int, int], ...], ...]
    description: str = ""

    def __post_init__(self) -> None:
        if self.num_s_blocks <= 0 or self.cores_per_s_block <= 0:
            raise ValueError("num_s_blocks and cores_per_s_block must be positive")
        if len(self.bank_map) != self.num_s_blocks:
            raise ValueError("bank_map length must equal num_s_blocks")

    @property
    def tree_depth(self) -> int:
        return len(self.tree_order)


@dataclass(frozen=True)
class SearchSpace:
    batch_parallel_factors: tuple[int, ...] = (1, 2, 4)
    head_parallel_factors: tuple[int, ...] = (1, 2, 4, 8)
    q_parallel_factors: tuple[int, ...] = (1, 2, 4)
    kv_parallel_factors: tuple[int, ...] = (1, 2, 4, 6, 8)
    device_parallel_factors: tuple[int, ...] = (1, 2, 4)
    q_chunk_sizes: tuple[int, ...] = (32, 64, 128, 256)
    k_chunk_sizes: tuple[int, ...] = (64, 128, 256)
    lane_group_capacities: tuple[int, ...] = (4, 6, 8)
    layout_policies: tuple[LayoutPolicy, ...] = ("default", "row_packed_by_head", "bandwidth_balanced")
    topology_modes: tuple[TopologyMode, ...] = ("independent", "sblock_multicast", "tree")
    page_size_strategies: tuple[PageSizeStrategy, ...] = ("auto_max_divisor", "auto_half")
    pipeline_depths: tuple[int, ...] = (1, 2, 3)
    trid_window_candidates: tuple[int, ...] = (4, 8, 14)
    k_cb_depths: tuple[int, ...] = (2, 3)
    v_cb_depths: tuple[int, ...] = (2, 3)
    math_fidelities: tuple[MathFidelity, ...] = ("LoFi", "HiFi2", "HiFi4")
    dual_noc_policies: tuple[bool, ...] = (False, True)
    sblock_topology_names: tuple[str, ...] = ("wh_sblock_6x4", "bh_sblock_8x8")
    decode_heads_per_lane_cap: int = 8


@dataclass(frozen=True)
class CostWeights:
    latency: float = 1.0
    dram_traffic: float = 0.02
    noc_traffic: float = 0.02
    inactive_core_penalty: float = 0.15
    imbalance_penalty: float = 0.25
    l1_overuse_penalty: float = 10.0


@dataclass(frozen=True)
class PolicyBucketConfig:
    mode: CacheKeyMode = "sequence_pow2"
    batch_multiple: int = 1
    head_multiple: int = 8

    def normalize_workload(self, workload: MLAWorkloadSpec) -> dict[str, Any]:
        if self.mode == "exact":
            return _semantic_workload_payload(workload)
        return {
            "mode": workload.mode,
            "causal": workload.causal,
            "paged": workload.paged,
            "batch_size": _round_up_to_multiple(workload.batch_size, self.batch_multiple),
            "seq_len_q": _next_power_of_two(workload.seq_len_q),
            "seq_len_kv": _next_power_of_two(workload.seq_len_kv),
            "num_q_heads": _round_up_to_multiple(workload.num_q_heads, self.head_multiple),
            "num_kv_heads": workload.num_kv_heads,
            "kv_lora_rank": workload.kv_lora_rank,
            "d_rope": workload.d_rope,
            "head_dim_qk": workload.head_dim_qk,
            "head_dim_v": workload.head_dim_v,
            "q_dtype_bytes": workload.q_dtype_bytes,
            "kv_dtype_bytes": workload.kv_dtype_bytes,
            "output_dtype_bytes": workload.output_dtype_bytes,
            "block_size": workload.block_size,
        }

    def normalize_hardware(self, hardware: HardwareTopologySpec) -> dict[str, Any]:
        if self.mode == "exact":
            return _semantic_hardware_payload(hardware)
        return {
            "arch": hardware.arch,
            "compute_grid_x": hardware.compute_grid_x,
            "compute_grid_y": hardware.compute_grid_y,
            "dram_bank_endpoints": hardware.dram_bank_endpoints,
            "noc_count": hardware.noc_count,
            "num_devices": hardware.num_devices,
        }


@dataclass(frozen=True)
class ParallelismPlan5D:
    batch_parallel_factor: int
    head_parallel_factor: int
    q_parallel_factor: int
    kv_parallel_factor: int
    device_parallel_factor: int

    @property
    def total_parallelism(self) -> int:
        return (
            self.batch_parallel_factor
            * self.head_parallel_factor
            * self.q_parallel_factor
            * self.kv_parallel_factor
            * self.device_parallel_factor
        )


@dataclass(frozen=True)
class GroupingPlan:
    batch_group_size: int
    head_group_size: int
    q_chunk_group_size: int
    kv_chunk_group_size: int
    active_lane_count: int
    provisioned_kv_parallel_factor: int
    effective_kv_parallel_factor: int
    effective_s_block_count: int


@dataclass(frozen=True)
class CompileTimeConfig:
    q_chunk_size: int
    k_chunk_size: int
    layout_policy: LayoutPolicy
    topology_mode: TopologyMode
    sblock_topology_name: str | None
    lane_group_capacity: int
    pipeline_depth: int
    trid_window: int
    page_size_strategy: PageSizeStrategy
    k_page_size_bytes: int
    k_num_pages: int
    k_cb_depth: int
    v_cb_depth: int
    dual_noc_policy: bool
    math_fidelity: MathFidelity
    bank_map: tuple[int, ...] = ()
    tree_order: tuple[tuple[tuple[int, int], ...], ...] = ()


@dataclass(frozen=True)
class RuntimeConfig:
    parallelism_5d: ParallelismPlan5D
    grouping: GroupingPlan
    use_balanced_q_parallel: bool
    reduction_tree_depth: int
    k_reuse_groups: int


@dataclass(frozen=True)
class PlanMetrics:
    analytical_score: float
    selected_latency_ms: float
    estimated_latency_ms: float
    measured_latency_ms: float | None
    selection_source: str
    q_preamble_ms: float
    compute_ms: float
    dram_ms: float
    noc_ms: float
    sync_ms: float
    dram_bytes: float
    noc_bytes: float
    l1_usage_bytes_per_core: int
    l1_usage_ratio: float
    active_cores: int
    provisioned_cores: int
    active_core_ratio: float
    provisioned_core_ratio: float
    imbalance_score: float
    q_num_chunks: int
    kv_num_chunks: int
    k_reuse_groups: int
    reader_ms: float
    reduction_ms: float
    reader_backpressure_ms: float
    writer_backpressure_ms: float
    sender_hotspot_ms: float
    overlap_residual_ms: float
    page_control_ms: float
    pipeline_bubble_ms: float
    l1_pressure_ms: float
    complexity_penalty_ms: float
    dual_noc_effectiveness: float
    unused_trid_ratio: float


@dataclass(frozen=True)
class PlanCandidate:
    candidate_key: str
    compile_time_config: CompileTimeConfig
    runtime_config: RuntimeConfig
    metrics: PlanMetrics

    @property
    def parallelism_5d(self) -> ParallelismPlan5D:
        return self.runtime_config.parallelism_5d

    @property
    def q_chunk_size(self) -> int:
        return self.compile_time_config.q_chunk_size

    @property
    def k_chunk_size(self) -> int:
        return self.compile_time_config.k_chunk_size

    @property
    def lane_group_capacity(self) -> int:
        return self.compile_time_config.lane_group_capacity

    @property
    def layout_policy(self) -> LayoutPolicy:
        return self.compile_time_config.layout_policy

    @property
    def topology_mode(self) -> TopologyMode:
        return self.compile_time_config.topology_mode

    @property
    def pipeline_depth(self) -> int:
        return self.compile_time_config.pipeline_depth

    @property
    def dual_noc_policy(self) -> bool:
        return self.compile_time_config.dual_noc_policy

    @property
    def math_fidelity(self) -> MathFidelity:
        return self.compile_time_config.math_fidelity

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlanCandidate":
        compile_time = payload["compile_time_config"]
        runtime = payload["runtime_config"]
        grouping = runtime["grouping"]
        metric_payload = {
            "reader_ms": 0.0,
            "reduction_ms": 0.0,
            "reader_backpressure_ms": 0.0,
            "writer_backpressure_ms": 0.0,
            "sender_hotspot_ms": 0.0,
            "overlap_residual_ms": 0.0,
            "page_control_ms": 0.0,
            "pipeline_bubble_ms": 0.0,
            "l1_pressure_ms": 0.0,
            "complexity_penalty_ms": 0.0,
            "dual_noc_effectiveness": 0.0,
            "unused_trid_ratio": 0.0,
            **payload["metrics"],
        }
        return cls(
            candidate_key=payload["candidate_key"],
            compile_time_config=CompileTimeConfig(
                q_chunk_size=compile_time["q_chunk_size"],
                k_chunk_size=compile_time["k_chunk_size"],
                layout_policy=compile_time["layout_policy"],
                topology_mode=compile_time["topology_mode"],
                sblock_topology_name=compile_time["sblock_topology_name"],
                lane_group_capacity=compile_time["lane_group_capacity"],
                pipeline_depth=compile_time["pipeline_depth"],
                trid_window=compile_time["trid_window"],
                page_size_strategy=compile_time["page_size_strategy"],
                k_page_size_bytes=compile_time["k_page_size_bytes"],
                k_num_pages=compile_time["k_num_pages"],
                k_cb_depth=compile_time["k_cb_depth"],
                v_cb_depth=compile_time["v_cb_depth"],
                dual_noc_policy=compile_time["dual_noc_policy"],
                math_fidelity=compile_time["math_fidelity"],
                bank_map=tuple(compile_time.get("bank_map", [])),
                tree_order=_tupleize_tree_order(compile_time.get("tree_order", [])),
            ),
            runtime_config=RuntimeConfig(
                parallelism_5d=ParallelismPlan5D(**runtime["parallelism_5d"]),
                grouping=GroupingPlan(**grouping),
                use_balanced_q_parallel=runtime["use_balanced_q_parallel"],
                reduction_tree_depth=runtime["reduction_tree_depth"],
                k_reuse_groups=runtime["k_reuse_groups"],
            ),
            metrics=PlanMetrics(**metric_payload),
        )


@dataclass(frozen=True)
class TuningResult:
    workload_signature: str
    policy_cache_key: str
    cache_key_mode: CacheKeyMode
    cache_hit: bool
    cache_source_signature: str | None
    reranked_top_k: int
    measured_candidate_count: int
    workload: MLAWorkloadSpec
    hardware: HardwareTopologySpec
    best_plan: PlanCandidate
    top_candidates: tuple[PlanCandidate, ...]
    searched_candidate_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_signature": self.workload_signature,
            "policy_cache_key": self.policy_cache_key,
            "cache_key_mode": self.cache_key_mode,
            "cache_hit": self.cache_hit,
            "cache_source_signature": self.cache_source_signature,
            "reranked_top_k": self.reranked_top_k,
            "measured_candidate_count": self.measured_candidate_count,
            "workload": asdict(self.workload),
            "hardware": asdict(self.hardware),
            "best_plan": self.best_plan.to_dict(),
            "top_candidates": [candidate.to_dict() for candidate in self.top_candidates],
            "searched_candidate_count": self.searched_candidate_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TuningResult":
        return cls(
            workload_signature=payload["workload_signature"],
            policy_cache_key=payload.get("policy_cache_key", payload["workload_signature"]),
            cache_key_mode=payload.get("cache_key_mode", "exact"),
            cache_hit=payload["cache_hit"],
            cache_source_signature=payload.get("cache_source_signature"),
            reranked_top_k=payload.get("reranked_top_k", 0),
            measured_candidate_count=payload.get("measured_candidate_count", 0),
            workload=MLAWorkloadSpec(**payload["workload"]),
            hardware=HardwareTopologySpec(**payload["hardware"]),
            best_plan=PlanCandidate.from_dict(payload["best_plan"]),
            top_candidates=tuple(PlanCandidate.from_dict(item) for item in payload["top_candidates"]),
            searched_candidate_count=payload["searched_candidate_count"],
        )


@dataclass
class PolicyCache:
    path: Path
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "PolicyCache":
        cache_path = Path(path)
        if cache_path.exists():
            records = json.loads(cache_path.read_text())
            return cls(path=cache_path, records=records)
        return cls(path=cache_path)

    def get(self, cache_key: str) -> TuningResult | None:
        payload = self.records.get(cache_key)
        if payload is None:
            return None
        return TuningResult.from_dict(payload)

    def put(self, result: TuningResult) -> None:
        self.records[result.policy_cache_key] = result.to_dict()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, indent=2, sort_keys=True))


@dataclass
class MeasurementDB:
    path: Path
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "MeasurementDB":
        db_path = Path(path)
        records: dict[str, dict[str, Any]] = {}
        if db_path.exists():
            payload = json.loads(db_path.read_text())
            for key, value in payload.items():
                if isinstance(value, dict):
                    record = dict(value)
                    record["latency_ms"] = float(value["latency_ms"])
                    records[key] = record
                else:
                    records[key] = {"latency_ms": float(value)}
        return cls(path=db_path, records=records)

    def get(self, candidate_key: str) -> float | None:
        payload = self.records.get(candidate_key)
        if payload is None:
            return None
        return float(payload["latency_ms"])

    def put(self, candidate_key: str, latency_ms: float, metadata: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"latency_ms": float(latency_ms)}
        if metadata:
            payload.update(metadata)
        self.records[candidate_key] = payload

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, indent=2, sort_keys=True))


class BasicMLAAutotuner:
    """Offline analytical autotuner for MLA plan selection."""

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        weights: CostWeights | None = None,
        *,
        heuristic_pruning: bool = True,
    ):
        self.search_space = search_space or SearchSpace()
        self.weights = weights or CostWeights()
        self.heuristic_pruning = heuristic_pruning

    def tune(
        self,
        workload: MLAWorkloadSpec,
        hardware: HardwareTopologySpec,
        *,
        top_k: int = 5,
        policy_cache: PolicyCache | None = None,
        bucket_config: PolicyBucketConfig | None = None,
        measurement_db: MeasurementDB | None = None,
        rerank_top_k: int = 0,
        calibrated_cost_model: object | None = None,
    ) -> TuningResult:
        bucket_config = bucket_config or PolicyBucketConfig()
        workload_signature = self._exact_signature(workload, hardware)
        policy_cache_key = self._policy_cache_key(workload, hardware, bucket_config)

        if policy_cache is not None:
            cached = policy_cache.get(policy_cache_key)
            if cached is not None:
                best_plan = cached.best_plan
                top_candidates = cached.top_candidates[:top_k]
                if calibrated_cost_model is not None:
                    from .cost_model import apply_calibration_to_candidate

                    best_plan = apply_calibration_to_candidate(
                        best_plan, calibrated_cost_model, cost_weights=self.weights
                    )
                    top_candidates = tuple(
                        apply_calibration_to_candidate(c, calibrated_cost_model, cost_weights=self.weights)
                        for c in top_candidates
                    )
                return TuningResult(
                    workload_signature=workload_signature,
                    policy_cache_key=policy_cache_key,
                    cache_key_mode=bucket_config.mode,
                    cache_hit=True,
                    cache_source_signature=cached.workload_signature,
                    reranked_top_k=cached.reranked_top_k,
                    measured_candidate_count=cached.measured_candidate_count,
                    workload=workload,
                    hardware=hardware,
                    best_plan=best_plan,
                    top_candidates=top_candidates,
                    searched_candidate_count=cached.searched_candidate_count,
                )

        candidates = self._enumerate_candidates(workload, hardware)
        if not candidates:
            raise RuntimeError("no feasible candidate found")

        if calibrated_cost_model is not None:
            from .cost_model import apply_calibration_to_candidate

            candidates = [
                apply_calibration_to_candidate(c, calibrated_cost_model, cost_weights=self.weights)
                for c in candidates
            ]

        candidates.sort(
            key=lambda candidate: (
                candidate.metrics.analytical_score,
                candidate.metrics.selected_latency_ms,
                candidate.metrics.l1_pressure_ms,
                candidate.metrics.complexity_penalty_ms,
                candidate.metrics.l1_usage_ratio,
                candidate.candidate_key,
            )
        )
        measured_count = 0
        if measurement_db is not None and rerank_top_k > 0:
            candidates, measured_count = self._apply_measurement_rerank(
                workload, hardware, candidates, measurement_db, rerank_top_k
            )

        result = TuningResult(
            workload_signature=workload_signature,
            policy_cache_key=policy_cache_key,
            cache_key_mode=bucket_config.mode,
            cache_hit=False,
            cache_source_signature=None,
            reranked_top_k=max(0, rerank_top_k),
            measured_candidate_count=measured_count,
            workload=workload,
            hardware=hardware,
            best_plan=candidates[0],
            top_candidates=tuple(candidates[:top_k]),
            searched_candidate_count=len(candidates),
        )
        if policy_cache is not None:
            policy_cache.put(result)
            policy_cache.save()
        return result

    def _enumerate_candidates(
        self,
        workload: MLAWorkloadSpec,
        hardware: HardwareTopologySpec,
    ) -> list[PlanCandidate]:
        batch_factors = self._bounded_factors(self.search_space.batch_parallel_factors, workload.batch_size)
        head_factors = self._candidate_head_factors(
            workload, self._bounded_factors(self.search_space.head_parallel_factors, workload.num_q_heads)
        )
        device_factors = self._bounded_factors(self.search_space.device_parallel_factors, hardware.num_devices)

        candidates: list[PlanCandidate] = []
        q_chunk_sizes = self.search_space.q_chunk_sizes
        if workload.mode == "decode" and workload.seq_len_q <= min(self.search_space.q_chunk_sizes):
            q_chunk_sizes = (min(self.search_space.q_chunk_sizes),)

        topology_modes = self._candidate_topology_modes(workload)
        for q_chunk_size in q_chunk_sizes:
            q_num_chunks = workload.q_num_chunks(q_chunk_size)
            q_factors = self._bounded_factors(self.search_space.q_parallel_factors, q_num_chunks)
            if workload.mode == "decode":
                q_factors = (1,)

            for k_chunk_size in self.search_space.k_chunk_sizes:
                kv_num_chunks = workload.kv_num_chunks(k_chunk_size)
                base_kv_factors = self._bounded_factors(self.search_space.kv_parallel_factors, kv_num_chunks)
                if workload.mode == "prefill":
                    base_kv_factors = (1,)

                for topology_mode in topology_modes:
                    topology_choices = self._topology_choices(workload, hardware, topology_mode)
                    if not topology_choices:
                        continue

                    for topology in topology_choices:
                        kv_factors = (
                            (topology.num_s_blocks,)
                            if topology is not None and workload.mode == "decode"
                            else base_kv_factors
                        )
                        layout_policies = self._candidate_layout_policies(workload, topology_mode)
                        for (
                            batch_parallel_factor,
                            head_parallel_factor,
                            q_parallel_factor,
                            kv_parallel_factor,
                            device_parallel_factor,
                        ) in itertools.product(
                            batch_factors,
                            head_factors,
                            q_factors,
                            kv_factors,
                            device_factors,
                        ):
                            parallelism = ParallelismPlan5D(
                                batch_parallel_factor=batch_parallel_factor,
                                head_parallel_factor=head_parallel_factor,
                                q_parallel_factor=q_parallel_factor,
                                kv_parallel_factor=kv_parallel_factor,
                                device_parallel_factor=device_parallel_factor,
                            )
                            lane_group_capacities = self._candidate_lane_group_capacities(parallelism, topology)
                            dual_noc_policies = self._candidate_dual_noc_policies(
                                workload, hardware, parallelism, topology_mode
                            )

                            for lane_group_capacity in lane_group_capacities:
                                seen_compile_configs: set[tuple[Any, ...]] = set()
                                for layout_policy in layout_policies:
                                    for page_size_strategy in self.search_space.page_size_strategies:
                                        _, k_num_pages = _select_page_size(
                                            total_bytes=k_chunk_size * workload.head_dim_qk * workload.kv_dtype_bytes,
                                            max_page_size=hardware.max_noc_page_size,
                                            strategy=page_size_strategy,
                                        )
                                        pipeline_depths = self._candidate_pipeline_depths(
                                            workload, hardware, topology_mode, k_num_pages
                                        )
                                        v_cb_depths = self._candidate_v_cb_depths(workload)
                                        for pipeline_depth in pipeline_depths:
                                            trid_windows = self._candidate_trid_windows(
                                                hardware, pipeline_depth, k_num_pages
                                            )
                                            for (
                                                trid_window,
                                                k_cb_depth,
                                                v_cb_depth,
                                                math_fidelity,
                                                dual_noc_policy,
                                            ) in itertools.product(
                                                trid_windows,
                                                self.search_space.k_cb_depths,
                                                v_cb_depths,
                                                self.search_space.math_fidelities,
                                                dual_noc_policies,
                                            ):
                                                compile_time_config = self._build_compile_time_config(
                                                    workload=workload,
                                                    hardware=hardware,
                                                    q_chunk_size=q_chunk_size,
                                                    k_chunk_size=k_chunk_size,
                                                    layout_policy=layout_policy,
                                                    topology_mode=topology_mode,
                                                    topology=topology,
                                                    lane_group_capacity=lane_group_capacity,
                                                    pipeline_depth=pipeline_depth,
                                                    trid_window=trid_window,
                                                    page_size_strategy=page_size_strategy,
                                                    k_cb_depth=k_cb_depth,
                                                    v_cb_depth=v_cb_depth,
                                                    dual_noc_policy=dual_noc_policy,
                                                    math_fidelity=math_fidelity,
                                                )
                                                compile_key = self._compile_time_dedup_key(compile_time_config)
                                                if compile_key in seen_compile_configs:
                                                    continue
                                                seen_compile_configs.add(compile_key)
                                                if not self._check_feasibility(
                                                    workload, hardware, parallelism, compile_time_config, topology
                                                ):
                                                    continue

                                                candidate = self._build_candidate(
                                                    workload=workload,
                                                    hardware=hardware,
                                                    parallelism=parallelism,
                                                    compile_time_config=compile_time_config,
                                                    topology=topology,
                                                )
                                                if candidate.metrics.l1_usage_ratio > 1.0:
                                                    continue
                                                candidates.append(candidate)
        return candidates

    def _topology_choices(
        self,
        workload: MLAWorkloadSpec,
        hardware: HardwareTopologySpec,
        topology_mode: TopologyMode,
    ) -> tuple[SBlockTopologySpec | None, ...]:
        if topology_mode == "independent":
            return (None,)
        if workload.mode != "decode":
            return ()
        candidates = [
            TOPOLOGY_LIBRARY[name]
            for name in self.search_space.sblock_topology_names
            if name in TOPOLOGY_LIBRARY and TOPOLOGY_LIBRARY[name].arch == hardware.arch
        ]
        return tuple(candidates)

    def _candidate_topology_modes(self, workload: MLAWorkloadSpec) -> tuple[TopologyMode, ...]:
        if self.heuristic_pruning and workload.mode != "decode":
            return ("independent",)
        return self.search_space.topology_modes

    def _candidate_head_factors(
        self,
        workload: MLAWorkloadSpec,
        head_factors: tuple[int, ...],
    ) -> tuple[int, ...]:
        if not self.heuristic_pruning or workload.mode != "decode":
            return head_factors
        filtered = tuple(
            value
            for value in head_factors
            if _ceil_div(workload.num_q_heads, value) <= self.search_space.decode_heads_per_lane_cap
        )
        return filtered or head_factors

    def _candidate_layout_policies(
        self,
        workload: MLAWorkloadSpec,
        topology_mode: TopologyMode,
    ) -> tuple[LayoutPolicy, ...]:
        if not self.heuristic_pruning:
            return self.search_space.layout_policies
        if workload.mode == "decode" and topology_mode != "independent":
            preferred = ("bandwidth_balanced", "row_packed_by_head")
        elif workload.mode == "decode":
            preferred = ("bandwidth_balanced", "default")
        else:
            preferred = self.search_space.layout_policies
        filtered = tuple(policy for policy in preferred if policy in self.search_space.layout_policies)
        return filtered or self.search_space.layout_policies

    def _candidate_lane_group_capacities(
        self,
        parallelism: ParallelismPlan5D,
        topology: SBlockTopologySpec | None,
    ) -> tuple[int, ...]:
        if topology is not None:
            return (topology.cores_per_s_block,)
        active_lane_count = (
            parallelism.batch_parallel_factor
            * parallelism.head_parallel_factor
            * parallelism.q_parallel_factor
        )
        return (max(1, active_lane_count),)

    def _candidate_pipeline_depths(
        self,
        workload: MLAWorkloadSpec,
        hardware: HardwareTopologySpec,
        topology_mode: TopologyMode,
        k_num_pages: int,
    ) -> tuple[int, ...]:
        depths = tuple(
            depth
            for depth in self.search_space.pipeline_depths
            if depth <= hardware.max_trid_window and depth <= max(1, k_num_pages)
        )
        if not depths:
            depth = min(min(self.search_space.pipeline_depths), hardware.max_trid_window, max(1, k_num_pages))
            return (max(1, depth),)
        if not self.heuristic_pruning:
            return depths
        min_depth = 1
        if workload.mode == "decode" and topology_mode != "independent" and k_num_pages > 1:
            min_depth = 2
        filtered = tuple(depth for depth in depths if depth >= min_depth)
        return filtered or depths

    def _candidate_trid_windows(
        self,
        hardware: HardwareTopologySpec,
        pipeline_depth: int,
        k_num_pages: int,
    ) -> tuple[int, ...]:
        candidates = tuple(
            window
            for window in self.search_space.trid_window_candidates
            if pipeline_depth <= window <= hardware.max_trid_window
        )
        if not candidates or not self.heuristic_pruning:
            return candidates
        anchor = min(hardware.max_trid_window, max(pipeline_depth, k_num_pages))
        lower = max((window for window in candidates if window <= anchor), default=None)
        upper = min((window for window in candidates if window >= anchor), default=None)
        selected = {window for window in (lower, upper) if window is not None}
        if len(selected) == 1:
            lighter = max((window for window in candidates if window < anchor), default=None)
            if lighter is not None:
                selected.add(lighter)
        return tuple(sorted(selected)) or candidates

    def _candidate_v_cb_depths(self, workload: MLAWorkloadSpec) -> tuple[int, ...]:
        if self.heuristic_pruning and workload.mode == "prefill":
            return (min(self.search_space.v_cb_depths),)
        return self.search_space.v_cb_depths

    def _candidate_dual_noc_policies(
        self,
        workload: MLAWorkloadSpec,
        hardware: HardwareTopologySpec,
        parallelism: ParallelismPlan5D,
        topology_mode: TopologyMode,
    ) -> tuple[bool, ...]:
        if not hardware.supports_dual_noc:
            return (False,)
        if not self.heuristic_pruning:
            return self.search_space.dual_noc_policies
        if workload.mode == "prefill":
            return (False,)
        if topology_mode == "independent" and parallelism.kv_parallel_factor == 1:
            return (False,)
        return self.search_space.dual_noc_policies

    def _compile_time_dedup_key(self, compile_time_config: CompileTimeConfig) -> tuple[Any, ...]:
        return (
            compile_time_config.q_chunk_size,
            compile_time_config.k_chunk_size,
            compile_time_config.layout_policy,
            compile_time_config.topology_mode,
            compile_time_config.sblock_topology_name,
            compile_time_config.lane_group_capacity,
            compile_time_config.pipeline_depth,
            compile_time_config.trid_window,
            compile_time_config.k_page_size_bytes,
            compile_time_config.k_num_pages,
            compile_time_config.k_cb_depth,
            compile_time_config.v_cb_depth,
            compile_time_config.dual_noc_policy,
            compile_time_config.math_fidelity,
            compile_time_config.bank_map,
            compile_time_config.tree_order,
        )

    def _build_compile_time_config(
        self,
        *,
        workload: MLAWorkloadSpec,
        hardware: HardwareTopologySpec,
        q_chunk_size: int,
        k_chunk_size: int,
        layout_policy: LayoutPolicy,
        topology_mode: TopologyMode,
        topology: SBlockTopologySpec | None,
        lane_group_capacity: int,
        pipeline_depth: int,
        trid_window: int,
        page_size_strategy: PageSizeStrategy,
        k_cb_depth: int,
        v_cb_depth: int,
        dual_noc_policy: bool,
        math_fidelity: MathFidelity,
    ) -> CompileTimeConfig:
        k_chunk_bytes = k_chunk_size * workload.head_dim_qk * workload.kv_dtype_bytes
        k_page_size_bytes, k_num_pages = _select_page_size(
            total_bytes=k_chunk_bytes,
            max_page_size=hardware.max_noc_page_size,
            strategy=page_size_strategy,
        )
        return CompileTimeConfig(
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            layout_policy=layout_policy,
            topology_mode=topology_mode,
            sblock_topology_name=None if topology is None else topology.name,
            lane_group_capacity=lane_group_capacity,
            pipeline_depth=pipeline_depth,
            trid_window=trid_window,
            page_size_strategy=page_size_strategy,
            k_page_size_bytes=k_page_size_bytes,
            k_num_pages=k_num_pages,
            k_cb_depth=k_cb_depth,
            v_cb_depth=v_cb_depth,
            dual_noc_policy=dual_noc_policy,
            math_fidelity=math_fidelity,
            bank_map=() if topology is None else topology.bank_map,
            tree_order=() if topology is None else topology.tree_order,
        )

    def _check_feasibility(
        self,
        workload: MLAWorkloadSpec,
        hardware: HardwareTopologySpec,
        parallelism: ParallelismPlan5D,
        compile_time_config: CompileTimeConfig,
        topology: SBlockTopologySpec | None,
    ) -> bool:
        if compile_time_config.q_chunk_size <= 0 or compile_time_config.k_chunk_size <= 0:
            return False
        if compile_time_config.q_chunk_size % 32 != 0 or compile_time_config.k_chunk_size % 32 != 0:
            return False
        if compile_time_config.pipeline_depth > compile_time_config.trid_window:
            return False
        if compile_time_config.trid_window > hardware.max_trid_window:
            return False
        if compile_time_config.k_cb_depth < compile_time_config.pipeline_depth:
            return False
        if compile_time_config.dual_noc_policy and not hardware.supports_dual_noc:
            return False
        if compile_time_config.topology_mode != "independent" and not hardware.supports_multicast:
            return False
        if compile_time_config.topology_mode == "tree" and not hardware.supports_tree:
            return False

        if parallelism.device_parallel_factor > hardware.num_devices:
            return False
        if workload.mode == "decode" and parallelism.q_parallel_factor != 1:
            return False
        if workload.mode == "prefill" and parallelism.kv_parallel_factor != 1:
            return False

        q_num_chunks = workload.q_num_chunks(compile_time_config.q_chunk_size)
        kv_num_chunks = workload.kv_num_chunks(compile_time_config.k_chunk_size)
        if parallelism.q_parallel_factor > q_num_chunks or parallelism.kv_parallel_factor > kv_num_chunks:
            return False

        head_group_size = _ceil_div(workload.num_q_heads, parallelism.head_parallel_factor)
        if workload.mode == "decode" and head_group_size > self.search_space.decode_heads_per_lane_cap:
            return False

        active_lane_count = (
            parallelism.batch_parallel_factor
            * parallelism.head_parallel_factor
            * parallelism.q_parallel_factor
        )
        provisioned_cores = parallelism.total_parallelism
        if provisioned_cores > hardware.total_cores:
            return False

        if topology is not None:
            if topology.arch != hardware.arch:
                return False
            if compile_time_config.lane_group_capacity != topology.cores_per_s_block:
                return False
            if topology.num_s_blocks > hardware.dram_bank_endpoints:
                return False
            if parallelism.kv_parallel_factor != topology.num_s_blocks:
                return False
            if active_lane_count > topology.cores_per_s_block:
                return False
        else:
            if compile_time_config.topology_mode != "independent":
                if active_lane_count > compile_time_config.lane_group_capacity:
                    return False
                if compile_time_config.lane_group_capacity > hardware.dram_bank_endpoints:
                    return False
            elif compile_time_config.lane_group_capacity < active_lane_count:
                return False

        if compile_time_config.v_cb_depth < 2 and parallelism.kv_parallel_factor > 1:
            return False
        return True

    def _build_candidate(
        self,
        *,
        workload: MLAWorkloadSpec,
        hardware: HardwareTopologySpec,
        parallelism: ParallelismPlan5D,
        compile_time_config: CompileTimeConfig,
        topology: SBlockTopologySpec | None,
    ) -> PlanCandidate:
        metrics, runtime_config = self._estimate_metrics(
            workload=workload,
            hardware=hardware,
            parallelism=parallelism,
            compile_time_config=compile_time_config,
            topology=topology,
        )
        candidate_key = self._measurement_key(workload, hardware, compile_time_config, runtime_config)
        return PlanCandidate(
            candidate_key=candidate_key,
            compile_time_config=compile_time_config,
            runtime_config=runtime_config,
            metrics=metrics,
        )

    def _estimate_metrics(
        self,
        *,
        workload: MLAWorkloadSpec,
        hardware: HardwareTopologySpec,
        parallelism: ParallelismPlan5D,
        compile_time_config: CompileTimeConfig,
        topology: SBlockTopologySpec | None,
    ) -> tuple[PlanMetrics, RuntimeConfig]:
        q_num_chunks = workload.q_num_chunks(compile_time_config.q_chunk_size)
        kv_num_chunks = workload.kv_num_chunks(compile_time_config.k_chunk_size)

        batch_group_size = _ceil_div(workload.batch_size, parallelism.batch_parallel_factor)
        head_group_size = _ceil_div(workload.num_q_heads, parallelism.head_parallel_factor)
        q_chunk_group_size = _ceil_div(q_num_chunks, parallelism.q_parallel_factor)
        kv_chunk_group_size = _ceil_div(kv_num_chunks, parallelism.kv_parallel_factor)
        active_lane_count = (
            parallelism.batch_parallel_factor
            * parallelism.head_parallel_factor
            * parallelism.q_parallel_factor
        )

        provisioned_kv_parallel_factor = parallelism.kv_parallel_factor
        effective_kv_parallel_factor = min(kv_num_chunks, provisioned_kv_parallel_factor)
        if topology is not None:
            effective_s_block_count = min(kv_num_chunks, topology.num_s_blocks)
            effective_kv_parallel_factor = effective_s_block_count
        else:
            effective_s_block_count = effective_kv_parallel_factor

        provisioned_cores = parallelism.total_parallelism
        active_cores = (
            parallelism.batch_parallel_factor
            * parallelism.head_parallel_factor
            * parallelism.q_parallel_factor
            * effective_kv_parallel_factor
            * parallelism.device_parallel_factor
        )
        active_core_ratio = active_cores / hardware.total_cores
        provisioned_core_ratio = provisioned_cores / hardware.total_cores

        imbalance_score = (
            _slack_ratio(workload.batch_size, parallelism.batch_parallel_factor)
            + _slack_ratio(workload.num_q_heads, parallelism.head_parallel_factor)
            + _slack_ratio(q_num_chunks, parallelism.q_parallel_factor)
            + _slack_ratio(kv_num_chunks, max(1, effective_kv_parallel_factor))
        ) / 4.0

        q_bytes_total = (
            workload.batch_size
            * workload.num_q_heads
            * workload.seq_len_q
            * workload.head_dim_qk
            * workload.q_dtype_bytes
        )
        k_bytes_base = (
            workload.batch_size
            * workload.num_kv_heads
            * workload.seq_len_kv
            * workload.head_dim_qk
            * workload.kv_dtype_bytes
        )
        out_bytes_total = (
            workload.batch_size
            * workload.num_q_heads
            * workload.seq_len_q
            * workload.head_dim_v
            * workload.output_dtype_bytes
        )

        shared_k_parallel_factor = parallelism.head_parallel_factor * parallelism.q_parallel_factor
        if compile_time_config.topology_mode == "independent" or workload.num_kv_heads > 1:
            k_reuse_groups = max(1, shared_k_parallel_factor)
        else:
            k_reuse_groups = _ceil_div(shared_k_parallel_factor, compile_time_config.lane_group_capacity)
        dram_bytes = q_bytes_total + out_bytes_total + k_bytes_base * k_reuse_groups

        q_slice_bytes = (
            batch_group_size
            * head_group_size
            * min(workload.seq_len_q, compile_time_config.q_chunk_size)
            * workload.head_dim_qk
            * workload.q_dtype_bytes
        )
        k_cb_bytes = compile_time_config.k_cb_depth * compile_time_config.k_chunk_size * workload.head_dim_qk * workload.kv_dtype_bytes
        v_cb_bytes = compile_time_config.v_cb_depth * compile_time_config.k_chunk_size * workload.head_dim_v * workload.output_dtype_bytes
        stats_bytes = head_group_size * workload.head_dim_v * workload.output_dtype_bytes * 2
        page_bytes = compile_time_config.pipeline_depth * compile_time_config.k_page_size_bytes
        l1_usage_bytes = int(q_slice_bytes + k_cb_bytes + v_cb_bytes + stats_bytes + page_bytes)
        l1_usage_ratio = l1_usage_bytes / hardware.l1_bytes_per_core

        compute_flops = (
            2.0
            * workload.batch_size
            * workload.num_q_heads
            * workload.seq_len_q
            * workload.seq_len_kv
            * (workload.head_dim_qk + workload.head_dim_v)
        )
        if workload.causal and workload.mode == "prefill":
            compute_flops *= 0.5

        use_balanced_q_parallel = (
            workload.mode == "prefill"
            and workload.causal
            and q_num_chunks % parallelism.q_parallel_factor == 0
            and q_chunk_group_size % 2 == 0
        )

        compute_efficiency = 0.64
        if compile_time_config.layout_policy == "row_packed_by_head":
            compute_efficiency += 0.04
        elif compile_time_config.layout_policy == "bandwidth_balanced":
            compute_efficiency += 0.07
        if use_balanced_q_parallel:
            compute_efficiency += 0.03
        if compile_time_config.k_cb_depth > compile_time_config.pipeline_depth:
            compute_efficiency += 0.02
        compute_efficiency += 0.05 * active_core_ratio
        compute_efficiency -= 0.12 * imbalance_score
        compute_efficiency = _clamp(compute_efficiency, 0.35, 0.90)

        throughput_flops_s = (
            active_cores
            * _fidelity_flops_per_cycle(compile_time_config.math_fidelity)
            * hardware.clock_GHz
            * 1e9
            * compute_efficiency
        )
        compute_ms = compute_flops / throughput_flops_s * 1e3

        reader_slots = max(
            1,
            min(
                compile_time_config.pipeline_depth,
                compile_time_config.trid_window,
                compile_time_config.k_cb_depth,
                compile_time_config.k_num_pages,
            ),
        )
        buffer_slack = max(0, compile_time_config.k_cb_depth - compile_time_config.pipeline_depth)
        page_overlap_gain = min(1.0, compile_time_config.trid_window / max(1, compile_time_config.k_num_pages))
        dram_efficiency = 0.56 + 0.07 * (compile_time_config.pipeline_depth - 1) + 0.06 * page_overlap_gain
        dram_efficiency += 0.02 * min(2, buffer_slack)
        if compile_time_config.layout_policy == "bandwidth_balanced":
            dram_efficiency += 0.08
        if compile_time_config.topology_mode != "independent":
            dram_efficiency += 0.05
        dram_efficiency = _clamp(dram_efficiency, 0.40, 0.92)
        effective_dram_bw = hardware.dram_bandwidth_GBs * dram_efficiency
        dram_ms = dram_bytes / 1e9 / effective_dram_bw * 1e3

        q_preamble_bytes = 0.0
        if compile_time_config.topology_mode != "independent" and effective_kv_parallel_factor > 1:
            q_preamble_bytes = q_bytes_total * (effective_kv_parallel_factor - 1) / effective_kv_parallel_factor

        k_noc_bytes = 0.0
        if compile_time_config.topology_mode != "independent" and shared_k_parallel_factor > 1:
            k_noc_bytes = (
                k_bytes_base
                * max(0, shared_k_parallel_factor - k_reuse_groups)
                / max(1, effective_kv_parallel_factor)
            )
            if compile_time_config.topology_mode == "tree":
                k_noc_bytes *= 0.85

        reduction_tree_depth = self._active_tree_depth(topology, effective_s_block_count)
        if reduction_tree_depth == 0 and effective_kv_parallel_factor > 1:
            reduction_tree_depth = math.ceil(math.log2(effective_kv_parallel_factor))
        reduction_bytes = out_bytes_total * reduction_tree_depth if effective_kv_parallel_factor > 1 else 0.0

        control_bytes = 0.05 * (q_preamble_bytes + k_noc_bytes + reduction_bytes)
        noc_bytes = q_preamble_bytes + k_noc_bytes + reduction_bytes + control_bytes

        noc_efficiency = 0.50 + 0.08 * (compile_time_config.pipeline_depth - 1)
        noc_efficiency += 0.04 * page_overlap_gain
        if compile_time_config.layout_policy == "row_packed_by_head":
            noc_efficiency += 0.05
        elif compile_time_config.layout_policy == "bandwidth_balanced":
            noc_efficiency += 0.08
        if compile_time_config.topology_mode == "tree":
            noc_efficiency += 0.05 if effective_kv_parallel_factor >= 4 else -0.03
        noc_efficiency = _clamp(noc_efficiency, 0.35, 0.90)

        dual_noc_effectiveness = 0.0
        if compile_time_config.dual_noc_policy and hardware.supports_dual_noc and noc_bytes > 0.0:
            noc_to_dram_ratio = noc_bytes / max(dram_bytes, 1.0)
            dual_noc_effectiveness = _clamp(
                0.55 * min(1.0, noc_to_dram_ratio)
                + 0.15 * (compile_time_config.topology_mode != "independent")
                + 0.10 * page_overlap_gain,
                0.0,
                0.90,
            )
        noc_bandwidth_scale = 1.0 + dual_noc_effectiveness
        effective_noc_bw = hardware.noc_bandwidth_GBs * noc_bandwidth_scale * noc_efficiency

        q_preamble_ms = q_preamble_bytes / 1e9 / effective_noc_bw * 1e3
        reader_noc_ms = (k_noc_bytes + 0.5 * control_bytes) / 1e9 / effective_noc_bw * 1e3

        reduction_ms = 0.0
        if reduction_bytes > 0.0 or control_bytes > 0.0:
            reduction_ms = (reduction_bytes + 0.5 * control_bytes) / 1e9 / effective_noc_bw * 1e3
        reduction_ms += 0.00025 * reduction_tree_depth * max(
            1.0, head_group_size / max(1, self.search_space.decode_heads_per_lane_cap)
        )

        critical_page_transactions = max(1, kv_chunk_group_size) * compile_time_config.k_num_pages
        page_control_per_transfer_ms = 0.00008 if workload.mode == "prefill" else 0.00012
        page_control_ms = critical_page_transactions * page_control_per_transfer_ms
        if compile_time_config.topology_mode != "independent":
            page_control_ms *= 1.15
        if compile_time_config.dual_noc_policy:
            page_control_ms *= 1.05

        dram_page_ms = compile_time_config.k_page_size_bytes / 1e9 / effective_dram_bw * 1e3
        noc_page_ms = 0.0
        if critical_page_transactions > 0 and noc_bytes > 0.0:
            noc_page_ms = (k_noc_bytes + reduction_bytes + control_bytes) / critical_page_transactions / 1e9 / effective_noc_bw * 1e3

        pipeline_bubble_ratio = max(0.0, 1.0 - reader_slots / max(1, compile_time_config.k_num_pages))
        pipeline_bubble_ms = (
            (dram_page_ms + noc_page_ms)
            * pipeline_bubble_ratio
            * max(1, kv_chunk_group_size)
            * 0.75
        )

        reader_ms = max(dram_ms, reader_noc_ms) + page_control_ms + pipeline_bubble_ms

        page_pipeline_pressure = compile_time_config.k_num_pages / max(1, reader_slots)
        reader_backpressure_factor = _clamp(
            0.015 * max(0, kv_chunk_group_size - 1)
            + 0.05 * max(0.0, page_pipeline_pressure - 1.0)
            + 0.10 * max(0.0, l1_usage_ratio - 0.30)
            + 0.04 * (compile_time_config.topology_mode != "independent")
            + 0.03 * max(0, effective_kv_parallel_factor - 1)
            - 0.04 * min(2, buffer_slack)
            - 0.03 * max(0, compile_time_config.v_cb_depth - 2),
            0.0,
            0.45 if workload.mode == "decode" else 0.20,
        )
        reader_backpressure_ms = reader_ms * reader_backpressure_factor

        sender_hotspot_ms = 0.0
        if effective_kv_parallel_factor > 1:
            sender_base_ms = 0.5 * reader_noc_ms + 0.25 * q_preamble_ms
            if topology is not None:
                sender_base_ms += (
                    (k_bytes_base / max(1, topology.num_s_blocks)) / 1e9 / effective_noc_bw * 1e3
                )
            else:
                sender_base_ms += 0.5 * reduction_ms
            sender_hotspot_factor = _clamp(
                0.04 * max(0, effective_kv_parallel_factor - 1)
                + 0.03 * max(0, active_lane_count - 1)
                + 0.05 * max(0.0, page_pipeline_pressure - 1.0)
                - 0.06 * dual_noc_effectiveness
                - 0.03 * (compile_time_config.topology_mode == "tree")
                - 0.02 * max(0, compile_time_config.pipeline_depth - 2),
                0.0,
                0.35,
            )
            sender_hotspot_ms = sender_base_ms * sender_hotspot_factor

        writer_backpressure_factor = _clamp(
            0.06 * max(0, effective_kv_parallel_factor - 1)
            + 0.05 * max(0.0, head_group_size / max(1, self.search_space.decode_heads_per_lane_cap) - 1.0)
            + 0.04 * (compile_time_config.topology_mode != "independent")
            + 0.03 * max(0.0, l1_usage_ratio - 0.35)
            - 0.07 * max(0, compile_time_config.v_cb_depth - 2)
            - 0.04 * dual_noc_effectiveness,
            0.0,
            0.35 if workload.mode == "decode" else 0.18,
        )
        writer_backpressure_ms = (reduction_ms + 0.15 * reader_ms) * writer_backpressure_factor

        overlap_residual_factor = _clamp(
            0.42
            - 0.06 * (compile_time_config.pipeline_depth - 1)
            - 0.04 * page_overlap_gain
            - 0.03 * min(2, buffer_slack)
            - 0.04 * dual_noc_effectiveness,
            0.08,
            0.45,
        )
        overlap_residual_ms = min(reader_ms, compute_ms) * overlap_residual_factor

        l1_pressure_ms = 0.0015 * l1_usage_ratio + _soft_knee_penalty(
            l1_usage_ratio,
            knee=0.45,
            span=0.35,
            scale=0.04,
        )

        useful_trid_need = min(
            compile_time_config.k_num_pages,
            compile_time_config.pipeline_depth + compile_time_config.k_cb_depth - 1,
        )
        unused_trid_ratio = max(0.0, compile_time_config.trid_window - useful_trid_need) / hardware.max_trid_window
        complexity_penalty_ms = 0.001 * unused_trid_ratio
        if compile_time_config.dual_noc_policy:
            complexity_penalty_ms += 0.0015 * (1.0 - dual_noc_effectiveness)
        complexity_penalty_ms += 0.0006 * max(0, compile_time_config.k_cb_depth - compile_time_config.pipeline_depth - 1)
        complexity_penalty_ms += 0.0006 * max(0, compile_time_config.v_cb_depth - 2)

        noc_ms = reader_noc_ms + reduction_ms
        sync_ms = 0.001 * (compile_time_config.pipeline_depth - 1) + 0.0005 * max(0, effective_kv_parallel_factor - 1)

        estimated_latency_ms = (
            q_preamble_ms
            + max(reader_ms + reader_backpressure_ms + sender_hotspot_ms, compute_ms)
            + overlap_residual_ms
            + reduction_ms
            + writer_backpressure_ms
            + sync_ms
            + l1_pressure_ms
            + complexity_penalty_ms
        )

        l1_overuse = max(0.0, l1_usage_ratio - 1.0)
        analytical_score = (
            self.weights.latency * estimated_latency_ms
            + self.weights.dram_traffic * (dram_bytes / 1e6)
            + self.weights.noc_traffic * (noc_bytes / 1e6)
            + self.weights.inactive_core_penalty * (1.0 - active_core_ratio)
            + self.weights.imbalance_penalty * imbalance_score
            + self.weights.l1_overuse_penalty * (l1_overuse**2)
        )

        runtime_config = RuntimeConfig(
            parallelism_5d=parallelism,
            grouping=GroupingPlan(
                batch_group_size=batch_group_size,
                head_group_size=head_group_size,
                q_chunk_group_size=q_chunk_group_size,
                kv_chunk_group_size=kv_chunk_group_size,
                active_lane_count=active_lane_count,
                provisioned_kv_parallel_factor=provisioned_kv_parallel_factor,
                effective_kv_parallel_factor=effective_kv_parallel_factor,
                effective_s_block_count=effective_s_block_count,
            ),
            use_balanced_q_parallel=use_balanced_q_parallel,
            reduction_tree_depth=reduction_tree_depth,
            k_reuse_groups=k_reuse_groups,
        )
        metrics = PlanMetrics(
            analytical_score=analytical_score,
            selected_latency_ms=estimated_latency_ms,
            estimated_latency_ms=estimated_latency_ms,
            measured_latency_ms=None,
            selection_source="analytical",
            q_preamble_ms=q_preamble_ms,
            compute_ms=compute_ms,
            dram_ms=dram_ms,
            noc_ms=noc_ms,
            sync_ms=sync_ms,
            dram_bytes=dram_bytes,
            noc_bytes=noc_bytes,
            l1_usage_bytes_per_core=l1_usage_bytes,
            l1_usage_ratio=l1_usage_ratio,
            active_cores=active_cores,
            provisioned_cores=provisioned_cores,
            active_core_ratio=active_core_ratio,
            provisioned_core_ratio=provisioned_core_ratio,
            imbalance_score=imbalance_score,
            q_num_chunks=q_num_chunks,
            kv_num_chunks=kv_num_chunks,
            k_reuse_groups=k_reuse_groups,
            reader_ms=reader_ms,
            reduction_ms=reduction_ms,
            reader_backpressure_ms=reader_backpressure_ms,
            writer_backpressure_ms=writer_backpressure_ms,
            sender_hotspot_ms=sender_hotspot_ms,
            overlap_residual_ms=overlap_residual_ms,
            page_control_ms=page_control_ms,
            pipeline_bubble_ms=pipeline_bubble_ms,
            l1_pressure_ms=l1_pressure_ms,
            complexity_penalty_ms=complexity_penalty_ms,
            dual_noc_effectiveness=dual_noc_effectiveness,
            unused_trid_ratio=unused_trid_ratio,
        )
        return metrics, runtime_config

    def _active_tree_depth(self, topology: SBlockTopologySpec | None, active_s_blocks: int) -> int:
        if topology is None or active_s_blocks <= 1:
            return 0
        depth = 0
        for step in topology.tree_order:
            if any(dst < active_s_blocks and src < active_s_blocks for dst, src in step):
                depth += 1
        return depth

    def _apply_measurement_rerank(
        self,
        workload: MLAWorkloadSpec,
        hardware: HardwareTopologySpec,
        candidates: list[PlanCandidate],
        measurement_db: MeasurementDB,
        rerank_top_k: int,
    ) -> tuple[list[PlanCandidate], int]:
        rerank_top_k = min(rerank_top_k, len(candidates))
        extra_measured_indices = [
            index
            for index, candidate in enumerate(candidates[rerank_top_k:], start=rerank_top_k)
            if candidate.candidate_key in measurement_db.records
        ]
        rerank_indices = list(range(rerank_top_k)) + extra_measured_indices
        reranked: list[PlanCandidate] = []
        measured_count = 0
        for index in rerank_indices:
            candidate = candidates[index]
            measured = measurement_db.get(candidate.candidate_key)
            if measured is not None:
                measured_count += 1
                updated_metrics = replace(
                    candidate.metrics,
                    selected_latency_ms=measured,
                    measured_latency_ms=measured,
                    selection_source="measurement_db",
                )
                candidate = replace(candidate, metrics=updated_metrics)
            reranked.append(candidate)

        reranked.sort(
            key=lambda candidate: (
                candidate.metrics.selected_latency_ms,
                0 if candidate.metrics.measured_latency_ms is not None else 1,
                candidate.metrics.analytical_score,
                candidate.candidate_key,
            )
        )
        extra_measured_index_set = set(extra_measured_indices)
        remaining_candidates = [
            candidate
            for index, candidate in enumerate(candidates[rerank_top_k:], start=rerank_top_k)
            if index not in extra_measured_index_set
        ]
        return reranked + remaining_candidates, measured_count

    def _bounded_factors(self, factors: Sequence[int], upper_bound: int) -> tuple[int, ...]:
        bounded = sorted({value for value in factors if 1 <= value <= max(1, upper_bound)})
        if not bounded:
            return (1,)
        return tuple(bounded)

    def _exact_signature(self, workload: MLAWorkloadSpec, hardware: HardwareTopologySpec) -> str:
        payload = {
            "workload": _semantic_workload_payload(workload),
            "hardware": _semantic_hardware_payload(hardware),
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()

    def _policy_cache_key(
        self,
        workload: MLAWorkloadSpec,
        hardware: HardwareTopologySpec,
        bucket_config: PolicyBucketConfig,
    ) -> str:
        payload = {
            "bucket_mode": bucket_config.mode,
            "workload_bucket": bucket_config.normalize_workload(workload),
            "hardware_bucket": bucket_config.normalize_hardware(hardware),
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()

    def _measurement_key(
        self,
        workload: MLAWorkloadSpec,
        hardware: HardwareTopologySpec,
        compile_time_config: CompileTimeConfig,
        runtime_config: RuntimeConfig,
    ) -> str:
        payload = {
            "workload": _semantic_workload_payload(workload),
            "hardware": _semantic_hardware_payload(hardware),
            "compile_time_config": asdict(compile_time_config),
            "runtime_config": asdict(runtime_config),
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()


PRESET_HARDWARE: dict[str, HardwareTopologySpec] = {
    "wormhole_b0": HardwareTopologySpec(
        name="wormhole_b0",
        arch="wormhole_b0",
        compute_grid_x=8,
        compute_grid_y=8,
        dram_bank_endpoints=6,
        dram_bandwidth_GBs=258.0,
        noc_bandwidth_GBs=32.0,
        l1_bytes_per_core=1_499_136,
        clock_GHz=1.0,
        max_noc_page_size=8_192,
        max_trid_window=14,
    ),
    # Single Wormhole chip on an N300 card (N300 = 2× WH; use num_devices=2 for full-card)
    "wormhole_n300": HardwareTopologySpec(
        name="wormhole_n300",
        arch="wormhole_b0",
        compute_grid_x=8,
        compute_grid_y=8,
        dram_bank_endpoints=6,
        dram_bandwidth_GBs=258.0,
        noc_bandwidth_GBs=32.0,
        l1_bytes_per_core=1_499_136,
        clock_GHz=1.0,
        max_noc_page_size=8_192,
        max_trid_window=14,
        num_devices=1,
    ),
    "blackhole": HardwareTopologySpec(
        name="blackhole",
        arch="blackhole",
        compute_grid_x=11,
        compute_grid_y=10,
        dram_bank_endpoints=8,
        dram_bandwidth_GBs=512.0,
        noc_bandwidth_GBs=86.4,
        l1_bytes_per_core=1_572_864,
        clock_GHz=1.35,
        max_noc_page_size=16_384,
        max_trid_window=14,
    ),
}

TOPOLOGY_LIBRARY: dict[str, SBlockTopologySpec] = {
    "wh_sblock_6x4": SBlockTopologySpec(
        name="wh_sblock_6x4",
        arch="wormhole_b0",
        num_s_blocks=6,
        cores_per_s_block=4,
        block_shape=(2, 2),
        bank_map=(1, 2, 0, 4, 9, 8),
        tree_order=(
            ((0, 1), (2, 3), (4, 5)),
            ((0, 2),),
            ((0, 4),),
        ),
        description="Current Wormhole-style 6 block x 4 core topology",
    ),
    "bh_sblock_8x8": SBlockTopologySpec(
        name="bh_sblock_8x8",
        arch="blackhole",
        num_s_blocks=8,
        cores_per_s_block=8,
        block_shape=(4, 2),
        bank_map=(1, 3, 2, 0, 5, 7, 6, 4),
        tree_order=(
            ((0, 1), (2, 3), (4, 5), (6, 7)),
            ((0, 2), (4, 6)),
            ((0, 4),),
        ),
        description="Current Blackhole-style 8 block x 8 core topology",
    ),
}

PRESET_WORKLOADS: dict[str, MLAWorkloadSpec] = {
    "flash_decode_wh": MLAWorkloadSpec(
        name="flash_decode_wh",
        mode="decode",
        causal=False,
        paged=False,
        batch_size=1,
        seq_len_q=1,
        seq_len_kv=4096,
        num_q_heads=32,
        num_kv_heads=1,
        kv_lora_rank=512,
        d_rope=64,
    ),
    "flash_decode_n300": MLAWorkloadSpec(
        name="flash_decode_n300",
        mode="decode",
        causal=False,
        paged=False,
        batch_size=1,
        seq_len_q=1,
        seq_len_kv=4096,
        num_q_heads=32,
        num_kv_heads=1,
        kv_lora_rank=512,
        d_rope=64,
    ),
    "flash_decode_bh": MLAWorkloadSpec(
        name="flash_decode_bh",
        mode="decode",
        causal=False,
        paged=False,
        batch_size=1,
        seq_len_q=1,
        seq_len_kv=4096,
        num_q_heads=64,
        num_kv_heads=1,
        kv_lora_rank=512,
        d_rope=64,
    ),
    "mla_prefill_wh": MLAWorkloadSpec(
        name="mla_prefill_wh",
        mode="prefill",
        causal=True,
        paged=False,
        batch_size=1,
        seq_len_q=1024,
        seq_len_kv=1024,
        num_q_heads=32,
        num_kv_heads=1,
        kv_lora_rank=512,
        d_rope=64,
    ),
    "mla_prefill_bh": MLAWorkloadSpec(
        name="mla_prefill_bh",
        mode="prefill",
        causal=True,
        paged=False,
        batch_size=1,
        seq_len_q=4096,
        seq_len_kv=4096,
        num_q_heads=64,
        num_kv_heads=1,
        kv_lora_rank=512,
        d_rope=64,
    ),
}

PRESET_TOPOLOGY_FOR_WORKLOAD: dict[str, str] = {
    "flash_decode_wh": "wormhole_b0",
    "flash_decode_n300": "wormhole_n300",
    "flash_decode_bh": "blackhole",
    "mla_prefill_wh": "wormhole_b0",
    "mla_prefill_bh": "blackhole",
}


def get_preset_hardware(name: str) -> HardwareTopologySpec:
    return PRESET_HARDWARE[name]


def get_preset_workload(name: str) -> MLAWorkloadSpec:
    return PRESET_WORKLOADS[name]


def get_topology_preset(name: str) -> SBlockTopologySpec:
    return TOPOLOGY_LIBRARY[name]


def _load_spec(path: str | Path, cls: type[MLAWorkloadSpec] | type[HardwareTopologySpec]) -> Any:
    payload = json.loads(Path(path).read_text())
    return cls(**payload)


def _summarize_plan(candidate: PlanCandidate) -> list[str]:
    metrics = candidate.metrics
    parallelism = candidate.parallelism_5d
    grouping = candidate.runtime_config.grouping
    return [
        (
            "5D parallelism: "
            f"B={parallelism.batch_parallel_factor}, "
            f"H={parallelism.head_parallel_factor}, "
            f"Q={parallelism.q_parallel_factor}, "
            f"KV={parallelism.kv_parallel_factor}, "
            f"D={parallelism.device_parallel_factor}"
        ),
        (
            "Grouping: "
            f"head_group_size={grouping.head_group_size}, "
            f"q_chunk_group_size={grouping.q_chunk_group_size}, "
            f"kv_chunk_group_size={grouping.kv_chunk_group_size}, "
            f"active_lanes={grouping.active_lane_count}, "
            f"effective_s_blocks={grouping.effective_s_block_count}"
        ),
        (
            "Compile-time: "
            f"q_chunk={candidate.q_chunk_size}, "
            f"k_chunk={candidate.k_chunk_size}, "
            f"layout={candidate.layout_policy}, "
            f"topology={candidate.topology_mode}, "
            f"sblock={candidate.compile_time_config.sblock_topology_name}, "
            f"page_size={candidate.compile_time_config.k_page_size_bytes}, "
            f"pipeline_depth={candidate.pipeline_depth}, "
            f"trid_window={candidate.compile_time_config.trid_window}, "
            f"dual_noc={candidate.dual_noc_policy}, "
            f"fidelity={candidate.math_fidelity}"
        ),
        (
            "Metrics: "
            f"selected_ms={metrics.selected_latency_ms:.6f}, "
            f"estimated_ms={metrics.estimated_latency_ms:.6f}, "
            f"reader_ms={metrics.reader_ms:.6f}, "
            f"compute_ms={metrics.compute_ms:.6f}, "
            f"reduce_ms={metrics.reduction_ms:.6f}, "
            f"reader_bp_ms={metrics.reader_backpressure_ms:.6f}, "
            f"writer_bp_ms={metrics.writer_backpressure_ms:.6f}, "
            f"sender_hotspot_ms={metrics.sender_hotspot_ms:.6f}, "
            f"dram_ms={metrics.dram_ms:.6f}, "
            f"noc_ms={metrics.noc_ms:.6f}, "
            f"overlap_residual_ms={metrics.overlap_residual_ms:.6f}, "
            f"l1_ratio={metrics.l1_usage_ratio:.3f}, "
            f"l1_pressure_ms={metrics.l1_pressure_ms:.6f}, "
            f"complexity_ms={metrics.complexity_penalty_ms:.6f}, "
            f"active_cores={metrics.active_cores}/{metrics.provisioned_cores}, "
            f"selection_source={metrics.selection_source}"
        ),
        f"Candidate key: {candidate.candidate_key}",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refined offline analytical autotuner for MLA planning")
    parser.add_argument(
        "--wh-profile-json",
        help="WH profile JSON (simple or detailed) used by profile-derived calibration flows.",
    )
    parser.add_argument(
        "--build-wh-profile-measurement-db",
        help="Build a MeasurementDB from WH profile JSON (simple or detailed) and exit.",
    )
    parser.add_argument(
        "--measurement-db-output",
        help="Output path for generated MeasurementDB JSON when using --build-wh-profile-measurement-db.",
    )
    parser.add_argument(
        "--profile-measurement-selection-mode",
        choices=("reference_current_a", "analytical_best"),
        default="reference_current_a",
        help="How WH profile measurements are mapped onto autotuner candidate keys.",
    )
    parser.add_argument(
        "--profile-measurement-reference-source",
        choices=("profile_harness", "mla1d_defaults", "hybrid_auto"),
        default="profile_harness",
        help="Which reference A-path config source is used when building MeasurementDB entries from profile JSON.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_WORKLOADS.keys()),
        help="Use a built-in workload + hardware preset",
    )
    parser.add_argument("--workload-json", help="Path to workload JSON. Use with --hardware-json if no preset.")
    parser.add_argument("--hardware-json", help="Path to hardware JSON. Use with --workload-json if no preset.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top candidates to print")
    parser.add_argument("--cache-file", help="Optional cache file for workload-signature -> best plan")
    parser.add_argument(
        "--cache-key-mode",
        choices=("exact", "sequence_pow2"),
        default="sequence_pow2",
        help="Cache key mode. sequence_pow2 enables bucketed policy reuse.",
    )
    parser.add_argument("--measurement-db", help="Optional JSON measurement DB keyed by candidate_key")
    parser.add_argument("--rerank-top-k", type=int, default=0, help="Measured reranking depth over analytical top-K")
    parser.add_argument(
        "--calibration-json",
        help="Load a hardware-calibrated cost model JSON; ranking uses calibrated latency (see autotuner.cost_model).",
    )
    parser.add_argument(
        "--fit-calibration-output",
        help=(
            "Fit CalibratedCostModel from either --measurement-db against enumerated candidates "
            "or directly from --wh-profile-json, write JSON, then exit."
        ),
    )
    parser.add_argument(
        "--calibration-feature-mode",
        choices=("auto", "scalar", "breakdown", "extended"),
        default="auto",
        help="Feature set for linear calibration (default: auto).",
    )
    parser.add_argument(
        "--calibration-ridge",
        type=float,
        default=1e-4,
        help="Ridge regularization for calibration normal equations.",
    )
    parser.add_argument(
        "--disable-heuristic-pruning",
        action="store_true",
        help="Disable workload-aware search pruning and run the fuller cartesian search.",
    )
    parser.add_argument("--output-json", help="Optional JSON output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.fit_calibration_output:
        if args.wh_profile_json and args.measurement_db:
            parser.error("use either --measurement-db or --wh-profile-json with --fit-calibration-output")
        from .cost_model import fit_calibrated_cost_model_for_workload, fit_calibrated_cost_model_from_wh_profile

        if args.wh_profile_json:
            model, report = fit_calibrated_cost_model_from_wh_profile(
                args.wh_profile_json,
                heuristic_pruning=not args.disable_heuristic_pruning,
                selection_mode=args.profile_measurement_selection_mode,
                reference_source=args.profile_measurement_reference_source,
                feature_mode=args.calibration_feature_mode,
                ridge=args.calibration_ridge,
            )
            model.save(args.fit_calibration_output)
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
            return 0

        if not args.measurement_db:
            parser.error("--measurement-db or --wh-profile-json is required with --fit-calibration-output")

        if args.preset:
            workload = get_preset_workload(args.preset)
            hardware = get_preset_hardware(PRESET_TOPOLOGY_FOR_WORKLOAD[args.preset])
        else:
            if not args.workload_json or not args.hardware_json:
                parser.error("--fit-calibration-output requires --preset or both --workload-json and --hardware-json")
            workload = _load_spec(args.workload_json, MLAWorkloadSpec)
            hardware = _load_spec(args.hardware_json, HardwareTopologySpec)
        measurement_db = MeasurementDB.load(args.measurement_db)
        autotuner = BasicMLAAutotuner(heuristic_pruning=not args.disable_heuristic_pruning)
        model, report = fit_calibrated_cost_model_for_workload(
            measurement_db,
            workload,
            hardware,
            autotuner=autotuner,
            feature_mode=args.calibration_feature_mode,
            ridge=args.calibration_ridge,
        )
        model.save(args.fit_calibration_output)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.build_wh_profile_measurement_db:
        if not args.measurement_db_output:
            parser.error("--measurement-db-output is required with --build-wh-profile-measurement-db")
        from .profile_measurement_bridge import build_wh_profile_measurement_db

        summary = build_wh_profile_measurement_db(
            profile_results_path=args.build_wh_profile_measurement_db,
            output_path=args.measurement_db_output,
            heuristic_pruning=not args.disable_heuristic_pruning,
            selection_mode=args.profile_measurement_selection_mode,
            reference_source=args.profile_measurement_reference_source,
        )
        print(f"measurement_db_path: {summary.output_path}")
        print(f"entry_count: {summary.entry_count}")
        print(f"reference_source: {summary.reference_source}")
        for entry in summary.entries:
            print(
                "entry: "
                f"profile_case={entry['profile_case']}, "
                f"candidate_key={entry['candidate_key']}, "
                f"latency_ms={entry['latency_ms']:.6f}, "
                f"selection_mode={entry['selection_mode']}, "
                f"reference_source={entry['reference_source']}, "
                f"analytical_ms={entry['analytical_latency_ms']:.6f}"
            )
        return 0

    if args.preset:
        workload = get_preset_workload(args.preset)
        hardware = get_preset_hardware(PRESET_TOPOLOGY_FOR_WORKLOAD[args.preset])
    else:
        if not args.workload_json or not args.hardware_json:
            parser.error("either --preset or both --workload-json and --hardware-json are required")
        workload = _load_spec(args.workload_json, MLAWorkloadSpec)
        hardware = _load_spec(args.hardware_json, HardwareTopologySpec)

    policy_cache = PolicyCache.load(args.cache_file) if args.cache_file else None
    bucket_config = PolicyBucketConfig(mode=args.cache_key_mode)
    measurement_db = MeasurementDB.load(args.measurement_db) if args.measurement_db else None
    calibrated_model = None
    if args.calibration_json:
        from .cost_model import CalibratedCostModel

        calibrated_model = CalibratedCostModel.load(args.calibration_json)

    autotuner = BasicMLAAutotuner(heuristic_pruning=not args.disable_heuristic_pruning)
    result = autotuner.tune(
        workload,
        hardware,
        top_k=args.top_k,
        policy_cache=policy_cache,
        bucket_config=bucket_config,
        measurement_db=measurement_db,
        rerank_top_k=args.rerank_top_k,
        calibrated_cost_model=calibrated_model,
    )

    print(f"workload_signature: {result.workload_signature}")
    print(f"policy_cache_key: {result.policy_cache_key}")
    print(f"cache_key_mode: {result.cache_key_mode}")
    print(f"cache_hit: {result.cache_hit}")
    print(f"cache_source_signature: {result.cache_source_signature}")
    print(f"searched_candidate_count: {result.searched_candidate_count}")
    print(f"reranked_top_k: {result.reranked_top_k}")
    print(f"measured_candidate_count: {result.measured_candidate_count}")
    print("best_plan:")
    for line in _summarize_plan(result.best_plan):
        print(f"  {line}")
    print("top_candidates:")
    for index, candidate in enumerate(result.top_candidates, start=1):
        print(f"  [{index}] analytical_score={candidate.metrics.analytical_score:.6f}")
        for line in _summarize_plan(candidate):
            print(f"      {line}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
