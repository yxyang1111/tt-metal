"""Build autotuner MeasurementDB entries from WH profile results."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .basic_autotuner import (
    BasicMLAAutotuner,
    MLAWorkloadSpec,
    MeasurementDB,
    PlanCandidate,
    get_preset_hardware,
)

ProfileMeasurementSelectionMode = Literal["reference_current_a", "analytical_best"]
ReferenceCurrentASource = Literal["profile_harness", "mla1d_defaults", "hybrid_auto"]


def _candidate_sort_key(candidate: PlanCandidate) -> tuple[Any, ...]:
    metrics = candidate.metrics
    return (
        metrics.analytical_score,
        metrics.selected_latency_ms,
        metrics.l1_pressure_ms,
        metrics.complexity_penalty_ms,
        metrics.l1_usage_ratio,
        candidate.candidate_key,
    )


@dataclass(frozen=True)
class MeasurementDBBuildSummary:
    output_path: str
    selection_mode: ProfileMeasurementSelectionMode
    reference_source: ReferenceCurrentASource
    entry_count: int
    entries: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class MappedProfileMeasurement:
    profile_case: str
    profile_mode: str
    profile_variant: str | None
    workload: MLAWorkloadSpec
    candidate: PlanCandidate
    latency_ms: float
    selection_mode: str
    reference_config: ReferenceCurrentAConfig
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ReferenceCurrentAConfig:
    mode: str
    source: str
    q_chunk_size: int
    k_chunk_size: int
    topology_mode: str = "independent"
    dual_noc_policy: bool = False
    math_fidelity: str = "HiFi4"
    target_layout_policy: str = "default"
    batch_parallel_factor: int = 1
    device_parallel_factor: int = 1
    q_parallel_factor: int | None = None
    kv_parallel_factor: int | None = None
    target_head_group_size: int | None = None
    target_active_lane_count: int | None = None
    target_active_cores: int | None = None
    max_cores_per_head_batch: int | None = None


def _extract_profile_entries(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    if "wh_profile" in payload:
        return tuple(payload["wh_profile"])
    if "profiles" in payload:
        return tuple(payload["profiles"])
    raise RuntimeError("profile JSON does not contain `wh_profile` or `profiles` entries")


def _detect_profile_payload_kind(payload: dict[str, Any]) -> str:
    if "wh_profile" in payload:
        return "wh_profile"
    if "profiles" in payload:
        return "profiles"
    return "unknown"


def _resolve_profile_mode(profile_entry: dict[str, Any]) -> str:
    if "mode" in profile_entry:
        return str(profile_entry["mode"])
    case_name = str(profile_entry["case"])
    return "prefill" if case_name.startswith("prefill") else "decode"


def _profile_case_to_workload(profile_entry: dict[str, Any]) -> MLAWorkloadSpec:
    case_name = str(profile_entry["case"])
    profile_workload = profile_entry["workload"]
    mode = _resolve_profile_mode(profile_entry)
    seq_len = int(profile_workload["seq_len"])
    batch = int(profile_workload["batch"])
    num_heads = int(profile_workload["num_heads"])
    num_kv_heads = int(profile_workload.get("num_kv_heads", 1))
    kv_lora_rank = int(profile_workload.get("kv_lora_rank", 512))
    qk_rope_head_dim = int(profile_workload.get("qk_rope_head_dim", 64))
    block_size = int(profile_workload.get("block_size", 64))

    if mode == "prefill":
        return MLAWorkloadSpec(
            name=f"wh_profile_{case_name}",
            mode="prefill",
            causal=True,
            paged=False,
            batch_size=batch,
            seq_len_q=seq_len,
            seq_len_kv=seq_len,
            num_q_heads=num_heads,
            num_kv_heads=num_kv_heads,
            kv_lora_rank=kv_lora_rank,
            d_rope=qk_rope_head_dim,
            block_size=block_size,
        )

    return MLAWorkloadSpec(
        name=f"wh_profile_{case_name}",
        mode="decode",
        causal=False,
        paged=False,
        batch_size=batch,
        seq_len_q=1,
        seq_len_kv=seq_len,
        num_q_heads=num_heads,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        d_rope=qk_rope_head_dim,
        block_size=block_size,
    )


def _profile_latency_ms(profile_entry: dict[str, Any]) -> float:
    kernel_ns = profile_entry["column_stats"]["DEVICE KERNEL DURATION [ns]"]["avg_ns"]
    return float(kernel_ns) / 1e6


def _rank_candidates(
    autotuner: BasicMLAAutotuner,
    workload: MLAWorkloadSpec,
) -> list[PlanCandidate]:
    hardware = get_preset_hardware("wormhole_b0")
    candidates = autotuner._enumerate_candidates(workload, hardware)
    if not candidates:
        raise RuntimeError(f"no feasible candidate found for profile workload {workload.name}")
    candidates.sort(key=_candidate_sort_key)
    return candidates


def _target_q_chunk_size(workload: MLAWorkloadSpec, profile_entry: dict[str, Any]) -> int:
    profile_workload = profile_entry.get("workload", {})
    if "q_chunk_size" in profile_workload:
        return int(profile_workload["q_chunk_size"])
    if workload.mode == "decode":
        return 32
    rounded_heads = max(32, int(math.ceil(workload.num_q_heads / 32.0) * 32))
    return 1 << (rounded_heads - 1).bit_length()


def _target_k_chunk_size(profile_entry: dict[str, Any]) -> int:
    return int(profile_entry.get("workload", {}).get("k_chunk_size", 128))


def _resolve_reference_current_a_config(
    workload: MLAWorkloadSpec,
    profile_entry: dict[str, Any],
) -> ReferenceCurrentAConfig:
    q_chunk_size = _target_q_chunk_size(workload, profile_entry)
    k_chunk_size = _target_k_chunk_size(profile_entry)
    if workload.mode == "decode":
        max_cores_per_head_batch = int(profile_entry.get("workload", {}).get("max_cores_per_head_batch", 4))
        target_head_group_size = min(8, workload.num_q_heads)
        target_active_lane_count = max(1, math.ceil(workload.num_q_heads / target_head_group_size))
        return ReferenceCurrentAConfig(
            mode="decode",
            source="profile_harness_decode",
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            q_parallel_factor=1,
            kv_parallel_factor=max_cores_per_head_batch,
            target_head_group_size=target_head_group_size,
            target_active_lane_count=target_active_lane_count,
            target_active_cores=target_active_lane_count * max_cores_per_head_batch,
            max_cores_per_head_batch=max_cores_per_head_batch,
        )

    return ReferenceCurrentAConfig(
        mode="prefill",
        source="profile_harness_prefill",
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        kv_parallel_factor=1,
        target_head_group_size=max(1, workload.num_q_heads),
        target_active_lane_count=1,
        target_active_cores=max(1, workload.batch_size),
    )


def _resolve_reference_current_a_from_mla1d_defaults(workload: MLAWorkloadSpec) -> ReferenceCurrentAConfig:
    if workload.mode == "decode":
        return ReferenceCurrentAConfig(
            mode="decode",
            source="mla1d_decode_defaults",
            q_chunk_size=0,
            k_chunk_size=128,
            q_parallel_factor=1,
            max_cores_per_head_batch=16,
        )
    return ReferenceCurrentAConfig(
        mode="prefill",
        source="mla1d_prefill_defaults",
        q_chunk_size=128,
        k_chunk_size=128,
        kv_parallel_factor=1,
    )


def _resolve_reference_current_a_config_for_source(
    workload: MLAWorkloadSpec,
    profile_entry: dict[str, Any],
    reference_source: ReferenceCurrentASource,
    profile_payload_kind: str,
) -> ReferenceCurrentAConfig:
    if reference_source == "profile_harness":
        return _resolve_reference_current_a_config(workload, profile_entry)
    if reference_source == "mla1d_defaults":
        return _resolve_reference_current_a_from_mla1d_defaults(workload)
    if profile_payload_kind in {"wh_profile", "profiles"}:
        return _resolve_reference_current_a_config(workload, profile_entry)
    return _resolve_reference_current_a_from_mla1d_defaults(workload)


def _optional_distance(actual: int, target: int | None) -> int:
    if target is None:
        return 0
    return abs(actual - target)


def _reference_current_a_decode_score(
    candidate: PlanCandidate,
    workload: MLAWorkloadSpec,
    reference_config: ReferenceCurrentAConfig,
) -> tuple[Any, ...]:
    parallelism = candidate.runtime_config.parallelism_5d
    grouping = candidate.runtime_config.grouping
    compile_time = candidate.compile_time_config

    return (
        compile_time.topology_mode != reference_config.topology_mode,
        compile_time.sblock_topology_name is not None,
        compile_time.dual_noc_policy != reference_config.dual_noc_policy,
        compile_time.layout_policy != reference_config.target_layout_policy,
        compile_time.math_fidelity != reference_config.math_fidelity,
        (reference_config.q_parallel_factor is not None and parallelism.q_parallel_factor != reference_config.q_parallel_factor),
        parallelism.device_parallel_factor != reference_config.device_parallel_factor,
        parallelism.batch_parallel_factor != reference_config.batch_parallel_factor,
        _optional_distance(grouping.active_lane_count, reference_config.target_active_lane_count),
        _optional_distance(grouping.head_group_size, reference_config.target_head_group_size),
        _optional_distance(parallelism.kv_parallel_factor, reference_config.kv_parallel_factor),
        _optional_distance(candidate.metrics.active_cores, reference_config.target_active_cores),
        abs(candidate.k_chunk_size - reference_config.k_chunk_size),
        abs(candidate.q_chunk_size - reference_config.q_chunk_size),
        _candidate_sort_key(candidate),
    )


def _reference_current_a_prefill_score(
    candidate: PlanCandidate,
    workload: MLAWorkloadSpec,
    reference_config: ReferenceCurrentAConfig,
) -> tuple[Any, ...]:
    parallelism = candidate.runtime_config.parallelism_5d
    grouping = candidate.runtime_config.grouping
    compile_time = candidate.compile_time_config

    return (
        compile_time.topology_mode != reference_config.topology_mode,
        compile_time.sblock_topology_name is not None,
        compile_time.dual_noc_policy != reference_config.dual_noc_policy,
        compile_time.math_fidelity != reference_config.math_fidelity,
        parallelism.device_parallel_factor != reference_config.device_parallel_factor,
        compile_time.layout_policy != reference_config.target_layout_policy,
        (reference_config.kv_parallel_factor is not None and parallelism.kv_parallel_factor != reference_config.kv_parallel_factor),
        abs(candidate.k_chunk_size - reference_config.k_chunk_size),
        abs(candidate.q_chunk_size - reference_config.q_chunk_size),
        _optional_distance(grouping.head_group_size, reference_config.target_head_group_size),
        abs(grouping.batch_group_size - max(1, workload.batch_size)),
        _candidate_sort_key(candidate),
    )


def _reference_current_a_score(
    candidate: PlanCandidate,
    workload: MLAWorkloadSpec,
    reference_config: ReferenceCurrentAConfig,
) -> tuple[Any, ...]:
    if workload.mode == "decode":
        return _reference_current_a_decode_score(candidate, workload, reference_config)
    return _reference_current_a_prefill_score(candidate, workload, reference_config)


def _select_candidate(
    candidates: list[PlanCandidate],
    workload: MLAWorkloadSpec,
    reference_config: ReferenceCurrentAConfig,
    selection_mode: ProfileMeasurementSelectionMode,
) -> tuple[PlanCandidate, str]:
    if selection_mode == "analytical_best":
        return candidates[0], "analytical_best"
    return min(candidates, key=lambda candidate: _reference_current_a_score(candidate, workload, reference_config)), (
        "reference_current_a"
    )


def _profile_metadata(
    profile_entry: dict[str, Any],
    workload: MLAWorkloadSpec,
    candidate: PlanCandidate,
    reference_config: ReferenceCurrentAConfig,
    selection_label: str,
    latency_ms: float,
) -> dict[str, Any]:
    analysis = profile_entry.get("analysis", {})
    ratios = analysis.get("ratios", {})
    reader_breakdown = profile_entry.get("custom_reader_breakdown", {})
    reader_shares = reader_breakdown.get("stage_shares", {})
    reader_source_breakdown = profile_entry.get("custom_reader_source_breakdown", {})
    reader_source_shares = reader_source_breakdown.get("stage_shares", {})
    writer_breakdown = profile_entry.get("custom_writer_breakdown", {})
    writer_shares = writer_breakdown.get("stage_shares", {})
    writer_source_breakdown = profile_entry.get("custom_writer_source_breakdown", {})
    writer_source_shares = writer_source_breakdown.get("stage_shares", {})

    return {
        "profile_case": profile_entry["case"],
        "profile_mode": _resolve_profile_mode(profile_entry),
        "profile_variant": profile_entry.get("variant"),
        "profile_latency_ms": latency_ms,
        "profile_selection_mode": selection_label,
        "profile_classification": analysis.get("classification"),
        "profile_reason": ", ".join(analysis.get("reasons", [])),
        "profile_ncrisc_share": ratios.get("ncrisc_share"),
        "profile_brisc_share": ratios.get("brisc_share"),
        "profile_trisc1_share": ratios.get("trisc1_share"),
        "profile_reader_reserve_share": reader_shares.get("SDPA-PAGED-RESERVE-SUM"),
        "profile_reader_issue_share": reader_shares.get("SDPA-PAGED-ISSUE-SUM"),
        "profile_reader_wait_share": reader_shares.get("SDPA-PAGED-WAIT-SUM"),
        "profile_reader_k_reserve_share": reader_source_shares.get("SDPA-K-RESERVE-SUM"),
        "profile_reader_k_issue_share": reader_source_shares.get("SDPA-K-ISSUE-SUM"),
        "profile_reader_k_wait_share": reader_source_shares.get("SDPA-K-WAIT-SUM"),
        "profile_reader_v_reserve_share": reader_source_shares.get("SDPA-V-RESERVE-SUM"),
        "profile_reader_v_issue_share": reader_source_shares.get("SDPA-V-ISSUE-SUM"),
        "profile_reader_v_wait_share": reader_source_shares.get("SDPA-V-WAIT-SUM"),
        "profile_writer_cb_wait_share": writer_shares.get("SDPA-WRITER-CB-WAIT-SUM"),
        "profile_writer_issue_share": writer_shares.get("SDPA-WRITER-ISSUE-SUM"),
        "profile_writer_sender_cb_wait_share": writer_source_shares.get("SDPA-WRITER-SENDER-CB-WAIT-SUM"),
        "profile_writer_root_cb_wait_share": writer_source_shares.get("SDPA-WRITER-ROOT-CB-WAIT-SUM"),
        "profile_writer_tree_child_wait_share": writer_source_shares.get("SDPA-WRITER-TREE-CHILD-WAIT-SUM"),
        "profile_writer_output_gather_wait_share": writer_source_shares.get("SDPA-WRITER-OUTPUT-GATHER-WAIT-SUM"),
        "analytical_latency_ms": candidate.metrics.selected_latency_ms,
        "analytical_score": candidate.metrics.analytical_score,
        "mapped_workload_name": workload.name,
        "mapped_candidate_topology_mode": candidate.compile_time_config.topology_mode,
        "mapped_candidate_math_fidelity": candidate.compile_time_config.math_fidelity,
        "reference_source": reference_config.source,
        "reference_q_chunk_size": reference_config.q_chunk_size,
        "reference_k_chunk_size": reference_config.k_chunk_size,
        "reference_topology_mode": reference_config.topology_mode,
        "reference_dual_noc_policy": reference_config.dual_noc_policy,
        "reference_math_fidelity": reference_config.math_fidelity,
        "reference_max_cores_per_head_batch": reference_config.max_cores_per_head_batch,
    }


def map_wh_profile_measurements(
    profile_results_path: str | Path,
    *,
    heuristic_pruning: bool = True,
    selection_mode: ProfileMeasurementSelectionMode = "reference_current_a",
    reference_source: ReferenceCurrentASource = "profile_harness",
) -> tuple[MappedProfileMeasurement, ...]:
    payload = json.loads(Path(profile_results_path).read_text())
    profile_payload_kind = _detect_profile_payload_kind(payload)
    wh_profile_entries = _extract_profile_entries(payload)
    if not wh_profile_entries:
        raise RuntimeError("profile JSON does not contain any profile entries")

    autotuner = BasicMLAAutotuner(heuristic_pruning=heuristic_pruning)
    mapped_measurements: list[MappedProfileMeasurement] = []
    for profile_entry in wh_profile_entries:
        workload = _profile_case_to_workload(profile_entry)
        reference_config = _resolve_reference_current_a_config_for_source(
            workload, profile_entry, reference_source, profile_payload_kind
        )
        candidates = _rank_candidates(autotuner, workload)
        candidate, selection_label = _select_candidate(candidates, workload, reference_config, selection_mode)
        latency_ms = _profile_latency_ms(profile_entry)
        metadata = _profile_metadata(profile_entry, workload, candidate, reference_config, selection_label, latency_ms)
        mapped_measurements.append(
            MappedProfileMeasurement(
                profile_case=str(profile_entry["case"]),
                profile_mode=_resolve_profile_mode(profile_entry),
                profile_variant=profile_entry.get("variant"),
                workload=workload,
                candidate=candidate,
                latency_ms=latency_ms,
                selection_mode=selection_label,
                reference_config=reference_config,
                metadata=metadata,
            )
        )
    return tuple(mapped_measurements)


def build_wh_profile_measurement_db(
    profile_results_path: str | Path,
    output_path: str | Path,
    *,
    heuristic_pruning: bool = True,
    selection_mode: ProfileMeasurementSelectionMode = "reference_current_a",
    reference_source: ReferenceCurrentASource = "profile_harness",
) -> MeasurementDBBuildSummary:
    output_path = Path(output_path)
    measurement_db = MeasurementDB(path=output_path)
    mapped_measurements = map_wh_profile_measurements(
        profile_results_path,
        heuristic_pruning=heuristic_pruning,
        selection_mode=selection_mode,
        reference_source=reference_source,
    )

    summary_entries: list[dict[str, Any]] = []
    for mapped in mapped_measurements:
        measurement_db.put(mapped.candidate.candidate_key, mapped.latency_ms, metadata=mapped.metadata)
        summary_entries.append(
            {
                "profile_case": mapped.profile_case,
                "profile_mode": mapped.profile_mode,
                "candidate_key": mapped.candidate.candidate_key,
                "latency_ms": mapped.latency_ms,
                "selection_mode": mapped.selection_mode,
                "analytical_latency_ms": mapped.candidate.metrics.selected_latency_ms,
                "reference_source": mapped.reference_config.source,
            }
        )

    measurement_db.save()
    return MeasurementDBBuildSummary(
        output_path=str(output_path),
        selection_mode=selection_mode,
        reference_source=reference_source,
        entry_count=len(summary_entries),
        entries=tuple(summary_entries),
    )
