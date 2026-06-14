"""Hardware-calibrated cost model for MLA autotuner analytical latency.

The analytical path in :class:`BasicMLAAutotuner` produces a scalar
``estimated_latency_ms`` plus a detailed breakdown in :class:`PlanMetrics`.
This module fits a **linear calibration** from those features to measured
kernel time (e.g. Wormhole N300 profile / device measurement DB), so the
tuner can rank candidates using **calibrated** latency instead of raw theory.

Typical workflow
----------------
1. Collect ``MeasurementDB`` JSON (``candidate_key`` -> ``latency_ms``), e.g.
   via ``profile_measurement_bridge.build_wh_profile_measurement_db``.
2. :func:`fit_calibrated_cost_model_from_measurement_db` matches keys to
   enumerated candidates and fits weights.
3. Pass the returned :class:`CalibratedCostModel` into ``tune(...,
   calibrated_cost_model=...)`` or save/load JSON for offline reuse.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from .basic_autotuner import (
    BasicMLAAutotuner,
    CostWeights,
    HardwareTopologySpec,
    MLAWorkloadSpec,
    MeasurementDB,
    PlanCandidate,
    PlanMetrics,
)

FeatureMode = Literal["scalar", "breakdown", "extended"]
CalibrationFeatureMode = Literal["auto", "scalar", "breakdown", "extended"]
LossMode = Literal["mse", "relative"]


def _gaussian_elimination_solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve A x = b for square A (copying inputs)."""
    n = len(b)
    mat = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot_row][col]) < 1e-18:
            raise ValueError("singular or ill-conditioned matrix in calibration fit")
        mat[col], mat[pivot_row] = mat[pivot_row], mat[col]
        piv = mat[col][col]
        for j in range(col, n + 1):
            mat[col][j] /= piv
        for r in range(n):
            if r == col:
                continue
            factor = mat[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                mat[r][j] -= factor * mat[col][j]
    return [mat[i][n] for i in range(n)]


def _ridge_normal_solve(
    xtx: list[list[float]],
    xty: list[float],
    ridge: float,
) -> list[float]:
    n = len(xty)
    aug = [[xtx[i][j] + (ridge if i == j else 0.0) for j in range(n)] for i in range(n)]
    return _gaussian_elimination_solve(aug, xty[:])


def _ols_fit(
    rows: list[list[float]],
    targets: list[float],
    *,
    ridge: float,
    sample_weights: list[float] | None = None,
) -> list[float]:
    """Weighted ridge regression.  Minimises  Σ wᵢ(ŷᵢ − yᵢ)².

    When *sample_weights* is ``None`` every sample has unit weight (standard
    OLS).  For MAPE-targeting loss, pass ``wᵢ = 1/yᵢ²`` so the objective
    becomes ``Σ (ŷᵢ/yᵢ − 1)²`` — directly minimising mean squared relative
    error regardless of absolute scale.
    """
    if not rows or len(rows) != len(targets):
        raise ValueError("rows and targets must be non-empty and equal length")
    p = len(rows[0])
    if p == 0:
        raise ValueError("feature dimension must be positive")
    for row in rows:
        if len(row) != p:
            raise ValueError("inconsistent feature row width")
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for k, (row, y) in enumerate(zip(rows, targets)):
        w = sample_weights[k] if sample_weights is not None else 1.0
        for i in range(p):
            xty[i] += w * row[i] * y
            for j in range(p):
                xtx[i][j] += w * row[i] * row[j]
    return _ridge_normal_solve(xtx, xty, ridge)


def extract_feature_vector(metrics: PlanMetrics, mode: FeatureMode) -> list[float]:
    """Build feature vector with leading 1.0 for intercept."""
    if mode == "scalar":
        return [1.0, metrics.estimated_latency_ms]
    if mode == "breakdown":
        return [
            1.0,
            metrics.compute_ms,
            metrics.dram_ms,
            metrics.noc_ms,
            metrics.sync_ms,
            metrics.reader_ms,
            metrics.reduction_ms,
        ]
    # extended
    reader_path = (
        metrics.reader_ms
        + metrics.reader_backpressure_ms
        + metrics.sender_hotspot_ms
        + metrics.page_control_ms
        + metrics.pipeline_bubble_ms
    )
    return [
        1.0,
        metrics.compute_ms,
        metrics.dram_ms,
        metrics.noc_ms,
        metrics.sync_ms,
        reader_path,
        metrics.reduction_ms,
        metrics.writer_backpressure_ms,
        metrics.overlap_residual_ms,
        metrics.l1_pressure_ms,
        metrics.complexity_penalty_ms,
        metrics.q_preamble_ms,
        float(metrics.l1_usage_ratio),
        float(metrics.imbalance_score),
    ]


FEATURE_LABELS: dict[FeatureMode, tuple[str, ...]] = {
    "scalar": ("bias", "estimated_latency_ms"),
    "breakdown": (
        "bias",
        "compute_ms",
        "dram_ms",
        "noc_ms",
        "sync_ms",
        "reader_ms",
        "reduction_ms",
    ),
    "extended": (
        "bias",
        "compute_ms",
        "dram_ms",
        "noc_ms",
        "sync_ms",
        "reader_path_ms",
        "reduction_ms",
        "writer_backpressure_ms",
        "overlap_residual_ms",
        "l1_pressure_ms",
        "complexity_penalty_ms",
        "q_preamble_ms",
        "l1_usage_ratio",
        "imbalance_score",
    ),
}


def calibration_feature_dimension(mode: FeatureMode) -> int:
    return len(FEATURE_LABELS[mode])


def resolve_calibration_feature_mode(sample_count: int, requested_mode: CalibrationFeatureMode) -> FeatureMode:
    if requested_mode != "auto":
        required_samples = calibration_feature_dimension(requested_mode)
        if sample_count < required_samples:
            raise ValueError(
                f"need at least {required_samples} samples for feature_mode={requested_mode}, got {sample_count}"
            )
        return requested_mode

    for mode in ("extended", "breakdown", "scalar"):
        if sample_count >= calibration_feature_dimension(mode):
            return mode
    raise ValueError(f"need at least 2 samples for feature_mode=auto, got {sample_count}")


def _dot(weights: Sequence[float], features: Sequence[float]) -> float:
    return sum(w * f for w, f in zip(weights, features))


@dataclass(frozen=True)
class CalibrationReport:
    """Diagnostics after fitting to measured latencies."""

    sample_count: int
    feature_mode: FeatureMode
    ridge: float
    weights: tuple[float, ...]
    feature_labels: tuple[str, ...]
    rmse_ms: float
    mae_ms: float
    mape: float
    max_abs_error_ms: float
    mean_measured_ms: float
    mean_predicted_ms: float
    r_squared: float
    loss_mode: LossMode = "mse"
    matched_keys: tuple[str, ...] = ()
    skipped_keys: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "feature_mode": self.feature_mode,
            "ridge": self.ridge,
            "loss_mode": self.loss_mode,
            "weights": list(self.weights),
            "feature_labels": list(self.feature_labels),
            "rmse_ms": self.rmse_ms,
            "mae_ms": self.mae_ms,
            "mape": self.mape,
            "max_abs_error_ms": self.max_abs_error_ms,
            "mean_measured_ms": self.mean_measured_ms,
            "mean_predicted_ms": self.mean_predicted_ms,
            "r_squared": self.r_squared,
            "matched_keys": list(self.matched_keys),
            "skipped_keys": list(self.skipped_keys),
        }


@dataclass
class CalibratedCostModel:
    """Linear map from feature vector to calibrated latency (milliseconds)."""

    feature_mode: FeatureMode
    weights: tuple[float, ...]
    ridge: float = 1e-6
    loss_mode: LossMode = "mse"
    hardware_name: str | None = None
    calibration_report: CalibrationReport | None = None

    def predict(self, metrics: PlanMetrics) -> float:
        features = extract_feature_vector(metrics, self.feature_mode)
        if len(features) != len(self.weights):
            raise ValueError(
                f"feature length {len(features)} != weights length {len(self.weights)}"
            )
        raw = _dot(self.weights, features)
        if raw <= 0.0 or not math.isfinite(raw):
            return float(metrics.estimated_latency_ms)
        return raw

    def save(self, path: str | Path) -> None:
        payload = {
            "feature_mode": self.feature_mode,
            "weights": list(self.weights),
            "ridge": self.ridge,
            "loss_mode": self.loss_mode,
            "hardware_name": self.hardware_name,
            "calibration_report": self.calibration_report.to_dict() if self.calibration_report else None,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "CalibratedCostModel":
        payload = json.loads(Path(path).read_text())
        report_payload = payload.get("calibration_report")
        report: CalibrationReport | None = None
        if report_payload:
            rp = dict(report_payload)
            rp["weights"] = tuple(float(w) for w in rp["weights"])
            rp["matched_keys"] = tuple(rp.get("matched_keys", []))
            rp["skipped_keys"] = tuple(rp.get("skipped_keys", []))
            rp.setdefault("loss_mode", "mse")
            report = CalibrationReport(**rp)
        return cls(
            feature_mode=payload["feature_mode"],
            weights=tuple(float(w) for w in payload["weights"]),
            ridge=float(payload.get("ridge", 1e-6)),
            loss_mode=payload.get("loss_mode", "mse"),
            hardware_name=payload.get("hardware_name"),
            calibration_report=report,
        )


def _recompute_analytical_score(
    metrics: PlanMetrics,
    *,
    new_latency_ms: float,
    weights: CostWeights,
) -> float:
    return (
        weights.latency * new_latency_ms
        + weights.dram_traffic * (metrics.dram_bytes / 1e6)
        + weights.noc_traffic * (metrics.noc_bytes / 1e6)
        + weights.inactive_core_penalty * (1.0 - metrics.active_core_ratio)
        + weights.imbalance_penalty * metrics.imbalance_score
        + weights.l1_overuse_penalty * (max(0.0, metrics.l1_usage_ratio - 1.0) ** 2)
    )


def apply_calibration_to_candidate(
    candidate: PlanCandidate,
    model: CalibratedCostModel,
    *,
    cost_weights: CostWeights,
) -> PlanCandidate:
    """Replace ``selected_latency_ms`` / ``analytical_score`` using calibrated latency."""
    calibrated = model.predict(candidate.metrics)
    new_metrics = PlanMetrics(
        **{
            **asdict(candidate.metrics),
            "selected_latency_ms": calibrated,
            "analytical_score": _recompute_analytical_score(
                candidate.metrics,
                new_latency_ms=calibrated,
                weights=cost_weights,
            ),
        }
    )
    return PlanCandidate(
        candidate_key=candidate.candidate_key,
        compile_time_config=candidate.compile_time_config,
        runtime_config=candidate.runtime_config,
        metrics=new_metrics,
    )


def _metrics_for_calibration_report(
    y: list[float],
    y_hat: list[float],
) -> tuple[float, float, float, float, float, float, float]:
    n = len(y)
    if n == 0:
        raise ValueError("empty calibration set")
    err = [y[i] - y_hat[i] for i in range(n)]
    mae = sum(abs(e) for e in err) / n
    rmse = math.sqrt(sum(e * e for e in err) / n)
    mape = sum(abs(err[i]) / max(abs(y[i]), 1e-12) for i in range(n)) / n
    max_abs = max(abs(e) for e in err)
    mean_y = sum(y) / n
    mean_hat = sum(y_hat) / n
    ss_tot = sum((y[i] - mean_y) ** 2 for i in range(n))
    ss_res = sum(err[i] ** 2 for i in range(n))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0
    return rmse, mae, mape, max_abs, mean_y, mean_hat, r2


def fit_calibrated_cost_model(
    samples: Iterable[tuple[PlanMetrics, float]],
    *,
    feature_mode: CalibrationFeatureMode = "auto",
    ridge: float = 1e-4,
    hardware_name: str | None = None,
    loss_mode: LossMode = "mse",
) -> tuple[CalibratedCostModel, CalibrationReport]:
    """Fit weights so calibrated latency matches measured kernel time.

    Parameters
    ----------
    samples
        Pairs ``(analytical PlanMetrics, measured_latency_ms)``.
    feature_mode
        ``scalar``: only ``estimated_latency_ms``; ``breakdown``: main pipe
        stages; ``extended``: breakdown + backpressure / overlap terms.
    ridge
        Tikhonov regularization on the normal equations (helps when features
        are correlated or sample count is small).
    loss_mode
        ``mse``: standard least squares (minimises absolute error, biased
        toward large-valued samples).
        ``relative``: inverse-variance weighted least squares with
        ``wᵢ = 1/yᵢ²``, so the objective becomes
        ``Σ (ŷᵢ/yᵢ − 1)²`` — directly targeting mean squared *relative*
        error.  This equalises each sample's contribution regardless of
        its absolute magnitude, which is critical when decode latencies
        (~0.04 ms) and prefill latencies (~55 ms) span three orders of
        magnitude.
    """
    metrics_list: list[PlanMetrics] = []
    targets: list[float] = []
    for metrics, measured in samples:
        metrics_list.append(metrics)
        targets.append(float(measured))

    if not metrics_list:
        raise ValueError("samples must be non-empty")

    resolved_feature_mode = resolve_calibration_feature_mode(len(metrics_list), feature_mode)
    rows = [extract_feature_vector(metrics, resolved_feature_mode) for metrics in metrics_list]

    sample_weights: list[float] | None = None
    if loss_mode == "relative":
        sample_weights = [1.0 / max(t, 1e-9) ** 2 for t in targets]

    weights_list = _ols_fit(rows, targets, ridge=ridge, sample_weights=sample_weights)
    weights = tuple(weights_list)
    y_hat = [_dot(weights, row) for row in rows]
    rmse, mae, mape, max_abs, mean_y, mean_hat, r2 = _metrics_for_calibration_report(targets, y_hat)
    labels = FEATURE_LABELS[resolved_feature_mode]
    if len(labels) != len(weights):
        labels = tuple(f"f{i}" for i in range(len(weights)))

    report = CalibrationReport(
        sample_count=len(targets),
        feature_mode=resolved_feature_mode,
        ridge=ridge,
        weights=weights,
        feature_labels=labels,
        rmse_ms=rmse,
        mae_ms=mae,
        mape=mape,
        max_abs_error_ms=max_abs,
        mean_measured_ms=mean_y,
        mean_predicted_ms=mean_hat,
        r_squared=r2,
        loss_mode=loss_mode,
    )
    model = CalibratedCostModel(
        feature_mode=resolved_feature_mode,
        weights=weights,
        ridge=ridge,
        loss_mode=loss_mode,
        hardware_name=hardware_name,
        calibration_report=report,
    )
    return model, report


def fit_calibrated_cost_model_from_measurement_db(
    measurement_db: MeasurementDB,
    candidate_by_key: dict[str, PlanCandidate],
    *,
    feature_mode: CalibrationFeatureMode = "auto",
    ridge: float = 1e-4,
    hardware_name: str | None = None,
    loss_mode: LossMode = "mse",
) -> tuple[CalibratedCostModel, CalibrationReport]:
    """Fit using a :class:`MeasurementDB` and a pre-built candidate map."""
    samples: list[tuple[PlanMetrics, float]] = []
    matched: list[str] = []
    skipped: list[str] = []
    for key, record in measurement_db.records.items():
        candidate = candidate_by_key.get(key)
        if candidate is None:
            skipped.append(key)
            continue
        samples.append((candidate.metrics, float(record["latency_ms"])))
        matched.append(key)

    if not samples:
        raise RuntimeError(
            "no measurement keys matched candidates; check workload/hardware alignment"
        )

    model, report = fit_calibrated_cost_model(
        samples,
        feature_mode=feature_mode,
        ridge=ridge,
        hardware_name=hardware_name,
        loss_mode=loss_mode,
    )
    report = replace(report, matched_keys=tuple(matched), skipped_keys=tuple(skipped))
    model = replace(model, calibration_report=report)
    return model, report


def fit_calibrated_cost_model_from_wh_profile(
    profile_results_path: str | Path,
    *,
    heuristic_pruning: bool = True,
    selection_mode: Literal["reference_current_a", "analytical_best"] = "reference_current_a",
    reference_source: Literal["profile_harness", "mla1d_defaults", "hybrid_auto"] = "profile_harness",
    feature_mode: CalibrationFeatureMode = "auto",
    ridge: float = 1e-4,
    hardware_name: str | None = "wormhole_b0",
    loss_mode: LossMode = "mse",
) -> tuple[CalibratedCostModel, CalibrationReport]:
    from .profile_measurement_bridge import map_wh_profile_measurements

    mapped_measurements = map_wh_profile_measurements(
        profile_results_path,
        heuristic_pruning=heuristic_pruning,
        selection_mode=selection_mode,
        reference_source=reference_source,
    )
    if not mapped_measurements:
        raise RuntimeError("no WH profile measurements were mapped to autotuner candidates")

    samples = [(mapped.candidate.metrics, mapped.latency_ms) for mapped in mapped_measurements]
    matched_keys = tuple(mapped.candidate.candidate_key for mapped in mapped_measurements)
    model, report = fit_calibrated_cost_model(
        samples,
        feature_mode=feature_mode,
        ridge=ridge,
        hardware_name=hardware_name,
        loss_mode=loss_mode,
    )
    report = replace(report, matched_keys=matched_keys)
    model = replace(model, calibration_report=report)
    return model, report


def build_candidate_map(
    autotuner: BasicMLAAutotuner,
    workload: MLAWorkloadSpec,
    hardware: HardwareTopologySpec,
) -> dict[str, PlanCandidate]:
    """Enumerate all feasible candidates and index by ``candidate_key``."""
    candidates = autotuner._enumerate_candidates(workload, hardware)
    return {c.candidate_key: c for c in candidates}


def fit_calibrated_cost_model_for_workload(
    measurement_db: MeasurementDB,
    workload: MLAWorkloadSpec,
    hardware: HardwareTopologySpec,
    *,
    autotuner: BasicMLAAutotuner | None = None,
    feature_mode: CalibrationFeatureMode = "auto",
    ridge: float = 1e-4,
    loss_mode: LossMode = "mse",
) -> tuple[CalibratedCostModel, CalibrationReport]:
    """Convenience: enumerate candidates for ``(workload, hardware)`` then fit."""
    tuner = autotuner or BasicMLAAutotuner()
    cmap = build_candidate_map(tuner, workload, hardware)
    return fit_calibrated_cost_model_from_measurement_db(
        measurement_db,
        cmap,
        feature_mode=feature_mode,
        ridge=ridge,
        hardware_name=getattr(hardware, "name", None),
        loss_mode=loss_mode,
    )


# ---------------------------------------------------------------------------
# Mode-stratified calibration
# ---------------------------------------------------------------------------

Mode = Literal["decode", "prefill"]


@dataclass
class StratifiedCalibratedCostModel:
    """Separate calibration models for decode and prefill.

    MLA decode (memory-bound, Q_len=1) and prefill (compute-bound, Q_len=L)
    have fundamentally different bottleneck structures, so a single linear
    model struggles to fit both regimes simultaneously.  This wrapper holds
    independent sub-models and dispatches ``predict`` based on the workload
    mode.
    """

    decode_model: CalibratedCostModel
    prefill_model: CalibratedCostModel

    def predict(self, metrics: PlanMetrics, *, mode: Mode | None = None) -> float:
        """Return calibrated latency, dispatching to the per-mode sub-model.

        If *mode* is ``None``, infer from ``metrics.q_num_chunks``: a single
        Q chunk (``q_num_chunks <= 1``) implies decode.
        """
        if mode is None:
            mode = "decode" if metrics.q_num_chunks <= 1 else "prefill"
        if mode == "decode":
            return self.decode_model.predict(metrics)
        return self.prefill_model.predict(metrics)

    def save(self, path: str | Path) -> None:
        payload = {
            "stratified": True,
            "decode": {
                "feature_mode": self.decode_model.feature_mode,
                "weights": list(self.decode_model.weights),
                "ridge": self.decode_model.ridge,
                "loss_mode": self.decode_model.loss_mode,
                "hardware_name": self.decode_model.hardware_name,
                "calibration_report": (
                    self.decode_model.calibration_report.to_dict()
                    if self.decode_model.calibration_report
                    else None
                ),
            },
            "prefill": {
                "feature_mode": self.prefill_model.feature_mode,
                "weights": list(self.prefill_model.weights),
                "ridge": self.prefill_model.ridge,
                "loss_mode": self.prefill_model.loss_mode,
                "hardware_name": self.prefill_model.hardware_name,
                "calibration_report": (
                    self.prefill_model.calibration_report.to_dict()
                    if self.prefill_model.calibration_report
                    else None
                ),
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "StratifiedCalibratedCostModel":
        payload = json.loads(Path(path).read_text())
        if not payload.get("stratified"):
            raise ValueError("JSON is not a stratified calibration model")

        def _load_sub(sub: dict[str, Any]) -> CalibratedCostModel:
            report_payload = sub.get("calibration_report")
            report: CalibrationReport | None = None
            if report_payload:
                rp = dict(report_payload)
                rp["weights"] = tuple(float(w) for w in rp["weights"])
                rp["matched_keys"] = tuple(rp.get("matched_keys", []))
                rp["skipped_keys"] = tuple(rp.get("skipped_keys", []))
                rp.setdefault("loss_mode", "mse")
                report = CalibrationReport(**rp)
            return CalibratedCostModel(
                feature_mode=sub["feature_mode"],
                weights=tuple(float(w) for w in sub["weights"]),
                ridge=float(sub.get("ridge", 1e-6)),
                loss_mode=sub.get("loss_mode", "mse"),
                hardware_name=sub.get("hardware_name"),
                calibration_report=report,
            )

        return cls(
            decode_model=_load_sub(payload["decode"]),
            prefill_model=_load_sub(payload["prefill"]),
        )


def fit_stratified_cost_model(
    decode_samples: Iterable[tuple[PlanMetrics, float]],
    prefill_samples: Iterable[tuple[PlanMetrics, float]],
    *,
    feature_mode: CalibrationFeatureMode = "auto",
    ridge: float = 1e-4,
    hardware_name: str | None = None,
    loss_mode: LossMode = "mse",
) -> tuple[StratifiedCalibratedCostModel, CalibrationReport, CalibrationReport]:
    """Fit independent calibration models for decode and prefill."""
    decode_model, decode_report = fit_calibrated_cost_model(
        decode_samples,
        feature_mode=feature_mode,
        ridge=ridge,
        hardware_name=hardware_name,
        loss_mode=loss_mode,
    )
    prefill_model, prefill_report = fit_calibrated_cost_model(
        prefill_samples,
        feature_mode=feature_mode,
        ridge=ridge,
        hardware_name=hardware_name,
        loss_mode=loss_mode,
    )
    model = StratifiedCalibratedCostModel(
        decode_model=decode_model,
        prefill_model=prefill_model,
    )
    return model, decode_report, prefill_report
