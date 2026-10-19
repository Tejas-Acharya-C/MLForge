"""MLForge Offline HTML Audit Report Generation Module (Milestone 4.2).

Transforms saved evaluation artifacts, manifests, diagnostics, robustness curves,
and acceptance decisions into a self-contained, offline HTML audit report.

SECURITY & AUDIT GUARANTEES
---------------------------
1. Standalone & Offline: Zero external fonts, CDNs, remote images, scripts, or network requests.
2. Safe Presentation: NEVER deserializes or executes model (.joblib/.pickle) artifacts.
3. Strict Output Confinement: Protects against path traversal in artifact resolution and report destination.
4. Robust Escaping: HTML-escapes all user-supplied, dataset, model, and diagnostic strings.
5. Atomic Output: Writes to a temporary file and atomically replaces destination to prevent partial writes.
6. Immutable Source: Never mutates existing manifests or evidence as a side effect.
"""

from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import zipfile

from mlforge.application.artifacts import load_run_artifacts
from mlforge.application.packaging import verify_audit_bundle
from mlforge.core.contracts import (
    AcceptanceDecision,
    AcceptanceOutcome,
    AcceptancePolicy,
    ArtifactManifest,
    AuditBundleManifest,
    CalibrationBin,
    CalibrationCurveResult,
    ComparisonReport,
    CriterionEvaluation,
    CriterionOutcome,
    CriterionSeverity,
    ErrorDiagnosticsResult,
    EvaluationResult,
    FoldErrorDiagnostic,
    FoldResult,
    MetricComparisonResult,
    MetricDirection,
    MetricSummary,
    ModelEvaluationResult,
    PerturbationType,
    ReleaseCertificate,
    RobustnessCurve,
    RobustnessEvaluationResult,
    RobustnessPerturbationResult,
    StatisticalTestResult,
    SubgroupMetricDiagnostic,
    TestStatus,
    ThresholdAnalysisResult,
    ThresholdMetrics,
)
from mlforge.core.exceptions import (
    ArtifactError,
    ArtifactLoadError,
    ArtifactSecurityError,
    ArtifactSerializationError,
    AuditBundleError,
)

REPORT_SCHEMA_VERSION = "1.0.0"


def _esc(val: Any) -> str:
    """HTML-escape a value safely."""
    if val is None:
        return ""
    return html.escape(str(val), quote=True)


def _fmt_float(val: Optional[float], decimals: int = 4) -> str:
    """Format float values with defensive handling of None, NaN, and Inf."""
    if val is None:
        return '<span class="val-na">N/A</span>'
    if isinstance(val, (int, float)):
        if math.isnan(val) or math.isinf(val):
            return '<span class="badge badge-warn">undefined</span>'
        return f"{val:.{decimals}f}"
    return _esc(str(val))


def _fmt_pct(val: Optional[float], decimals: int = 2) -> str:
    """Format percentage values safely."""
    if val is None:
        return '<span class="val-na">N/A</span>'
    if isinstance(val, (int, float)):
        if math.isnan(val) or math.isinf(val):
            return '<span class="badge badge-warn">undefined</span>'
        return f"{val:.{decimals}f}%"
    return _esc(str(val))


def _fmt_bytes(num_bytes: Optional[int]) -> str:
    """Format byte counts into human-readable strings."""
    if num_bytes is None:
        return "N/A"
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


# ---------------------------------------------------------------------------
# Inline SVG Visualization Renderers (100% Offline, Zero JS, Accessible)
# ---------------------------------------------------------------------------

def _render_calibration_svg(cal: Optional[CalibrationCurveResult]) -> str:
    """Render inline SVG reliability diagram for probability calibration."""
    if not cal or not cal.bins:
        return '<div class="chart-empty">No calibration data available for visualization.</div>'

    width, height = 360, 260
    pad_left, pad_right, pad_top, pad_bottom = 45, 20, 20, 40
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    # Diagonal ideal line: (0,0) to (1,1)
    d_x0, d_y0 = pad_left, pad_top + plot_h
    d_x1, d_y1 = pad_left + plot_w, pad_top

    points_svg = []
    poly_pts = []

    for b in cal.bins:
        obs_rate = getattr(b, "observed_positive_rate", None)
        if obs_rate is None:
            obs_rate = getattr(b, "empirical_positive_rate", None)
        if b.is_empty or b.mean_predicted_prob is None or obs_rate is None:
            continue
        cx = pad_left + max(0.0, min(1.0, b.mean_predicted_prob)) * plot_w
        cy = pad_top + (1.0 - max(0.0, min(1.0, obs_rate))) * plot_h
        poly_pts.append(f"{cx:.1f},{cy:.1f}")
        points_svg.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" class="svg-point" '
            f'data-info="Bin {b.bin_index}: Pred {_fmt_float(b.mean_predicted_prob, 3)}, '
            f'Obs {_fmt_float(obs_rate, 3)}, N={b.sample_count}"/>'
        )

    polyline_svg = ""
    if len(poly_pts) >= 2:
        polyline_svg = f'<polyline points="{" ".join(poly_pts)}" fill="none" class="svg-line-primary" stroke-width="2"/>'

    svg = f"""
    <svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" aria-label="Probability Calibration Reliability Diagram">
        <!-- Grid and Axes -->
        <rect x="{pad_left}" y="{pad_top}" width="{plot_w}" height="{plot_h}" fill="#161b22" stroke="#30363d"/>
        <line x1="{pad_left}" y1="{pad_top + plot_h/2}" x2="{pad_left + plot_w}" y2="{pad_top + plot_h/2}" stroke="#21262d" stroke-dasharray="2,2"/>
        <line x1="{pad_left + plot_w/2}" y1="{pad_top}" x2="{pad_left + plot_w/2}" y2="{pad_top + plot_h}" stroke="#21262d" stroke-dasharray="2,2"/>

        <!-- Ideal diagonal -->
        <line x1="{d_x0}" y1="{d_y0}" x2="{d_x1}" y2="{d_y1}" stroke="#8b949e" stroke-dasharray="4,4" stroke-width="1.5"/>

        <!-- Model calibration curve -->
        {polyline_svg}
        {"".join(points_svg)}

        <!-- Axis Labels -->
        <text x="{pad_left}" y="{pad_top + plot_h + 15}" class="svg-axis-lbl">0.0</text>
        <text x="{pad_left + plot_w/2}" y="{pad_top + plot_h + 15}" text-anchor="middle" class="svg-axis-lbl">0.5</text>
        <text x="{pad_left + plot_w}" y="{pad_top + plot_h + 15}" text-anchor="end" class="svg-axis-lbl">1.0</text>
        <text x="{pad_left + plot_w/2}" y="{pad_top + plot_h + 32}" text-anchor="middle" class="svg-title-lbl">Mean Predicted Probability</text>

        <text x="{pad_left - 8}" y="{pad_top + plot_h}" text-anchor="end" class="svg-axis-lbl">0.0</text>
        <text x="{pad_left - 8}" y="{pad_top + plot_h/2 + 4}" text-anchor="end" class="svg-axis-lbl">0.5</text>
        <text x="{pad_left - 8}" y="{pad_top + 8}" text-anchor="end" class="svg-axis-lbl">1.0</text>
        <text x="{pad_left - 26}" y="{pad_top + plot_h/2}" text-anchor="middle" transform="rotate(-90 {pad_left - 26} {pad_top + plot_h/2})" class="svg-title-lbl">Empirical Positive Rate</text>
    </svg>
    """
    return svg


def _render_threshold_svg(ta: Optional[ThresholdAnalysisResult]) -> str:
    """Render inline SVG trade-off curve across decision cutoffs."""
    if not ta or not ta.threshold_metrics:
        return '<div class="chart-empty">No threshold sweep data available for visualization.</div>'

    width, height = 360, 260
    pad_left, pad_right, pad_top, pad_bottom = 45, 20, 20, 40
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    f1_pts, prec_pts, rec_pts = [], [], []
    for tm in ta.threshold_metrics:
        x = pad_left + max(0.0, min(1.0, tm.threshold)) * plot_w
        if tm.f1_score is not None:
            f1_pts.append(f"{x:.1f},{pad_top + (1.0 - max(0.0, min(1.0, tm.f1_score))) * plot_h:.1f}")
        if tm.precision is not None:
            prec_pts.append(f"{x:.1f},{pad_top + (1.0 - max(0.0, min(1.0, tm.precision))) * plot_h:.1f}")
        if tm.recall is not None:
            rec_pts.append(f"{x:.1f},{pad_top + (1.0 - max(0.0, min(1.0, tm.recall))) * plot_h:.1f}")

    lines = []
    if len(f1_pts) >= 2:
        lines.append(f'<polyline points="{" ".join(f1_pts)}" fill="none" stroke="#58a6ff" stroke-width="2"/>')
    if len(prec_pts) >= 2:
        lines.append(f'<polyline points="{" ".join(prec_pts)}" fill="none" stroke="#3fb950" stroke-width="1.5" stroke-dasharray="3,3"/>')
    if len(rec_pts) >= 2:
        lines.append(f'<polyline points="{" ".join(rec_pts)}" fill="none" stroke="#d29922" stroke-width="1.5" stroke-dasharray="1,2"/>')

    svg = f"""
    <svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" aria-label="Decision Threshold Trade-Off Curve">
        <rect x="{pad_left}" y="{pad_top}" width="{plot_w}" height="{plot_h}" fill="#161b22" stroke="#30363d"/>
        <line x1="{pad_left}" y1="{pad_top + plot_h/2}" x2="{pad_left + plot_w}" y2="{pad_top + plot_h/2}" stroke="#21262d" stroke-dasharray="2,2"/>
        {"".join(lines)}

        <!-- Axis Labels -->
        <text x="{pad_left}" y="{pad_top + plot_h + 15}" class="svg-axis-lbl">0.0</text>
        <text x="{pad_left + plot_w/2}" y="{pad_top + plot_h + 15}" text-anchor="middle" class="svg-axis-lbl">0.5</text>
        <text x="{pad_left + plot_w}" y="{pad_top + plot_h + 15}" text-anchor="end" class="svg-axis-lbl">1.0</text>
        <text x="{pad_left + plot_w/2}" y="{pad_top + plot_h + 32}" text-anchor="middle" class="svg-title-lbl">Decision Threshold</text>

        <text x="{pad_left - 8}" y="{pad_top + plot_h}" text-anchor="end" class="svg-axis-lbl">0.0</text>
        <text x="{pad_left - 8}" y="{pad_top + 8}" text-anchor="end" class="svg-axis-lbl">1.0</text>
        <text x="{pad_left - 26}" y="{pad_top + plot_h/2}" text-anchor="middle" transform="rotate(-90 {pad_left - 26} {pad_top + plot_h/2})" class="svg-title-lbl">Metric Value</text>
    </svg>
    """
    return svg


def _render_robustness_svg(curves: List[RobustnessCurve]) -> str:
    """Render inline SVG degradation trajectory across stress levels."""
    if not curves:
        return '<div class="chart-empty">No robustness curves available for visualization.</div>'

    width, height = 360, 260
    pad_left, pad_right, pad_top, pad_bottom = 45, 20, 20, 40
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    colors = ["#58a6ff", "#3fb950", "#d29922", "#bc8cff"]
    curve_elements = []

    for c_idx, curve in enumerate(curves):
        color = colors[c_idx % len(colors)]
        pts = []
        for pt in curve.points:
            if not pt.is_usable or pt.retained_ratio is None:
                continue
            x = pad_left + max(0.0, min(1.0, pt.severity_level)) * plot_w
            y = pad_top + (1.0 - max(0.0, min(1.2, pt.retained_ratio))) * (plot_h / 1.2)
            pts.append(f"{x:.1f},{y:.1f}")
            curve_elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')

        if len(pts) >= 2:
            curve_elements.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2"/>')

    # Reference 1.0 retention baseline
    y_one = pad_top + (1.0 - (1.0 / 1.2)) * plot_h

    svg = f"""
    <svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" aria-label="Robustness Degradation Trajectory">
        <rect x="{pad_left}" y="{pad_top}" width="{plot_w}" height="{plot_h}" fill="#161b22" stroke="#30363d"/>
        <line x1="{pad_left}" y1="{y_one}" x2="{pad_left + plot_w}" y2="{y_one}" stroke="#8b949e" stroke-dasharray="3,3" stroke-width="1"/>
        <text x="{pad_left + plot_w - 4}" y="{y_one - 4}" text-anchor="end" class="svg-axis-lbl">1.0 (Clean)</text>

        {"".join(curve_elements)}

        <!-- Axis Labels -->
        <text x="{pad_left}" y="{pad_top + plot_h + 15}" class="svg-axis-lbl">0.0</text>
        <text x="{pad_left + plot_w/2}" y="{pad_top + plot_h + 15}" text-anchor="middle" class="svg-axis-lbl">0.5</text>
        <text x="{pad_left + plot_w}" y="{pad_top + plot_h + 15}" text-anchor="end" class="svg-axis-lbl">1.0</text>
        <text x="{pad_left + plot_w/2}" y="{pad_top + plot_h + 32}" text-anchor="middle" class="svg-title-lbl">Perturbation Severity</text>

        <text x="{pad_left - 8}" y="{pad_top + plot_h}" text-anchor="end" class="svg-axis-lbl">0.0</text>
        <text x="{pad_left - 8}" y="{pad_top + 8}" text-anchor="end" class="svg-axis-lbl">1.2</text>
        <text x="{pad_left - 26}" y="{pad_top + plot_h/2}" text-anchor="middle" transform="rotate(-90 {pad_left - 26} {pad_top + plot_h/2})" class="svg-title-lbl">Retained Performance</text>
    </svg>
    """
    return svg


# ---------------------------------------------------------------------------
# Section HTML Builders
# ---------------------------------------------------------------------------

def _build_header_section(
    manifest: Union[ArtifactManifest, AuditBundleManifest],
    evaluation: EvaluationResult,
    source_type: str,
) -> str:
    """Section A: Header and Run Identity."""
    run_id = manifest.run_id or evaluation.dataset_metadata.checksum_sha256[:8]
    ds_hash = manifest.dataset_checksum or evaluation.dataset_metadata.checksum_sha256
    model_provenance = getattr(manifest, "model_provenance", None) or evaluation.candidate_result.model_provenance or "unknown"
    cand_fold_idx = getattr(manifest, "candidate_model_fold_index", None) or evaluation.candidate_result.fitted_fold_index
    base_fold_idx = getattr(manifest, "baseline_model_fold_index", None) or (
        evaluation.baseline_result.fitted_fold_index if evaluation.baseline_result else None
    )

    prov_str = _esc(model_provenance)
    if cand_fold_idx is not None:
        prov_str += f" (Candidate Fold #{cand_fold_idx})"
    if base_fold_idx is not None:
        prov_str += f" (Baseline Fold #{base_fold_idx})"

    source_badge = (
        '<span class="badge badge-accent">Verified Audit Bundle (.zip)</span>'
        if source_type == "audit_bundle"
        else '<span class="badge badge-muted">Local Run Directory</span>'
    )

    return f"""
    <header class="report-header">
        <div class="header-top">
            <div>
                <span class="brand-title">MLForge</span>
                <span class="report-subtitle">Model Evaluation &amp; Reliability Audit Report</span>
            </div>
            <div>
                {source_badge}
                <span class="badge badge-muted">Schema v{_esc(manifest.schema_version)}</span>
            </div>
        </div>

        <div class="identity-grid">
            <div class="id-item">
                <span class="id-label">Run Identifier</span>
                <span class="id-value font-mono">{_esc(run_id)}</span>
            </div>
            <div class="id-item">
                <span class="id-label">Dataset SHA-256</span>
                <span class="id-value font-mono truncate" title="{_esc(ds_hash)}">{_esc(ds_hash[:16])}...</span>
            </div>
            <div class="id-item">
                <span class="id-label">Split Protocol</span>
                <span class="id-value">{_esc(evaluation.reproducibility.split_strategy)} ({evaluation.reproducibility.num_folds} Folds)</span>
            </div>
            <div class="id-item">
                <span class="id-label">Random Seed</span>
                <span class="id-value font-mono">{evaluation.reproducibility.random_seed}</span>
            </div>
            <div class="id-item">
                <span class="id-label">Candidate Model</span>
                <span class="id-value">{_esc(evaluation.candidate_result.model_name)} <span class="text-muted">({_esc(evaluation.candidate_result.model_type)})</span></span>
            </div>
            <div class="id-item">
                <span class="id-label">Baseline Model</span>
                <span class="id-value">{_esc(evaluation.baseline_result.model_name if evaluation.baseline_result else "None (Single Model Evaluation)")}</span>
            </div>
            <div class="id-item full-width">
                <span class="id-label">Model Estimator Provenance</span>
                <span class="id-value font-mono">{prov_str}</span>
            </div>
        </div>
        <div class="notice-bar">
            <strong>Cryptographic Notice:</strong> Checksum verification confirms local artifact integrity against the run manifest. It does not establish external author authenticity or guarantee model trustworthiness.
        </div>
    </header>
    """


def _build_executive_summary_section(
    manifest: Union[ArtifactManifest, AuditBundleManifest],
    evaluation: EvaluationResult,
    decision: Optional[AcceptanceDecision],
) -> str:
    """Section B: Executive Evaluation Summary."""
    cand = evaluation.candidate_result
    base = evaluation.baseline_result

    # Decision banner
    if decision:
        outcome = decision.outcome.value if hasattr(decision.outcome, "value") else str(decision.outcome)
        outcome_upper = outcome.upper()
        if outcome_upper == "PASS":
            outcome_class = "banner-pass"
            outcome_icon = "[PASS]"
        elif outcome_upper == "FAIL":
            outcome_class = "banner-fail"
            outcome_icon = "[FAIL]"
        else:
            outcome_class = "banner-inconclusive"
            outcome_icon = "[INCONCLUSIVE]"

        decision_html = f"""
        <div class="decision-banner {outcome_class}">
            <div class="decision-status">
                <span class="decision-badge">{outcome_icon}</span>
                <span class="decision-title">Acceptance Decision: {outcome_upper}</span>
            </div>
            <div class="decision-meta">
                <span>Passed: <strong>{decision.passed_rules_count}</strong></span>
                <span>Failed: <strong>{decision.failed_rules_count}</strong></span>
                <span>Inconclusive: <strong>{decision.inconclusive_rules_count}</strong></span>
                <span>Advisories: <strong>{len(decision.advisory_warnings)}</strong></span>
            </div>
            <p class="decision-rationale">{_esc(decision.summary_rationale)}</p>
        </div>
        """
    else:
        decision_html = """
        <div class="decision-banner banner-neutral">
            <div class="decision-status">
                <span class="decision-badge">[NO DECISION]</span>
                <span class="decision-title">No Acceptance Decision Evaluated</span>
            </div>
            <p class="decision-rationale">This evaluation run does not contain a saved acceptance decision or policy evaluation artifact.</p>
        </div>
        """

    # Key metric cards
    key_metrics = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "brier_score", "log_loss"]
    cards_html = []

    for m in key_metrics:
        cand_summary = cand.metric_summaries.get(m)
        cand_val = cand_summary.mean if cand_summary else None
        base_summary = base.metric_summaries.get(m) if base else None
        base_val = base_summary.mean if base_summary else None

        delta_str = ""
        if cand_val is not None and base_val is not None:
            delta = cand_val - base_val
            sign = "+" if delta > 0 else ""
            delta_str = f'<span class="card-delta font-mono">Δ {sign}{delta:.4f}</span>'

        label = m.replace("_", " ").title()
        cards_html.append(f"""
        <div class="metric-card">
            <span class="card-label">{label}</span>
            <span class="card-value font-mono">{_fmt_float(cand_val, 4)}</span>
            <div class="card-sub">
                <span class="text-muted">Base: {_fmt_float(base_val, 4)}</span>
                {delta_str}
            </div>
        </div>
        """)

    # Warning notices
    warnings_html = ""
    all_warnings = list(evaluation.warnings)
    if decision and decision.advisory_warnings:
        all_warnings.extend(decision.advisory_warnings)

    if all_warnings:
        warn_items = "".join(f"<li>{_esc(w)}</li>" for w in all_warnings)
        warnings_html = f"""
        <div class="advisory-box">
            <h4>Advisories &amp; Evaluation Warnings ({len(all_warnings)})</h4>
            <ul>{warn_items}</ul>
        </div>
        """

    return f"""
    <section class="report-section" id="executive-summary">
        <h2>Executive Evaluation Summary</h2>
        {decision_html}
        <div class="metrics-card-grid">
            {"".join(cards_html)}
        </div>
        {warnings_html}
    </section>
    """


def _build_performance_section(evaluation: EvaluationResult) -> str:
    """Section C: Detailed Classification Performance."""
    cand = evaluation.candidate_result
    base = evaluation.baseline_result

    # 1. Cross-validation aggregate table
    rows = []
    for metric_name, summary in sorted(cand.metric_summaries.items()):
        ci_str = "N/A"
        if summary.confidence_interval_95 and len(summary.confidence_interval_95) == 2:
            ci_str = f"[{_fmt_float(summary.confidence_interval_95[0], 4)}, {_fmt_float(summary.confidence_interval_95[1], 4)}]"

        rows.append(f"""
        <tr>
            <td class="font-mono font-bold">{_esc(metric_name)}</td>
            <td class="text-right font-mono">{_fmt_float(summary.mean, 4)}</td>
            <td class="text-right font-mono">{_fmt_float(summary.std, 4)}</td>
            <td class="text-center font-mono">{ci_str}</td>
            <td class="text-right font-mono">{_fmt_float(summary.median, 4)}</td>
            <td class="text-right font-mono">{_fmt_float(summary.min_value, 4)}</td>
            <td class="text-right font-mono">{_fmt_float(summary.max_value, 4)}</td>
            <td class="text-center font-mono">{summary.valid_fold_count} / {summary.valid_fold_count + summary.undefined_fold_count}</td>
        </tr>
        """)

    aggregate_table = f"""
    <div class="table-container">
        <table class="data-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th class="text-right">Mean</th>
                    <th class="text-right">Std Dev</th>
                    <th class="text-center">95% CI</th>
                    <th class="text-right">Median</th>
                    <th class="text-right">Min</th>
                    <th class="text-right">Max</th>
                    <th class="text-center">Valid Folds</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    </div>
    """

    # 2. Fold breakdown table
    fold_headers = ["Fold", "Train N", "Val N", "Accuracy", "Balanced Acc", "Macro F1", "ROC-AUC", "Brier", "Fit (s)"]
    fold_rows = []
    for f in cand.fold_results:
        m = f.metrics
        fold_rows.append(f"""
        <tr>
            <td class="font-mono text-center">#{f.fold_index}</td>
            <td class="font-mono text-right">{f.train_indices_count}</td>
            <td class="font-mono text-right">{f.val_indices_count}</td>
            <td class="font-mono text-right">{_fmt_float(m.get('accuracy').value if m.get('accuracy') else None, 4)}</td>
            <td class="font-mono text-right">{_fmt_float(m.get('balanced_accuracy').value if m.get('balanced_accuracy') else None, 4)}</td>
            <td class="font-mono text-right">{_fmt_float(m.get('macro_f1').value if m.get('macro_f1') else None, 4)}</td>
            <td class="font-mono text-right">{_fmt_float(m.get('roc_auc').value if m.get('roc_auc') else None, 4)}</td>
            <td class="font-mono text-right">{_fmt_float(m.get('brier_score').value if m.get('brier_score') else None, 4)}</td>
            <td class="font-mono text-right">{f.fit_duration_seconds:.2f}</td>
        </tr>
        """)

    fold_table = f"""
    <div class="table-container">
        <table class="data-table">
            <thead>
                <tr>{"".join(f'<th class="text-center">{h}</th>' for h in fold_headers)}</tr>
            </thead>
            <tbody>
                {"".join(fold_rows)}
            </tbody>
        </table>
    </div>
    """

    # 3. Confusion matrix
    cm_html = ""
    if cand.aggregate_confusion_matrix and cand.fold_results and cand.fold_results[0].class_labels:
        labels = cand.fold_results[0].class_labels
        matrix = cand.aggregate_confusion_matrix
        th_cols = "".join(f'<th class="text-center font-mono">Pred {_esc(lbl)}</th>' for lbl in labels)
        cm_rows = []
        for r_idx, row in enumerate(matrix):
            cells = "".join(f'<td class="font-mono text-center cm-cell">{val}</td>' for val in row)
            cm_rows.append(f'<tr><th class="font-mono">True {_esc(labels[r_idx])}</th>{cells}</tr>')

        cm_html = f"""
        <div class="cm-wrapper">
            <h4>Aggregate Out-of-Fold Confusion Matrix (Candidate)</h4>
            <table class="cm-table">
                <thead><tr><th></th>{th_cols}</tr></thead>
                <tbody>{"".join(cm_rows)}</tbody>
            </table>
        </div>
        """

    return f"""
    <section class="report-section" id="performance">
        <h2>Classification Performance</h2>
        <h3>Cross-Validation Summary Statistics</h3>
        {aggregate_table}
        <h3>Fold-Level Partition Metrics</h3>
        {fold_table}
        {cm_html}
    </section>
    """


def _build_comparison_section(evaluation: EvaluationResult) -> str:
    """Section D: Candidate-versus-Baseline Comparison & Statistical Tests."""
    comp = evaluation.comparison_report
    if not comp:
        return """
        <section class="report-section" id="comparison">
            <h2>Candidate-versus-Baseline Comparison</h2>
            <div class="info-box">
                No baseline model was paired with this evaluation. Single-model runs do not generate comparative significance metrics.
            </div>
        </section>
        """

    rows = []
    for m_name, mc in sorted(comp.metric_comparisons.items()):
        pd = mc.paired_differences
        dir_label = "Higher is better" if mc.direction == MetricDirection.HIGHER_IS_BETTER else "Lower is better"

        ttest_str = '<span class="val-na">N/A</span>'
        if mc.corrected_resampled_t_test:
            tt = mc.corrected_resampled_t_test
            if tt.status == TestStatus.COMPLETED and tt.p_value is not None:
                sig_badge = (
                    '<span class="badge badge-success">p &lt; α (Sig)</span>'
                    if tt.statistically_significant
                    else '<span class="badge badge-muted">Not Sig</span>'
                )
                ttest_str = f"t={_fmt_float(tt.statistic, 3)}, p={_fmt_float(tt.p_value, 4)} {sig_badge}"
            else:
                ttest_str = f'<span class="text-muted">{_esc(tt.unavailable_reason or "Unavailable")}</span>'

        wilc_str = '<span class="val-na">N/A</span>'
        if mc.wilcoxon_signed_rank_test:
            wt = mc.wilcoxon_signed_rank_test
            if wt.status == TestStatus.COMPLETED and wt.p_value is not None:
                sig_badge = (
                    '<span class="badge badge-success">p &lt; α</span>'
                    if wt.statistically_significant
                    else '<span class="badge badge-muted">Not Sig</span>'
                )
                wilc_str = f"W={_fmt_float(wt.statistic, 1)}, p={_fmt_float(wt.p_value, 4)} {sig_badge}"
            else:
                wilc_str = f'<span class="text-muted">{_esc(wt.unavailable_reason or "Unavailable")}</span>'

        rows.append(f"""
        <tr>
            <td class="font-mono font-bold">{_esc(m_name)}</td>
            <td class="text-muted">{dir_label}</td>
            <td class="text-right font-mono font-bold">{_fmt_float(pd.mean_delta, 4)}</td>
            <td class="text-right font-mono">{_fmt_float(pd.std_delta, 4)}</td>
            <td>{ttest_str}</td>
            <td>{wilc_str}</td>
        </tr>
        """)

    table_html = f"""
    <div class="table-container">
        <table class="data-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Direction</th>
                    <th class="text-right">Mean Δ (Cand - Base)</th>
                    <th class="text-right">Std Dev Δ</th>
                    <th>Nadeau-Bengio Corrected t-Test</th>
                    <th>Wilcoxon Signed-Rank Test</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    </div>
    """

    return f"""
    <section class="report-section" id="comparison">
        <h2>Candidate-versus-Baseline Comparison</h2>
        <p class="section-desc">
            Paired difference analysis between <strong>{_esc(comp.candidate_model_name)}</strong> and baseline <strong>{_esc(comp.baseline_model_name)}</strong> across {comp.num_folds} folds.
        </p>
        {table_html}
        <div class="notice-bar">
            <strong>Methodology Note:</strong> Cross-validation folds violate the independence assumptions of naive paired t-tests. MLForge applies the Nadeau-Bengio corrected resampled t-test to adjust variance for overlapping training partitions.
        </div>
    </section>
    """


def _build_calibration_section(evaluation: EvaluationResult) -> str:
    """Section E: Calibration and Threshold Analysis."""
    cand = evaluation.candidate_result
    cal = cand.calibration
    ta = cand.threshold_analysis

    if not cal and not ta:
        return """
        <section class="report-section" id="calibration">
            <h2>Calibration &amp; Threshold Analysis</h2>
            <div class="info-box">
                Probability calibration and threshold analysis were not computed for this run (e.g. estimator lacks predict_proba or multiclass task).
            </div>
        </section>
        """

    cal_content = ""
    if cal:
        svg_cal = _render_calibration_svg(cal)
        bin_rows = []
        for b in cal.bins:
            status_badge = (
                '<span class="badge badge-warn">Empty</span>'
                if b.is_empty
                else '<span class="badge badge-muted">Active</span>'
            )
            low = getattr(b, "lower_bound", getattr(b, "bin_lower", 0.0))
            high = getattr(b, "upper_bound", getattr(b, "bin_upper", 1.0))
            obs_rate = getattr(b, "observed_positive_rate", getattr(b, "empirical_positive_rate", None))
            bin_rows.append(f"""
            <tr>
                <td class="font-mono text-center">#{b.bin_index}</td>
                <td class="font-mono text-center">[{low:.2f}, {high:.2f})</td>
                <td class="font-mono text-right">{b.sample_count}</td>
                <td class="font-mono text-right">{_fmt_float(b.mean_predicted_prob, 3)}</td>
                <td class="font-mono text-right">{_fmt_float(obs_rate, 3)}</td>
                <td class="text-center">{status_badge}</td>
            </tr>
            """)

        cal_content = f"""
        <h3>Probability Calibration (Reliability Diagram)</h3>
        <div class="viz-grid">
            <div class="viz-chart">
                {svg_cal}
                <div class="chart-stats">
                    <span>ECE: <strong>{_fmt_float(cal.expected_calibration_error, 4)}</strong></span>
                    <span>MCE: <strong>{_fmt_float(cal.max_calibration_error, 4)}</strong></span>
                    <span>Brier: <strong>{_fmt_float(cal.brier_score, 4)}</strong></span>
                    <span>Log Loss: <strong>{_fmt_float(cal.log_loss, 4)}</strong></span>
                </div>
            </div>
            <div class="viz-table">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th class="text-center">Bin</th>
                            <th class="text-center">Range</th>
                            <th class="text-right">N</th>
                            <th class="text-right">Pred Prob</th>
                            <th class="text-right">Empirical</th>
                            <th class="text-center">Status</th>
                        </tr>
                    </thead>
                    <tbody>{"".join(bin_rows)}</tbody>
                </table>
            </div>
        </div>
        """

    ta_content = ""
    if ta:
        svg_ta = _render_threshold_svg(ta)
        th_rows = []
        # Sample representative points if threshold sweep has many cutoffs
        step = max(1, len(ta.threshold_metrics) // 10)
        sample_metrics = ta.threshold_metrics[::step]
        if ta.threshold_metrics[-1] not in sample_metrics:
            sample_metrics.append(ta.threshold_metrics[-1])

        for tm in sample_metrics:
            th_rows.append(f"""
            <tr>
                <td class="font-mono font-bold text-center">{tm.threshold:.2f}</td>
                <td class="font-mono text-right">{tm.true_positives}</td>
                <td class="font-mono text-right">{tm.false_positives}</td>
                <td class="font-mono text-right">{tm.true_negatives}</td>
                <td class="font-mono text-right">{tm.false_negatives}</td>
                <td class="font-mono text-right">{_fmt_float(tm.precision, 3)}</td>
                <td class="font-mono text-right">{_fmt_float(tm.recall, 3)}</td>
                <td class="font-mono text-right">{_fmt_float(tm.f1_score, 3)}</td>
            </tr>
            """)

        ta_content = f"""
        <h3>Decision Threshold Sweep Analysis (Positive Class: {_esc(ta.positive_class_label)})</h3>
        <div class="viz-grid">
            <div class="viz-chart">
                {svg_ta}
                <div class="chart-stats legend-row">
                    <span style="color: #58a6ff;">&horbar; F1 Score</span>
                    <span style="color: #3fb950;">&ndash;&ndash; Precision</span>
                    <span style="color: #d29922;">&middot;&middot; Recall</span>
                </div>
            </div>
            <div class="viz-table">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th class="text-center">Cutoff</th>
                            <th class="text-right">TP</th>
                            <th class="text-right">FP</th>
                            <th class="text-right">TN</th>
                            <th class="text-right">FN</th>
                            <th class="text-right">Precision</th>
                            <th class="text-right">Recall</th>
                            <th class="text-right">F1</th>
                        </tr>
                    </thead>
                    <tbody>{"".join(th_rows)}</tbody>
                </table>
            </div>
        </div>
        """

    return f"""
    <section class="report-section" id="calibration">
        <h2>Calibration &amp; Threshold Analysis</h2>
        {cal_content}
        {ta_content}
    </section>
    """


def _build_error_diagnostics_section(evaluation: EvaluationResult) -> str:
    """Section F: Error Diagnostics and Subgroup Breakdown."""
    cand = evaluation.candidate_result
    diag = cand.error_diagnostics
    if not diag:
        return """
        <section class="report-section" id="diagnostics">
            <h2>Error Diagnostics &amp; Subgroup Analysis</h2>
            <div class="info-box">
                No error diagnostic or subgroup breakdown evidence was generated for this evaluation run.
            </div>
        </section>
        """

    # Fold errors
    f_rows = []
    for fd in diag.fold_diagnostics:
        f_rows.append(f"""
        <tr>
            <td class="font-mono text-center">#{fd.fold_index}</td>
            <td class="font-mono text-right">{fd.total_samples}</td>
            <td class="font-mono text-right">{fd.error_count}</td>
            <td class="font-mono text-right">{_fmt_pct(fd.error_rate * 100)}</td>
            <td class="font-mono text-right">{fd.false_positives if fd.false_positives is not None else "N/A"}</td>
            <td class="font-mono text-right">{fd.false_negatives if fd.false_negatives is not None else "N/A"}</td>
        </tr>
        """)

    fold_diag_html = f"""
    <div class="table-container">
        <table class="data-table">
            <thead>
                <tr>
                    <th class="text-center">Fold</th>
                    <th class="text-right">Total N</th>
                    <th class="text-right">Errors</th>
                    <th class="text-right">Error Rate</th>
                    <th class="text-right">False Positives</th>
                    <th class="text-right">False Negatives</th>
                </tr>
            </thead>
            <tbody>{"".join(f_rows)}</tbody>
        </table>
    </div>
    """

    # Subgroups
    subgroup_html = ""
    if diag.subgroup_diagnostics:
        sg_rows = []
        for col_name, sg_list in diag.subgroup_diagnostics.items():
            for sg in sg_list:
                status = (
                    '<span class="badge badge-muted">Usable</span>'
                    if sg.is_usable
                    else f'<span class="badge badge-warn" title="{_esc(sg.unusable_reason)}">Low Support</span>'
                )
                sg_rows.append(f"""
                <tr>
                    <td class="font-mono">{_esc(col_name)}</td>
                    <td class="font-mono font-bold">{_esc(sg.subgroup_value)}</td>
                    <td class="font-mono text-right">{sg.sample_count}</td>
                    <td class="font-mono text-right">{sg.error_count}</td>
                    <td class="font-mono text-right">{_fmt_pct(sg.error_rate * 100)}</td>
                    <td class="font-mono text-right">{_fmt_float(sg.accuracy, 3)}</td>
                    <td class="font-mono text-right">{_fmt_float(sg.precision, 3)}</td>
                    <td class="font-mono text-right">{_fmt_float(sg.recall, 3)}</td>
                    <td class="text-center">{status}</td>
                </tr>
                """)

        subgroup_html = f"""
        <h3>Subgroup Slice Diagnostics</h3>
        <div class="table-container">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Feature Column</th>
                        <th>Subgroup Value</th>
                        <th class="text-right">Samples</th>
                        <th class="text-right">Errors</th>
                        <th class="text-right">Error Rate</th>
                        <th class="text-right">Accuracy</th>
                        <th class="text-right">Precision</th>
                        <th class="text-right">Recall</th>
                        <th class="text-center">Status</th>
                    </tr>
                </thead>
                <tbody>{"".join(sg_rows)}</tbody>
            </table>
        </div>
        """

    return f"""
    <section class="report-section" id="diagnostics">
        <h2>Error Diagnostics &amp; Subgroup Analysis</h2>
        <div class="summary-stats">
            <span>Total Evaluated Samples: <strong>{diag.total_samples}</strong></span>
            <span>Total Misclassifications: <strong>{diag.total_errors}</strong></span>
            <span>Overall Error Rate: <strong>{_fmt_pct(diag.overall_error_rate * 100)}</strong></span>
        </div>
        <h3>Fold-Level Error Rates</h3>
        {fold_diag_html}
        {subgroup_html}
    </section>
    """


def _build_robustness_section(evaluation: EvaluationResult) -> str:
    """Section G: Robustness and Stress Testing."""
    cand = evaluation.candidate_result
    rob = cand.robustness
    if not rob or not rob.curves:
        return """
        <section class="report-section" id="robustness">
            <h2>Robustness &amp; Controlled Stress Testing</h2>
            <div class="info-box">
                Controlled feature perturbation and stress testing were not evaluated for this model run.
            </div>
        </section>
        """

    svg_rob = _render_robustness_svg(rob.curves)

    rows = []
    for curve in rob.curves:
        aurdc_str = _fmt_float(curve.aurdc, 4)
        for pt in curve.points:
            rows.append(f"""
            <tr>
                <td class="font-mono font-bold">{_esc(curve.perturbation_type.value)}</td>
                <td class="font-mono text-center">{pt.severity_level:.2f}</td>
                <td class="font-mono text-right">{_fmt_float(pt.clean_score, 4)}</td>
                <td class="font-mono text-right">{_fmt_float(pt.perturbed_score, 4)}</td>
                <td class="font-mono text-right font-bold">{_fmt_float(pt.retained_ratio, 4)}</td>
                <td class="font-mono text-right">{_fmt_pct(pt.performance_drop_percentage)}</td>
                <td class="font-mono text-right font-bold">{aurdc_str}</td>
            </tr>
            """)

    return f"""
    <section class="report-section" id="robustness">
        <h2>Robustness &amp; Controlled Stress Testing</h2>
        <div class="viz-grid">
            <div class="viz-chart">
                {svg_rob}
            </div>
            <div class="viz-table">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Perturbation Type</th>
                            <th class="text-center">Severity</th>
                            <th class="text-right">Clean Score</th>
                            <th class="text-right">Perturbed</th>
                            <th class="text-right">Retention</th>
                            <th class="text-right">Drop %</th>
                            <th class="text-right">AURDC</th>
                        </tr>
                    </thead>
                    <tbody>{"".join(rows)}</tbody>
                </table>
            </div>
        </div>
    </section>
    """


def _build_acceptance_section(
    decision: Optional[AcceptanceDecision],
    policy: Optional[AcceptancePolicy],
    certificate: Optional[ReleaseCertificate] = None,
) -> str:
    """Section H: Acceptance Decision and Release Authorization."""
    cert_html = ""
    if certificate:
        cert_outcome = certificate.release_gate_outcome.value if hasattr(certificate.release_gate_outcome, "value") else str(certificate.release_gate_outcome)
        cert_outcome_badge = (
            '<span class="badge badge-muted">CERTIFICATE ISSUED &mdash; VERIFICATION NOT ESTABLISHED</span>'
            if cert_outcome == "passed"
            else f'<span class="badge badge-fail">GATE {cert_outcome.upper()}</span>'
        )
        cert_html = f"""
        <div class="certificate-box" style="margin-top: 24px; background: var(--bg-surface-elevated); border: 1px solid var(--border-color); border-radius: 8px; padding: 18px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h3 style="font-size: 1.1rem; color: var(--accent);">Signed Release Certificate &mdash; {cert_outcome_badge}</h3>
                <span class="badge badge-muted font-mono">{_esc(certificate.signature_algorithm)}</span>
            </div>
            <div class="identity-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px;">
                <div class="id-item">
                    <span class="id-label">Certificate ID</span>
                    <span class="id-value font-mono">{_esc(certificate.certificate_id)}</span>
                </div>
                <div class="id-item">
                    <span class="id-label">Signing Key ID</span>
                    <span class="id-value font-mono">{_esc(certificate.key_id)}</span>
                </div>
                <div class="id-item">
                    <span class="id-label">Target Environment</span>
                    <span class="id-value">{_esc(certificate.target_environment)}</span>
                </div>
                <div class="id-item">
                    <span class="id-label">Issued Timestamp</span>
                    <span class="id-value">{_esc(certificate.issued_at)}</span>
                </div>
                <div class="id-item full-width" style="grid-column: 1 / -1;">
                    <span class="id-label">Payload Digest (SHA-256)</span>
                    <span class="id-value font-mono">{_esc(certificate.payload_digest_sha256)}</span>
                </div>
                <div class="id-item full-width" style="grid-column: 1 / -1;">
                    <span class="id-label">Cryptographic Signature ({_esc(certificate.signature_algorithm)})</span>
                    <span class="id-value font-mono text-muted">{_esc(certificate.signature[:48])}...</span>
                </div>
            </div>
            <div class="notice-bar" style="margin-top: 12px; font-size: 12px;">
                <strong>Verification Notice:</strong> This offline report displays the self-contained claims of an issued release certificate. Offline rendering does NOT establish cryptographic validity, signer trust, or release authorization. Authorization requires programmatic verification via <code>verify_release_certificate(cert, trusted_keys=..., expected_source=...)</code>.
            </div>
        </div>
        """
    else:
        cert_html = """
        <div class="certificate-box" style="margin-top: 24px; background: var(--bg-surface-elevated); border: 1px solid var(--border-color); border-radius: 8px; padding: 18px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="font-size: 1.1rem; color: var(--text-muted);">Release Certificate Status</h3>
                <span class="badge badge-muted">NO CERTIFICATE ISSUED</span>
            </div>
            <p class="text-muted" style="margin-top: 8px; font-size: 13px;">
                No cryptographically signed release certificate was issued for this run. A passing acceptance decision alone does not authorize model release.
            </p>
        </div>
        """

    if not decision:
        return f"""
        <section class="report-section" id="acceptance">
            <h2>Acceptance Decision &amp; Governance</h2>
            <div class="info-box">
                No acceptance decision was evaluated for this run. Model was not subjected to automated policy release gates.
            </div>
            {cert_html}
        </section>
        """

    rules_rows = []
    for crit in decision.criteria_evaluations:
        sev_badge = (
            '<span class="badge badge-fail">BLOCKER</span>'
            if crit.severity == CriterionSeverity.BLOCKER
            else '<span class="badge badge-muted">ADVISORY</span>'
        )

        outcome_val = crit.outcome.value if hasattr(crit.outcome, "value") else str(crit.outcome)
        out_upper = outcome_val.upper()
        if out_upper == "SATISFIED":
            out_badge = '<span class="badge badge-success">SATISFIED</span>'
        elif out_upper == "VIOLATED":
            out_badge = '<span class="badge badge-fail">VIOLATED</span>'
        elif out_upper == "INCONCLUSIVE":
            out_badge = '<span class="badge badge-warn">INCONCLUSIVE</span>'
        else:
            out_badge = '<span class="badge badge-muted">SKIPPED</span>'

        op_str = crit.operator.value if hasattr(crit.operator, "value") else str(crit.operator)

        rules_rows.append(f"""
        <tr>
            <td class="font-mono font-bold">{_esc(crit.rule_id)}</td>
            <td>{sev_badge}</td>
            <td class="font-mono">{_esc(crit.metric)}</td>
            <td class="font-mono text-center">{_esc(op_str)} {_fmt_float(crit.threshold_value, 4)}</td>
            <td class="font-mono text-right">{_fmt_float(crit.observed_value, 4)}</td>
            <td class="text-center">{out_badge}</td>
            <td class="font-mono text-muted">{_esc(crit.reason_code)}</td>
            <td>{_esc(crit.message)}</td>
        </tr>
        """)

    pol_meta = f"Policy: <strong>{_esc(decision.policy_id)}</strong> (v{_esc(decision.policy_version)})"
    if policy:
        pol_meta += f" &mdash; <em>{_esc(policy.name)}</em>"

    sig_html = ""
    if decision.hash_signature:
        sig_html = f'<div class="hash-signature">Decision Cryptographic Signature: <span class="font-mono">{_esc(decision.hash_signature)}</span></div>'

    return f"""
    <section class="report-section" id="acceptance">
        <h2>Acceptance Decision &amp; Governance</h2>
        <div class="policy-header">
            <div>{pol_meta}</div>
            <div class="text-muted">Evaluated: {_esc(decision.evaluated_at)}</div>
        </div>

        <div class="table-container">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Rule ID</th>
                        <th>Severity</th>
                        <th>Metric</th>
                        <th class="text-center">Condition</th>
                        <th class="text-right">Observed</th>
                        <th class="text-center">Verdict</th>
                        <th>Code</th>
                        <th>Explanation</th>
                    </tr>
                </thead>
                <tbody>{"".join(rules_rows)}</tbody>
            </table>
        </div>
        {sig_html}
        {cert_html}
    </section>
    """


def _build_provenance_section(
    manifest: Union[ArtifactManifest, AuditBundleManifest],
    evaluation: EvaluationResult,
) -> str:
    """Section I: Artifact Integrity and Provenance Inventory."""
    file_rows = []

    # File checksums from manifest
    if isinstance(manifest, AuditBundleManifest):
        for f in manifest.files:
            file_rows.append(f"""
            <tr>
                <td class="font-mono font-bold">{_esc(f.path)}</td>
                <td><span class="badge badge-muted">{_esc(f.artifact_type)}</span></td>
                <td class="font-mono text-right">{_fmt_bytes(f.size_bytes)}</td>
                <td class="font-mono text-muted">{_esc(f.sha256[:16])}...</td>
                <td class="text-center"><span class="badge badge-success">Verified</span></td>
            </tr>
            """)
    else:
        for path_name, sha in sorted(manifest.file_checksums.items()):
            art_type = "model" if "models/" in path_name else ("manifest" if "manifest" in path_name else "evidence")
            file_rows.append(f"""
            <tr>
                <td class="font-mono font-bold">{_esc(path_name)}</td>
                <td><span class="badge badge-muted">{art_type}</span></td>
                <td class="font-mono text-right">&mdash;</td>
                <td class="font-mono text-muted">{_esc(sha[:16])}...</td>
                <td class="text-center"><span class="badge badge-success">Checksummed</span></td>
            </tr>
            """)

    model_status_cand = "Present (.joblib)" if manifest.has_models else "None"
    model_prov = getattr(manifest, "model_provenance", None) or evaluation.candidate_result.model_provenance or "N/A"
    fold_idx = getattr(manifest, "candidate_model_fold_index", None) or evaluation.candidate_result.fitted_fold_index

    return f"""
    <section class="report-section" id="provenance">
        <h2>Artifact Integrity &amp; Provenance Inventory</h2>

        <div class="provenance-cards">
            <div class="prov-card">
                <span class="prov-label">Candidate Model File</span>
                <span class="prov-val">{_esc(model_status_cand)}</span>
            </div>
            <div class="prov-card">
                <span class="prov-label">Training Partition Provenance</span>
                <span class="prov-val font-mono">{_esc(model_prov)}</span>
            </div>
            <div class="prov-card">
                <span class="prov-label">Fitted Partition Fold</span>
                <span class="prov-val font-mono">Fold #{fold_idx if fold_idx is not None else "N/A"}</span>
            </div>
        </div>

        <h3>Indexed File Checksums (SHA-256)</h3>
        <div class="table-container">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Archive Path</th>
                        <th>Artifact Type</th>
                        <th class="text-right">Size</th>
                        <th>SHA-256 Digest</th>
                        <th class="text-center">Integrity Status</th>
                    </tr>
                </thead>
                <tbody>{"".join(file_rows)}</tbody>
            </table>
        </div>

        <div class="trust-tier-box">
            <h4>Integrity &amp; Trust Model Tiers</h4>
            <ul>
                <li><strong>Tier 1 (File Existence):</strong> Verified locally on disk.</li>
                <li><strong>Tier 2 (Cryptographic Integrity):</strong> SHA-256 digest matches run manifest byte-for-byte.</li>
                <li><strong>Tier 3 (Bundle Consistency):</strong> Packaged audit bundle structure verified without deserializing models.</li>
                <li><strong>Tier 4 (External Trust / Author Authenticity):</strong> <em>Unverified.</em> Cryptographic checksums prove internal consistency, not external origin. Never deserialize untrusted models.</li>
            </ul>
        </div>
    </section>
    """


def _build_methodology_section() -> str:
    """Section J: Evaluation Methodology, Caveats, and Limitations."""
    return """
    <section class="report-section" id="methodology">
        <h2>Evaluation Methodology &amp; Scientific Limitations</h2>
        <div class="methodology-grid">
            <div class="method-card">
                <h4>Leakage-Safe Partition Isolation</h4>
                <p>
                    All preprocessing transformations (scaling, imputation, one-hot encoding) are encapsulated within fold-local pipelines and fitted exclusively on training splits. Out-of-fold evaluation predictions never incorporate global dataset statistics.
                </p>
            </div>
            <div class="method-card">
                <h4>Estimator Provenance Semantics</h4>
                <p>
                    The serialized model artifact is the fitted pipeline from the final cross-validation fold (Fold K-1), not an estimator retrained on the entire dataset. Reported performance reflects cross-validation uncertainty, not a single optimistic partition.
                </p>
            </div>
            <div class="method-card">
                <h4>Resampled Fold Dependence</h4>
                <p>
                    K-fold cross-validation folds share training instances, violating independence assumptions of naive t-tests. The Nadeau-Bengio corrected resampled t-test introduces a variance penalty to mitigate false positives in statistical significance tests.
                </p>
            </div>
            <div class="method-card">
                <h4>Survivorship &amp; Support Safeguards</h4>
                <p>
                    Under stress testing and subgroup analysis, slices with insufficient sample support or non-converged estimators are conservatively flagged as unusable. Areas under degradation curves (AURDC) are undefined if any intermediate severity fails.
                </p>
            </div>
        </div>
    </section>
    """


# ---------------------------------------------------------------------------
# Embedded CSS Stylesheet (Self-Contained, Zero External Resources)
# ---------------------------------------------------------------------------

REPORT_CSS = """
:root {
    --bg-base: #0e1117;
    --bg-surface: #161b22;
    --bg-surface-elevated: #21262d;
    --border-color: #30363d;
    --border-subtle: #21262d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #6e7681;
    --accent: #58a6ff;
    --accent-hover: #79c0ff;
    --success: #3fb950;
    --success-bg: rgba(63, 185, 80, 0.12);
    --fail: #f85149;
    --fail-bg: rgba(248, 81, 73, 0.12);
    --warn: #d29922;
    --warn-bg: rgba(210, 153, 34, 0.12);
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    --font-mono: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, monospace;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    background-color: var(--bg-base);
    color: var(--text-primary);
    font-family: var(--font-sans);
    font-size: 14px;
    line-height: 1.6;
    padding: 24px;
    -webkit-font-smoothing: antialiased;
}

.report-container {
    max-width: 1200px;
    margin: 0 auto;
}

/* Typography & Utilities */
h1, h2, h3, h4 { color: var(--text-primary); font-weight: 600; }
h2 { font-size: 1.4rem; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border-color); }
h3 { font-size: 1.1rem; margin: 20px 0 12px 0; color: var(--text-secondary); }
h4 { font-size: 0.95rem; margin-bottom: 8px; }
.font-mono { font-family: var(--font-mono); }
.font-bold { font-weight: 600; }
.text-right { text-align: right; }
.text-center { text-align: center; }
.text-muted { color: var(--text-muted); }
.truncate { max-width: 200px; display: inline-block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; vertical-align: bottom; }

/* Badges */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border: 1px solid transparent;
}
.badge-muted { background: var(--bg-surface-elevated); color: var(--text-secondary); border-color: var(--border-color); }
.badge-accent { background: rgba(88, 166, 255, 0.15); color: var(--accent); border-color: rgba(88, 166, 255, 0.4); }
.badge-success { background: var(--success-bg); color: var(--success); border-color: rgba(63, 185, 80, 0.4); }
.badge-fail { background: var(--fail-bg); color: var(--fail); border-color: rgba(248, 81, 73, 0.4); }
.badge-warn { background: var(--warn-bg); color: var(--warn); border-color: rgba(210, 153, 34, 0.4); }
.val-na { color: var(--text-muted); font-style: italic; }

/* Header */
.report-header {
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 24px;
}
.header-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 20px;
}
.brand-title {
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #fff;
    margin-right: 12px;
}
.report-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
}
.identity-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
    margin-bottom: 16px;
}
.id-item {
    display: flex;
    flex-direction: column;
}
.id-item.full-width {
    grid-column: 1 / -1;
}
.id-label {
    font-size: 11px;
    text-transform: uppercase;
    color: var(--text-muted);
    font-weight: 600;
    margin-bottom: 2px;
}
.id-value {
    font-size: 13px;
    color: var(--text-primary);
}
.notice-bar {
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 16px;
}

/* Sections */
.report-section {
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 24px;
}
.section-desc {
    color: var(--text-secondary);
    margin-bottom: 16px;
}

/* Decision Banners */
.decision-banner {
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 20px;
    border: 1px solid;
}
.banner-pass { background: var(--success-bg); border-color: rgba(63, 185, 80, 0.4); }
.banner-fail { background: var(--fail-bg); border-color: rgba(248, 81, 73, 0.4); }
.banner-inconclusive { background: var(--warn-bg); border-color: rgba(210, 153, 34, 0.4); }
.banner-neutral { background: var(--bg-surface-elevated); border-color: var(--border-color); }
.decision-status {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}
.decision-badge {
    font-family: var(--font-mono);
    font-size: 1.1rem;
    font-weight: 700;
}
.banner-pass .decision-badge { color: var(--success); }
.banner-fail .decision-badge { color: var(--fail); }
.banner-inconclusive .decision-badge { color: var(--warn); }
.banner-neutral .decision-badge { color: var(--text-secondary); }
.decision-title {
    font-size: 1.1rem;
    font-weight: 600;
}
.decision-meta {
    display: flex;
    gap: 20px;
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 8px;
}
.decision-meta strong { color: var(--text-primary); }
.decision-rationale {
    font-size: 13px;
    color: var(--text-primary);
}

/* Metric Cards */
.metrics-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}
.metric-card {
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 12px;
    display: flex;
    flex-direction: column;
}
.card-label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 4px;
}
.card-value {
    font-size: 1.4rem;
    font-weight: 600;
    color: #fff;
    margin-bottom: 6px;
    font-variant-numeric: tabular-nums;
}
.card-sub {
    font-size: 11px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.card-delta {
    color: var(--accent);
    font-weight: 600;
}

/* Tables */
.table-container {
    overflow-x: auto;
    margin-bottom: 16px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
}
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.data-table th, .data-table td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border-color);
    text-align: left;
    white-space: nowrap;
}
.data-table th {
    background: var(--bg-surface-elevated);
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 12px;
}
.data-table tr:last-child td { border-bottom: none; }
.data-table tbody tr:hover { background: rgba(255, 255, 255, 0.02); }

/* Confusion Matrix */
.cm-wrapper {
    margin-top: 20px;
}
.cm-table {
    border-collapse: collapse;
    margin-top: 8px;
}
.cm-table th, .cm-table td {
    border: 1px solid var(--border-color);
    padding: 8px 16px;
    font-size: 13px;
}
.cm-table th { background: var(--bg-surface-elevated); }
.cm-cell { background: #161b22; font-weight: 600; }

/* Visualizations */
.viz-grid {
    display: grid;
    grid-template-columns: 380px 1fr;
    gap: 20px;
    align-items: start;
    margin-bottom: 20px;
}
@media (max-width: 900px) {
    .viz-grid { grid-template-columns: 1fr; }
}
.viz-chart {
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 12px;
}
.chart-svg {
    width: 100%;
    height: auto;
    display: block;
}
.svg-axis-lbl { font-size: 10px; fill: var(--text-muted); font-family: var(--font-mono); }
.svg-title-lbl { font-size: 11px; fill: var(--text-secondary); font-family: var(--font-sans); }
.svg-line-primary { stroke: var(--accent); }
.svg-point { fill: var(--accent); }
.chart-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--border-color);
}
.chart-stats strong { color: var(--text-primary); font-family: var(--font-mono); }
.legend-row span { font-size: 11px; font-weight: 600; }
.chart-empty {
    padding: 40px;
    text-align: center;
    color: var(--text-muted);
    font-style: italic;
    background: var(--bg-surface-elevated);
    border-radius: 6px;
}

/* Policy / Provenance / Limitations */
.policy-header {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    margin-bottom: 12px;
}
.hash-signature {
    margin-top: 12px;
    font-size: 11px;
    color: var(--text-muted);
}
.provenance-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
    margin-bottom: 20px;
}
.prov-card {
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 12px;
}
.prov-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; display: block; margin-bottom: 4px; }
.prov-val { font-size: 13px; color: var(--text-primary); font-weight: 600; }
.trust-tier-box, .advisory-box, .info-box {
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 16px;
    margin-top: 16px;
}
.trust-tier-box ul, .advisory-box ul {
    margin-left: 20px;
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 8px;
}
.trust-tier-box li, .advisory-box li { margin-bottom: 4px; }
.info-box { color: var(--text-secondary); font-size: 13px; }
.methodology-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 16px;
}
.method-card {
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 16px;
}
.method-card h4 { color: var(--accent); margin-bottom: 6px; }
.method-card p { font-size: 12px; color: var(--text-secondary); line-height: 1.5; }

/* Navigation Jump Bar */
.report-nav {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 20px;
    padding: 8px 12px;
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 6px;
}
.nav-link {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 12px;
    font-weight: 500;
    padding: 4px 8px;
    border-radius: 4px;
}
.nav-link:hover, .nav-link:focus {
    color: var(--text-primary);
    background: var(--bg-surface-elevated);
    outline: none;
}
"""


# ---------------------------------------------------------------------------
# Public Reporting API
# ---------------------------------------------------------------------------

def load_report_evidence(
    source: Union[str, Path],
) -> Dict[str, Any]:
    """Load and validate all evaluation evidence, decision, policy, and manifest for reporting.

    Parameters
    ----------
    source : str or Path
        Directory containing serialized run artifacts, or a .zip audit bundle archive.

    Returns
    -------
    dict
        Dictionary containing validated contracts:
        - 'manifest': ArtifactManifest or AuditBundleManifest
        - 'evaluation': EvaluationResult
        - 'decision': Optional[AcceptanceDecision]
        - 'policy': Optional[AcceptancePolicy]
        - 'source_type': 'run_dir' or 'audit_bundle'
        - 'source_path': Path

    Raises
    ------
    ArtifactLoadError
        If source is invalid, missing required evidence, or malformed.
    AuditBundleError
        If audit bundle verification fails.
    """
    path = Path(source).resolve()

    if not path.exists():
        raise ArtifactLoadError(f"Evidence source '{path}' does not exist.")

    # 1. Audit Bundle Archive (.zip)
    if path.is_file() and path.suffix.lower() == ".zip":
        verification = verify_audit_bundle(path, raise_on_error=True)
        manifest = verification.manifest

        with zipfile.ZipFile(path, mode="r") as zf:
            # evaluation.json
            try:
                with zf.open("evaluation.json") as ef:
                    eval_dict = json.loads(ef.read().decode("utf-8"))
                evaluation = EvaluationResult.model_validate(eval_dict)
            except Exception as exc:
                raise ArtifactLoadError(f"Failed to load evaluation evidence from bundle '{path}': {exc}") from exc

            # decision.json (optional)
            decision = None
            if "decision.json" in zf.namelist():
                try:
                    with zf.open("decision.json") as df:
                        dec_dict = json.loads(df.read().decode("utf-8"))
                    decision = AcceptanceDecision.model_validate(dec_dict)
                except Exception as exc:
                    raise ArtifactLoadError(f"Failed to load decision from bundle '{path}': {exc}") from exc

            # policy.json (optional)
            policy = None
            if "policy.json" in zf.namelist():
                try:
                    with zf.open("policy.json") as pf:
                        pol_dict = json.loads(pf.read().decode("utf-8"))
                    policy = AcceptancePolicy.model_validate(pol_dict)
                except Exception as exc:
                    raise ArtifactLoadError(f"Failed to load policy from bundle '{path}': {exc}") from exc

            # release/release_certificate.json (optional)
            certificate = None
            if "release/release_certificate.json" in zf.namelist():
                try:
                    with zf.open("release/release_certificate.json") as cf:
                        cert_dict = json.loads(cf.read().decode("utf-8"))
                    certificate = ReleaseCertificate.model_validate(cert_dict)
                except Exception:
                    certificate = None

        return {
            "manifest": manifest,
            "evaluation": evaluation,
            "decision": decision,
            "policy": policy,
            "certificate": certificate,
            "source_type": "audit_bundle",
            "source_path": path,
        }

    # 2. Local Run Directory
    elif path.is_dir():
        loaded = load_run_artifacts(path)

        cert_path = path / "release" / "release_certificate.json"
        certificate = None
        if cert_path.exists():
            try:
                with open(cert_path, "r", encoding="utf-8") as cf:
                    cert_dict = json.load(cf)
                certificate = ReleaseCertificate.model_validate(cert_dict)
            except Exception:
                certificate = None

        return {
            "manifest": loaded["manifest"],
            "evaluation": loaded["evaluation"],
            "decision": loaded.get("decision"),
            "policy": loaded.get("policy"),
            "certificate": certificate,
            "source_type": "run_dir",
            "source_path": path,
        }

    else:
        raise ArtifactLoadError(
            f"Evidence source '{path}' must be a valid directory or a .zip archive file."
        )


def generate_html_report(
    source: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    overwrite: bool = True,
) -> Path:
    """Generate a self-contained, offline HTML audit report from saved run artifacts or an audit bundle.

    SECURITY GUARANTEES
    -------------------
    1. NEVER deserializes or executes model (.joblib/.pickle) artifacts.
    2. NEVER mutates existing manifests or evidence files.
    3. Confines file resolution and writes atomically to prevent partial output.
    4. Escapes all dynamic strings to defend against script injection.
    5. Works 100% offline without remote scripts, fonts, CDNs, or network dependencies.

    Parameters
    ----------
    source : str or Path
        Directory containing serialized run artifacts, or path to a .zip audit bundle.
    output_path : str or Path, optional
        Destination path for report.html. If None, saved into source run directory or adjacent to bundle.
    overwrite : bool, default=True
        If True, overwrites existing report file. If False, raises ArtifactSerializationError.

    Returns
    -------
    Path
        Absolute path to the created HTML report file.

    Raises
    ------
    ArtifactLoadError
        If evidence loading fails or required evidence is missing.
    ArtifactSecurityError
        If path traversal is detected in destination path resolution.
    ArtifactSerializationError
        If writing the report fails or overwrite=False conflicts.
    """
    evidence = load_report_evidence(source)
    manifest = evidence["manifest"]
    evaluation = evidence["evaluation"]
    decision = evidence["decision"]
    policy = evidence["policy"]
    certificate = evidence.get("certificate")
    source_type = evidence["source_type"]
    source_path = evidence["source_path"]

    # Resolve output path
    if output_path is not None:
        target_file = Path(output_path).resolve()
    else:
        if source_type == "run_dir":
            target_file = (source_path / "report.html").resolve()
        else:
            target_file = (source_path.parent / f"{manifest.run_id}_report.html").resolve()

    # Confine destination path against pure root escaping or invalid targets
    if target_file.is_dir():
        target_file = (target_file / "report.html").resolve()

    if not overwrite and target_file.exists():
        raise ArtifactSerializationError(
            f"Report file '{target_file}' already exists and overwrite is set to False."
        )

    target_file.parent.mkdir(parents=True, exist_ok=True)

    # Build sections
    header_html = _build_header_section(manifest, evaluation, source_type)
    exec_html = _build_executive_summary_section(manifest, evaluation, decision)
    perf_html = _build_performance_section(evaluation)
    comp_html = _build_comparison_section(evaluation)
    cal_html = _build_calibration_section(evaluation)
    diag_html = _build_error_diagnostics_section(evaluation)
    rob_html = _build_robustness_section(evaluation)
    accept_html = _build_acceptance_section(decision, policy, certificate)
    prov_html = _build_provenance_section(manifest, evaluation)
    method_html = _build_methodology_section()

    nav_html = """
    <nav class="report-nav" aria-label="Report sections">
        <a href="#executive-summary" class="nav-link">Executive Summary</a>
        <a href="#performance" class="nav-link">Performance</a>
        <a href="#comparison" class="nav-link">Comparison</a>
        <a href="#calibration" class="nav-link">Calibration</a>
        <a href="#diagnostics" class="nav-link">Diagnostics</a>
        <a href="#robustness" class="nav-link">Robustness</a>
        <a href="#acceptance" class="nav-link">Acceptance</a>
        <a href="#provenance" class="nav-link">Artifacts &amp; Provenance</a>
        <a href="#methodology" class="nav-link">Methodology</a>
    </nav>
    """

    doc_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="generator" content="MLForge {REPORT_SCHEMA_VERSION}">
    <title>MLForge Audit Report &mdash; {_esc(manifest.run_id)}</title>
    <style>
{REPORT_CSS}
    </style>
</head>
<body>
    <div class="report-container">
        {header_html}
        {nav_html}
        {exec_html}
        {perf_html}
        {comp_html}
        {cal_html}
        {diag_html}
        {rob_html}
        {accept_html}
        {prov_html}
        {method_html}
    </div>
</body>
</html>
"""

    # Atomic write to destination
    tmp_path = target_file.with_name(f"{target_file.name}.tmp_{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(doc_html)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, target_file)
        return target_file
    except Exception as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise ArtifactSerializationError(
            f"Failed to write HTML report to '{target_file}': {exc}"
        ) from exc
