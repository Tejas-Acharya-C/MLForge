import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import ConfusionMatrix from '../components/ConfusionMatrix';
import FoldDistributionChart from '../components/FoldDistributionChart';
import { EvaluationStatusResponse, AcceptanceDecision } from '../types/api';

interface Props {
  runId: string;
  onNavigate: (screen: string, runId?: string) => void;
}

export const EvaluationResultsScreen: React.FC<Props> = ({ runId, onNavigate }) => {
  const [data, setData] = useState<any | null>(null);
  const [folds, setFolds] = useState<any[] | null>(null);
  const [acceptance, setAcceptance] = useState<AcceptanceDecision | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [runStatus, setRunStatus] = useState<EvaluationStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const [cancelling, setCancelling] = useState<boolean>(false);
  const [selectedChartMetric, setSelectedChartMetric] = useState<string>('accuracy');

  useEffect(() => {
    let mounted = true;
    let pollTimer: any = null;
    let timerInterval: any = null;

    const fetchStatusAndData = async () => {
      try {
        const status = await api.getEvaluationStatus(runId);
        if (!mounted) return;
        setRunStatus(status);

        if (status.state === 'created' || status.state === 'running') {
          setLoading(false);
          pollTimer = setTimeout(fetchStatusAndData, 2000);
        } else if (status.state === 'completed') {
          // Fetch final scientific results, fold metrics, and acceptance decision
          const [resData, foldData, accData] = await Promise.all([
            api.getEvaluationResults(runId),
            api.getFoldResults(runId).catch(() => null),
            api.getAcceptanceDecision(runId).catch(() => null),
          ]);
          if (!mounted) return;
          setData(resData);
          setFolds(foldData);
          setAcceptance(accData);
          setLoading(false);
        } else if (status.state === 'failed') {
          setError(status.error_message || 'The evaluation run terminated with a failure state.');
          setLoading(false);
        } else if (status.state === 'cancelled') {
          setError('This evaluation run was cancelled by the user.');
          setLoading(false);
        }
      } catch (err: any) {
        if (!mounted) return;
        setError(err.message || 'Evaluation run not found or server error.');
        setLoading(false);
      }
    };

    timerInterval = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1);
    }, 1000);

    fetchStatusAndData();

    return () => {
      mounted = false;
      if (pollTimer) clearTimeout(pollTimer);
      if (timerInterval) clearInterval(timerInterval);
    };
  }, [runId]);

  const handleCancelRun = async () => {
    setCancelling(true);
    try {
      await api.cancelEvaluation(runId, 'Evaluation aborted by user via workbench');
      const status = await api.getEvaluationStatus(runId);
      setRunStatus(status);
      setError('Evaluation run successfully cancelled.');
    } catch (err: any) {
      alert(`Failed to cancel run: ${err.message}`);
    } finally {
      setCancelling(false);
    }
  };

  // 1. Initial network loading state
  if (loading && !runStatus) {
    return (
      <div className="empty-state" style={{ padding: '64px 0' }}>
        <div className="dot-pulse" style={{ marginBottom: '16px' }} />
        <div style={{ fontWeight: 600, fontSize: '16px', marginBottom: '4px' }}>Initializing Run Context</div>
        <div style={{ color: 'var(--color-ink-muted)' }}>Connecting to evaluation engine for run {runId}...</div>
      </div>
    );
  }

  // 2. Active in-flight evaluation state
  if (runStatus && (runStatus.state === 'created' || runStatus.state === 'running')) {
    return (
      <div className="card" style={{ padding: '36px', textAlign: 'center', margin: '24px 0' }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
          <span className="dot-pulse" style={{ width: '10px', height: '10px' }} />
          <span className="badge badge-running" style={{ fontSize: '13px', padding: '4px 12px' }}>
            Stratified Cross-Validation in Progress
          </span>
        </div>
        <h2 style={{ margin: '0 0 8px 0', fontSize: '22px' }}>Executing Evaluation Pipeline</h2>
        <p style={{ color: 'var(--color-ink-secondary)', maxWidth: '520px', margin: '0 auto 20px', fontSize: '14px' }}>
          The local engine is fitting out-of-fold candidate pipelines, evaluating naive baseline anchors, and computing paired statistical tests without data leakage.
        </p>

        <div className="metric-card" style={{ display: 'inline-block', minWidth: '220px', margin: '0 auto 24px' }}>
          <div className="metric-card-label">Active Elapsed Duration</div>
          <div className="metric-card-value font-mono tabular-nums">{elapsedSeconds}s</div>
          <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
            Engine status: <code>{runStatus.state}</code>
          </div>
        </div>

        <div>
          <button
            className="btn btn-danger"
            onClick={handleCancelRun}
            disabled={cancelling}
            style={{ marginRight: '10px' }}
          >
            {cancelling ? 'Aborting...' : 'Abort Evaluation Run'}
          </button>
          <button className="btn btn-secondary" onClick={() => onNavigate('overview')}>
            View Run History
          </button>
        </div>
      </div>
    );
  }

  // 3. Execution failure state
  if (error || !data) {
    return (
      <div className="card" style={{ borderColor: 'var(--color-crimson-border)', backgroundColor: 'var(--color-surface)', margin: '24px 0' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
          <span className="status-badge status-failed">Execution Failed</span>
          <span className="mono" style={{ fontSize: '13px', color: 'var(--color-ink-muted)' }}>Run {runId}</span>
        </div>
        <h3 style={{ color: 'var(--color-crimson)', margin: '0 0 8px 0', fontSize: '18px' }}>
          {runStatus?.state === 'cancelled' ? 'Run Cancelled' : 'Evaluation Failure'}
        </h3>
        <p style={{ color: 'var(--color-ink-primary)', fontSize: '13px', marginBottom: '16px', lineHeight: 1.5 }}>
          {error || 'No evaluation results recorded for this run.'}
        </p>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button className="btn btn-primary" onClick={() => onNavigate('new')}>
            Configure New Evaluation
          </button>
          <button className="btn btn-secondary" onClick={() => onNavigate('overview')}>
            Return to Run History
          </button>
        </div>
      </div>
    );
  }

  // 4. Completed scientific results workspace
  const candResult = data.candidate_result || {};
  const baseResult = data.baseline_result || {};
  const comparison = data.comparison_report || {};
  const candSummaries = candResult.metric_summaries || {};
  const baseSummaries = baseResult.metric_summaries || {};

  // Accuracy comparison
  const candAcc = candSummaries['accuracy']?.mean ?? null;
  const baseAcc = baseSummaries['accuracy']?.mean ?? null;
  const deltaAcc = candAcc !== null && baseAcc !== null ? candAcc - baseAcc : null;

  // Macro F1 comparison (honest null/undefined preservation)
  const candF1 = candSummaries['macro_f1']?.mean ?? candSummaries['f1_macro']?.mean ?? null;
  const baseF1 = baseSummaries['macro_f1']?.mean ?? baseSummaries['f1_macro']?.mean ?? null;
  const deltaF1 = candF1 !== null && baseF1 !== null ? candF1 - baseF1 : null;

  // Statistical test on accuracy
  const accStatTest = comparison.metric_comparisons?.['accuracy']?.corrected_resampled_t_test;

  const metricKeys = Array.from(new Set([...Object.keys(candSummaries), ...Object.keys(baseSummaries)]));

  // Acceptance outcome
  const outcome = acceptance?.outcome || runStatus?.acceptance_outcome || 'INCONCLUSIVE';
  const isPass = outcome === 'PASS';
  const isFail = outcome === 'FAIL';
  const isInconclusive = outcome === 'INCONCLUSIVE';

  // Find any inconclusive blocker criteria
  const inconclusiveCriteria = acceptance?.criteria_evaluations?.filter((c) => c.outcome === 'inconclusive') || [];
  const violatedCriteria = acceptance?.criteria_evaluations?.filter((c) => c.outcome === 'violated') || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px', flexWrap: 'wrap' }}>
            <span className="mono" style={{ fontSize: '13px', color: 'var(--color-ink-muted)' }}>Run {runId}</span>
            <span className="status-badge status-completed">Evaluated</span>
            <span className="status-badge" style={{ background: 'var(--color-stone-surface)', color: 'var(--color-ink)' }}>
              {candResult.model_type || 'Candidate'} vs {baseResult.model_type || 'Dummy Baseline'}
            </span>
            <span
              className={`badge ${
                isPass ? 'badge-pass' : isFail ? 'badge-fail' : 'badge-inconclusive'
              }`}
            >
              Acceptance: {outcome}
            </span>
          </div>
          <h1 style={{ margin: 0, fontSize: '22px', letterSpacing: '-0.02em' }}>Cross-Validation Performance Results</h1>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <button className="btn btn-secondary" onClick={() => onNavigate('reliability', runId)}>
            Reliability & Slices &rarr;
          </button>
          <button className="btn btn-secondary" onClick={() => onNavigate('acceptance', runId)}>
            Release Governance &rarr;
          </button>
          <button className="btn btn-secondary" onClick={() => onNavigate('details', runId)}>
            Artifact Ledger &rarr;
          </button>
        </div>
      </div>

      {/* Acceptance Decision Detailed Diagnostic Banner */}
      <div
        className={`diagnostic-banner ${
          isPass
            ? 'diagnostic-banner-pass'
            : isFail
            ? 'diagnostic-banner-fail'
            : 'diagnostic-banner-inconclusive'
        }`}
        role="region"
        aria-label="Acceptance Policy Verdict"
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
              <span
                className={`badge ${
                  isPass ? 'badge-pass' : isFail ? 'badge-fail' : 'badge-inconclusive'
                }`}
                style={{ fontSize: '12px', fontWeight: 700 }}
              >
                VERDICT: {outcome}
              </span>
              <span style={{ fontSize: '13px', fontWeight: 600 }}>
                {isPass
                  ? 'All Mandatory Blocker Criteria Conclusively Satisfied'
                  : isFail
                  ? 'Policy Violation: Model Failed Blocker Threshold'
                  : 'Decision Inconclusive: Insufficient Statistical Evidence'}
              </span>
            </div>
            <p style={{ margin: '4px 0 0 0', fontSize: '13px', color: 'var(--color-ink-primary)', lineHeight: 1.5 }}>
              {acceptance?.summary_rationale ||
                (isPass
                  ? 'Candidate model meets all required performance and statistical superiority thresholds.'
                  : isInconclusive
                  ? 'Evaluation evidence is incomplete or statistically unavailable for one or more blocker rules.'
                  : 'Candidate model breached mandatory threshold constraints.')}
            </p>

            {/* Inconclusive Explanations */}
            {isInconclusive && inconclusiveCriteria.length > 0 && (
              <div style={{ marginTop: '12px', background: 'rgba(217, 119, 6, 0.08)', padding: '10px 14px', borderRadius: 'var(--radius-sm)' }}>
                <strong style={{ fontSize: '12px', color: 'var(--color-amber)' }}>
                  Inconclusive Diagnostic Breakdown ({inconclusiveCriteria.length} rule{inconclusiveCriteria.length > 1 ? 's' : ''}):
                </strong>
                <ul style={{ margin: '6px 0 0 0', paddingLeft: '18px', fontSize: '12px', color: 'var(--color-ink-primary)' }}>
                  {inconclusiveCriteria.map((c) => (
                    <li key={c.rule_id} style={{ marginBottom: '4px' }}>
                      <strong>Rule <code>{c.rule_id}</code></strong> ({c.metric} {c.operator} {c.threshold_value ?? '0'}):{' '}
                      <span className="mono" style={{ color: 'var(--color-amber)', fontWeight: 600 }}>[{c.reason_code}]</span>{' '}
                      {c.message}
                    </li>
                  ))}
                </ul>
                <div style={{ fontSize: '11px', color: 'var(--color-ink-secondary)', marginTop: '6px' }}>
                  <em>Note: Missing secondary metrics on naive baselines do not invalidate independent scalar comparisons (e.g. Accuracy comparison remains fully valid).</em>
                </div>
              </div>
            )}

            {/* Violated Criteria */}
            {isFail && violatedCriteria.length > 0 && (
              <div style={{ marginTop: '12px', background: 'rgba(220, 38, 38, 0.08)', padding: '10px 14px', borderRadius: 'var(--radius-sm)' }}>
                <strong style={{ fontSize: '12px', color: 'var(--color-crimson)' }}>
                  Violated Blocker Criteria ({violatedCriteria.length}):
                </strong>
                <ul style={{ margin: '6px 0 0 0', paddingLeft: '18px', fontSize: '12px', color: 'var(--color-ink-primary)' }}>
                  {violatedCriteria.map((c) => (
                    <li key={c.rule_id} style={{ marginBottom: '4px' }}>
                      <strong>Rule <code>{c.rule_id}</code></strong> ({c.metric} {c.operator} {c.threshold_value}):{' '}
                      <span className="mono" style={{ color: 'var(--color-crimson)' }}>[{c.reason_code}]</span>{' '}
                      {c.message}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {acceptance?.hash_signature && (
            <div style={{ textAlign: 'right', fontSize: '11px', color: 'var(--color-ink-muted)' }}>
              <div>SHA-256 Decision Hash:</div>
              <code className="font-mono" style={{ fontSize: '11px' }}>
                {acceptance.hash_signature.substring(0, 16)}...
              </code>
            </div>
          )}
        </div>
      </div>

      {/* Primary KPI Callout Strip */}
      <div className="grid grid-3">
        {/* Accuracy Comparison */}
        <div className="card">
          <div className="metric-label">Candidate vs Baseline Accuracy</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
            <span className="metric-value mono" style={{ color: 'var(--color-emerald)' }}>
              {candAcc !== null ? candAcc.toFixed(4) : 'N/A'}
            </span>
            <span style={{ fontSize: '13px', color: 'var(--color-ink-muted)' }}>
              vs {baseAcc !== null ? baseAcc.toFixed(4) : 'N/A'} (Baseline)
            </span>
          </div>
          <div style={{ fontSize: '12px', marginTop: '6px' }}>
            {deltaAcc !== null ? (
              <span className={deltaAcc > 0 ? 'diff-positive' : deltaAcc < 0 ? 'diff-negative' : 'diff-neutral'}>
                Net Advantage: {deltaAcc > 0 ? `+${deltaAcc.toFixed(4)}` : deltaAcc.toFixed(4)}
              </span>
            ) : (
              <span style={{ color: 'var(--color-ink-muted)' }}>Paired contrast unavailable</span>
            )}
          </div>
          {accStatTest && (
            <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
              Nadeau-Bengio: <code>p={accStatTest.p_value !== null ? accStatTest.p_value.toFixed(4) : 'N/A'}</code>{' '}
              {accStatTest.statistically_significant ? '(p < 0.05)' : '(not significant)'}
            </div>
          )}
        </div>

        {/* Macro F1 Comparison */}
        <div className="card">
          <div className="metric-label">Macro F1 Score</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
            <span className="metric-value mono" style={{ color: candF1 !== null ? 'var(--color-ink-primary)' : 'var(--color-amber)' }}>
              {candF1 !== null ? candF1.toFixed(4) : 'Undefined'}
            </span>
            <span style={{ fontSize: '13px', color: 'var(--color-ink-muted)' }}>
              vs {baseF1 !== null ? baseF1.toFixed(4) : 'Undefined (Dummy)'}
            </span>
          </div>
          <div style={{ fontSize: '12px', marginTop: '6px' }}>
            {deltaF1 !== null ? (
              <span className={deltaF1 > 0 ? 'diff-positive' : deltaF1 < 0 ? 'diff-negative' : 'diff-neutral'}>
                &Delta;: {deltaF1 > 0 ? `+${deltaF1.toFixed(4)}` : deltaF1.toFixed(4)}
              </span>
            ) : (
              <span style={{ color: 'var(--color-ink-muted)' }}>Baseline F1 undefined (0 minority predictions)</span>
            )}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
            Strategy: Prior Probability Dummy Anchor
          </div>
        </div>

        {/* Statistical Test Summary */}
        <div className="card">
          <div className="metric-label">Paired Statistical Superiority</div>
          <div className="metric-value mono" style={{ color: accStatTest?.statistically_significant ? 'var(--color-emerald)' : 'var(--color-ink-muted)', fontSize: '18px' }}>
            {accStatTest?.statistically_significant ? 'Statistically Superior' : accStatTest ? 'Inconclusive / N.S.' : 'Test Unavailable'}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--color-ink-secondary)', marginTop: '4px' }}>
            {accStatTest
              ? `t = ${accStatTest.statistic !== null ? accStatTest.statistic.toFixed(2) : 'N/A'}, df = ${accStatTest.degrees_of_freedom ?? 4}`
              : 'Corrected resampled t-test (Nadeau & Bengio)'}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
            Corrects for cross-validation fold training overlap
          </div>
        </div>
      </div>

      {/* Comprehensive Metric Comparison Table */}
      <div className="card">
        <div className="panel-header">
          <div>
            <div className="panel-title">Comprehensive Metric Comparison Ledger</div>
            <div className="panel-description">
              Out-of-fold performance estimates with sample standard deviations and paired hypothesis testing.
            </div>
          </div>
        </div>

        <div className="data-table-container">
          <table className="data-table" aria-label="Metric comparison table">
            <thead>
              <tr>
                <th>Metric Name</th>
                <th>Candidate Out-of-Fold (&mu; &plusmn; &sigma;)</th>
                <th>Uncorrected 95% CI</th>
                <th>Baseline Mean (&mu;)</th>
                <th>Net Difference (&Delta;)</th>
                <th>Corrected Paired t-Test (p-value)</th>
              </tr>
            </thead>
            <tbody>
              {metricKeys.map((key) => {
                const cSum = candSummaries[key];
                const bSum = baseSummaries[key];

                const cMean = cSum?.mean;
                const bMean = bSum?.mean;
                const cStd = cSum?.std;
                const ci = cSum?.confidence_interval_95;

                const hasDiff = cMean !== null && cMean !== undefined && bMean !== null && bMean !== undefined;
                const diff = hasDiff ? cMean - bMean : null;

                const pTest = comparison.metric_comparisons?.[key]?.corrected_resampled_t_test;

                return (
                  <tr key={key}>
                    <td className="mono" style={{ fontWeight: 600 }}>{key}</td>
                    <td className="mono tabular-nums">
                      {cMean !== null && cMean !== undefined ? (
                        <span>
                          <strong>{cMean.toFixed(4)}</strong>
                          {cStd !== undefined && cStd !== null && (
                            <span style={{ color: 'var(--color-ink-muted)' }}> &plusmn; {cStd.toFixed(4)}</span>
                          )}
                        </span>
                      ) : (
                        <span style={{ color: 'var(--color-amber)', fontSize: '12px' }}>
                          Undefined {cSum?.undefined_reasons?.[0] ? `(${cSum.undefined_reasons[0]})` : ''}
                        </span>
                      )}
                    </td>
                    <td className="mono tabular-nums" style={{ color: 'var(--color-ink-muted)' }}>
                      {ci ? `[${ci[0].toFixed(4)}, ${ci[1].toFixed(4)}]` : '—'}
                    </td>
                    <td className="mono tabular-nums">
                      {bMean !== null && bMean !== undefined ? (
                        bMean.toFixed(4)
                      ) : (
                        <span style={{ color: 'var(--color-ink-muted)', fontSize: '12px' }}>
                          Undefined (single-class)
                        </span>
                      )}
                    </td>
                    <td className="mono tabular-nums">
                      {diff !== null ? (
                        <span className={diff > 0 ? 'diff-positive' : diff < 0 ? 'diff-negative' : 'diff-neutral'}>
                          {diff > 0 ? `+${diff.toFixed(4)}` : diff.toFixed(4)}
                        </span>
                      ) : (
                        <span style={{ color: 'var(--color-ink-muted)' }}>&mdash;</span>
                      )}
                    </td>
                    <td className="mono tabular-nums" style={{ fontSize: '12px' }}>
                      {pTest ? (
                        pTest.p_value !== null ? (
                          <span style={{ color: pTest.p_value < 0.05 ? 'var(--color-emerald)' : 'var(--color-ink-muted)' }}>
                            p = {pTest.p_value.toFixed(4)} {pTest.p_value < 0.05 ? '(p < 0.05)' : '(n.s.)'}
                          </span>
                        ) : (
                          <span style={{ color: 'var(--color-amber)' }}>{pTest.status || 'Unavailable'}</span>
                        )
                      ) : (
                        <span style={{ color: 'var(--color-ink-muted)' }}>&mdash;</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop: '10px', fontSize: '11px', color: 'var(--color-ink-muted)', lineHeight: 1.4 }}>
          * 95% Confidence Intervals are calculated over cross-validation fold scores via Student's t-distribution. They estimate mean fold performance across this sample partition and do not account for training overlap variance (Bengio & Grandvalet, 2004). Paired baseline tests use the Nadeau-Bengio corrected estimator.
        </div>
      </div>

      {/* Fold Distribution Variance */}
      {folds && folds.length > 0 && (
        <div className="card">
          <div className="panel-header">
            <div>
              <div className="panel-title">Cross-Validation Fold Stability & Distribution</div>
              <div className="panel-description">
                Per-fold score breakdown comparing candidate versus dummy baseline across {folds.length} stratified partitions.
              </div>
            </div>
            <div style={{ display: 'flex', gap: '6px' }}>
              {['accuracy', 'macro_f1'].map((m) => (
                <button
                  key={m}
                  className={`btn btn-sm ${selectedChartMetric === m ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setSelectedChartMetric(m)}
                >
                  {m === 'accuracy' ? 'Accuracy' : 'Macro F1'}
                </button>
              ))}
            </div>
          </div>
          <FoldDistributionChart
            candidateFolds={folds}
            baselineFolds={baseResult.fold_results || []}
            metricName={selectedChartMetric}
          />
        </div>
      )}

      {/* Confusion Matrix Section */}
      <div className="grid grid-2">
        <div className="card">
          <h3 style={{ margin: '0 0 4px 0', fontSize: '15px' }}>Candidate Confusion Matrix</h3>
          <p style={{ margin: '0 0 16px 0', fontSize: '12px', color: 'var(--color-ink-muted)' }}>
            Aggregated cross-validation predictions for candidate estimator.
          </p>
          {candResult.aggregate_confusion_matrix ? (
            <ConfusionMatrix
              matrix={candResult.aggregate_confusion_matrix}
              classLabels={data.dataset_metadata?.class_labels || ['Class 0', 'Class 1']}
            />
          ) : (
            <div className="empty-state">No candidate confusion matrix recorded</div>
          )}
        </div>

        <div className="card">
          <h3 style={{ margin: '0 0 4px 0', fontSize: '15px' }}>Baseline Confusion Matrix</h3>
          <p style={{ margin: '0 0 16px 0', fontSize: '12px', color: 'var(--color-ink-muted)' }}>
            Aggregated cross-validation predictions for naive baseline (majority class anchor).
          </p>
          {baseResult.aggregate_confusion_matrix ? (
            <ConfusionMatrix
              matrix={baseResult.aggregate_confusion_matrix}
              classLabels={data.dataset_metadata?.class_labels || ['Class 0', 'Class 1']}
            />
          ) : (
            <div className="empty-state">No baseline confusion matrix recorded</div>
          )}
        </div>
      </div>

      {/* Evaluation Methodology Notice */}
      <div className="card" style={{ background: 'var(--color-stone-surface)', borderColor: 'var(--color-stone-border)' }}>
        <h4 style={{ margin: '0 0 6px 0', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Methodological Notice & Scientific Evidence Standard
        </h4>
        <p style={{ margin: 0, fontSize: '12px', color: 'var(--color-ink-secondary)', lineHeight: '1.5' }}>
          Reported performance metrics reflect cross-validation observations evaluated strictly out-of-fold with zero data leakage across feature preprocessing or scaling. Cross-validation variance establishes internal empirical consistency under the observed dataset distribution, not unconditional generalization guarantees to non-stationary future production populations. An acceptance verdict of <code>PASS</code> certifies policy compliance, not automated release authorization.
        </p>
      </div>
    </div>
  );
};

export default EvaluationResultsScreen;
