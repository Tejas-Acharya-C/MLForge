import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import { CalibrationCurveResult, RobustnessEvaluationResult } from '../types/api';
import CalibrationChart from '../components/CalibrationChart';
import RobustnessChart from '../components/RobustnessChart';

interface Props {
  runId: string;
  onNavigate: (screen: string, runId?: string) => void;
}

export const ReliabilityAnalysisScreen: React.FC<Props> = ({ runId, onNavigate }) => {
  const [calibration, setCalibration] = useState<CalibrationCurveResult | null>(null);
  const [subgroups, setSubgroups] = useState<any | null>(null);
  const [robustness, setRobustness] = useState<RobustnessEvaluationResult | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [robustnessRunning, setRobustnessRunning] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [robustnessMsg, setRobustnessMsg] = useState<string | null>(null);

  // Robustness run configuration state
  const [selectedPerturbations, setSelectedPerturbations] = useState<string[]>([
    'gaussian_noise',
    'missingness',
    'label_noise',
  ]);

  useEffect(() => {
    let mounted = true;
    const fetchReliabilityData = async () => {
      setLoading(true);
      setError(null);
      try {
        const [calData, subData, robData] = await Promise.all([
          api.getCalibrationDiagnostics(runId).catch(() => null),
          api.getSubgroupDiagnostics(runId).catch(() => null),
          api.getRobustnessResults(runId).catch(() => null),
        ]);
        if (!mounted) return;
        setCalibration(calData);
        setSubgroups(subData);
        setRobustness(robData);
      } catch (err: any) {
        if (!mounted) return;
        setError(err.message || 'Failed to load reliability diagnostics');
      } finally {
        if (mounted) setLoading(false);
      }
    };
    fetchReliabilityData();
    return () => {
      mounted = false;
    };
  }, [runId]);

  const handleRunRobustness = async () => {
    if (selectedPerturbations.length === 0) {
      setRobustnessMsg('Please select at least one perturbation type.');
      return;
    }
    setRobustnessRunning(true);
    setRobustnessMsg(null);
    try {
      const res = await api.runRobustnessTest(runId, {
        perturbations: selectedPerturbations,
        severity_levels: [0.05, 0.10, 0.25, 0.50],
      });
      setRobustness(res);
      setRobustnessMsg('Robustness stress test completed and recorded.');
    } catch (err: any) {
      setRobustnessMsg(`Robustness test failed: ${err.message || 'Unknown error'}`);
    } finally {
      setRobustnessRunning(false);
    }
  };

  const togglePerturbation = (type: string) => {
    if (selectedPerturbations.includes(type)) {
      setSelectedPerturbations(selectedPerturbations.filter((p) => p !== type));
    } else {
      setSelectedPerturbations([...selectedPerturbations, type]);
    }
  };

  if (loading) {
    return (
      <div className="empty-state" style={{ padding: '64px 0' }}>
        <div className="status-badge status-running" style={{ marginBottom: '16px' }}>
          Analyzing Reliability
        </div>
        <div style={{ color: 'var(--color-ink-muted)' }}>Retrieving calibration curves, subgroup disparities, and stress profiles for run {runId}...</div>
      </div>
    );
  }

  const ece = calibration?.expected_calibration_error ?? null;
  const brier = calibration?.brier_score ?? null;
  const mce = calibration?.max_calibration_error ?? null;

  const hasRobustnessCurves = Boolean(robustness?.curves && robustness.curves.length > 0);
  const aurdc = hasRobustnessCurves ? (robustness?.curves?.[0]?.aurdc ?? null) : null;
  const isResilient = aurdc !== null ? aurdc >= 0.70 : null;

  // Flatten subgroups list and compute average error rate across slices for disparity comparison
  const allSlices: any[] = [];
  if (subgroups?.subgroup_diagnostics) {
    Object.entries(subgroups.subgroup_diagnostics).forEach(([col, items]: [string, any]) => {
      if (Array.isArray(items)) {
        items.forEach((item) => allSlices.push({ ...item, colName: col }));
      }
    });
  }
  const avgErrorRate = allSlices.length > 0
    ? allSlices.reduce((acc, s) => acc + (s.error_rate || 0), 0) / allSlices.length
    : 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px', flexWrap: 'wrap' }}>
            <span className="mono" style={{ fontSize: '13px', color: 'var(--color-ink-muted)' }}>Run {runId}</span>
            <span className="status-badge status-completed">Diagnostics</span>
          </div>
          <h2 style={{ margin: 0, fontSize: '20px' }}>Reliability, Calibration & Robustness Laboratory</h2>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <button className="btn btn-secondary" onClick={() => onNavigate('results', runId)}>
            &larr; Performance Results
          </button>
          <button className="btn btn-secondary" onClick={() => onNavigate('governance', runId)}>
            Release Gate &rarr;
          </button>
          <button className="btn btn-secondary" onClick={() => onNavigate('artifacts', runId)}>
            Artifact Ledger &rarr;
          </button>
        </div>
      </div>

      {error && (
        <div className="card" style={{ borderColor: 'var(--color-crimson-border)' }}>
          <p style={{ margin: 0, color: 'var(--color-crimson)' }}>{error}</p>
        </div>
      )}

      {/* Section 1: Probability Calibration & Diagnostics */}
      <div className="grid grid-2">
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '8px' }}>
            <div>
              <h3 style={{ margin: 0, fontSize: '15px' }}>Reliability Diagram & Calibration</h3>
              <div style={{ fontSize: '12px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
                Forecast posterior probabilities versus observed empirical rates
              </div>
            </div>
            {calibration && (
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                <span className="badge badge-neutral">
                  ECE: <strong className="mono tabular-nums">{ece !== null ? ece.toFixed(4) : 'N/A'}</strong>
                </span>
                <span className="badge badge-neutral">
                  Brier: <strong className="mono tabular-nums">{brier !== null ? brier.toFixed(4) : 'N/A'}</strong>
                </span>
              </div>
            )}
          </div>

          {calibration ? (
            <CalibrationChart data={calibration} />
          ) : (
            <div className="empty-state">No probability calibration data recorded for this run</div>
          )}
        </div>

        {/* Confidence Distribution & Pathology */}
        <div className="card">
          <h3 style={{ margin: '0 0 6px 0', fontSize: '15px' }}>Calibration Properties & Risk Profile</h3>
          <p style={{ margin: '0 0 16px 0', fontSize: '13px', color: 'var(--color-ink-muted)' }}>
            Examining calibration properties and overconfident risk zones.
          </p>

          {calibration ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div className="grid grid-2">
                <div style={{ padding: '14px', background: 'var(--color-stone-surface)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-stone-border)' }}>
                  <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Maximum Calibration Error (MCE)</div>
                  <div className="mono tabular-nums" style={{ fontSize: '18px', fontWeight: 600, color: 'var(--color-amber)', marginTop: '4px' }}>
                    {mce !== null ? mce.toFixed(4) : 'Not Evaluated'}
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
                    Worst-case discrepancy across bins
                  </div>
                </div>
                <div style={{ padding: '14px', background: 'var(--color-stone-surface)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-stone-border)' }}>
                  <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Calibration Assessment</div>
                  <div className="mono" style={{ fontSize: '15px', fontWeight: 600, color: ece !== null && ece < 0.1 ? 'var(--color-emerald)' : 'var(--color-amber)', marginTop: '4px' }}>
                    {ece !== null
                      ? ece < 0.05
                        ? 'Well Calibrated'
                        : ece < 0.12
                        ? 'Moderate Distortion'
                        : 'Severe Miscalibration'
                      : 'Unavailable'}
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
                    ECE &lt; 0.05 threshold guideline
                  </div>
                </div>
              </div>

              <div style={{ fontSize: '13px', color: 'var(--color-ink-secondary)', lineHeight: 1.6, padding: '12px', background: 'var(--color-stone-bg)', borderRadius: 'var(--radius-sm)' }}>
                <p style={{ margin: '0 0 8px 0' }}>
                  <strong>Operational Interpretation:</strong> Calibration quantifies whether predicted probabilities accurately represent real frequencies of occurrence. If uncalibrated, downstream decision thresholds cannot guarantee bounded false-alarm rates.
                </p>
                <p style={{ margin: 0 }}>
                  A Brier score of <strong className="mono font-semibold">{brier !== null ? brier.toFixed(4) : 'N/A'}</strong> measures mean squared probability error against true binary outcomes.
                </p>
              </div>
            </div>
          ) : (
            <div className="empty-state">Calibration diagnostics not available</div>
          )}
        </div>
      </div>

      {/* Section 2: Subgroup Disparities */}
      <div className="card">
        <h3 style={{ margin: '0 0 6px 0', fontSize: '15px' }}>Subgroup Slice Performance & Error Disparity</h3>
        <p style={{ margin: '0 0 16px 0', fontSize: '13px', color: 'var(--color-ink-muted)' }}>
          Assessing classification performance stability across observed demographic or categorical cohorts.
        </p>

        {allSlices.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Cohort / Slice</th>
                  <th>Sample Size (N)</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>Error Rate</th>
                  <th>Disparity Ratio (vs Mean)</th>
                </tr>
              </thead>
              <tbody>
                {allSlices.map((slice: any, idx: number) => {
                  const errorRate = slice.error_rate || 0;
                  const disparityRatio = avgErrorRate > 0 ? errorRate / avgErrorRate : 1.0;
                  const isSmallSample = slice.sample_count < 30;

                  return (
                    <tr key={`${slice.colName}-${idx}`}>
                      <td className="mono" style={{ fontWeight: 600 }}>
                        {slice.subgroup_column || slice.colName} = {slice.subgroup_value}
                      </td>
                      <td className="mono">
                        {slice.sample_count}
                        {isSmallSample && (
                          <span style={{ fontSize: '11px', color: 'var(--color-amber)', marginLeft: '6px' }} title="Small sample size: estimates exhibit elevated variance">
                            (N&lt;30)
                          </span>
                        )}
                      </td>
                      <td className="mono">{(slice.accuracy || 0).toFixed(4)}</td>
                      <td className="mono">{(slice.precision || 0).toFixed(4)}</td>
                      <td className="mono">{(slice.recall || 0).toFixed(4)}</td>
                      <td className="mono" style={{ color: errorRate > 0.3 ? 'var(--color-crimson)' : 'var(--color-ink-primary)' }}>
                        {errorRate.toFixed(4)}
                      </td>
                      <td className="mono font-semibold" style={{ color: disparityRatio > 1.25 ? 'var(--color-amber)' : 'var(--color-ink-primary)' }}>
                        {disparityRatio.toFixed(2)}x
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div style={{ marginTop: '10px', fontSize: '11px', color: 'var(--color-ink-muted)' }}>
              Note: Disparity ratio compares individual slice error rate against cohort mean ({avgErrorRate.toFixed(4)}). Slices with N&lt;30 have wide estimation intervals.
            </div>
          </div>
        ) : (
          <div className="empty-state">No subgroup disparity attributes registered for this evaluation run</div>
        )}
      </div>

      {/* Section 3: Robustness Stress Testing Workbench */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '8px' }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '15px' }}>Perturbation Stress Testing & Robustness Lab</h3>
            <div style={{ fontSize: '12px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
              Evaluate degradation profiles under covariate shifts, additive noise, and feature dropouts
            </div>
          </div>
          {hasRobustnessCurves && (
            <span className="status-badge" style={{ background: 'var(--color-stone-surface)', color: 'var(--color-ink)' }}>
              Status: {isResilient ? <span style={{ color: 'var(--color-emerald)' }}>Resilient</span> : <span style={{ color: 'var(--color-crimson)' }}>Degraded</span>}
            </span>
          )}
        </div>

        {/* Configuration controls */}
        <div style={{ padding: '16px', background: 'var(--color-stone-surface)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-stone-border)', marginBottom: '16px' }}>
          <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '8px' }}>Perturbation Stress Scenarios</div>
          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', marginBottom: '16px' }}>
            {[
              { id: 'gaussian_noise', label: 'Gaussian Covariate Noise' },
              { id: 'missingness', label: 'Feature Missingness (MCAR)' },
              { id: 'label_noise', label: 'Label Inversion Noise' },
            ].map((p) => (
              <label key={p.id} style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '13px' }}>
                <input
                  type="checkbox"
                  checked={selectedPerturbations.includes(p.id)}
                  onChange={() => togglePerturbation(p.id)}
                  disabled={robustnessRunning}
                />
                <span>{p.label}</span>
              </label>
            ))}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <button
              className="btn btn-primary"
              onClick={handleRunRobustness}
              disabled={robustnessRunning}
            >
              {robustnessRunning ? 'Executing Stress Tests...' : 'Execute Robustness Stress Tests'}
            </button>
            <span style={{ fontSize: '12px', color: 'var(--color-ink-muted)' }}>
              Evaluates test points at severities: 5%, 10%, 25%, 50%
            </span>
          </div>

          {robustnessMsg && (
            <div style={{ marginTop: '12px', fontSize: '13px', color: robustnessMsg.includes('failed') ? 'var(--color-crimson)' : 'var(--color-emerald)' }}>
              {robustnessMsg}
            </div>
          )}
        </div>

        {/* Robustness Results Display */}
        {robustness && robustness.curves && robustness.curves.length > 0 ? (
          <div>
            <div className="grid grid-2" style={{ marginBottom: '16px' }}>
              <div className="card" style={{ background: 'var(--color-stone-bg)' }}>
                <div className="metric-label">Area Under Retention Curve (AURDC)</div>
                <div className="metric-value mono" style={{ color: isResilient ? 'var(--color-emerald)' : 'var(--color-crimson)' }}>
                  {aurdc !== null ? aurdc.toFixed(4) : 'N/A'}
                </div>
                <div style={{ fontSize: '12px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
                  Target retention &ge; 0.7000 over 50% perturbation envelope
                </div>
              </div>

              <div className="card" style={{ background: 'var(--color-stone-bg)' }}>
                <div className="metric-label">Stress Test Decision</div>
                <div className="metric-value mono" style={{ color: isResilient ? 'var(--color-emerald)' : 'var(--color-crimson)' }}>
                  {isResilient ? 'PASSED CRITERIA' : 'FAILED RESILIENCE'}
                </div>
                <div style={{ fontSize: '12px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
                  Measured retention across {robustness.curves.length} perturbation curve(s).
                </div>
              </div>
            </div>

            <RobustnessChart curves={robustness.curves} />
          </div>
        ) : (
          <div className="empty-state">
            No robustness stress tests have been executed for this model yet. Run the stress scenarios above.
          </div>
        )}
      </div>
    </div>
  );
};

export default ReliabilityAnalysisScreen;
