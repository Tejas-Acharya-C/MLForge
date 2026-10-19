import React, { useEffect, useState, useMemo } from 'react';
import { api } from '../api/client';
import { EvaluationSummaryItem, RunState } from '../types/api';

interface LabOverviewProps {
  onSelectRun?: (runId: string, targetScreen?: 'results' | 'reliability' | 'details' | 'acceptance') => void;
  onNewEvaluation?: (preset?: string) => void;
  onNavigate?: (screen: string, runId?: string) => void;
}

export const LabOverview: React.FC<LabOverviewProps> = ({ onSelectRun, onNewEvaluation, onNavigate }) => {
  const handleSelect = (runId: string, targetScreen: 'results' | 'reliability' | 'details' | 'acceptance' = 'results') => {
    if (onSelectRun) {
      onSelectRun(runId, targetScreen);
    } else if (onNavigate) {
      onNavigate(targetScreen, runId);
    }
  };

  const handleNew = (preset?: string) => {
    if (onNewEvaluation) {
      onNewEvaluation(preset);
    } else if (onNavigate) {
      onNavigate('new_evaluation');
    }
  };

  const [runs, setRuns] = useState<EvaluationSummaryItem[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [stateFilter, setStateFilter] = useState<string>('all');
  const [outcomeFilter, setOutcomeFilter] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const loadRuns = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await api.listEvaluations({
        state: stateFilter !== 'all' ? stateFilter : undefined,
        limit: 50,
      });
      setRuns(resp.items);
      setTotal(resp.total);
    } catch (err: any) {
      setError(err.message || 'Failed to connect to MLForge backend API.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadRuns();
  }, [stateFilter]);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(text);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Filtered runs based on search term and outcome filter
  const filteredRuns = useMemo(() => {
    return runs.filter((r) => {
      // Outcome filter
      if (outcomeFilter !== 'all') {
        if (r.acceptance_outcome !== outcomeFilter) return false;
      }
      // Search term
      if (searchTerm.trim()) {
        const query = searchTerm.toLowerCase().trim();
        const runIdMatch = r.run_id.toLowerCase().includes(query);
        const modelMatch = (r.candidate_model_type || '').toLowerCase().includes(query);
        const sourceMatch = (r.metadata?.source || '').toLowerCase().includes(query);
        const targetMatch = (r.metadata?.target_column || '').toLowerCase().includes(query);
        if (!runIdMatch && !modelMatch && !sourceMatch && !targetMatch) return false;
      }
      return true;
    });
  }, [runs, outcomeFilter, searchTerm]);

  // Aggregate dashboard stats
  const stats = useMemo(() => {
    const totalCount = runs.length;
    const completedCount = runs.filter((r) => r.state === 'completed').length;
    const passCount = runs.filter((r) => r.acceptance_outcome === 'PASS').length;
    const runningCount = runs.filter((r) => r.state === 'running' || r.state === 'created').length;
    const passRate = completedCount > 0 ? ((passCount / completedCount) * 100).toFixed(0) : '—';
    return { totalCount, completedCount, passCount, runningCount, passRate };
  }, [runs]);

  const getStatusBadge = (state: RunState, outcome?: string | null) => {
    if (state === 'running') {
      return (
        <span className="badge badge-running" style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
          <span className="dot-pulse" style={{ width: '6px', height: '6px' }} />
          Running
        </span>
      );
    }
    if (state === 'failed') {
      return <span className="badge badge-fail">Execution Failed</span>;
    }
    if (state === 'cancelled') {
      return <span className="badge badge-neutral">Cancelled</span>;
    }
    if (outcome === 'PASS') {
      return <span className="badge badge-pass" title="All mandatory blocker criteria satisfied">PASS</span>;
    }
    if (outcome === 'FAIL') {
      return <span className="badge badge-fail" title="One or more blocker criteria violated">FAIL</span>;
    }
    if (outcome === 'INCONCLUSIVE') {
      return (
        <span
          className="badge badge-inconclusive"
          title="Evidence insufficient or statistically unavailable"
        >
          INCONCLUSIVE
        </span>
      );
    }
    return <span className="badge badge-neutral">{state}</span>;
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Top Banner & Control Toolbar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <span className="env-badge">Local Offline Lab</span>
            <span style={{ fontSize: '12px', color: 'var(--color-ink-muted)' }}>SQLite Persistence</span>
          </div>
          <h1 style={{ margin: 0, fontSize: '24px', letterSpacing: '-0.02em' }}>Laboratory Run History</h1>
          <p style={{ color: 'var(--color-ink-secondary)', fontSize: '13px', marginTop: '4px', maxWidth: '700px' }}>
            Permanent audit ledger of cross-validation evaluations, baseline statistical comparisons, calibration diagnostics, and Ed25519 cryptographic release records.
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button className="btn btn-secondary btn-sm" onClick={loadRuns} disabled={loading} aria-label="Refresh run ledger">
            {loading ? 'Refreshing...' : 'Refresh Ledger'}
          </button>
          <button className="btn btn-primary" onClick={() => handleNew()} aria-label="Configure new evaluation run">
            + New Evaluation Run
          </button>
        </div>
      </div>

      {/* Summary KPI Cards */}
      <div className="grid grid-4" style={{ gap: '16px' }}>
        <div className="metric-card">
          <div className="metric-card-label">Total Recorded Runs</div>
          <div className="metric-card-value">{total}</div>
          <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
            Persisted in <code>artifacts/mlforge.db</code>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-card-label">Completed Evaluations</div>
          <div className="metric-card-value" style={{ color: 'var(--color-ink-primary)' }}>
            {stats.completedCount}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
            Stratified cross-validation finished
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-card-label">Acceptance Pass Rate</div>
          <div className="metric-card-value" style={{ color: stats.passCount > 0 ? 'var(--color-emerald)' : 'var(--color-ink-muted)' }}>
            {stats.passRate === '—' ? '—' : `${stats.passRate}%`}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
            {stats.passCount} of {stats.completedCount} satisfied policy
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-card-label">Active / Queued Runs</div>
          <div className="metric-card-value" style={{ color: stats.runningCount > 0 ? '#2563eb' : 'var(--color-ink-muted)' }}>
            {stats.runningCount}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
            {stats.runningCount > 0 ? 'Evaluating in background' : 'Engine idle and ready'}
          </div>
        </div>
      </div>

      {/* Search and Filter Toolbar */}
      <div className="card" style={{ padding: '16px', marginBottom: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', flexWrap: 'wrap' }}>
          {/* Search bar */}
          <div style={{ flex: '1 1 240px', minWidth: '200px' }}>
            <input
              type="text"
              className="form-input"
              placeholder="Search by Run ID, dataset, or model type..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{ padding: '6px 12px', fontSize: '13px' }}
              aria-label="Search evaluation runs"
            />
          </div>

          {/* Lifecycle state pills */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--color-ink-muted)', marginRight: '4px' }}>State:</span>
            {['all', 'completed', 'running', 'failed', 'cancelled'].map((st) => (
              <button
                key={st}
                className="btn btn-sm"
                style={{
                  textTransform: 'capitalize',
                  backgroundColor: stateFilter === st ? 'var(--color-accent-subtle)' : 'transparent',
                  borderColor: stateFilter === st ? '#bfdbfe' : 'var(--color-border-subtle)',
                  color: stateFilter === st ? 'var(--color-accent-primary)' : 'var(--color-ink-secondary)',
                  fontWeight: stateFilter === st ? 600 : 400,
                  fontSize: '11px',
                  padding: '3px 8px',
                }}
                onClick={() => setStateFilter(st)}
                aria-pressed={stateFilter === st}
              >
                {st}
              </button>
            ))}
          </div>

          {/* Acceptance outcome pills */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--color-ink-muted)', marginRight: '4px' }}>Outcome:</span>
            {['all', 'PASS', 'FAIL', 'INCONCLUSIVE'].map((out) => (
              <button
                key={out}
                className="btn btn-sm"
                style={{
                  backgroundColor: outcomeFilter === out ? 'var(--color-surface-sunken)' : 'transparent',
                  borderColor: outcomeFilter === out ? 'var(--color-border-strong)' : 'var(--color-border-subtle)',
                  color: outcomeFilter === out ? 'var(--color-ink-primary)' : 'var(--color-ink-muted)',
                  fontWeight: outcomeFilter === out ? 600 : 400,
                  fontSize: '11px',
                  padding: '3px 8px',
                }}
                onClick={() => setOutcomeFilter(out)}
                aria-pressed={outcomeFilter === out}
              >
                {out}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Error Callout */}
      {error && (
        <div className="callout callout-error" role="alert">
          <div style={{ fontWeight: 600, marginBottom: '2px' }}>API Connection Error</div>
          <div>{error}</div>
          <button className="btn btn-secondary btn-sm" style={{ marginTop: '8px' }} onClick={loadRuns}>
            Retry Connection
          </button>
        </div>
      )}

      {/* Loading Skeleton */}
      {loading && (
        <div className="card" style={{ padding: '48px', textAlign: 'center', color: 'var(--color-ink-muted)' }}>
          <div className="dot-pulse" style={{ marginBottom: '12px' }} />
          <div>Synchronizing with local SQLite run database...</div>
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && filteredRuns.length === 0 && (
        <div className="empty-state">
          <div className="empty-state-title">No evaluation runs match criteria</div>
          <p className="empty-state-text">
            {runs.length === 0
              ? 'No evaluations are recorded yet. Start by executing one of our deterministic classification benchmarks.'
              : 'No runs matched your search query and filters. Clear filters to see all recorded runs.'}
          </p>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '12px', flexWrap: 'wrap' }}>
            {runs.length === 0 ? (
              <>
                <button className="btn btn-primary" onClick={() => handleNew('churn')}>
                  Run Customer Churn Benchmark
                </button>
                <button className="btn btn-secondary" onClick={() => handleNew('breast_cancer')}>
                  Run Breast Cancer Benchmark
                </button>
              </>
            ) : (
              <button
                className="btn btn-secondary"
                onClick={() => {
                  setStateFilter('all');
                  setOutcomeFilter('all');
                  setSearchTerm('');
                }}
              >
                Reset All Filters
              </button>
            )}
          </div>
        </div>
      )}

      {/* Runs Table */}
      {!loading && !error && filteredRuns.length > 0 && (
        <div className="data-table-container">
          <table className="data-table" aria-label="Evaluation runs ledger">
            <thead>
              <tr>
                <th>Lifecycle State</th>
                <th>Run Identifier</th>
                <th>Dataset & Target</th>
                <th>Candidate Model</th>
                <th style={{ textAlign: 'right' }}>Accuracy</th>
                <th style={{ textAlign: 'right' }}>Macro F1 (95% CI)</th>
                <th>Acceptance Outcome</th>
                <th>Timestamp (UTC)</th>
                <th style={{ textAlign: 'right' }}>Workbench Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredRuns.map((run) => {
                const f1 = run.summary_metrics['macro_f1'];
                const acc = run.summary_metrics['accuracy'];
                const f1Detail = run.metric_details.find((m) => m.metric_name === 'macro_f1');
                const isCopied = copiedId === run.run_id;

                return (
                  <tr key={run.run_id}>
                    <td>{getStatusBadge(run.state, run.acceptance_outcome)}</td>
                    <td className="tabular-nums font-mono">
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span
                          style={{
                            color: 'var(--color-accent-primary)',
                            cursor: 'pointer',
                            fontWeight: 600,
                            textDecoration: 'underline',
                          }}
                          onClick={() => handleSelect(run.run_id, 'results')}
                          title="Open performance results"
                        >
                          {run.run_id}
                        </span>
                        <button
                          onClick={() => copyToClipboard(run.run_id)}
                          style={{
                            border: 'none',
                            background: 'transparent',
                            cursor: 'pointer',
                            fontSize: '11px',
                            color: isCopied ? 'var(--color-emerald)' : 'var(--color-ink-muted)',
                            padding: '2px',
                          }}
                          title="Copy Run ID"
                          aria-label={`Copy Run ID ${run.run_id}`}
                        >
                          {isCopied ? '✓' : '⧉'}
                        </button>
                      </div>
                    </td>
                    <td>
                      <div>
                        <strong>{run.metadata.source || 'Tabular Data'}</strong>
                        {run.metadata.target_column && (
                          <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)' }}>
                            target: <code>{run.metadata.target_column}</code>
                          </div>
                        )}
                      </div>
                    </td>
                    <td>
                      <span style={{ textTransform: 'capitalize' }}>
                        {run.candidate_model_type?.replace(/_/g, ' ') || 'Logistic Regression'}
                      </span>
                    </td>
                    <td style={{ textAlign: 'right' }} className="tabular-nums">
                      {acc !== undefined && acc !== null ? (
                        <strong>{acc.toFixed(4)}</strong>
                      ) : (
                        <span style={{ color: 'var(--color-ink-muted)' }}>&mdash;</span>
                      )}
                    </td>
                    <td style={{ textAlign: 'right' }} className="tabular-nums">
                      {f1 !== undefined && f1 !== null ? (
                        <div>
                          <span>{f1.toFixed(4)}</span>
                          {f1Detail?.ci_lower !== undefined && f1Detail?.ci_upper !== undefined && (
                            <div style={{ fontSize: '10px', color: 'var(--color-ink-muted)' }}>
                              [{f1Detail.ci_lower.toFixed(3)}, {f1Detail.ci_upper.toFixed(3)}]
                            </div>
                          )}
                        </div>
                      ) : (
                        <span style={{ color: 'var(--color-ink-muted)' }}>&mdash;</span>
                      )}
                    </td>
                    <td>
                      {run.acceptance_outcome ? (
                        getStatusBadge(run.state, run.acceptance_outcome)
                      ) : (
                        <span style={{ color: 'var(--color-ink-muted)', fontSize: '12px' }}>Not evaluated</span>
                      )}
                    </td>
                    <td className="tabular-nums font-mono" style={{ fontSize: '12px', color: 'var(--color-ink-secondary)' }}>
                      {run.created_at.replace('T', ' ').substring(0, 19)}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div style={{ display: 'inline-flex', gap: '4px', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                        <button
                          className="btn btn-secondary btn-sm"
                          onClick={() => handleSelect(run.run_id, 'results')}
                          title="View cross-validation metrics"
                        >
                          Results
                        </button>
                        <button
                          className="btn btn-secondary btn-sm"
                          onClick={() => handleSelect(run.run_id, 'reliability')}
                          title="View calibration and perturbation analysis"
                        >
                          Slices
                        </button>
                        <button
                          className="btn btn-secondary btn-sm"
                          onClick={() => handleSelect(run.run_id, 'acceptance')}
                          title="Inspect release gate compliance and certificates"
                        >
                          Gate
                        </button>
                        <button
                          className="btn btn-secondary btn-sm"
                          onClick={() => handleSelect(run.run_id, 'details')}
                          title="Inspect artifact ledger and audit archive"
                        >
                          Audit
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default LabOverview;
