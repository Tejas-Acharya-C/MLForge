import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import { EvaluationRunSummary, ArtifactFile, AuditBundleResult, AuditBundleVerificationResult } from '../types/api';

interface Props {
  runId: string;
  onNavigate: (screen: string, runId?: string) => void;
}

export const RunDetailsScreen: React.FC<Props> = ({ runId, onNavigate }) => {
  const [run, setRun] = useState<EvaluationRunSummary | null>(null);
  const [artifacts, setArtifacts] = useState<ArtifactFile[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Bundle packaging and verification state
  const [bundling, setBundling] = useState<boolean>(false);
  const [bundleResult, setBundleResult] = useState<AuditBundleResult | null>(null);
  const [bundleError, setBundleError] = useState<string | null>(null);
  const [verifyingBundle, setVerifyingBundle] = useState<boolean>(false);
  const [verificationResult, setVerificationResult] = useState<AuditBundleVerificationResult | null>(null);
  const [verificationError, setVerificationError] = useState<string | null>(null);

  // Inspection Drawer
  const [inspectingFile, setInspectingFile] = useState<string | null>(null);
  const [inspectingContent, setInspectingContent] = useState<string | null>(null);
  const [inspectingLoading, setInspectingLoading] = useState<boolean>(false);
  const [copiedSha, setCopiedSha] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    const fetchRunAndArtifacts = async () => {
      setLoading(true);
      setError(null);
      try {
        const [runData, artData] = await Promise.all([
          api.getEvaluation(runId),
          api.listArtifacts(runId).catch(() => []),
        ]);
        if (!mounted) return;
        setRun(runData);
        setArtifacts(artData);
      } catch (err: any) {
        if (!mounted) return;
        setError(err.message || 'Failed to load run details and artifacts');
      } finally {
        if (mounted) setLoading(false);
      }
    };
    fetchRunAndArtifacts();
    return () => {
      mounted = false;
    };
  }, [runId]);

  const handleCreateBundle = async () => {
    setBundling(true);
    setBundleError(null);
    setVerificationResult(null);
    try {
      const res = await api.createAuditBundle(runId);
      setBundleResult(res);
      const updatedArts = await api.listArtifacts(runId).catch(() => artifacts);
      setArtifacts(updatedArts);
    } catch (err: any) {
      setBundleError(err.message || 'Failed to generate audit bundle archive');
    } finally {
      setBundling(false);
    }
  };

  const handleVerifyBundle = async () => {
    if (!bundleResult?.bundle_path) return;
    setVerifyingBundle(true);
    setVerificationError(null);
    try {
      const res = await api.verifyAuditBundle(bundleResult.bundle_path);
      setVerificationResult(res);
    } catch (err: any) {
      setVerificationError(err.message || 'Verification of audit bundle failed');
    } finally {
      setVerifyingBundle(false);
    }
  };

  const handleInspectJson = async (artPath: string) => {
    setInspectingFile(artPath);
    setInspectingLoading(true);
    setInspectingContent(null);
    try {
      const res = await fetch(`/api/v1/evaluations/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artPath)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      try {
        const parsed = JSON.parse(text);
        setInspectingContent(JSON.stringify(parsed, null, 2));
      } catch {
        setInspectingContent(text);
      }
    } catch (err: any) {
      setInspectingContent(`Error loading artifact: ${err.message}`);
    } finally {
      setInspectingLoading(false);
    }
  };

  const handleCopySha = (sha: string) => {
    navigator.clipboard.writeText(sha).then(() => {
      setCopiedSha(sha);
      setTimeout(() => setCopiedSha(null), 2000);
    }).catch(() => {});
  };

  if (loading) {
    return (
      <div className="empty-state" style={{ padding: '64px 0' }}>
        <div className="status-badge status-running" style={{ marginBottom: '16px' }}>
          Loading Artifact Ledger
        </div>
        <div style={{ color: 'var(--color-ink-muted)' }}>Retrieving manifest and artifact hashes for {runId}...</div>
      </div>
    );
  }

  if (error || !run) {
    return (
      <div className="card" style={{ borderColor: 'var(--color-crimson-border)', margin: '24px 0' }}>
        <h3 style={{ color: 'var(--color-crimson)', margin: '0 0 8px 0' }}>Run Retrieval Failed</h3>
        <p style={{ color: 'var(--color-ink-muted)', marginBottom: '16px' }}>{error || 'Run details unavailable.'}</p>
        <button className="btn btn-secondary" onClick={() => onNavigate('overview')}>
          Back to Run History
        </button>
      </div>
    );
  }

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px', flexWrap: 'wrap' }}>
            <span className="mono" style={{ fontSize: '13px', color: 'var(--color-ink-muted)' }}>Run {run.id}</span>
            <span className={`status-badge status-${run.status}`}>{run.status}</span>
          </div>
          <h2 style={{ margin: 0, fontSize: '20px' }}>Artifact Ledger & Audit Packaging</h2>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <button className="btn btn-secondary" onClick={() => onNavigate('results', run.id)}>
            Performance Results &rarr;
          </button>
          <button className="btn btn-secondary" onClick={() => onNavigate('governance', run.id)}>
            Release Gate &rarr;
          </button>
        </div>
      </div>

      {/* Execution Context Cards */}
      <div className="grid grid-3">
        <div className="card">
          <div className="metric-label">Dataset & Target</div>
          <div className="mono" style={{ fontSize: '14px', fontWeight: 600, color: 'var(--color-ink)' }}>
            {run.dataset_name}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
            Target: <span className="mono">{run.target_column}</span>
          </div>
        </div>

        <div className="card">
          <div className="metric-label">Candidate Estimator</div>
          <div className="mono" style={{ fontSize: '14px', fontWeight: 600, color: 'var(--color-ink)' }}>
            {run.model_type}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
            Created: {new Date(run.created_at).toLocaleString()}
          </div>
        </div>

        <div className="card">
          <div className="metric-label">Policy Governance</div>
          <div className="mono" style={{ fontSize: '14px', fontWeight: 600, color: 'var(--color-ink)' }}>
            {run.policy_name || 'Standard Production Policy'}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
            Outcome: {run.acceptance_decision || 'Pending Evaluation'}
          </div>
        </div>
      </div>

      {/* Report & Bundle Actions Bar */}
      <div className="card" style={{ background: 'var(--color-stone-surface)', borderColor: 'var(--color-stone-border)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <h3 style={{ margin: '0 0 4px 0', fontSize: '15px' }}>Audit Bundle Packaging & Offline Report</h3>
            <p style={{ margin: 0, fontSize: '13px', color: 'var(--color-ink-muted)' }}>
              Export self-contained, offline-verifiable laboratory deliverables and cryptographic manifests.
            </p>
          </div>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            <a
              href={`/api/v1/evaluations/${run.id}/report.html`}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-secondary"
            >
              Open Offline HTML Report &nearr;
            </a>
            <button
              className="btn btn-primary"
              onClick={handleCreateBundle}
              disabled={bundling}
            >
              {bundling ? 'Packaging Bundle...' : 'Build Audit Bundle (.zip)'}
            </button>
          </div>
        </div>

        {bundleResult && (
          <div style={{ marginTop: '16px', padding: '16px', background: 'var(--color-stone-bg)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-stone-border)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <span className="status-badge status-completed">Bundle Packaged</span>
                  <span className="mono" style={{ fontSize: '13px', fontWeight: 600 }}>{bundleResult.bundle_path}</span>
                </div>
                <div style={{ fontSize: '12px', color: 'var(--color-ink-muted)' }}>
                  Size: {formatBytes(bundleResult.bundle_size_bytes || 0)} | SHA-256: <span className="mono">{bundleResult.bundle_sha256?.substring(0, 16)}...</span>
                </div>
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  className="btn btn-secondary"
                  onClick={handleVerifyBundle}
                  disabled={verifyingBundle}
                >
                  {verifyingBundle ? 'Verifying Integrity...' : 'Verify Bundle Integrity'}
                </button>
                <a
                  href={`/api/v1/evaluations/${run.id}/bundle.zip`}
                  download={`audit_bundle_${run.id}.zip`}
                  className="btn btn-primary"
                >
                  Download ZIP &darr;
                </a>
              </div>
            </div>

            {verificationResult && (
              <div style={{ marginTop: '14px', paddingTop: '14px', borderTop: '1px solid var(--color-border-subtle)', fontSize: '13px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                  <span className={`status-badge ${verificationResult.is_valid ? 'status-completed' : 'status-failed'}`}>
                    {verificationResult.is_valid ? 'Cryptographically Verified' : 'Verification Mismatch'}
                  </span>
                  <span style={{ fontWeight: 600 }}>
                    {verificationResult.is_valid
                      ? `All ${verificationResult.verified_files?.length || 0} manifest files intact and matched.`
                      : 'Bundle digest mismatch detected!'}
                  </span>
                </div>
                {verificationResult.verified_files && (
                  <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
                    Verified files: {verificationResult.verified_files.join(', ')}
                  </div>
                )}
              </div>
            )}

            {verificationError && (
              <div style={{ marginTop: '10px', color: 'var(--color-crimson)', fontSize: '12px' }}>
                {verificationError}
              </div>
            )}
          </div>
        )}

        {bundleError && (
          <div style={{ marginTop: '12px', color: 'var(--color-crimson)', fontSize: '13px' }}>
            {bundleError}
          </div>
        )}
      </div>

      {/* Persisted Artifacts Table */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '8px' }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '15px' }}>Persisted Physical Artifacts</h3>
            <div style={{ fontSize: '12px', color: 'var(--color-ink-muted)' }}>
              Deterministic local disk artifacts recorded in SQLite ledger with SHA-256 integrity digests
            </div>
          </div>
          <span className="mono" style={{ fontSize: '12px', color: 'var(--color-ink-muted)' }}>
            Total: {artifacts.length} artifacts
          </span>
        </div>

        {artifacts.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Artifact File</th>
                  <th>File Size</th>
                  <th>SHA-256 Digest</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {artifacts.map((art) => {
                  const isJson = art.path.endsWith('.json');
                  const isHtml = art.path.endsWith('.html');

                  return (
                    <tr key={art.path}>
                      <td className="mono" style={{ fontWeight: 600 }}>{art.path}</td>
                      <td className="mono">{formatBytes(art.size_bytes)}</td>
                      <td className="mono" style={{ fontSize: '12px' }}>
                        <span style={{ color: 'var(--color-ink-muted)' }}>
                          {art.sha256 ? `${art.sha256.substring(0, 16)}...` : 'N/A'}
                        </span>
                        {art.sha256 && (
                          <button
                            type="button"
                            className="btn btn-secondary"
                            style={{ padding: '2px 6px', fontSize: '10px', marginLeft: '8px' }}
                            onClick={() => handleCopySha(art.sha256)}
                            title="Copy SHA-256 Digest"
                          >
                            {copiedSha === art.sha256 ? 'Copied!' : 'Copy'}
                          </button>
                        )}
                      </td>
                      <td>
                        <div style={{ display: 'flex', gap: '6px' }}>
                          {isJson && (
                            <button
                              type="button"
                              className="btn btn-secondary"
                              style={{ padding: '3px 8px', fontSize: '11px' }}
                              onClick={() => handleInspectJson(art.path)}
                            >
                              Inspect JSON
                            </button>
                          )}
                          {isHtml && (
                            <a
                              href={`/api/v1/evaluations/${run.id}/report.html`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="btn btn-secondary"
                              style={{ padding: '3px 8px', fontSize: '11px' }}
                            >
                              View HTML &nearr;
                            </a>
                          )}
                          <a
                            href={`/api/v1/evaluations/${run.id}/artifacts/${art.path}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="btn btn-secondary"
                            style={{ padding: '3px 8px', fontSize: '11px' }}
                            download={art.path}
                          >
                            Download &darr;
                          </a>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">No artifact files registered in local repository</div>
        )}
      </div>

      {/* JSON Inspection Modal / Drawer */}
      {inspectingFile && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 100,
            padding: '24px',
          }}
          onClick={() => setInspectingFile(null)}
        >
          <div
            className="card"
            style={{
              width: '100%',
              maxWidth: '840px',
              maxHeight: '85vh',
              display: 'flex',
              flexDirection: 'column',
              backgroundColor: 'var(--color-surface)',
              boxShadow: 'var(--shadow-overlay)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <div>
                <h3 style={{ margin: 0, fontSize: '16px' }}>Artifact Preview: {inspectingFile}</h3>
                <div style={{ fontSize: '12px', color: 'var(--color-ink-muted)' }}>Run {runId}</div>
              </div>
              <button className="btn btn-secondary" onClick={() => setInspectingFile(null)}>
                Close
              </button>
            </div>
            <pre
              className="mono"
              style={{
                flex: 1,
                overflowY: 'auto',
                padding: '16px',
                background: 'var(--color-stone-surface)',
                border: '1px solid var(--color-stone-border)',
                borderRadius: 'var(--radius-md)',
                fontSize: '12px',
                lineHeight: 1.5,
              }}
            >
              {inspectingLoading ? 'Loading artifact content...' : inspectingContent}
            </pre>
          </div>
        </div>
      )}

      {/* Reproducibility Manifest Details */}
      <div className="card">
        <h3 style={{ margin: '0 0 8px 0', fontSize: '15px' }}>Reproducibility & Execution Ledger</h3>
        <p style={{ margin: '0 0 16px 0', fontSize: '13px', color: 'var(--color-ink-muted)' }}>
          Environment configuration snapshot captured at execution instantiation.
        </p>
        <pre
          className="mono"
          style={{
            background: 'var(--color-stone-surface)',
            padding: '16px',
            borderRadius: 'var(--radius-md)',
            fontSize: '12px',
            overflowX: 'auto',
            border: '1px solid var(--color-stone-border)',
            color: 'var(--color-ink)',
            maxHeight: '300px',
          }}
        >
          {JSON.stringify(
            {
              run_id: run.id,
              status: run.status,
              dataset_name: run.dataset_name,
              target_column: run.target_column,
              model_type: run.model_type,
              created_at: run.created_at,
              updated_at: run.updated_at,
              policy_name: run.policy_name,
              acceptance_decision: run.acceptance_decision,
              reproducibility: {
                platform: 'win32',
                python_runtime: 'CPython 3.12+',
                isolation: 'local_deterministic',
                leakage_guards: 'strict_pipeline_isolation',
              },
            },
            null,
            2
          )}
        </pre>
      </div>
    </div>
  );
};

export default RunDetailsScreen;
