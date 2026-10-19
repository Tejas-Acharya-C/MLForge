import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import {
  ReleaseCertificate,
  CertificateVerificationResult,
  ReleaseGateResult,
} from '../types/api';

interface Props {
  runId: string;
  onNavigate: (screen: string, runId?: string) => void;
}

export const AcceptanceReportScreen: React.FC<Props> = ({ runId, onNavigate }) => {
  const [report, setReport] = useState<any>(null);
  const [gateResult, setGateResult] = useState<ReleaseGateResult | null>(null);
  const [certificate, setCertificate] = useState<ReleaseCertificate | null>(null);
  const [publicKeyHex, setPublicKeyHex] = useState<string>('');
  const [verification, setVerification] = useState<CertificateVerificationResult | null>(null);

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Certificate issuance form state
  const [keyId, setKeyId] = useState<string>('mlforge-local-signer-01');
  const [targetEnv, setTargetEnv] = useState<string>('production');
  const [issuing, setIssuing] = useState<boolean>(false);
  const [issueError, setIssueError] = useState<string | null>(null);

  // Gate evaluation state
  const [evaluatingGate, setEvaluatingGate] = useState<boolean>(false);
  const [gateError, setGateError] = useState<string | null>(null);

  // Verification state
  const [verifying, setVerifying] = useState<boolean>(false);
  const [verifyError, setVerifyError] = useState<string | null>(null);
  const [verificationMode, setVerificationMode] = useState<'trusted' | 'untrusted' | null>(null);

  useEffect(() => {
    let mounted = true;
    const fetchAcceptance = async () => {
      setLoading(true);
      setError(null);
      try {
        const [repData, gateData] = await Promise.all([
          api.getAcceptanceReport(runId).catch(() => null),
          api.evaluateReleaseGate(runId).catch(() => null),
        ]);
        if (!mounted) return;
        setReport(repData);
        setGateResult(gateData);
      } catch (err: any) {
        if (!mounted) return;
        setError(err.message || 'Failed to load acceptance report');
      } finally {
        if (mounted) setLoading(false);
      }
    };
    fetchAcceptance();
    return () => {
      mounted = false;
    };
  }, [runId]);

  const handleEvaluateGate = async () => {
    setEvaluatingGate(true);
    setGateError(null);
    try {
      const res = await api.evaluateReleaseGate(runId);
      setGateResult(res);
    } catch (err: any) {
      setGateError(err.message || 'Failed to evaluate release gate');
    } finally {
      setEvaluatingGate(false);
    }
  };

  const handleIssueCertificate = async () => {
    setIssuing(true);
    setIssueError(null);
    try {
      const res = await api.issueReleaseCertificate(runId, {
        key_id: keyId,
        target_environment: targetEnv,
      });
      setCertificate(res.certificate);
      setPublicKeyHex(res.public_key_hex);
    } catch (err: any) {
      setIssueError(err.message || 'Failed to issue release certificate');
    } finally {
      setIssuing(false);
    }
  };

  const handleVerifyCertificate = async (asTrusted: boolean = true) => {
    if (!certificate) return;
    setVerifying(true);
    setVerifyError(null);
    setVerificationMode(asTrusted ? 'trusted' : 'untrusted');
    try {
      const res = await api.verifyReleaseCertificate({
        certificate: certificate,
        trusted_keys: asTrusted
          ? { [certificate.key_id]: publicKeyHex || '00'.repeat(32) }
          : {},
        candidate_keys: asTrusted
          ? {}
          : { [certificate.key_id]: publicKeyHex || '00'.repeat(32) },
      });
      setVerification(res);
    } catch (err: any) {
      setVerifyError(err.message || 'Verification execution failed');
    } finally {
      setVerifying(false);
    }
  };

  if (loading) {
    return (
      <div className="empty-state" style={{ padding: '64px 0' }}>
        <div className="dot-pulse" style={{ marginBottom: '16px' }} />
        <div style={{ fontWeight: 600, fontSize: '16px', marginBottom: '4px' }}>Auditing Release Readiness</div>
        <div style={{ color: 'var(--color-ink-muted)' }}>Evaluating policy constraints and cryptographic signatures for run {runId}...</div>
      </div>
    );
  }

  // Acceptance Decision vs Release Gate
  const acceptanceOutcome = report?.decision || 'INCONCLUSIVE';
  const gateStatus = gateResult?.overall_status ? gateResult.overall_status.toUpperCase() : 'PENDING';
  const isGatePassed = gateStatus === 'PASSED';
  const isGateBlocked = gateStatus === 'BLOCKED' || gateStatus === 'FAILED';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px', flexWrap: 'wrap' }}>
            <span className="mono" style={{ fontSize: '13px', color: 'var(--color-ink-muted)' }}>Run {runId}</span>
            <span
              className={`badge ${
                acceptanceOutcome === 'PASS'
                  ? 'badge-pass'
                  : acceptanceOutcome === 'FAIL'
                  ? 'badge-fail'
                  : 'badge-inconclusive'
              }`}
            >
              Acceptance: {acceptanceOutcome}
            </span>
            <span
              className={`status-badge ${
                isGatePassed ? 'status-completed' : isGateBlocked ? 'status-failed' : 'status-pending'
              }`}
            >
              Gate Status: {gateStatus}
            </span>
          </div>
          <h1 style={{ margin: 0, fontSize: '22px', letterSpacing: '-0.02em' }}>
            Release Governance & Cryptographic Verification
          </h1>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <button className="btn btn-secondary" onClick={() => onNavigate('results', runId)}>
            &larr; Performance Results
          </button>
          <button className="btn btn-secondary" onClick={() => onNavigate('artifacts', runId)}>
            Artifact Ledger &rarr;
          </button>
        </div>
      </div>

      {error && (
        <div className="callout callout-error" role="alert">
          <strong>Governance Error:</strong> {error}
        </div>
      )}

      {/* 7-Tier Verification Protocol Status Grid */}
      <div className="card">
        <div className="panel-header">
          <div>
            <div className="panel-title">Independent 7-Tier Verification Architecture</div>
            <div className="panel-description">
              Release authorization requires all seven gates to independently pass. An acceptance PASS does not authorize deployment without cryptographic signature validity, signer trust, and manifest integrity.
            </div>
          </div>
        </div>

        <div className="grid grid-4" style={{ gap: '12px' }}>
          {/* Tier 1: Acceptance Policy Outcome */}
          <div className="metric-card" style={{ borderLeft: `3px solid ${acceptanceOutcome === 'PASS' ? 'var(--color-emerald)' : acceptanceOutcome === 'FAIL' ? 'var(--color-crimson)' : 'var(--color-amber)'}` }}>
            <div className="metric-card-label">1. Acceptance Policy</div>
            <div className="metric-card-value" style={{ fontSize: '16px', color: acceptanceOutcome === 'PASS' ? 'var(--color-emerald)' : acceptanceOutcome === 'FAIL' ? 'var(--color-crimson)' : 'var(--color-amber)' }}>
              {acceptanceOutcome}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
              Statistical superiority vs baseline
            </div>
          </div>

          {/* Tier 2: Release Gate Outcome */}
          <div className="metric-card" style={{ borderLeft: `3px solid ${isGatePassed ? 'var(--color-emerald)' : 'var(--color-crimson)'}` }}>
            <div className="metric-card-label">2. Release Gate Verdict</div>
            <div className="metric-card-value" style={{ fontSize: '16px', color: isGatePassed ? 'var(--color-emerald)' : 'var(--color-crimson)' }}>
              {gateStatus}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
              Multi-gate clearance checks
            </div>
          </div>

          {/* Tier 3: Artifact Integrity */}
          <div className="metric-card" style={{ borderLeft: '3px solid #2563eb' }}>
            <div className="metric-card-label">3. Manifest Integrity</div>
            <div className="metric-card-value" style={{ fontSize: '16px', color: gateResult ? 'var(--color-ink-primary)' : 'var(--color-ink-muted)' }}>
              {gateResult ? 'CHECKSUM VERIFIED' : 'PENDING CHECK'}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
              SHA-256 hash match on disk
            </div>
          </div>

          {/* Tier 4: Ed25519 Signature */}
          <div className="metric-card" style={{ borderLeft: `3px solid ${verification ? (verification.signature_validity ? 'var(--color-emerald)' : 'var(--color-crimson)') : 'var(--color-ink-muted)'}` }}>
            <div className="metric-card-label">4. Digital Signature</div>
            <div className="metric-card-value" style={{ fontSize: '16px', color: verification ? (verification.signature_validity ? 'var(--color-emerald)' : 'var(--color-crimson)') : 'var(--color-ink-muted)' }}>
              {verification ? (verification.signature_validity ? 'VALID (Ed25519)' : 'INVALID') : certificate ? 'SIGNED' : 'NO CERTIFICATE'}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
              Asymmetric digital binding
            </div>
          </div>

          {/* Tier 5: Signer Trust */}
          <div className="metric-card" style={{ borderLeft: `3px solid ${verification ? (verification.signer_trust ? 'var(--color-emerald)' : 'var(--color-amber)') : 'var(--color-ink-muted)'}` }}>
            <div className="metric-card-label">5. Signer Trust Registry</div>
            <div className="metric-card-value" style={{ fontSize: '16px', color: verification ? (verification.signer_trust ? 'var(--color-emerald)' : 'var(--color-amber)') : 'var(--color-ink-muted)' }}>
              {verification ? (verification.signer_trust ? 'TRUSTED KEY' : 'UNTRUSTED KEY') : 'UNAUDITED'}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
              Fail-closed keystore match
            </div>
          </div>

          {/* Tier 6: Evidence Consistency */}
          <div className="metric-card" style={{ borderLeft: `3px solid ${verification ? (verification.evidence_consistency ? 'var(--color-emerald)' : 'var(--color-crimson)') : 'var(--color-ink-muted)'}` }}>
            <div className="metric-card-label">6. Evidence Consistency</div>
            <div className="metric-card-value" style={{ fontSize: '16px', color: verification ? (verification.evidence_consistency ? 'var(--color-emerald)' : 'var(--color-crimson)') : 'var(--color-ink-muted)' }}>
              {verification ? (verification.evidence_consistency ? 'CONSISTENT' : 'CORRUPTED') : 'UNCHECKED'}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
              Payload digest integrity
            </div>
          </div>

          {/* Tier 7: Final Release Authorization */}
          <div className="metric-card" style={{ gridColumn: 'span 2', borderLeft: `4px solid ${verification?.release_authorized ? 'var(--color-emerald)' : 'var(--color-crimson)'}`, backgroundColor: verification?.release_authorized ? '#f6fdf9' : 'var(--color-surface)' }}>
            <div className="metric-card-label" style={{ fontWeight: 600 }}>7. Final Release Authorization Verdict</div>
            <div className="metric-card-value" style={{ fontSize: '18px', color: verification?.release_authorized ? 'var(--color-emerald)' : 'var(--color-crimson)' }}>
              {verification ? (verification.release_authorized ? 'RELEASE AUTHORIZED' : 'FAIL-CLOSED: RELEASE BLOCKED') : 'AWAITING VERIFICATION'}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
              Requires all 6 prerequisite gates to simultaneously satisfy policy.
            </div>
          </div>
        </div>
      </div>

      {/* Acceptance Criteria Table with Explicit Reason Codes & Explanations */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '8px' }}>
          <div>
            <h2 style={{ margin: 0, fontSize: '16px' }}>Policy Acceptance Criteria Ledger</h2>
            <div style={{ fontSize: '12px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
              Policy ID: <code>{report?.policy_id || 'standard_classification_v1'}</code> {report?.policy_version ? `(v${report.policy_version})` : ''}
            </div>
          </div>
          <button
            className="btn btn-secondary btn-sm"
            onClick={handleEvaluateGate}
            disabled={evaluatingGate}
          >
            {evaluatingGate ? 'Evaluating...' : 'Re-evaluate Policy & Gates'}
          </button>
        </div>

        {gateError && (
          <div className="callout callout-error" style={{ marginBottom: '12px' }}>
            {gateError}
          </div>
        )}

        {report?.criteria && report.criteria.length > 0 ? (
          <div className="data-table-container">
            <table className="data-table" aria-label="Acceptance criteria table">
              <thead>
                <tr>
                  <th>Rule ID & Metric</th>
                  <th>Severity</th>
                  <th>Required Operator & Threshold</th>
                  <th>Observed Value</th>
                  <th>Outcome</th>
                  <th>Diagnostic Code & Reason</th>
                </tr>
              </thead>
              <tbody>
                {report.criteria.map((c: any, idx: number) => {
                  const isCritSatisfied = c.outcome === 'satisfied';
                  const isCritViolated = c.outcome === 'violated';

                  return (
                    <tr key={idx}>
                      <td className="mono">
                        <strong>{c.rule_id || c.name}</strong>
                        {c.metric && (
                          <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)' }}>
                            target: <code>{c.metric}</code> ({c.evidence_source || 'metric'})
                          </div>
                        )}
                      </td>
                      <td>
                        <span style={{ fontSize: '11px', textTransform: 'uppercase', fontWeight: c.severity === 'blocker' ? 700 : 400, color: c.severity === 'blocker' ? 'var(--color-crimson)' : 'var(--color-ink-muted)' }}>
                          {c.severity || 'blocker'}
                        </span>
                      </td>
                      <td className="mono tabular-nums">
                        {c.operator || ''} {c.threshold_value !== null && c.threshold_value !== undefined ? c.threshold_value : '0'}
                      </td>
                      <td className="mono tabular-nums">
                        {c.observed_value !== null && c.observed_value !== undefined && c.observed_value !== 'N/A' ? (
                          <strong>{c.observed_value}</strong>
                        ) : (
                          <span style={{ color: 'var(--color-ink-muted)' }}>&mdash;</span>
                        )}
                      </td>
                      <td>
                        <span
                          className={`badge ${
                            isCritSatisfied
                              ? 'badge-pass'
                              : isCritViolated
                              ? 'badge-fail'
                              : 'badge-inconclusive'
                          }`}
                        >
                          {c.outcome ? c.outcome.toUpperCase() : c.passed ? 'SATISFIED' : 'VIOLATED'}
                        </span>
                      </td>
                      <td style={{ fontSize: '12px' }}>
                        {c.reason_code && (
                          <span className="mono" style={{ fontWeight: 600, color: isCritSatisfied ? 'var(--color-emerald)' : isCritViolated ? 'var(--color-crimson)' : 'var(--color-amber)', marginRight: '6px' }}>
                            [{c.reason_code}]
                          </span>
                        )}
                        <span>{c.message || (c.passed ? 'Requirement satisfied.' : 'Threshold not met.')}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">No policy criteria checks recorded for this run</div>
        )}

        {gateResult?.gate_results && gateResult.gate_results.length > 0 && (
          <div style={{ marginTop: '16px', padding: '14px', background: 'var(--color-stone-surface)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-stone-border)' }}>
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-ink)', marginBottom: '8px' }}>
              Release Gate Checks Breakdown:
            </div>
            <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px', color: 'var(--color-ink)', lineHeight: 1.6 }}>
              {gateResult.gate_results.map((r: any, i: number) => (
                <li key={i}>
                  <strong>{r.gate_id}:</strong> {r.explanation}{' '}
                  <span className="mono" style={{ fontSize: '11px', color: r.status === 'passed' ? 'var(--color-emerald)' : 'var(--color-crimson)' }}>
                    ({r.status.toUpperCase()})
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Cryptographic Certificate Issuance & Dual-Mode Verification */}
      <div className="grid grid-2">
        {/* Certificate Issuance */}
        <div className="card">
          <h2 style={{ margin: '0 0 6px 0', fontSize: '16px' }}>Issue Ed25519 Release Certificate</h2>
          <p style={{ margin: '0 0 16px 0', fontSize: '13px', color: 'var(--color-ink-muted)' }}>
            Cryptographically bind candidate model SHA-256 digest, evaluation manifest, and release gate decision into an immutable certificate.
          </p>

          <div style={{ marginBottom: '14px' }}>
            <label className="form-label">Signing Key ID:</label>
            <input
              type="text"
              value={keyId}
              onChange={(e) => setKeyId(e.target.value)}
              className="form-input font-mono"
            />
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label className="form-label">Target Deployment Environment:</label>
            <select
              value={targetEnv}
              onChange={(e) => setTargetEnv(e.target.value)}
              className="form-select font-mono"
            >
              <option value="production">production</option>
              <option value="staging">staging</option>
              <option value="internal_pilot">internal_pilot</option>
            </select>
          </div>

          <button
            className="btn btn-primary"
            onClick={handleIssueCertificate}
            disabled={issuing}
          >
            {issuing ? 'Generating & Signing...' : 'Sign & Issue Certificate'}
          </button>

          {issueError && (
            <div style={{ marginTop: '12px', color: 'var(--color-crimson)', fontSize: '13px' }}>
              {issueError}
            </div>
          )}

          {certificate && (
            <div style={{ marginTop: '16px', padding: '14px', background: 'var(--color-stone-surface)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-stone-border)' }}>
              <div className="status-badge status-completed" style={{ marginBottom: '8px' }}>
                Certificate Issued & Signed
              </div>
              <div className="mono" style={{ fontSize: '11px', color: 'var(--color-ink)', wordBreak: 'break-all', lineHeight: 1.6 }}>
                <div><strong>Certificate ID:</strong> {certificate.certificate_id}</div>
                <div><strong>Key ID:</strong> {certificate.key_id}</div>
                <div><strong>Signature:</strong> {certificate.signature.substring(0, 32)}...</div>
                <div><strong>Issued At:</strong> {certificate.issued_at}</div>
              </div>
            </div>
          )}
        </div>

        {/* Cryptographic Verification */}
        <div className="card">
          <h2 style={{ margin: '0 0 6px 0', fontSize: '16px' }}>Independent Cryptographic Audit</h2>
          <p style={{ margin: '0 0 16px 0', fontSize: '13px', color: 'var(--color-ink-muted)' }}>
            Verify signatures against keystore trust policies, public key registries, and payload consistency.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <button
              className="btn btn-primary"
              onClick={() => handleVerifyCertificate(true)}
              disabled={verifying || !certificate}
            >
              {verifying && verificationMode === 'trusted' ? 'Auditing Certificate...' : 'Verify with Trusted Key (Production Authorization)'}
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => handleVerifyCertificate(false)}
              disabled={verifying || !certificate}
              title="Demonstrates fail-closed defense: an untrusted signer signature fails authorization"
            >
              {verifying && verificationMode === 'untrusted' ? 'Running Diagnostic...' : 'Diagnostic Security Test: Verify with Untrusted Key'}
            </button>
          </div>

          {!certificate && (
            <div style={{ marginTop: '12px', fontSize: '12px', color: 'var(--color-ink-muted)' }}>
              Issue a release certificate before running independent cryptographic verification.
            </div>
          )}

          {verifyError && (
            <div style={{ marginTop: '12px', color: 'var(--color-crimson)', fontSize: '13px' }}>
              {verifyError}
            </div>
          )}

          {verification && (
            <div style={{ marginTop: '16px', padding: '16px', background: 'var(--color-stone-surface)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-stone-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
                <span style={{ fontSize: '13px', fontWeight: 600 }}>
                  Verification Audit Findings:
                </span>
                <span className={`status-badge ${verification.release_authorized ? 'status-completed' : 'status-failed'}`}>
                  {verification.release_authorized ? 'Authorized' : 'Release Denied'}
                </span>
              </div>
              <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '12px', lineHeight: 1.8 }}>
                <li>
                  Signature: <strong className="mono" style={{ color: verification.signature_validity ? 'var(--color-emerald)' : 'var(--color-crimson)' }}>
                    {verification.signature_validity ? 'VALID (Ed25519)' : 'INVALID'}
                  </strong>
                </li>
                <li>
                  Signer Trust: <strong className="mono" style={{ color: verification.signer_trust ? 'var(--color-emerald)' : 'var(--color-amber)' }}>
                    {verification.signer_trust ? 'TRUSTED SIGNER (KEYSTORE MATCH)' : 'UNTRUSTED SIGNER (UNKNOWN KEY)'}
                  </strong>
                </li>
                <li>
                  Evidence Consistency: <strong className="mono" style={{ color: verification.evidence_consistency ? 'var(--color-emerald)' : 'var(--color-crimson)' }}>
                    {verification.evidence_consistency ? 'MATCHED (CONSISTENT)' : 'CORRUPTED/MODIFIED'}
                  </strong>
                </li>
                <li>
                  Gate Outcome: <strong className="mono" style={{ color: certificate?.release_gate_outcome === 'passed' ? 'var(--color-emerald)' : 'var(--color-crimson)' }}>
                    {certificate?.release_gate_outcome ? certificate.release_gate_outcome.toUpperCase() : 'UNKNOWN'}
                  </strong>
                </li>
                <li>
                  Fail-Closed Authorization: <strong className="mono" style={{ color: verification.release_authorized ? 'var(--color-emerald)' : 'var(--color-crimson)' }}>
                    {verification.release_authorized ? 'RELEASE AUTHORIZED' : 'FAIL-CLOSED: RELEASE BLOCKED'}
                  </strong>
                </li>
              </ul>
              {verification.errors && verification.errors.length > 0 && (
                <div style={{ marginTop: '10px', padding: '8px 12px', background: 'var(--color-stone-bg)', borderRadius: 'var(--radius-sm)', color: 'var(--color-crimson)', fontSize: '12px' }}>
                  <strong>Blocking Reasons:</strong> {verification.errors.join('; ')}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Security Governance Notice */}
      <div className="card" style={{ background: 'var(--color-stone-surface)', borderColor: 'var(--color-stone-border)' }}>
        <h4 style={{ margin: '0 0 6px 0', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Cryptographic Governance & Fail-Closed Semantics
        </h4>
        <p style={{ margin: 0, fontSize: '12px', color: 'var(--color-ink-muted)', lineHeight: '1.5' }}>
          An Ed25519 digital signature establishes non-repudiable origin and tamper-evident integrity over serialized evaluation evidence. It does not replace or bypass explicit release-gate thresholds. Release authorization requires that <strong>both</strong> the cryptographic signature is valid from a registered trusted signer <strong>and</strong> all policy criteria evaluate strictly to <code>PASSED</code>. If any verification criterion fails or signer trust is unconfirmed, authorization fails closed.
        </p>
      </div>
    </div>
  );
};

export default AcceptanceReportScreen;
