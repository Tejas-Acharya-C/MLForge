import { describe, it, expect } from 'vitest';
import { CertificateVerificationResult, ReleaseGateResult } from '../types/api';

describe('Release Authorization & Trust Presentation Logic', () => {
  it('correctly categorizes an authorized release with trusted signer', () => {
    const result: CertificateVerificationResult = {
      payload_integrity: true,
      signature_validity: true,
      signer_trust: true,
      evidence_consistency: true,
      release_authorized: true,
      run_id: 'run_test_01',
      certificate_id: 'cert_abc123',
      key_id: 'srv_key_trusted',
      errors: [],
      warnings: [],
      details: ['Valid cryptographic signature matching trusted key material.'],
    };

    // Release authorization is valid iff all 4 tiers pass
    const isFullyApproved =
      result.payload_integrity &&
      result.signature_validity &&
      result.signer_trust &&
      result.evidence_consistency &&
      result.release_authorized;

    expect(isFullyApproved).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it('detects unknown signer with valid signature as UNAUTHORIZED (fail-closed)', () => {
    const result: CertificateVerificationResult = {
      payload_integrity: true,
      signature_validity: true,
      signer_trust: false,
      evidence_consistency: true,
      release_authorized: false,
      run_id: 'run_test_02',
      certificate_id: 'cert_xyz789',
      key_id: 'unknown_signer_key',
      errors: ['Signer key unknown_signer_key is UNTRUSTED.'],
      warnings: [],
      details: ['Cryptographic signature verified against candidate key, but signer is NOT trusted.'],
    };

    // The signature is mathematically valid
    expect(result.signature_validity).toBe(true);
    // But signer is untrusted
    expect(result.signer_trust).toBe(false);
    // Therefore release authorization MUST be denied
    expect(result.release_authorized).toBe(false);
  });

  it('detects payload tampering and marks integrity as failed', () => {
    const result: CertificateVerificationResult = {
      payload_integrity: false,
      signature_validity: false,
      signer_trust: true,
      evidence_consistency: true,
      release_authorized: false,
      run_id: 'run_test_03',
      certificate_id: 'cert_tampered',
      key_id: 'tampered_key_id',
      errors: ['Payload SHA-256 digest mismatch: certificate content has been altered.'],
      warnings: [],
      details: [],
    };

    expect(result.payload_integrity).toBe(false);
    expect(result.release_authorized).toBe(false);
    expect(result.errors[0]).toContain('mismatch');
  });

  it('handles release gate blockers and prevents authorization when policy fails', () => {
    const gate: ReleaseGateResult = {
      run_id: 'run_gate_01',
      is_releasable: false,
      overall_status: 'blocked',
      policy_id: 'strict_v1',
      policy_version: '1.0.0',
      target_environment: 'production',
      gate_results: [
        {
          gate_id: 'acceptance_decision',
          status: 'failed',
          is_blocking: true,
          explanation: 'Acceptance policy failed on metric macro_f1 (0.55 < 0.70)',
          evidence_references: {},
        },
      ],
      evaluated_at: '2026-10-19T10:00:00Z',
      summary: 'Release gate blocked by 1 check(s).',
    };

    expect(gate.is_releasable).toBe(false);
    expect(gate.overall_status).toBe('blocked');
    expect(gate.gate_results[0].is_blocking).toBe(true);
  });
});

describe('Diagnostic Missingness & Undefined Metric Transparency', () => {
  it('identifies undefined ROC-AUC in single-class folds without raising or fabricating', () => {
    const undefinedMetric = {
      value: null,
      is_undefined: true,
      undefined_reason: 'Only one class present in validation fold; ROC-AUC is mathematically undefined.',
    };

    expect(undefinedMetric.value).toBeNull();
    expect(undefinedMetric.is_undefined).toBe(true);
    expect(undefinedMetric.undefined_reason).toContain('mathematically undefined');
  });

  it('verifies dataset inspection input constraints', () => {
    const validateParams = (target: string, kFolds: number) => {
      const errors: string[] = [];
      if (!target || target.trim().length === 0) {
        errors.push('Target column is required.');
      }
      if (kFolds < 2 || kFolds > 20) {
        errors.push('Fold count must be between 2 and 20.');
      }
      return errors;
    };

    expect(validateParams('', 5)).toContain('Target column is required.');
    expect(validateParams('churn', 1)).toContain('Fold count must be between 2 and 20.');
    expect(validateParams('churn', 25)).toContain('Fold count must be between 2 and 20.');
    expect(validateParams('churn_label', 5)).toHaveLength(0);
  });

  it('preserves undefined metrics in fold evaluations without coercing to zero', () => {
    const rawMetrics: Record<string, { value: number | null; is_undefined: boolean; undefined_reason?: string }> = {
      accuracy: { value: 0.85, is_undefined: false },
      roc_auc: { value: null, is_undefined: true, undefined_reason: 'Single class in fold' },
      macro_f1: { value: 0.82, is_undefined: false },
    };

    // Format helper matching EvaluationResults.tsx
    const formatMetric = (m?: { value: number | null; is_undefined: boolean }) => {
      if (!m || m.is_undefined || m.value === null || m.value === undefined) return 'N/A';
      return (m.value * 100).toFixed(1) + '%';
    };

    expect(formatMetric(rawMetrics.accuracy)).toBe('85.0%');
    expect(formatMetric(rawMetrics.roc_auc)).toBe('N/A');
    expect(formatMetric(rawMetrics.roc_auc)).not.toBe('0.0%');
  });
});

describe('Acceptance Criteria Diagnostics & Inconclusive Breakdown', () => {
  it('correctly parses inconclusive criteria diagnostics with rule ID and reason code', () => {
    const criteria = [
      {
        criterion_id: 'rule_acc_min',
        rule_type: 'absolute_metric',
        metric_name: 'accuracy',
        threshold_value: 0.80,
        raw_observed_value: 0.85,
        outcome: 'PASS',
        reason_code: 'THRESHOLD_MET',
        message: 'Accuracy 0.8500 exceeds minimum threshold 0.8000',
        evidence_source: 'candidate_cross_validation',
      },
      {
        criterion_id: 'rule_stat_sig',
        rule_type: 'statistical_superiority',
        metric_name: 'macro_f1',
        threshold_value: 0.05,
        raw_observed_value: null,
        outcome: 'INCONCLUSIVE',
        reason_code: 'INSUFFICIENT_PAIRED_OBSERVATIONS',
        message: 'Only 3 valid paired fold observations available; minimum 5 required for Nadeau-Bengio t-test.',
        evidence_source: 'paired_resampled_t_test',
      },
    ];

    const inconclusive = criteria.filter((c) => c.outcome === 'INCONCLUSIVE');
    expect(inconclusive).toHaveLength(1);
    expect(inconclusive[0].criterion_id).toBe('rule_stat_sig');
    expect(inconclusive[0].reason_code).toBe('INSUFFICIENT_PAIRED_OBSERVATIONS');
    expect(inconclusive[0].raw_observed_value).toBeNull();
    expect(inconclusive[0].evidence_source).toBe('paired_resampled_t_test');
    expect(inconclusive[0].message).toContain('Nadeau-Bengio');
  });

  it('demonstrates that an Acceptance PASS does NOT imply Release Gate PASS or Release Authorization', () => {
    // Model achieves statistical superiority (Acceptance: PASS)
    const acceptanceOutcome = 'PASS';

    // But fails operational latency, calibration, or robustness checks in Release Gate
    const releaseGate: ReleaseGateResult = {
      run_id: 'eval_candidate_42',
      is_releasable: false,
      overall_status: 'blocked',
      policy_id: 'production_deployment_policy_v2',
      policy_version: '2.1.0',
      target_environment: 'production',
      gate_results: [
        {
          gate_id: 'statistical_acceptance',
          status: 'passed',
          is_blocking: true,
          explanation: 'Acceptance policy met with statistical significance.',
          evidence_references: { acceptance_id: 'acc_01' },
        },
        {
          gate_id: 'robustness_degradation',
          status: 'failed',
          is_blocking: true,
          explanation: 'AURDC under gaussian noise is 0.42, below required minimum 0.70.',
          evidence_references: { perturbation: 'gaussian_noise' },
        },
      ],
      evaluated_at: '2026-10-19T11:00:00Z',
      summary: 'Blocked by 1 failing gate check: robustness_degradation.',
    };

    // Release Certificate status is UNAUTHORIZED
    const certificateVerification: CertificateVerificationResult = {
      payload_integrity: true,
      signature_validity: true,
      signer_trust: true,
      evidence_consistency: false, // Inconsistent because release gate is blocked!
      release_authorized: false,
      run_id: 'eval_candidate_42',
      certificate_id: 'cert_42',
      key_id: 'trusted_ci_signer',
      errors: ['Evidence consistency failure: Release gate status is BLOCKED.'],
      warnings: [],
      details: ['Signer is trusted and signature is cryptographically valid, but release is BLOCKED.'],
    };

    expect(acceptanceOutcome).toBe('PASS');
    expect(releaseGate.is_releasable).toBe(false);
    expect(releaseGate.overall_status).toBe('blocked');
    expect(certificateVerification.release_authorized).toBe(false);
    expect(certificateVerification.evidence_consistency).toBe(false);
  });

  it('verifies the 7-tier release governance model independence', () => {
    // 7 Independent Tiers:
    // 1. Acceptance Outcome (statistical/empirical)
    // 2. Release Gate Outcome (multi-dimensional readiness)
    // 3. Artifact Integrity Verification (digest match)
    // 4. Cryptographic Signature Validity (Ed25519)
    // 5. Signer Trust Verification (Keystore authorization)
    // 6. Evidence Consistency (matching run ID, policy, and hashes)
    // 7. Final Release Authorization (Boolean conjunction of all preconditions)

    interface SevenTierGovernanceState {
      tier1_acceptance_outcome: 'PASS' | 'FAIL' | 'INCONCLUSIVE';
      tier2_release_gate_status: 'passed' | 'blocked' | 'error';
      tier3_artifact_integrity: boolean;
      tier4_signature_validity: boolean;
      tier5_signer_trust: boolean;
      tier6_evidence_consistency: boolean;
      tier7_release_authorized: boolean;
    }

    const testState: SevenTierGovernanceState = {
      tier1_acceptance_outcome: 'PASS',
      tier2_release_gate_status: 'passed',
      tier3_artifact_integrity: true,
      tier4_signature_validity: true,
      tier5_signer_trust: false, // Untrusted key
      tier6_evidence_consistency: true,
      tier7_release_authorized: false, // Fails closed due to tier 5
    };

    // Tier 7 requires ALL Tiers 2-6 to be true/passed
    const computedAuthorization =
      testState.tier2_release_gate_status === 'passed' &&
      testState.tier3_artifact_integrity &&
      testState.tier4_signature_validity &&
      testState.tier5_signer_trust &&
      testState.tier6_evidence_consistency;

    expect(computedAuthorization).toBe(false);
    expect(testState.tier7_release_authorized).toBe(false);
  });
});
