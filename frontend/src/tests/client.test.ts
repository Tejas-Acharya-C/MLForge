import { describe, it, expect, vi, beforeEach } from 'vitest';
import { api } from '../api/client';

describe('MLForge Typed Frontend API Client', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  describe('Dataset & Policy Endpoints', () => {
    it('queries sample datasets and parses response', async () => {
      const mockDatasets = [
        {
          id: 'churn',
          name: 'Customer Churn',
          description: 'Synthetic churn',
          num_rows: 500,
          num_features: 12,
          default_target: 'churn_label',
        },
      ];

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockDatasets,
      } as any);

      const datasets = await api.getSampleDatasets();
      expect(datasets).toHaveLength(1);
      expect(datasets[0].id).toBe('churn');
      expect(globalThis.fetch).toHaveBeenCalledWith('/api/v1/datasets/sample');
    });

    it('submits dataset inspection request with JSON payload', async () => {
      const mockMeta = {
        num_rows: 500,
        num_features: 6,
        feature_names: ['age', 'tenure'],
        feature_types: { age: 'int64', tenure: 'int64' },
        numeric_feature_count: 2,
        categorical_feature_count: 0,
        target_name: 'churn_label',
        target_type: 'int64',
        class_distribution: { '0': 350, '1': 150 },
        class_labels: ['0', '1'],
        imbalance_ratio: 2.33,
        missing_value_counts: {},
        constant_columns: [],
        checksum_sha256: 'a'.repeat(64),
        warnings: [],
      };

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockMeta,
      } as any);

      const meta = await api.inspectDataset({ sample_name: 'churn', target_column: 'churn_label', k_folds: 5 });
      expect(meta.num_rows).toBe(500);
      expect(meta.target_name).toBe('churn_label');
      expect(globalThis.fetch).toHaveBeenCalledWith(
        '/api/v1/datasets/inspect',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sample_name: 'churn', target_column: 'churn_label', k_folds: 5 }),
        })
      );
    });

    it('retrieves standard acceptance policies', async () => {
      const mockPolicies = [
        {
          policy_id: 'standard_classification_v1',
          version: '1.0.0',
          name: 'Standard Tabular Policy',
          description: 'Production criteria',
          rules_count: 3,
          policy: { policy_id: 'standard_classification_v1', version: '1.0.0', rules: [] },
        },
      ];

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockPolicies,
      } as any);

      const policies = await api.getAcceptancePolicies();
      expect(policies).toHaveLength(1);
      expect(policies[0].policy_id).toBe('standard_classification_v1');
    });
  });

  describe('Evaluation Lifecycle Endpoints', () => {
    it('submits a new evaluation run', async () => {
      const mockSummary = {
        run_id: 'eval_abc123',
        state: 'created',
        created_at: '2026-10-19T10:00:00Z',
        candidate_model_type: 'logistic_regression',
        baseline_model_type: 'dummy_prior',
      };

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockSummary,
      } as any);

      const res = await api.submitEvaluation({
        sample_name: 'churn',
        target_column: 'churn_label',
        candidate_model_type: 'logistic_regression',
        num_folds: 5,
        random_state: 42,
      });

      expect(res.run_id).toBe('eval_abc123');
      expect(res.state).toBe('created');
    });

    it('lists evaluations with filter parameters', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ items: [], total: 0, limit: 10, offset: 0 }),
      } as any);

      await api.listEvaluations({ state: 'completed', limit: 10 });
      expect(globalThis.fetch).toHaveBeenCalledWith('/api/v1/evaluations?state=completed&limit=10');
    });

    it('cancels an active evaluation run', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ run_id: 'eval_cancel_me', state: 'cancelled' }),
      } as any);

      const status = await api.cancelEvaluation('eval_cancel_me', 'User abort');
      expect(status.state).toBe('cancelled');
    });
  });

  describe('Robustness Endpoints', () => {
    it('retrieves previously evaluated robustness curves via GET', async () => {
      const mockCurves = {
        curves: [
          {
            perturbation_type: 'gaussian_noise',
            metric_name: 'accuracy',
            clean_score: 0.90,
            area_under_degradation_curve: 0.88,
            points: [],
            warnings: [],
          },
        ],
        summary_retention: { gaussian_noise: 0.85 },
        is_usable: true,
        warnings: [],
      };

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockCurves,
      } as any);

      const res = await api.getRobustnessResults('eval_rob_1');
      expect(res.curves).toHaveLength(1);
      expect(res.curves[0].area_under_degradation_curve).toBe(0.88);
      expect(globalThis.fetch).toHaveBeenCalledWith('/api/v1/evaluations/eval_rob_1/robustness');
    });

    it('runs new robustness perturbations via POST', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ curves: [], is_usable: true }),
      } as any);

      await api.runRobustnessStressTest('eval_rob_1', { profile: 'noise', severity_levels: [0.05, 0.10] });
      expect(globalThis.fetch).toHaveBeenCalledWith(
        '/api/v1/evaluations/eval_rob_1/robustness',
        expect.objectContaining({ method: 'POST' })
      );
    });
  });

  describe('Release Certificate & Trust Endpoints', () => {
    it('verifies release certificate with trusted keys', async () => {
      const mockResult = {
        is_valid: true,
        payload_integrity: true,
        signature_validity: true,
        signer_trust: true,
        evidence_consistency: true,
        release_authorized: true,
        run_id: 'run_123',
        key_id: 'trusted_key_01',
        errors: [],
        warnings: [],
        details: ['Verified against server trusted keystore.'],
      };

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockResult,
      } as any);

      const res = await api.verifyReleaseCertificate({
        certificate: { run_id: 'run_123' } as any,
        trusted_keys: { trusted_key_01: 'a'.repeat(64) },
      });

      expect(res.release_authorized).toBe(true);
      expect(res.signer_trust).toBe(true);
    });

    it('verifies unknown-signer certificate failing authorization', async () => {
      const mockResult = {
        is_valid: false,
        payload_integrity: true,
        signature_validity: true,
        signer_trust: false,
        evidence_consistency: true,
        release_authorized: false,
        run_id: 'run_123',
        key_id: 'unknown_key_01',
        errors: ['Signer key unknown_key_01 is UNTRUSTED.'],
        warnings: [],
        details: [],
      };

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockResult,
      } as any);

      const res = await api.verifyReleaseCertificate({
        certificate: { run_id: 'run_123' } as any,
        candidate_keys: { unknown_key_01: 'b'.repeat(64) },
      });

      expect(res.signature_validity).toBe(true);
      expect(res.signer_trust).toBe(false);
      expect(res.release_authorized).toBe(false);
    });

    it('evaluates release gate policy without issuance', async () => {
      const mockGate = {
        is_releasable: true,
        overall_status: 'passed',
        gate_checks: [],
        policy: {} as any,
        evaluated_at: '2026-10-19T10:00:00Z',
        summary: 'All gate checks satisfied.',
      };

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockGate,
      } as any);

      const gate = await api.evaluateReleaseGate('run_123', { require_acceptance_pass: true });
      expect(gate.overall_status).toBe('passed');
      expect(gate.is_releasable).toBe(true);
    });
  });

  describe('Error Handling Robustness', () => {
    it('handles HTTP error responses with string detail', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        statusText: 'Not Found',
        json: async () => ({ detail: 'Run not found' }),
      } as any);

      await expect(api.getEvaluationDetail('missing-run')).rejects.toThrow('Run not found');
    });

    it('handles HTTP error responses with array detail (FastAPI validation error)', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 422,
        statusText: 'Unprocessable Entity',
        json: async () => ({
          detail: [{ loc: ['body', 'num_folds'], msg: 'ensure this value is greater than or equal to 2' }],
        }),
      } as any);

      await expect(
        api.submitEvaluation({ target_column: 'target', num_folds: 1 })
      ).rejects.toThrow('ensure this value is greater than or equal to 2');
    });

    it('handles non-JSON error responses gracefully using statusText fallback', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 502,
        statusText: 'Bad Gateway',
        json: async () => {
          throw new Error('Not JSON');
        },
      } as any);

      await expect(api.getEvaluationDetail('any-run')).rejects.toThrow('HTTP 502: Bad Gateway');
    });

    it('formats offline report and audit bundle URLs correctly', () => {
      expect(api.getReportHtmlUrl('run-123')).toBe('/api/v1/evaluations/run-123/report.html');
      expect(api.getBundleZipUrl('run-123')).toBe('/api/v1/evaluations/run-123/bundle.zip');
    });
  });
});
