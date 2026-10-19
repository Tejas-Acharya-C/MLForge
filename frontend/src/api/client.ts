/**
 * MLForge Typed REST API Client.
 * Communicates with the FastAPI backend at /api/v1.
 */

import {
  AcceptanceDecision,
  AuditBundleVerificationResult,
  CertificateVerificationResult,
  ComparisonReport,
  DatasetMetadata,
  EvaluationDetailResponse,
  EvaluationListResponse,
  EvaluationStatusResponse,
  EvaluationSummaryItem,
  FoldResult,
  ModelEvaluationResult,
  PolicyTemplateInfo,
  ReleaseCertificate,
  ReleaseGateResult,
  ReleasePolicy,
  RobustnessEvaluationResult,
  SampleDatasetInfo,
} from '../types/api';

const BASE_URL = '/api/v1';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    try {
      const errorJson = await res.json();
      if (errorJson.detail) {
        errorDetail = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
      }
    } catch {
      // Keep default error text
    }
    throw new Error(errorDetail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  // Datasets & Policies
  async getSampleDatasets(): Promise<SampleDatasetInfo[]> {
    const res = await fetch(`${BASE_URL}/datasets/sample`);
    return handleResponse<SampleDatasetInfo[]>(res);
  },

  async inspectDataset(req: {
    file_path?: string;
    sample_name?: string;
    target_column: string;
    k_folds?: number;
  }): Promise<DatasetMetadata> {
    const res = await fetch(`${BASE_URL}/datasets/inspect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    return handleResponse<DatasetMetadata>(res);
  },

  async getAcceptancePolicies(): Promise<PolicyTemplateInfo[]> {
    const res = await fetch(`${BASE_URL}/acceptance-policies`);
    return handleResponse<PolicyTemplateInfo[]>(res);
  },

  // Evaluations
  async submitEvaluation(req: {
    file_path?: string;
    sample_name?: string;
    target_column: string;
    candidate_model_type?: string;
    candidate_hyperparameters?: Record<string, any>;
    num_folds?: number;
    random_state?: number;
    subgroup_columns?: string[];
    policy_id?: string;
    run_id?: string;
    run_metadata?: Record<string, any>;
  }): Promise<EvaluationSummaryItem> {
    const res = await fetch(`${BASE_URL}/evaluations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    return handleResponse<EvaluationSummaryItem>(res);
  },

  async listEvaluations(params?: {
    state?: string;
    limit?: number;
    offset?: number;
    order_by?: string;
    descending?: boolean;
  }): Promise<EvaluationListResponse> {
    const q = new URLSearchParams();
    if (params?.state && params.state !== 'all') q.append('state', params.state);
    if (params?.limit) q.append('limit', String(params.limit));
    if (params?.offset) q.append('offset', String(params.offset));
    if (params?.order_by) q.append('order_by', params.order_by);
    if (params?.descending !== undefined) q.append('descending', String(params.descending));

    const res = await fetch(`${BASE_URL}/evaluations?${q.toString()}`);
    return handleResponse<EvaluationListResponse>(res);
  },

  async getEvaluationDetail(runId: string): Promise<EvaluationDetailResponse> {
    const res = await fetch(`${BASE_URL}/evaluations/${encodeURIComponent(runId)}`);
    return handleResponse<EvaluationDetailResponse>(res);
  },

  async getEvaluationStatus(runId: string): Promise<EvaluationStatusResponse> {
    const res = await fetch(`${BASE_URL}/evaluations/${encodeURIComponent(runId)}/status`);
    return handleResponse<EvaluationStatusResponse>(res);
  },

  async cancelEvaluation(runId: string, reason?: string): Promise<EvaluationStatusResponse> {
    const res = await fetch(`${BASE_URL}/evaluations/${encodeURIComponent(runId)}/cancel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason: reason || 'Cancelled by user via UI' }),
    });
    return handleResponse<EvaluationStatusResponse>(res);
  },

  // Scientific Results
  async getEvaluationResults(runId: string): Promise<{
    candidate_result: ModelEvaluationResult;
    baseline_result: ModelEvaluationResult;
    comparison_report?: ComparisonReport;
    dataset_metadata: DatasetMetadata;
    reproducibility: any;
    warnings: string[];
  }> {
    const res = await fetch(`${BASE_URL}/evaluations/${encodeURIComponent(runId)}/results`);
    return handleResponse(res);
  },

  async getFoldResults(runId: string): Promise<FoldResult[]> {
    const res = await fetch(`${BASE_URL}/evaluations/${encodeURIComponent(runId)}/results/folds`);
    return handleResponse<FoldResult[]>(res);
  },

  async getCalibrationDiagnostics(runId: string): Promise<any> {
    const res = await fetch(`${BASE_URL}/evaluations/${encodeURIComponent(runId)}/diagnostics/calibration`);
    return handleResponse(res);
  },

  async getSubgroupDiagnostics(runId: string): Promise<{ subgroup_diagnostics: Record<string, any[]> }> {
    const res = await fetch(`${BASE_URL}/evaluations/${encodeURIComponent(runId)}/diagnostics/subgroups`);
    return handleResponse(res);
  },

  async getAcceptanceDecision(runId: string): Promise<AcceptanceDecision> {
    const res = await fetch(`${BASE_URL}/evaluations/${encodeURIComponent(runId)}/acceptance`);
    return handleResponse<AcceptanceDecision>(res);
  },

  getReportHtmlUrl(runId: string): string {
    return `${BASE_URL}/evaluations/${encodeURIComponent(runId)}/report.html`;
  },

  getBundleZipUrl(runId: string): string {
    return `${BASE_URL}/evaluations/${encodeURIComponent(runId)}/bundle.zip`;
  },

  // Robustness
  async runRobustnessStressTest(
    runId: string,
    req: { profile: string; severity_levels?: number[]; random_state?: number }
  ): Promise<RobustnessEvaluationResult> {
    const res = await fetch(`${BASE_URL}/evaluations/${encodeURIComponent(runId)}/robustness`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    return handleResponse<RobustnessEvaluationResult>(res);
  },

  // Release Gate & Certificate
  async evaluateReleaseGate(runId: string, policy?: Partial<ReleasePolicy>): Promise<ReleaseGateResult> {
    const res = await fetch(`${BASE_URL}/evaluations/${encodeURIComponent(runId)}/release-gate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ policy }),
    });
    return handleResponse<ReleaseGateResult>(res);
  },

  async issueReleaseCertificate(
    runId: string,
    req: { key_id: string; target_environment: string; release_policy?: Partial<ReleasePolicy> }
  ): Promise<{ certificate: ReleaseCertificate; public_key_hex: string; certificate_path: string }> {
    const res = await fetch(`${BASE_URL}/evaluations/${encodeURIComponent(runId)}/certificate/issue`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    return handleResponse(res);
  },

  async verifyReleaseCertificate(req: {
    certificate?: any;
    certificate_path?: string;
    trusted_keys?: Record<string, string>;
    candidate_keys?: Record<string, string>;
    evidence_dir?: string;
    bundle_path?: string;
  }): Promise<CertificateVerificationResult> {
    const res = await fetch(`${BASE_URL}/verify-release`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    return handleResponse<CertificateVerificationResult>(res);
  },

  async createAuditBundle(runId: string): Promise<{ bundle_path: string; size_bytes: number; member_count: number; bundle_size_bytes?: number; bundle_sha256?: string }> {
    const res = await fetch(`${BASE_URL}/evaluations/${encodeURIComponent(runId)}/bundle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    const data = await handleResponse<any>(res);
    return {
      ...data,
      bundle_size_bytes: data.size_bytes || data.bundle_size_bytes || 0,
      bundle_sha256: data.checksum_sha256 || data.sha256 || data.bundle_sha256 || '',
    };
  },

  async verifyAuditBundle(bundlePath: string): Promise<AuditBundleVerificationResult> {
    const res = await fetch(`${BASE_URL}/verify-bundle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bundle_path: bundlePath }),
    });
    return handleResponse<AuditBundleVerificationResult>(res);
  },

  async checkHealth(): Promise<{ status: string; service?: string; version?: string }> {
    const res = await fetch(`${BASE_URL}/healthz`);
    return handleResponse(res);
  },

  async listArtifacts(runId: string): Promise<any[]> {
    const res = await fetch(`${BASE_URL}/evaluations/${encodeURIComponent(runId)}/artifacts`);
    const data = await handleResponse<any>(res);
    if (Array.isArray(data.artifacts)) {
      return data.artifacts;
    }
    if (data.files && typeof data.files === 'object') {
      return Object.entries(data.files).map(([p, info]: [string, any]) => ({
        path: p,
        name: p.split('/').pop() || p,
        size_bytes: info.size_bytes || 0,
        sha256: info.sha256 || '',
      }));
    }
    return [];
  },

  async getRobustnessResults(runId: string): Promise<any> {
    const res = await fetch(`${BASE_URL}/evaluations/${encodeURIComponent(runId)}/robustness`);
    return handleResponse(res);
  },

  async runRobustnessTest(
    runId: string,
    req: { perturbations?: string[]; severity_levels?: number[]; profile?: string; random_state?: number }
  ): Promise<any> {
    return this.runRobustnessStressTest(runId, {
      profile: req.profile || 'standard',
      severity_levels: req.severity_levels,
      random_state: req.random_state,
    });
  },

  async getEvaluation(runId: string): Promise<any> {
    const detail = await this.getEvaluationDetail(runId);
    return {
      id: detail.record.run_id,
      status: detail.record.state,
      dataset_name: detail.record.metadata?.dataset_name || detail.record.dataset_checksum?.substring(0, 12) || 'Synthetic Dataset',
      target_column: detail.record.metadata?.target_column || 'target',
      model_type: detail.record.candidate_model_type || 'Candidate Model',
      created_at: detail.record.created_at,
      updated_at: detail.record.updated_at,
      policy_name: detail.record.metadata?.policy_id || 'Standard Policy',
      acceptance_decision: detail.record.acceptance_outcome || (detail.has_decision ? 'Recorded' : 'Pending'),
      artifacts: detail.artifacts,
      lifecycle: detail.lifecycle_history,
      metrics: detail.metrics,
    };
  },

  async getAcceptanceReport(runId: string): Promise<any> {
    const dec = await this.getAcceptanceDecision(runId);
    return {
      decision: dec.outcome,
      policy_id: dec.policy_id,
      policy_version: dec.policy_version,
      criteria: dec.criteria_evaluations.map((c) => ({
        name: c.rule_id || c.metric,
        rule_id: c.rule_id,
        metric: c.metric,
        operator: c.operator,
        threshold_value: c.threshold_value,
        observed_value: c.observed_value !== null ? String(c.observed_value) : 'N/A',
        raw_observed_value: c.observed_value,
        outcome: c.outcome,
        reason_code: c.reason_code,
        message: c.message,
        evidence_source: c.evidence_source,
        passed: c.outcome === 'satisfied',
        severity: c.severity,
      })),
      rationale: dec.summary_rationale,
      passed_count: dec.passed_rules_count,
      failed_count: dec.failed_rules_count,
      inconclusive_count: dec.inconclusive_rules_count,
      advisory_warnings: dec.advisory_warnings || [],
      hash_signature: dec.hash_signature,
      raw_decision: dec,
    };
  },
};
