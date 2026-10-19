import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import { DatasetMetadata, PolicyTemplateInfo, SampleDatasetInfo } from '../types/api';

interface NewEvaluationProps {
  initialPreset?: string;
  onRunStarted?: (runId: string) => void;
  onCancel?: () => void;
  onNavigate?: (screen: string, runId?: string) => void;
}

export const NewEvaluation: React.FC<NewEvaluationProps> = ({ initialPreset, onRunStarted, onCancel, onNavigate }) => {
  // Dataset inputs
  const [samplePresets, setSamplePresets] = useState<SampleDatasetInfo[]>([]);
  const [selectedSample, setSelectedSample] = useState<string>(initialPreset || 'churn');
  const [filePath, setFilePath] = useState<string>('');
  const [useSample, setUseSample] = useState<boolean>(true);
  const [targetColumn, setTargetColumn] = useState<string>('churn_label');
  const [subgroupColumn, setSubgroupColumn] = useState<string>('region');

  // Inspected metadata
  const [metadata, setMetadata] = useState<DatasetMetadata | null>(null);
  const [inspecting, setInspecting] = useState<boolean>(false);
  const [inspectError, setInspectError] = useState<string | null>(null);

  // Hyperparameters
  const [cParam, setCParam] = useState<number>(1.0);
  const [penalty, setPenalty] = useState<string>('l2');
  const [solver, setSolver] = useState<string>('lbfgs');
  const [maxIter, setMaxIter] = useState<number>(200);

  // Validation protocol
  const [numFolds, setNumFolds] = useState<number>(5);
  const [randomSeed, setRandomSeed] = useState<number>(42);

  // Policies
  const [policies, setPolicies] = useState<PolicyTemplateInfo[]>([]);
  const [selectedPolicyId, setSelectedPolicyId] = useState<string>('standard_classification_v1');

  // Execution state
  const [executing, setExecuting] = useState<boolean>(false);
  const [execError, setExecError] = useState<string | null>(null);

  // Initial load
  useEffect(() => {
    api.getSampleDatasets().then(setSamplePresets).catch(() => {});
    api.getAcceptancePolicies().then(setPolicies).catch(() => {});
    handleInspect();
  }, [selectedSample, useSample]);

  const handleInspect = async () => {
    setInspecting(true);
    setInspectError(null);
    try {
      const meta = await api.inspectDataset({
        sample_name: useSample ? selectedSample : undefined,
        file_path: !useSample ? filePath : undefined,
        target_column: targetColumn,
        k_folds: numFolds,
      });
      setMetadata(meta);
      if (!meta.feature_names.includes(subgroupColumn)) {
        const catCols = Object.entries(meta.feature_types)
          .filter(([_, t]) => t === 'object' || t === 'category')
          .map(([col]) => col);
        setSubgroupColumn(catCols[0] || '');
      }
    } catch (err: any) {
      setInspectError(err.message || 'Dataset inspection failed.');
      setMetadata(null);
    } finally {
      setInspecting(false);
    }
  };

  const handlePresetChange = (presetId: string) => {
    setSelectedSample(presetId);
    setUseSample(true);
    if (presetId === 'churn') {
      setTargetColumn('churn_label');
      setSubgroupColumn('region');
    } else if (presetId === 'breast_cancer') {
      setTargetColumn('diagnosis');
      setSubgroupColumn('institution_tier');
    }
  };

  const handleExecute = async () => {
    setExecuting(true);
    setExecError(null);
    try {
      const result = await api.submitEvaluation({
        sample_name: useSample ? selectedSample : undefined,
        file_path: !useSample ? filePath : undefined,
        target_column: targetColumn,
        candidate_model_type: 'logistic_regression',
        candidate_hyperparameters: {
          C: cParam,
          penalty: penalty === 'none' ? null : penalty,
          solver: solver,
          max_iter: maxIter,
        },
        num_folds: numFolds,
        random_state: randomSeed,
        subgroup_columns: subgroupColumn ? [subgroupColumn] : undefined,
        policy_id: selectedPolicyId,
      });
      if (onRunStarted) {
        onRunStarted(result.run_id);
      } else if (onNavigate) {
        onNavigate('results', result.run_id);
      }
    } catch (err: any) {
      setExecError(err.message || 'Failed to submit evaluation.');
      setExecuting(false);
    }
  };

  const handleCancel = () => {
    if (onCancel) {
      onCancel();
    } else if (onNavigate) {
      onNavigate('overview');
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', paddingBottom: '80px' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h1>Configure Evaluation Run</h1>
          <p style={{ color: 'var(--color-ink-secondary)', fontSize: '13px', marginTop: '2px' }}>
            Specify dataset inputs, candidate hyperparameters, cross-validation protocol, and formal governance policies.
          </p>
        </div>
        <button className="btn btn-secondary" onClick={handleCancel}>
          Cancel
        </button>
      </div>

      {execError && (
        <div className="callout callout-error">
          <strong>Execution Error:</strong> {execError}
        </div>
      )}

      {/* Region 1: Dataset & Target Selection */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">1. Tabular Dataset & Target Selection</div>
            <div className="panel-description">Select a local dataset file or load one of the built-in benchmarks.</div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '16px', marginBottom: '16px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', cursor: 'pointer' }}>
            <input
              type="radio"
              checked={useSample}
              onChange={() => setUseSample(true)}
            />
            Built-in Benchmark Preset
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', cursor: 'pointer' }}>
            <input
              type="radio"
              checked={!useSample}
              onChange={() => setUseSample(false)}
            />
            Local File System Path (CSV / Parquet)
          </label>
        </div>

        {useSample ? (
          <div style={{ display: 'flex', gap: '10px', marginBottom: '16px', flexWrap: 'wrap' }}>
            {samplePresets.map((p) => (
              <button
                key={p.id}
                type="button"
                className={`btn ${selectedSample === p.id ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => handlePresetChange(p.id)}
              >
                {p.name} ({p.num_rows} rows)
              </button>
            ))}
          </div>
        ) : (
          <div className="form-group">
            <label className="form-label">Local File Path:</label>
            <div style={{ display: 'flex', gap: '8px' }}>
              <input
                type="text"
                className="form-input font-mono"
                placeholder="e.g. data/my_classification_data.csv"
                value={filePath}
                onChange={(e) => setFilePath(e.target.value)}
              />
              <button className="btn btn-secondary" onClick={handleInspect} disabled={inspecting || !filePath}>
                Inspect schema
              </button>
            </div>
            <div className="form-help">Must be a valid local CSV or Parquet file path without directory traversal.</div>
          </div>
        )}

        <div className="form-row">
          <div className="form-group">
            <label className="form-label">Classification Target Column:</label>
            <input
              type="text"
              className="form-input font-mono"
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              onBlur={handleInspect}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Subgroup Slice Feature (Categorical):</label>
            {metadata && (Object.keys(metadata.feature_types).length > 0) ? (
              <select
                className="form-select font-mono"
                value={subgroupColumn}
                onChange={(e) => setSubgroupColumn(e.target.value)}
              >
                <option value="">(None — Global Evaluation Only)</option>
                {/* Prioritize categorical/object columns */}
                <optgroup label="Categorical Columns (Inspected)">
                  {Object.entries(metadata.feature_types)
                    .filter(([col, type]) => col !== targetColumn && (type === 'object' || type === 'category' || type === 'string'))
                    .map(([col]) => (
                      <option key={col} value={col}>
                        {col} ({metadata.feature_types[col]})
                      </option>
                    ))}
                </optgroup>
                <optgroup label="All Other Features">
                  {Object.entries(metadata.feature_types)
                    .filter(([col, type]) => col !== targetColumn && type !== 'object' && type !== 'category' && type !== 'string')
                    .map(([col]) => (
                      <option key={col} value={col}>
                        {col} ({metadata.feature_types[col]})
                      </option>
                    ))}
                </optgroup>
              </select>
            ) : (
              <input
                type="text"
                className="form-input font-mono"
                placeholder="e.g. region or institution_tier"
                value={subgroupColumn}
                onChange={(e) => setSubgroupColumn(e.target.value)}
              />
            )}
            <div className="form-help">
              Evaluates subgroup slices to diagnose calibration parity and error disparities.
            </div>
          </div>
        </div>

        {inspectError && (
          <div className="callout callout-warning">
            <strong>Validation Notice:</strong> {inspectError}
          </div>
        )}

        {/* Schema Inspector */}
        {metadata && (
          <div style={{ marginTop: '12px', borderTop: '1px solid var(--color-border-subtle)', paddingTop: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '12px', flexWrap: 'wrap' }}>
              <div className="badge badge-neutral">Rows: <strong>{metadata.num_rows}</strong></div>
              <div className="badge badge-neutral">Features: <strong>{metadata.num_features}</strong> ({metadata.numeric_feature_count} numeric, {metadata.categorical_feature_count} categorical)</div>
              <div className="badge badge-neutral">Imbalance Ratio: <strong className="tabular-nums">{metadata.imbalance_ratio.toFixed(2)}:1</strong></div>
              <div className="badge badge-neutral font-mono" style={{ fontSize: '11px' }}>SHA256: {metadata.checksum_sha256.substring(0, 12)}...</div>
            </div>

            <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--color-ink-secondary)', marginBottom: '6px' }}>
              Class Distribution:
            </div>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
              {Object.entries(metadata.class_distribution).map(([cls, cnt]) => (
                <div key={cls} className="badge badge-neutral" style={{ padding: '4px 10px' }}>
                  <span>Class <code>{cls}</code>:</span>
                  <strong className="tabular-nums font-mono">{cnt} samples</strong>
                  <span style={{ opacity: 0.7 }}>({((cnt / metadata.num_rows) * 100).toFixed(1)}%)</span>
                </div>
              ))}
            </div>

            {metadata.warnings.length > 0 && (
              <div className="callout callout-warning" style={{ margin: '8px 0' }}>
                <div style={{ fontWeight: 600 }}>Data Quality Warnings:</div>
                <ul style={{ paddingLeft: '18px', marginTop: '4px' }}>
                  {metadata.warnings.map((w, idx) => (
                    <li key={idx}>{w}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Region 2: Model & Preprocessing Pipeline Configuration */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">2. Estimator & Hyperparameter Configuration</div>
            <div className="panel-description">Specify candidate Logistic Regression hyperparameters and baseline anchor.</div>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label className="form-label">Inverse Regularization (C):</label>
            <select
              className="form-select font-mono"
              value={cParam}
              onChange={(e) => setCParam(parseFloat(e.target.value))}
            >
              <option value="0.01">0.01 (Strong L2 Regularization)</option>
              <option value="0.1">0.10 (Moderate Regularization)</option>
              <option value="1.0">1.00 (Standard Default)</option>
              <option value="10.0">10.00 (Mild Regularization)</option>
              <option value="100.0">100.00 (Minimal Regularization)</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Penalty Norm:</label>
            <select
              className="form-select font-mono"
              value={penalty}
              onChange={(e) => setPenalty(e.target.value)}
            >
              <option value="l2">L2 (Ridge Penalty)</option>
              <option value="l1">L1 (Lasso Penalty)</option>
              <option value="none">None (Unpenalized)</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Optimization Solver:</label>
            <select
              className="form-select font-mono"
              value={solver}
              onChange={(e) => setSolver(e.target.value)}
            >
              <option value="lbfgs">lbfgs (Standard quasi-Newton)</option>
              <option value="saga">saga (Elastic-net / L1 compatible)</option>
              <option value="liblinear">liblinear (Coordinate descent)</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Max Iterations:</label>
            <input
              type="number"
              className="form-input font-mono"
              value={maxIter}
              min={50}
              max={1000}
              step={50}
              onChange={(e) => setMaxIter(parseInt(e.target.value, 10))}
            />
          </div>
        </div>

        <div className="callout callout-info" style={{ marginTop: '12px', fontSize: '12px' }}>
          <strong>Leakage-Safe Guarantee:</strong> Pipelines dynamically scale numerical features and one-hot encode categoricals strictly inside each training fold partition. Baseline anchor model: <code>DummyClassifier(strategy='prior')</code> evaluated on identical folds.
        </div>
      </div>

      {/* Region 3: Validation Protocol & Acceptance Policy */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">3. Validation Protocol & Governance Policy</div>
            <div className="panel-description">Define cross-validation partition bounds and select automated acceptance criteria.</div>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label className="form-label">Stratified K-Fold Count:</label>
            <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
              {[2, 3, 5, 10].map((k) => (
                <button
                  key={k}
                  type="button"
                  className={`btn ${numFolds === k ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ flex: 1, padding: '6px 4px', fontSize: '12px' }}
                  onClick={() => setNumFolds(k)}
                >
                  {k} Folds {k === 5 ? '(Std)' : ''}
                </button>
              ))}
            </div>
            <input
              type="range"
              min={2}
              max={10}
              value={numFolds}
              onChange={(e) => setNumFolds(parseInt(e.target.value, 10))}
              style={{ width: '100%' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
              <span>K={numFolds}: {Math.round(((numFolds - 1) / numFolds) * 100)}% Train / {Math.round((1 / numFolds) * 100)}% Validation per fold</span>
              <span className="font-mono">{numFolds} Partitions</span>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Deterministic Random Seed:</label>
            <input
              type="number"
              className="form-input font-mono"
              value={randomSeed}
              onChange={(e) => setRandomSeed(parseInt(e.target.value, 10))}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Acceptance Policy Template:</label>
            <select
              className="form-select"
              value={selectedPolicyId}
              onChange={(e) => setSelectedPolicyId(e.target.value)}
            >
              {policies.map((pol) => (
                <option key={pol.policy_id} value={pol.policy_id}>
                  {pol.name} ({pol.rules_count} criteria)
                </option>
              ))}
            </select>
            {policies.find((p) => p.policy_id === selectedPolicyId)?.description && (
              <div className="form-help">
                {policies.find((p) => p.policy_id === selectedPolicyId)?.description}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Region 4: Preflight Summary & Sticky Execution Bar */}
      <div
        style={{
          position: 'sticky',
          bottom: '16px',
          backgroundColor: 'var(--color-surface)',
          border: '1px solid var(--color-border-strong)',
          borderRadius: 'var(--radius-lg)',
          boxShadow: 'var(--shadow-overlay)',
          padding: '16px 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          zIndex: 50,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div>
            <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Execution Target
            </div>
            <div style={{ fontWeight: 600 }}>{useSample ? selectedSample : filePath || 'Local Dataset'}</div>
          </div>
          <div>
            <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Protocol
            </div>
            <div>{numFolds}-Fold Stratified CV (Seed {randomSeed})</div>
          </div>
          <div>
            <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Policy
            </div>
            <div>{selectedPolicyId}</div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button className="btn btn-secondary" onClick={handleCancel} disabled={executing}>
            Cancel
          </button>
          <button
            className="btn btn-primary"
            style={{ padding: '9px 20px', fontSize: '14px', fontWeight: 600 }}
            onClick={handleExecute}
            disabled={executing || (inspecting && !metadata)}
          >
            {executing ? 'Executing evaluation...' : 'Execute evaluation run'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default NewEvaluation;
