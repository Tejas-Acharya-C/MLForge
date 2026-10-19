import { renderToString } from 'react-dom/server';
import { describe, it, expect } from 'vitest';
import { ConfusionMatrix } from '../components/ConfusionMatrix';
import { CalibrationChart } from '../components/CalibrationChart';
import { RobustnessChart } from '../components/RobustnessChart';
import { FoldDistributionChart } from '../components/FoldDistributionChart';
import { CalibrationCurveResult, RobustnessCurve } from '../types/api';

// Unit testing pure routing logic matching App.tsx
function parseHash(hash: string) {
  const clean = hash.replace(/^#\/?/, '').trim();
  if (!clean || clean === 'runs' || clean === 'overview') {
    return { screen: 'overview', runId: null };
  }
  if (clean === 'runs/new' || clean === 'new_evaluation' || clean === 'new') {
    return { screen: 'new_evaluation', runId: null };
  }

  const parts = clean.split('/');
  if (parts[0] === 'runs' && parts.length >= 2) {
    const runId = parts[1];
    const sub = parts[2] || 'results';
    if (sub === 'results') return { screen: 'results', runId };
    if (sub === 'reliability' || sub === 'robustness') return { screen: 'reliability', runId };
    if (sub === 'artifacts' || sub === 'details') return { screen: 'details', runId };
    if (sub === 'governance' || sub === 'acceptance' || sub === 'release') return { screen: 'acceptance', runId };
  }

  if (clean.startsWith('results')) return { screen: 'results', runId: parts[1] || null };
  if (clean.startsWith('reliability')) return { screen: 'reliability', runId: parts[1] || null };
  if (clean.startsWith('details') || clean.startsWith('artifacts')) return { screen: 'details', runId: parts[1] || null };
  if (clean.startsWith('acceptance') || clean.startsWith('governance')) return { screen: 'acceptance', runId: parts[1] || null };

  return { screen: 'overview', runId: null };
}

function buildHash(screen: string, runId?: string | null): string {
  if (screen === 'overview') return '#/runs';
  if (screen === 'new_evaluation' || screen === 'new') return '#/runs/new';
  if (runId) {
    if (screen === 'results') return `#/runs/${runId}/results`;
    if (screen === 'reliability') return `#/runs/${runId}/reliability`;
    if (screen === 'details' || screen === 'artifacts') return `#/runs/${runId}/artifacts`;
    if (screen === 'acceptance' || screen === 'governance') return `#/runs/${runId}/governance`;
  }
  return '#/runs';
}

describe('MLForge URL Hash Routing Architecture', () => {
  it('parses root and overview hash correctly', () => {
    expect(parseHash('')).toEqual({ screen: 'overview', runId: null });
    expect(parseHash('#/')).toEqual({ screen: 'overview', runId: null });
    expect(parseHash('#/runs')).toEqual({ screen: 'overview', runId: null });
    expect(parseHash('#overview')).toEqual({ screen: 'overview', runId: null });
  });

  it('parses new evaluation route correctly', () => {
    expect(parseHash('#/runs/new')).toEqual({ screen: 'new_evaluation', runId: null });
    expect(parseHash('#new_evaluation')).toEqual({ screen: 'new_evaluation', runId: null });
  });

  it('parses run-specific subroutes correctly', () => {
    const runId = 'eval_run_20261019_abc123';
    expect(parseHash(`#/runs/${runId}/results`)).toEqual({ screen: 'results', runId });
    expect(parseHash(`#/runs/${runId}/reliability`)).toEqual({ screen: 'reliability', runId });
    expect(parseHash(`#/runs/${runId}/artifacts`)).toEqual({ screen: 'details', runId });
    expect(parseHash(`#/runs/${runId}/governance`)).toEqual({ screen: 'acceptance', runId });
  });

  it('handles legacy and alias routes gracefully', () => {
    const runId = 'test_run_1';
    expect(parseHash(`#/runs/${runId}/robustness`)).toEqual({ screen: 'reliability', runId });
    expect(parseHash(`#/runs/${runId}/details`)).toEqual({ screen: 'details', runId });
    expect(parseHash(`#/runs/${runId}/acceptance`)).toEqual({ screen: 'acceptance', runId });
  });

  it('generates canonical canonical URLs via buildHash', () => {
    const runId = 'run_456';
    expect(buildHash('overview')).toBe('#/runs');
    expect(buildHash('new')).toBe('#/runs/new');
    expect(buildHash('results', runId)).toBe('#/runs/run_456/results');
    expect(buildHash('reliability', runId)).toBe('#/runs/run_456/reliability');
    expect(buildHash('artifacts', runId)).toBe('#/runs/run_456/artifacts');
    expect(buildHash('details', runId)).toBe('#/runs/run_456/artifacts');
    expect(buildHash('governance', runId)).toBe('#/runs/run_456/governance');
    expect(buildHash('acceptance', runId)).toBe('#/runs/run_456/governance');
  });

  it('falls back to overview for unknown or invalid routes', () => {
    expect(parseHash('#/invalid/route/path')).toEqual({ screen: 'overview', runId: null });
  });
});

describe('Scientific Visualization Enhancements & Accessibility', () => {
  it('renders ConfusionMatrix with view mode toggles and semantic labels', () => {
    const matrix = [
      [90, 10],
      [5, 95],
    ];
    const labels = ['Class 0', 'Class 1'];
    const html = renderToString(<ConfusionMatrix matrix={matrix} classLabels={labels} title="Candidate Matrix" />);

    // Segmented toggles
    expect(html).toContain('Raw Counts');
    expect(html).toContain('Row % (Recall)');
    expect(html).toContain('Col % (Precision)');
    // Raw numbers
    expect(html).toContain('90');
    expect(html).toContain('95');
    // Explanatory methodological note
    expect(html).toContain('Showing absolute sample counts across cross-validation out-of-fold predictions.');
  });

  it('renders CalibrationChart with responsive viewBox and marginal sample count histogram', () => {
    const calData: CalibrationCurveResult = {
      bin_strategy: 'uniform',
      num_bins: 2,
      total_samples: 150,
      positive_class_label: 'positive',
      expected_calibration_error: 0.035,
      max_calibration_error: 0.070,
      brier_score: 0.089,
      log_loss: 0.25,
      is_usable: true,
      warnings: [],
      bins: [
        {
          bin_index: 0,
          lower_bound: 0.0,
          upper_bound: 0.5,
          sample_count: 85,
          mean_predicted_prob: 0.22,
          observed_positive_rate: 0.20,
          is_empty: false,
        },
        {
          bin_index: 1,
          lower_bound: 0.5,
          upper_bound: 1.0,
          sample_count: 65,
          mean_predicted_prob: 0.78,
          observed_positive_rate: 0.81,
          is_empty: false,
        },
      ],
    };

    const html = renderToString(<CalibrationChart data={calData} />);
    // Responsive viewBox
    expect(html).toContain('viewBox="0 0 520 420"');
    // Role for accessibility
    expect(html).toContain('role="img"');
    // Marginal sample count section
    expect(html).toContain('Empirical Sample Distribution (Bin Counts)');
    // Real bin sample counts
    expect(html).toContain('85');
    expect(html).toContain('65');
  });

  it('renders RobustnessChart with responsive viewBox and accessibility metadata', () => {
    const curves: RobustnessCurve[] = [
      {
        perturbation_type: 'gaussian_noise',
        metric_name: 'macro_f1',
        clean_score: 0.90,
        area_under_degradation_curve: 0.82,
        points: [
          {
            perturbation_type: 'gaussian_noise',
            severity_level: 0.1,
            metric_name: 'macro_f1',
            clean_score: 0.90,
            perturbed_score: 0.85,
            retained_ratio: 0.944,
            performance_drop_percentage: 5.6,
            is_usable: true,
          },
        ],
      },
    ];

    const html = renderToString(<RobustnessChart curves={curves} />);
    expect(html).toContain('viewBox="0 0 560 240"');
    expect(html).toContain('role="img"');
    expect(html).toContain('gaussian noise');
    expect(html).toContain('AURDC:');
    expect(html).toContain('0.820');
  });

  it('renders FoldDistributionChart with responsive viewBox and accessibility metadata', () => {
    const folds = [
      {
        fold_index: 0,
        train_indices_count: 80,
        val_indices_count: 20,
        metrics: {
          macro_f1: { value: 0.88, is_undefined: false },
        },
        confusion_matrix: [[10, 0], [0, 10]],
        class_labels: ['0', '1'],
        fit_duration_seconds: 0.01,
        predict_duration_seconds: 0.001,
      },
    ];

    const html = renderToString(<FoldDistributionChart candidateFolds={folds} metricName="macro_f1" />);
    expect(html).toContain('viewBox="0 0 560 200"');
    expect(html).toContain('role="img"');
    expect(html).toContain('Fold Stability Strip');
    expect(html).toContain('Candidate Model');
  });
});
