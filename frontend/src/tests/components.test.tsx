import { renderToString } from 'react-dom/server';
import { describe, it, expect } from 'vitest';

import { ConfusionMatrix } from '../components/ConfusionMatrix';
import { CalibrationChart } from '../components/CalibrationChart';
import { RobustnessChart } from '../components/RobustnessChart';
import { FoldDistributionChart } from '../components/FoldDistributionChart';
import { CalibrationCurveResult, RobustnessCurve } from '../types/api';

describe('MLForge Scientific Frontend Components', () => {
  describe('ConfusionMatrix Component', () => {
    it('renders confusion matrix with true/pred headers and cell counts', () => {
      const matrix = [
        [45, 5],
        [8, 42],
      ];
      const labels = ['Negative', 'Positive'];

      const html = renderToString(
        <ConfusionMatrix matrix={matrix} classLabels={labels} title="Test Matrix" />
      );

      expect(html).toContain('Test Matrix');
      expect(html).toContain('Negative');
      expect(html).toContain('Positive');
      expect(html).toContain('45');
      expect(html).toContain('42');
      expect(html).toContain('True \\ Pred');
    });

    it('renders clean fallback empty state when matrix is missing or empty', () => {
      const html = renderToString(<ConfusionMatrix matrix={[]} classLabels={[]} />);
      expect(html).toContain('No matrix data available.');
    });
  });

  describe('CalibrationChart Component', () => {
    it('renders calibration bins, strategy, and valid ECE / Brier score', () => {
      const calData: CalibrationCurveResult = {
        bin_strategy: 'uniform',
        num_bins: 2,
        total_samples: 100,
        positive_class_label: '1',
        expected_calibration_error: 0.0432,
        max_calibration_error: 0.0811,
        brier_score: 0.1245,
        log_loss: 0.321,
        is_usable: true,
        warnings: [],
        bins: [
          {
            bin_index: 0,
            lower_bound: 0.0,
            upper_bound: 0.2,
            sample_count: 50,
            mean_predicted_prob: 0.12,
            observed_positive_rate: 0.10,
            is_empty: false,
          },
          {
            bin_index: 1,
            lower_bound: 0.2,
            upper_bound: 0.4,
            sample_count: 50,
            mean_predicted_prob: 0.31,
            observed_positive_rate: 0.28,
            is_empty: false,
          },
        ],
      };

      const html = renderToString(<CalibrationChart data={calData} />);
      expect(html).toContain('Strategy:');
      expect(html).toContain('uniform');
      expect(html).toContain('ECE:');
      expect(html).toContain('0.0432');
      expect(html).toContain('Brier Score:');
      expect(html).toContain('0.1245');
    });

    it('handles null ECE and Brier score gracefully without fabricating numbers', () => {
      const calData: CalibrationCurveResult = {
        bin_strategy: 'quantile',
        num_bins: 0,
        total_samples: 0,
        positive_class_label: 'Class_A',
        expected_calibration_error: null,
        max_calibration_error: null,
        brier_score: null,
        log_loss: null,
        is_usable: false,
        warnings: [],
        bins: [],
      };

      const html = renderToString(<CalibrationChart data={calData} />);
      expect(html).toContain('quantile');
      expect(html).toContain('Class_A');
      // Should NOT render ECE or Brier score badges when null
      expect(html).not.toContain('ECE:');
      expect(html).not.toContain('Brier Score:');
    });
  });

  describe('RobustnessChart Component', () => {
    it('renders perturbation curves and AURDC badge when present', () => {
      const curves: RobustnessCurve[] = [
        {
          perturbation_type: 'gaussian_noise',
          metric_name: 'accuracy',
          clean_score: 0.88,
          area_under_degradation_curve: 0.845,
          points: [
            {
              perturbation_type: 'gaussian_noise',
              severity_level: 0.05,
              metric_name: 'accuracy',
              clean_score: 0.88,
              perturbed_score: 0.85,
              retained_ratio: 0.966,
              performance_drop_percentage: 3.4,
              is_usable: true,
            },
          ],
        },
      ];

      const html = renderToString(<RobustnessChart curves={curves} />);
      expect(html).toContain('gaussian noise');
      expect(html).toContain('AURDC:');
      expect(html).toContain('0.845');
    });

    it('renders without AURDC badge when area_under_degradation_curve is undefined (no fake fallback)', () => {
      const curves: RobustnessCurve[] = [
        {
          perturbation_type: 'category_dropout',
          metric_name: 'macro_f1',
          clean_score: 0.75,
          area_under_degradation_curve: undefined,
          points: [],
        },
      ];

      const html = renderToString(<RobustnessChart curves={curves} />);
      expect(html).toContain('category dropout');
      // Must NOT fabricate a default AURDC (e.g. 0.85)
      expect(html).not.toContain('AURDC:');
    });

    it('renders clean fallback empty state when curves are empty', () => {
      const html = renderToString(<RobustnessChart curves={[]} />);
      expect(html).toContain('No robustness curves available.');
    });
  });

  describe('FoldDistributionChart Component', () => {
    it('renders fold metrics and summary indicators', () => {
      const folds = [
        {
          fold_index: 0,
          train_indices_count: 80,
          val_indices_count: 20,
          metrics: {
            macro_f1: { value: 0.85, is_undefined: false },
            accuracy: { value: 0.88, is_undefined: false },
          },
          confusion_matrix: [[10, 0], [0, 10]],
          class_labels: ['0', '1'],
          fit_duration_seconds: 0.01,
          predict_duration_seconds: 0.001,
        },
        {
          fold_index: 1,
          train_indices_count: 80,
          val_indices_count: 20,
          metrics: {
            macro_f1: { value: 0.87, is_undefined: false },
            accuracy: { value: 0.90, is_undefined: false },
          },
          confusion_matrix: [[10, 0], [0, 10]],
          class_labels: ['0', '1'],
          fit_duration_seconds: 0.01,
          predict_duration_seconds: 0.001,
        },
      ];

      const html = renderToString(
        <FoldDistributionChart candidateFolds={folds} metricName="macro_f1" />
      );

      expect(html).toContain('Fold Stability Strip');
      expect(html).toContain('macro_f1');
      expect(html).toContain('Candidate');
      expect(html).toContain('Fold 1 Candidate: 0.8500');
      expect(html).toContain('Fold 2 Candidate: 0.8700');
    });

    it('handles empty fold list gracefully', () => {
      const html = renderToString(<FoldDistributionChart candidateFolds={[]} metricName="accuracy" />);
      expect(html).toContain('No fold data available.');
    });
  });
});
