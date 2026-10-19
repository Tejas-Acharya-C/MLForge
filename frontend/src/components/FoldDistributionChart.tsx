import React from 'react';
import { FoldResult } from '../types/api';

interface FoldDistributionChartProps {
  candidateFolds: FoldResult[];
  baselineFolds?: FoldResult[];
  metricName?: string;
}

export const FoldDistributionChart: React.FC<FoldDistributionChartProps> = ({
  candidateFolds,
  baselineFolds,
  metricName = 'macro_f1',
}) => {
  if (!candidateFolds || candidateFolds.length === 0) {
    return <div style={{ color: 'var(--color-ink-muted)', fontSize: '13px' }}>No fold data available.</div>;
  }

  const width = 560;
  const height = 200;
  const paddingLeft = 55;
  const paddingRight = 35;
  const paddingTop = 25;
  const paddingBottom = 40;

  const plotWidth = width - paddingLeft - paddingRight;
  const plotHeight = height - paddingTop - paddingBottom;

  // Extract metric scores for candidate and baseline
  const candScores: (number | null)[] = candidateFolds.map(
    (f) => f.metrics[metricName]?.value ?? null
  );
  const baseScores: (number | null)[] = baselineFolds
    ? baselineFolds.map((f) => f.metrics[metricName]?.value ?? null)
    : [];

  const validScores = [...candScores, ...baseScores].filter((v): v is number => v !== null);
  const minVal = validScores.length > 0 ? Math.max(0, Math.min(...validScores) - 0.05) : 0;
  const maxVal = validScores.length > 0 ? Math.min(1.0, Math.max(...validScores) + 0.05) : 1;

  const scaleY = (val: number) => {
    if (maxVal === minVal) return paddingTop + plotHeight / 2;
    return height - paddingBottom - ((val - minVal) / (maxVal - minVal)) * plotHeight;
  };

  const scaleX = (foldIdx: number, totalFolds: number) => {
    return paddingLeft + (foldIdx / Math.max(1, totalFolds - 1)) * plotWidth;
  };

  const nFolds = candidateFolds.length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', width: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-ink-secondary)' }}>
          Fold Stability Strip ({metricName})
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--color-accent-primary)' }} />
            <span style={{ fontWeight: 500 }}>Candidate Model</span>
          </div>
          {baselineFolds && baselineFolds.length > 0 && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--color-ink-muted)' }} />
              <span style={{ color: 'var(--color-ink-muted)' }}>Baseline Dummy</span>
            </div>
          )}
        </div>
      </div>

      <div style={{ width: '100%', overflowX: 'auto' }}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          style={{
            width: '100%',
            maxWidth: '100%',
            height: 'auto',
            background: 'var(--color-surface)',
            border: '1px solid var(--color-border-subtle)',
            borderRadius: 'var(--radius-md)',
            display: 'block',
          }}
          role="img"
          aria-label={`Cross-validation fold performance plot for ${metricName}`}
        >
          {/* Horizontal gridlines */}
          {[minVal, (minVal + maxVal) / 2, maxVal].map((tick, i) => (
            <g key={`y-tick-${i}`}>
              <line
                x1={paddingLeft}
                y1={scaleY(tick)}
                x2={width - paddingRight}
                y2={scaleY(tick)}
                stroke="var(--color-border-subtle)"
                strokeDasharray="3 3"
              />
              <text
                x={paddingLeft - 8}
                y={scaleY(tick) + 3}
                fontSize={10}
                fill="var(--color-ink-muted)"
                textAnchor="end"
                className="tabular-nums font-mono"
              >
                {tick.toFixed(3)}
              </text>
            </g>
          ))}

          {/* Baseline trajectory */}
          {baseScores.length > 1 && (
            <polyline
              points={baseScores
                .map((sc, i) => (sc !== null ? `${scaleX(i, nFolds)},${scaleY(sc)}` : null))
                .filter(Boolean)
                .join(' ')}
              fill="none"
              stroke="var(--color-ink-muted)"
              strokeWidth={1.5}
              strokeDasharray="4 4"
            />
          )}

          {/* Candidate trajectory */}
          {candScores.length > 1 && (
            <polyline
              points={candScores
                .map((sc, i) => (sc !== null ? `${scaleX(i, nFolds)},${scaleY(sc)}` : null))
                .filter(Boolean)
                .join(' ')}
              fill="none"
              stroke="var(--color-accent-primary)"
              strokeWidth={2.5}
            />
          )}

          {/* Baseline dots */}
          {baseScores.map((sc, i) => {
            if (sc === null) return null;
            return (
              <circle
                key={`base-${i}`}
                cx={scaleX(i, nFolds)}
                cy={scaleY(sc)}
                r={4}
                fill="var(--color-ink-muted)"
                stroke="var(--color-surface)"
                strokeWidth={1.5}
              >
                <title>{`Fold ${i + 1} Baseline: ${sc.toFixed(4)}`}</title>
              </circle>
            );
          })}

          {/* Candidate dots */}
          {candScores.map((sc, i) => {
            if (sc === null) return null;
            return (
              <circle
                key={`cand-${i}`}
                cx={scaleX(i, nFolds)}
                cy={scaleY(sc)}
                r={5}
                fill="var(--color-accent-primary)"
                stroke="var(--color-surface)"
                strokeWidth={2}
              >
                <title>{`Fold ${i + 1} Candidate: ${sc.toFixed(4)}`}</title>
              </circle>
            );
          })}

          {/* X-axis fold labels */}
          {candidateFolds.map((_, i) => (
            <text
              key={`x-lbl-${i}`}
              x={scaleX(i, nFolds)}
              y={height - paddingBottom + 18}
              fontSize={11}
              fill="var(--color-ink-secondary)"
              textAnchor="middle"
              className="font-mono"
            >
              Fold {i + 1}
            </text>
          ))}
        </svg>
      </div>
    </div>
  );
};

export default FoldDistributionChart;
