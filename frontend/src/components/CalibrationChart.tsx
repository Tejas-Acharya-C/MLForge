import React, { useState } from 'react';
import { CalibrationBin, CalibrationCurveResult } from '../types/api';

interface CalibrationChartProps {
  data: CalibrationCurveResult;
}

export const CalibrationChart: React.FC<CalibrationChartProps> = ({ data }) => {
  const [hoveredBin, setHoveredBin] = useState<CalibrationBin | null>(null);

  const viewBoxWidth = 520;
  const viewBoxHeight = 420;

  const paddingLeft = 55;
  const paddingRight = 35;
  const paddingTop = 25;
  const curveBottom = 260; // where curve ends
  const histTop = 295; // where histogram starts
  const histBottom = 380; // where histogram ends

  const plotWidth = viewBoxWidth - paddingLeft - paddingRight;
  const curveHeight = curveBottom - paddingTop;
  const histHeight = histBottom - histTop;

  // Scale coordinates: [0, 1] -> [paddingLeft, viewBoxWidth - paddingRight]
  const scaleX = (val: number) => paddingLeft + val * plotWidth;
  // Curve Y: [0, 1] -> [curveBottom, paddingTop]
  const scaleY = (val: number) => curveBottom - val * curveHeight;

  const nonNullBins = data.bins.filter(
    (b) => b.mean_predicted_prob !== null && b.observed_positive_rate !== null
  );

  // Maximum sample count across bins for histogram scaling
  const maxSampleCount = Math.max(1, ...data.bins.map((b) => b.sample_count || 0));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', width: '100%' }}>
      {/* Metrics Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
        <div className="badge badge-neutral">
          Strategy: <strong>{data.bin_strategy}</strong>
        </div>
        <div className="badge badge-neutral">
          Target Class: <strong>{data.positive_class_label}</strong>
        </div>
        {data.expected_calibration_error !== null && (
          <div className="badge badge-neutral">
            ECE: <strong className="tabular-nums font-mono">{data.expected_calibration_error.toFixed(4)}</strong>
          </div>
        )}
        {data.brier_score !== null && (
          <div className="badge badge-neutral">
            Brier Score: <strong className="tabular-nums font-mono">{data.brier_score.toFixed(4)}</strong>
          </div>
        )}
      </div>

      <div style={{ position: 'relative', width: '100%', overflowX: 'auto' }}>
        <svg
          viewBox={`0 0 ${viewBoxWidth} ${viewBoxHeight}`}
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
          aria-label="Reliability diagram showing predicted probabilities vs observed positive rates and marginal sample distributions"
        >
          {/* Main Curve Gridlines */}
          {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((tick) => (
            <g key={`grid-${tick}`}>
              <line
                x1={scaleX(tick)}
                y1={paddingTop}
                x2={scaleX(tick)}
                y2={curveBottom}
                stroke="var(--color-border-subtle)"
                strokeDasharray="3 3"
              />
              <line
                x1={scaleX(0)}
                y1={scaleY(tick)}
                x2={scaleX(1)}
                y2={scaleY(tick)}
                stroke="var(--color-border-subtle)"
                strokeDasharray="3 3"
              />
              {/* Y-axis tick labels */}
              <text
                x={paddingLeft - 8}
                y={scaleY(tick) + 3}
                fontSize={10}
                fill="var(--color-ink-muted)"
                textAnchor="end"
                className="tabular-nums font-mono"
              >
                {tick.toFixed(1)}
              </text>
            </g>
          ))}

          {/* Diagonal Reference Line (Perfect Calibration y = x) */}
          <line
            x1={scaleX(0)}
            y1={scaleY(0)}
            x2={scaleX(1)}
            y2={scaleY(1)}
            stroke="var(--color-ink-muted)"
            strokeWidth={1.5}
            strokeDasharray="4 4"
          />

          {/* Deviation Connectors */}
          {nonNullBins.map((bin) => {
            const x = scaleX(bin.mean_predicted_prob!);
            const y = scaleY(bin.observed_positive_rate!);
            const diagY = scaleY(bin.mean_predicted_prob!);
            return (
              <line
                key={`gap-${bin.bin_index}`}
                x1={x}
                y1={y}
                x2={x}
                y2={diagY}
                stroke={y < diagY ? 'var(--color-crimson)' : 'var(--color-accent-primary)'}
                strokeWidth={1.5}
                opacity={0.65}
              />
            );
          })}

          {/* Calibration Curve Trajectory Line */}
          {nonNullBins.length > 1 && (
            <polyline
              points={nonNullBins
                .map((b) => `${scaleX(b.mean_predicted_prob!)},${scaleY(b.observed_positive_rate!)}`)
                .join(' ')}
              fill="none"
              stroke="var(--color-accent-primary)"
              strokeWidth={2.5}
            />
          )}

          {/* Binned Points */}
          {nonNullBins.map((bin) => {
            const cx = scaleX(bin.mean_predicted_prob!);
            const cy = scaleY(bin.observed_positive_rate!);
            const isHovered = hoveredBin?.bin_index === bin.bin_index;
            return (
              <circle
                key={`pt-${bin.bin_index}`}
                cx={cx}
                cy={cy}
                r={isHovered ? 7 : 4.5}
                fill="var(--color-accent-primary)"
                stroke="var(--color-surface)"
                strokeWidth={2}
                style={{ cursor: 'pointer', transition: 'r 0.15s ease' }}
                onMouseEnter={() => setHoveredBin(bin)}
                onMouseLeave={() => setHoveredBin(null)}
              >
                <title>{`Bin ${bin.bin_index + 1}: Pred ${bin.mean_predicted_prob?.toFixed(3)}, Obs ${bin.observed_positive_rate?.toFixed(3)}, N=${bin.sample_count}`}</title>
              </circle>
            );
          })}

          {/* Curve Y-Axis Label */}
          <text
            x={-(paddingTop + curveHeight / 2)}
            y={14}
            fontSize={11}
            fill="var(--color-ink-secondary)"
            textAnchor="middle"
            fontWeight={600}
            transform="rotate(-90)"
          >
            Observed Positive Rate
          </text>

          {/* Section Divider / Label for Marginal Histogram */}
          <line
            x1={paddingLeft}
            y1={histTop - 12}
            x2={scaleX(1)}
            y2={histTop - 12}
            stroke="var(--color-border-strong)"
          />
          <text
            x={paddingLeft}
            y={histTop - 18}
            fontSize={10}
            fill="var(--color-ink-muted)"
            fontWeight={600}
            style={{ textTransform: 'uppercase', letterSpacing: '0.05em' }}
          >
            Empirical Sample Distribution (Bin Counts)
          </text>

          {/* Marginal Histogram Bars */}
          {data.bins.map((bin) => {
            const barX1 = scaleX(bin.lower_bound);
            const barX2 = scaleX(bin.upper_bound);
            const barW = Math.max(2, barX2 - barX1 - 2);
            const barH = (bin.sample_count / maxSampleCount) * histHeight;
            const barY = histBottom - barH;
            const isHovered = hoveredBin?.bin_index === bin.bin_index;

            return (
              <g key={`bar-${bin.bin_index}`}>
                <rect
                  x={barX1 + 1}
                  y={barY}
                  width={barW}
                  height={barH}
                  fill={isHovered ? 'var(--color-accent-primary)' : 'var(--color-stone-border)'}
                  opacity={isHovered ? 0.9 : 0.6}
                  rx={2}
                  style={{ cursor: 'pointer', transition: 'all 0.15s ease' }}
                  onMouseEnter={() => setHoveredBin(bin)}
                  onMouseLeave={() => setHoveredBin(null)}
                >
                  <title>{`Bin ${bin.bin_index + 1} [${bin.lower_bound.toFixed(2)}-${bin.upper_bound.toFixed(2)}]: ${bin.sample_count} samples`}</title>
                </rect>
                {bin.sample_count > 0 && (
                  <text
                    x={barX1 + barW / 2 + 1}
                    y={barY - 3}
                    fontSize={9}
                    fill="var(--color-ink-muted)"
                    textAnchor="middle"
                    className="tabular-nums font-mono"
                  >
                    {bin.sample_count}
                  </text>
                )}
              </g>
            );
          })}

          {/* Histogram Baseline & X-Axis Labels */}
          <line
            x1={paddingLeft}
            y1={histBottom}
            x2={scaleX(1)}
            y2={histBottom}
            stroke="var(--color-border-strong)"
          />

          {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((tick) => (
            <text
              key={`x-tick-${tick}`}
              x={scaleX(tick)}
              y={histBottom + 16}
              fontSize={10}
              fill="var(--color-ink-muted)"
              textAnchor="middle"
              className="tabular-nums font-mono"
            >
              {tick.toFixed(1)}
            </text>
          ))}

          {/* Shared X-Axis Label */}
          <text
            x={paddingLeft + plotWidth / 2}
            y={viewBoxHeight - 6}
            fontSize={11}
            fill="var(--color-ink-secondary)"
            textAnchor="middle"
            fontWeight={600}
          >
            Predicted Probability Interval [0.0 &rarr; 1.0]
          </text>
        </svg>

        {/* Hover Tooltip */}
        {hoveredBin && (
          <div
            style={{
              position: 'absolute',
              top: '12px',
              right: '12px',
              backgroundColor: 'var(--color-surface)',
              border: '1px solid var(--color-border-strong)',
              borderRadius: 'var(--radius-md)',
              padding: '10px 14px',
              boxShadow: 'var(--shadow-overlay)',
              fontSize: '12px',
              pointerEvents: 'none',
              zIndex: 10,
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: '4px' }}>
              Bin {hoveredBin.bin_index + 1} [{hoveredBin.lower_bound.toFixed(2)} - {hoveredBin.upper_bound.toFixed(2)}]
            </div>
            <div>Sample Count: <span className="tabular-nums font-mono font-semibold">{hoveredBin.sample_count}</span></div>
            <div>Mean Pred Prob: <span className="tabular-nums font-mono">{hoveredBin.mean_predicted_prob !== null ? hoveredBin.mean_predicted_prob.toFixed(4) : 'N/A'}</span></div>
            <div>Observed Pos Rate: <span className="tabular-nums font-mono">{hoveredBin.observed_positive_rate !== null ? hoveredBin.observed_positive_rate.toFixed(4) : 'N/A'}</span></div>
            {hoveredBin.mean_predicted_prob !== null && hoveredBin.observed_positive_rate !== null && (
              <div style={{ marginTop: '4px', paddingTop: '4px', borderTop: '1px solid var(--color-border-subtle)', color: hoveredBin.observed_positive_rate < hoveredBin.mean_predicted_prob ? 'var(--color-crimson)' : 'var(--color-emerald)' }}>
                {hoveredBin.observed_positive_rate < hoveredBin.mean_predicted_prob ? 'Overconfident (Obs < Pred)' : 'Underconfident (Obs > Pred)'}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default CalibrationChart;
