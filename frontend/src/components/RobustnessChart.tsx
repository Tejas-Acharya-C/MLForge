import React from 'react';
import { RobustnessCurve } from '../types/api';

interface RobustnessChartProps {
  curves: RobustnessCurve[];
}

export const RobustnessChart: React.FC<RobustnessChartProps> = ({ curves }) => {
  if (!curves || curves.length === 0) {
    return <div style={{ color: 'var(--color-ink-muted)', fontSize: '13px' }}>No robustness curves available.</div>;
  }

  const viewBoxWidth = 560;
  const viewBoxHeight = 240;
  const paddingLeft = 55;
  const paddingRight = 35;
  const paddingTop = 25;
  const paddingBottom = 45;

  const plotWidth = viewBoxWidth - paddingLeft - paddingRight;
  const plotHeight = viewBoxHeight - paddingTop - paddingBottom;

  // Determine maximum severity present in curves
  let maxSevFound = 0.5;
  curves.forEach((c) => {
    c.points.forEach((p) => {
      if (p.severity_level > maxSevFound) maxSevFound = p.severity_level;
    });
  });
  const maxSeverity = Math.max(0.5, Math.ceil(maxSevFound * 10) / 10);

  // X is severity level (0.0 to maxSeverity), Y is retained ratio (0.0 to 1.1)
  const scaleX = (sev: number) => paddingLeft + (sev / maxSeverity) * plotWidth;
  const scaleY = (ret: number) => viewBoxHeight - paddingBottom - (ret / 1.1) * plotHeight;

  // Scientific palette
  const colors = [
    'var(--color-accent-primary)',
    '#059669', // Emerald
    '#d97706', // Amber
    '#7c3aed', // Violet
  ];

  const xTicks = [0, 0.1, 0.2, 0.3, 0.4, 0.5].filter((s) => s <= maxSeverity);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', width: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
        {curves.map((c, i) => (
          <div key={`legend-${i}`} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: colors[i % colors.length] }} />
            <span style={{ textTransform: 'capitalize', fontWeight: 500 }}>{c.perturbation_type.replace(/_/g, ' ')}</span>
            {c.area_under_degradation_curve !== undefined && c.area_under_degradation_curve !== null && (
              <span className="tabular-nums font-mono badge badge-neutral" style={{ padding: '1px 6px', fontSize: '11px' }}>
                AURDC: {c.area_under_degradation_curve.toFixed(3)}
              </span>
            )}
          </div>
        ))}
      </div>

      <div style={{ width: '100%', overflowX: 'auto' }}>
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
          aria-label="Perturbation robustness stress-testing curves showing retention ratios across corruption severities"
        >
          {/* Y-axis gridlines: 0.0, 0.25, 0.50, 0.75, 1.0 */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((tick) => (
            <g key={`y-tick-${tick}`}>
              <line
                x1={paddingLeft}
                y1={scaleY(tick)}
                x2={viewBoxWidth - paddingRight}
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
                {(tick * 100).toFixed(0)}%
              </text>
            </g>
          ))}

          {/* X-axis ticks */}
          {xTicks.map((sev) => (
            <g key={`x-tick-${sev}`}>
              <line
                x1={scaleX(sev)}
                y1={scaleY(0)}
                x2={scaleX(sev)}
                y2={scaleY(1.05)}
                stroke="var(--color-border-subtle)"
                strokeDasharray="3 3"
              />
              <text
                x={scaleX(sev)}
                y={viewBoxHeight - paddingBottom + 16}
                fontSize={10}
                fill="var(--color-ink-muted)"
                textAnchor="middle"
                className="tabular-nums font-mono"
              >
                {(sev * 100).toFixed(0)}%
              </text>
            </g>
          ))}

          {/* 100% Retention Reference Line */}
          <line
            x1={scaleX(0)}
            y1={scaleY(1.0)}
            x2={scaleX(maxSeverity)}
            y2={scaleY(1.0)}
            stroke="var(--color-border-strong)"
            strokeWidth={1}
            strokeDasharray="4 4"
          />

          {/* Plot each perturbation curve */}
          {curves.map((curve, cIdx) => {
            const color = colors[cIdx % colors.length];
            const sortedPts = [...curve.points].sort((a, b) => a.severity_level - b.severity_level);
            const validPts = sortedPts.filter((p) => p.is_usable && p.retained_ratio !== null);

            const polylinePts: string[] = [];
            polylinePts.push(`${scaleX(0)},${scaleY(1.0)}`);
            validPts.forEach((p) => {
              polylinePts.push(`${scaleX(p.severity_level)},${scaleY(p.retained_ratio!)}`);
            });

            return (
              <g key={`curve-${cIdx}`}>
                <polyline
                  points={polylinePts.join(' ')}
                  fill="none"
                  stroke={color}
                  strokeWidth={2.5}
                />
                {/* Origin point */}
                <circle cx={scaleX(0)} cy={scaleY(1.0)} r={3.5} fill={color} />
                {/* Point dots */}
                {validPts.map((p, pIdx) => (
                  <circle
                    key={`pt-${cIdx}-${pIdx}`}
                    cx={scaleX(p.severity_level)}
                    cy={scaleY(p.retained_ratio!)}
                    r={5}
                    fill={color}
                    stroke="var(--color-surface)"
                    strokeWidth={1.5}
                  >
                    <title>
                      {`${curve.perturbation_type}: Severity ${(p.severity_level * 100).toFixed(0)}% -> Retained ${(p.retained_ratio! * 100).toFixed(1)}%`}
                    </title>
                  </circle>
                ))}
              </g>
            );
          })}

          {/* Axis Titles */}
          <text
            x={paddingLeft + plotWidth / 2}
            y={viewBoxHeight - 8}
            fontSize={11}
            fill="var(--color-ink-secondary)"
            textAnchor="middle"
            fontWeight={600}
          >
            Perturbation Severity Level
          </text>
          <text
            x={-(paddingTop + plotHeight / 2)}
            y={14}
            fontSize={11}
            fill="var(--color-ink-secondary)"
            textAnchor="middle"
            fontWeight={600}
            transform="rotate(-90)"
          >
            Performance Retention Ratio
          </text>
        </svg>
      </div>
    </div>
  );
};

export default RobustnessChart;
