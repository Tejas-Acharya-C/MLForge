import React, { useState } from 'react';

interface ConfusionMatrixProps {
  matrix: number[][];
  classLabels: string[];
  title?: string;
}

export type ConfusionMatrixViewMode = 'raw' | 'recall' | 'precision';

export const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({ matrix, classLabels, title }) => {
  const [viewMode, setViewMode] = useState<ConfusionMatrixViewMode>('raw');

  if (!matrix || matrix.length === 0) {
    return <div style={{ color: 'var(--color-ink-muted)', fontSize: '13px' }}>No matrix data available.</div>;
  }

  const numCols = matrix[0]?.length || 0;

  // Calculate totals, row sums, and column sums
  const total = matrix.reduce((acc, row) => acc + row.reduce((rAcc, val) => rAcc + val, 0), 0);
  const rowSums = matrix.map((row) => row.reduce((acc, v) => acc + v, 0));
  const colSums = Array.from({ length: numCols }, (_, c) =>
    matrix.reduce((acc, row) => acc + (row[c] || 0), 0)
  );

  let maxVal = 1;
  matrix.forEach((row) => {
    row.forEach((v) => {
      if (v > maxVal) maxVal = v;
    });
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
        {title && <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-ink-secondary)' }}>{title}</div>}
        <div style={{ display: 'flex', gap: '4px', backgroundColor: 'var(--color-stone-bg)', padding: '2px', borderRadius: 'var(--radius-sm)' }}>
          <button
            type="button"
            className={`btn ${viewMode === 'raw' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '3px 8px', fontSize: '11px', height: '24px', lineHeight: 1 }}
            onClick={() => setViewMode('raw')}
          >
            Raw Counts
          </button>
          <button
            type="button"
            className={`btn ${viewMode === 'recall' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '3px 8px', fontSize: '11px', height: '24px', lineHeight: 1 }}
            onClick={() => setViewMode('recall')}
            title="Row Normalized: True Class Recall"
          >
            Row % (Recall)
          </button>
          <button
            type="button"
            className={`btn ${viewMode === 'precision' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '3px 8px', fontSize: '11px', height: '24px', lineHeight: 1 }}
            onClick={() => setViewMode('precision')}
            title="Column Normalized: Predicted Class Precision"
          >
            Col % (Precision)
          </button>
        </div>
      </div>

      <div style={{ overflowX: 'auto', border: '1px solid var(--color-border-subtle)', borderRadius: 'var(--radius-md)', padding: '12px', background: 'var(--color-surface)' }}>
        <div style={{ display: 'grid', gridTemplateColumns: `auto repeat(${classLabels.length}, minmax(80px, 1fr))`, gap: '4px', alignItems: 'center' }}>
          {/* Header Row (Predicted) */}
          <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', textAlign: 'right', paddingRight: '8px' }}>
            True \ Pred
          </div>
          {classLabels.map((lbl, idx) => (
            <div
              key={`col-${idx}`}
              style={{
                fontSize: '11px',
                fontWeight: 600,
                color: 'var(--color-ink-secondary)',
                textAlign: 'center',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                padding: '4px',
              }}
              title={`Predicted: ${lbl}`}
            >
              {lbl}
            </div>
          ))}

          {/* Matrix Rows */}
          {matrix.map((row, rIdx) => {
            const rSum = rowSums[rIdx] || 1;
            return (
              <React.Fragment key={`row-${rIdx}`}>
                <div
                  style={{
                    fontSize: '11px',
                    fontWeight: 600,
                    color: 'var(--color-ink-secondary)',
                    textAlign: 'right',
                    paddingRight: '8px',
                    whiteSpace: 'nowrap',
                  }}
                  title={`True: ${classLabels[rIdx]}`}
                >
                  {classLabels[rIdx]}
                </div>
                {row.map((count, cIdx) => {
                  const isDiagonal = rIdx === cIdx;
                  const cSum = colSums[cIdx] || 1;

                  let primaryText = String(count);
                  let secondaryText = total > 0 ? `${((count / total) * 100).toFixed(1)}%` : '0%';
                  let ratio = maxVal > 0 ? count / maxVal : 0;

                  if (viewMode === 'recall') {
                    const rowPct = (count / rSum) * 100;
                    primaryText = `${rowPct.toFixed(1)}%`;
                    secondaryText = `n=${count}/${rSum}`;
                    ratio = rowPct / 100;
                  } else if (viewMode === 'precision') {
                    const colPct = cSum > 0 ? (count / cSum) * 100 : 0;
                    primaryText = `${colPct.toFixed(1)}%`;
                    secondaryText = `n=${count}/${cSum}`;
                    ratio = colPct / 100;
                  }

                  // Neutral dark blue for diagonal, subtle dark red for off-diagonal
                  const bg = isDiagonal
                    ? `rgba(37, 99, 235, ${Math.max(0.08, ratio * 0.45)})`
                    : count > 0
                    ? `rgba(220, 38, 38, ${Math.max(0.06, ratio * 0.35)})`
                    : 'var(--color-canvas-subtle)';

                  return (
                    <div
                      key={`cell-${rIdx}-${cIdx}`}
                      style={{
                        minHeight: '52px',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        backgroundColor: bg,
                        border: '1px solid var(--color-border-subtle)',
                        borderRadius: 'var(--radius-sm)',
                        padding: '4px',
                        cursor: 'default',
                      }}
                      title={`True: ${classLabels[rIdx]}, Pred: ${classLabels[cIdx]} — Count: ${count} (Row Recall: ${((count / rSum) * 100).toFixed(1)}%, Col Precision: ${cSum > 0 ? ((count / cSum) * 100).toFixed(1) : '0'}%)`}
                    >
                      <span className="tabular-nums font-mono" style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-ink-primary)' }}>
                        {primaryText}
                      </span>
                      <span className="tabular-nums font-mono" style={{ fontSize: '10px', color: 'var(--color-ink-muted)', opacity: 0.85 }}>
                        {secondaryText}
                      </span>
                    </div>
                  );
                })}
              </React.Fragment>
            );
          })}
        </div>
      </div>
      <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', fontStyle: 'italic' }}>
        {viewMode === 'raw' && 'Showing absolute sample counts across cross-validation out-of-fold predictions.'}
        {viewMode === 'recall' && 'Row normalized (True class recall): proportion of actual class instances correctly identified.'}
        {viewMode === 'precision' && 'Column normalized (Predicted class precision): proportion of positive calls that were true positives.'}
      </div>
    </div>
  );
};

export default ConfusionMatrix;
