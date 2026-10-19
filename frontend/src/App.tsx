import React, { useState, useEffect, useCallback } from 'react';
import LabOverviewScreen from './screens/LabOverview';
import NewEvaluationScreen from './screens/NewEvaluation';
import EvaluationResultsScreen from './screens/EvaluationResults';
import ReliabilityAnalysisScreen from './screens/ReliabilityAnalysis';
import RunDetailsScreen from './screens/RunDetails';
import AcceptanceReportScreen from './screens/AcceptanceReport';
import { api } from './api/client';

interface RouteState {
  screen: 'overview' | 'new_evaluation' | 'results' | 'reliability' | 'details' | 'acceptance' | 'not_found';
  runId: string | null;
}

function parseHash(hash: string): RouteState {
  const clean = hash.replace(/^#\/?/, '').trim();
  if (!clean || clean === 'runs' || clean === 'overview') {
    return { screen: 'overview', runId: null };
  }
  if (clean === 'runs/new' || clean === 'new_evaluation' || clean === 'new') {
    return { screen: 'new_evaluation', runId: null };
  }

  // Matches runs/:runId/:screen
  const parts = clean.split('/');
  if (parts[0] === 'runs' && parts.length >= 2) {
    const runId = parts[1];
    const sub = parts[2] || 'results';
    if (sub === 'results') return { screen: 'results', runId };
    if (sub === 'reliability' || sub === 'robustness') return { screen: 'reliability', runId };
    if (sub === 'artifacts' || sub === 'details') return { screen: 'details', runId };
    if (sub === 'governance' || sub === 'acceptance' || sub === 'release') return { screen: 'acceptance', runId };
  }

  // Fallback for legacy screen names in hash
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

export const App: React.FC = () => {
  const [route, setRoute] = useState<RouteState>(() => parseHash(window.location.hash));
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');

  // Handle URL hash changes (browser back/forward & direct links)
  useEffect(() => {
    const handleHashChange = () => {
      setRoute(parseHash(window.location.hash));
      window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    window.addEventListener('hashchange', handleHashChange);
    window.addEventListener('popstate', handleHashChange);

    // If no initial hash, set default #/runs
    if (!window.location.hash) {
      window.location.hash = '#/runs';
    }

    return () => {
      window.removeEventListener('hashchange', handleHashChange);
      window.removeEventListener('popstate', handleHashChange);
    };
  }, []);

  // Check backend health on mount and periodically
  useEffect(() => {
    let mounted = true;
    const checkHealth = async () => {
      try {
        const res = await api.checkHealth();
        if (mounted) {
          if (res.status === 'ok' || res.status === 'ready') {
            setBackendStatus('connected');
          } else {
            setBackendStatus('disconnected');
          }
        }
      } catch {
        if (mounted) setBackendStatus('disconnected');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleNavigate = useCallback((screen: string, runId?: string) => {
    const targetRunId = runId || route.runId;
    const newHash = buildHash(screen, targetRunId);
    if (window.location.hash !== newHash) {
      window.location.hash = newHash;
    } else {
      setRoute(parseHash(newHash));
    }
  }, [route.runId]);

  const { screen: currentScreen, runId: activeRunId } = route;

  // Breadcrumb generation
  const getBreadcrumb = () => {
    if (currentScreen === 'overview') {
      return <span>Run History</span>;
    }
    if (currentScreen === 'new_evaluation') {
      return (
        <>
          <a href="#/runs" style={{ color: 'var(--color-ink-secondary)', textDecoration: 'none' }}>
            Run History
          </a>
          <span style={{ color: 'var(--color-ink-muted)' }}>/</span>
          <span>Configure Evaluation</span>
        </>
      );
    }
    if (activeRunId) {
      const screenLabel =
        currentScreen === 'results'
          ? 'Performance Results'
          : currentScreen === 'reliability'
          ? 'Reliability & Robustness'
          : currentScreen === 'details'
          ? 'Artifact Ledger'
          : currentScreen === 'acceptance'
          ? 'Release Governance'
          : currentScreen;

      return (
        <>
          <a href="#/runs" style={{ color: 'var(--color-ink-secondary)', textDecoration: 'none' }}>
            Run History
          </a>
          <span style={{ color: 'var(--color-ink-muted)' }}>/</span>
          <span className="mono" style={{ fontWeight: 600 }}>{activeRunId}</span>
          <span style={{ color: 'var(--color-ink-muted)' }}>/</span>
          <span>{screenLabel}</span>
        </>
      );
    }
    return <span>Run History</span>;
  };

  return (
    <div className="layout">
      {/* Precision Scientific Header */}
      <header className="header" role="banner">
        <div className="header-inner">
          <div className="flex items-center gap-3">
            <a
              href="#/runs"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                textDecoration: 'none',
                color: 'inherit',
              }}
              aria-label="MLForge Laboratory Home"
            >
              <div
                style={{
                  width: '20px',
                  height: '20px',
                  background: 'var(--color-ink)',
                  borderRadius: '3px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--color-surface)',
                  fontSize: '12px',
                  fontWeight: 800,
                  letterSpacing: '-0.05em',
                }}
              >
                F
              </div>
              <span className="logo">MLForge</span>
            </a>
            <span
              style={{
                fontSize: '11px',
                color: 'var(--color-ink-muted)',
                letterSpacing: '0.04em',
                textTransform: 'uppercase',
                borderLeft: '1px solid var(--color-border-subtle)',
                paddingLeft: '10px',
              }}
            >
              Reliability Lab
            </span>
          </div>

          {/* Navigation Tabs */}
          <nav className="nav-tabs" aria-label="Main Navigation">
            <button
              className={`nav-tab ${currentScreen === 'overview' ? 'active' : ''}`}
              onClick={() => handleNavigate('overview')}
              aria-current={currentScreen === 'overview' ? 'page' : undefined}
            >
              Run History
            </button>
            <button
              className={`nav-tab ${currentScreen === 'new_evaluation' ? 'active' : ''}`}
              onClick={() => handleNavigate('new_evaluation')}
              aria-current={currentScreen === 'new_evaluation' ? 'page' : undefined}
            >
              + New Evaluation
            </button>
            {activeRunId && (
              <>
                <button
                  className={`nav-tab ${currentScreen === 'results' ? 'active' : ''}`}
                  onClick={() => handleNavigate('results', activeRunId)}
                  aria-current={currentScreen === 'results' ? 'page' : undefined}
                >
                  Results
                </button>
                <button
                  className={`nav-tab ${currentScreen === 'reliability' ? 'active' : ''}`}
                  onClick={() => handleNavigate('reliability', activeRunId)}
                  aria-current={currentScreen === 'reliability' ? 'page' : undefined}
                >
                  Reliability & Slices
                </button>
                <button
                  className={`nav-tab ${currentScreen === 'details' ? 'active' : ''}`}
                  onClick={() => handleNavigate('details', activeRunId)}
                  aria-current={currentScreen === 'details' ? 'page' : undefined}
                >
                  Artifacts
                </button>
                <button
                  className={`nav-tab ${currentScreen === 'acceptance' ? 'active' : ''}`}
                  onClick={() => handleNavigate('acceptance', activeRunId)}
                  aria-current={currentScreen === 'acceptance' ? 'page' : undefined}
                >
                  Release Gate
                </button>
              </>
            )}
          </nav>

          {/* Backend Status & Context Indicator */}
          <div className="flex items-center gap-3">
            {activeRunId && (
              <span className="mono" style={{ fontSize: '12px', color: 'var(--color-ink-muted)' }}>
                Active: <span style={{ color: 'var(--color-ink)', fontWeight: 600 }}>{activeRunId}</span>
              </span>
            )}
            <div className="flex items-center gap-2" style={{ fontSize: '12px' }}>
              <span
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background:
                    backendStatus === 'connected'
                      ? 'var(--color-emerald)'
                      : backendStatus === 'checking'
                      ? 'var(--color-amber)'
                      : 'var(--color-crimson)',
                  display: 'inline-block',
                }}
                aria-hidden="true"
              />
              <span style={{ color: 'var(--color-ink-muted)', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                {backendStatus === 'connected' ? 'Core 8000' : backendStatus === 'checking' ? 'Connecting...' : 'Core Offline'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Breadcrumb Bar */}
      <div
        style={{
          backgroundColor: 'var(--color-canvas-subtle)',
          borderBottom: '1px solid var(--color-border-subtle)',
          padding: '8px 24px',
          fontSize: '12px',
        }}
        aria-label="Breadcrumb Navigation"
      >
        <div style={{ maxWidth: '1400px', margin: '0 auto', display: 'flex', alignItems: 'center', gap: '8px' }}>
          {getBreadcrumb()}
        </div>
      </div>

      {/* Main Content Workspace */}
      <main className="content" role="main">
        {backendStatus === 'disconnected' && (
          <div className="card" style={{ borderColor: 'var(--color-crimson)', marginBottom: '24px', background: 'var(--color-surface)' }}>
            <div className="flex justify-between items-center flex-wrap gap-3">
              <div>
                <strong style={{ color: 'var(--color-crimson)', fontSize: '14px' }}>Backend API Core Offline</strong>
                <p style={{ margin: '4px 0 0 0', fontSize: '13px', color: 'var(--color-ink-muted)' }}>
                  Unable to connect to the MLForge local API server at <code>http://127.0.0.1:8000</code>. Ensure the server is running via <code>python -m mlforge serve</code>.
                </p>
              </div>
              <button className="btn btn-secondary" onClick={() => window.location.reload()}>
                Retry Connection
              </button>
            </div>
          </div>
        )}

        {currentScreen === 'overview' && (
          <LabOverviewScreen onNavigate={handleNavigate} />
        )}

        {currentScreen === 'new_evaluation' && (
          <NewEvaluationScreen onNavigate={handleNavigate} />
        )}

        {currentScreen === 'results' && (
          activeRunId ? (
            <EvaluationResultsScreen runId={activeRunId} onNavigate={handleNavigate} />
          ) : (
            <div className="empty-state">
              <div className="empty-state-title">No Evaluation Run Selected</div>
              <p className="empty-state-text">
                Please select a run from the history ledger to inspect performance metrics.
              </p>
              <button className="btn btn-primary" onClick={() => handleNavigate('overview')}>
                Browse Run History
              </button>
            </div>
          )
        )}

        {currentScreen === 'reliability' && (
          activeRunId ? (
            <ReliabilityAnalysisScreen runId={activeRunId} onNavigate={handleNavigate} />
          ) : (
            <div className="empty-state">
              <div className="empty-state-title">No Evaluation Run Selected</div>
              <p className="empty-state-text">
                Select an evaluation run to analyze probability calibration and stress profiles.
              </p>
              <button className="btn btn-primary" onClick={() => handleNavigate('overview')}>
                Browse Run History
              </button>
            </div>
          )
        )}

        {currentScreen === 'details' && (
          activeRunId ? (
            <RunDetailsScreen runId={activeRunId} onNavigate={handleNavigate} />
          ) : (
            <div className="empty-state">
              <div className="empty-state-title">No Evaluation Run Selected</div>
              <p className="empty-state-text">
                Select an evaluation run to view persisted artifacts and audit bundles.
              </p>
              <button className="btn btn-primary" onClick={() => handleNavigate('overview')}>
                Browse Run History
              </button>
            </div>
          )
        )}

        {currentScreen === 'acceptance' && (
          activeRunId ? (
            <AcceptanceReportScreen runId={activeRunId} onNavigate={handleNavigate} />
          ) : (
            <div className="empty-state">
              <div className="empty-state-title">No Evaluation Run Selected</div>
              <p className="empty-state-text">
                Select an evaluation run to evaluate policy criteria and cryptographic release certificates.
              </p>
              <button className="btn btn-primary" onClick={() => handleNavigate('overview')}>
                Browse Run History
              </button>
            </div>
          )
        )}
      </main>

      {/* Footer Notice */}
      <footer
        style={{
          borderTop: '1px solid var(--color-border-subtle)',
          padding: '16px 24px',
          fontSize: '12px',
          color: 'var(--color-ink-muted)',
          backgroundColor: 'var(--color-surface)',
        }}
      >
        <div style={{ maxWidth: '1400px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
          <div>
            MLForge local evaluation laboratory &bull; Tabular classification reliability & verification &bull; Strict zero-leakage protocol
          </div>
          <div className="mono">
            Local Core v0.5.0 &bull; Ed25519 Cryptographic Trust
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
