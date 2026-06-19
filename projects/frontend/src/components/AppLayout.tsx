import { Outlet } from "react-router-dom";

import { useVisionWorkflow } from "../hooks/useVisionWorkflow";
import { StatusPill } from "./StatusPill";

export function AppLayout(): React.JSX.Element {
  const { backendStatus, activeAction, error, clearError } = useVisionWorkflow();

  const tone =
    backendStatus === "ok" ? "ok" : backendStatus === "down" ? "error" : "neutral";

  return (
    <main className="app-shell">
      <header className="app-header">
        <p className="app-kicker">
          Vision to Blueprint
        </p>
        <h1 className="app-title">Modular Furniture Design Workspace</h1>
        <p className="highlight app-highlight">
          Frontend is modularized into views and components with routing, while backend remains the orchestration boundary.
        </p>
        <div className="app-status-row">
          <StatusPill label={`Backend: ${backendStatus}`} tone={tone} />
          {activeAction ? <StatusPill label={activeAction} tone="warning" /> : null}
          {error ? (
            <button className="secondary" onClick={clearError}>
              Clear Error
            </button>
          ) : null}
        </div>

      </header>

      {activeAction ? (
        <section className="global-feedback" role="status" aria-live="polite">
          <div className="global-feedback-row">
            <span className="global-feedback-spinner" aria-hidden="true" />
            <p>{activeAction}</p>
          </div>
          <div className="global-feedback-progress" aria-hidden="true" />
        </section>
      ) : null}

      <Outlet />

      {error ? (
        <section className="panel" style={{ marginTop: "0.8rem" }}>
          <h3 style={{ marginTop: 0 }}>Backend Error</h3>
          <p style={{ color: "#9f1f1f", marginBottom: 0 }}>{error}</p>
        </section>
      ) : null}
    </main>
  );
}
