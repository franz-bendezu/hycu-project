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
        <p style={{ margin: 0, letterSpacing: "0.08em", fontSize: "0.8rem", textTransform: "uppercase" }}>
          Vision to Blueprint
        </p>
        <h1>Modular Furniture Design Workspace</h1>
        <p className="highlight">
          Frontend is modularized into views and components with routing, while backend remains the orchestration boundary.
        </p>
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap" }}>
          <StatusPill label={`Backend: ${backendStatus}`} tone={tone} />
          {activeAction ? <StatusPill label={activeAction} tone="warning" /> : null}
          {error ? (
            <button className="secondary" onClick={clearError}>
              Clear Error
            </button>
          ) : null}
        </div>

      </header>

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
