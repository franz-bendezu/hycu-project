import { WorkspaceFlow } from "../components/WorkspaceFlow";

export function WorkspaceView(): React.JSX.Element {
  return (
    <section
      className="content-grid"
      style={{ gridTemplateColumns: "1fr" }}
      aria-label="Workspace route content"
    >
      <WorkspaceFlow />
    </section>
  );
}
