import { ModelInspector } from "../components/ModelInspector";
import { WorkspaceFlow } from "../components/WorkspaceFlow";

export function WorkspaceView(): React.JSX.Element {
  return (
    <section className="content-grid" aria-label="Workspace route content">
      <WorkspaceFlow />
      <ModelInspector />
    </section>
  );
}
