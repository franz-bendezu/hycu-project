import { ModelInspector } from "../components/ModelInspector";
import { TopologyAdjustmentsPanel } from "../components/TopologyAdjustmentsPanel";

export function ModelView(): React.JSX.Element {
  return (
    <section
      className="model-editor-grid"
      aria-label="Model route content"
    >
      <div className="model-editor-controls">
        <TopologyAdjustmentsPanel />
      </div>
      <div className="model-editor-preview">
        <ModelInspector />
      </div>
    </section>
  );
}