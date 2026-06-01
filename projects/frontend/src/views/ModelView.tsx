import { ModelInspector } from "../components/ModelInspector";
import { TopologyAdjustmentsPanel } from "../components/TopologyAdjustmentsPanel";

export function ModelView(): React.JSX.Element {
  return (
    <section
      className="content-grid"
      style={{ gridTemplateColumns: "1fr" }}
      aria-label="Model route content"
    >
      <TopologyAdjustmentsPanel />
      <ModelInspector />
    </section>
  );
}