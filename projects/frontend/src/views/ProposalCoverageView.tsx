import { Link } from "react-router-dom";

import { SectionCard } from "../components/SectionCard";
import { useVisionWorkflow } from "../hooks/useVisionWorkflow";

const coverage = [
  {
    item: "Upload image",
    mappedTo: "Workspace / Step 1",
    status: "Implemented",
  },
  {
    item: "Hybrid pipeline orchestration",
    mappedTo: "Backend API client boundary",
    status: "Implemented in integration boundary",
  },
  {
    item: "Adjust topology via sliders",
    mappedTo: "Workspace / Step 3",
    status: "Implemented",
  },
  {
    item: "Infer hardware",
    mappedTo: "Model Inspector / Hardware and warnings",
    status: "Implemented (display)",
  },
  {
    item: "Real-time 3D visualization",
    mappedTo: "Model Inspector / Real-time workspace",
    status: "Implemented (frontend pseudo-3D preview)",
  },
  {
    item: "Generate 2D nesting and blueprints",
    mappedTo: "Fabrication Output route",
    status: "Implemented (frontend package preview)",
  },
];

export function ProposalCoverageView(): React.JSX.Element {
  const { selectedProjectKey } = useVisionWorkflow();
  const workspacePath = selectedProjectKey ? `/projects/${selectedProjectKey}/workspace` : "/";
  const fabricationPath = selectedProjectKey ? `/projects/${selectedProjectKey}/fabrication-output` : "/";

  return (
    <section className="content-grid" style={{ gridTemplateColumns: "1fr" }}>
      <SectionCard
        title="Proposal and diagram coverage"
        subtitle="This route maps proposal diagrams to implemented frontend surfaces"
      >
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left", borderBottom: "1px solid #d5cfbe", padding: "0.35rem" }}>Diagram feature</th>
              <th style={{ textAlign: "left", borderBottom: "1px solid #d5cfbe", padding: "0.35rem" }}>Frontend mapping</th>
              <th style={{ textAlign: "left", borderBottom: "1px solid #d5cfbe", padding: "0.35rem" }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {coverage.map((row) => (
              <tr key={row.item}>
                <td style={{ borderBottom: "1px solid #ebe6d8", padding: "0.35rem" }}>{row.item}</td>
                <td style={{ borderBottom: "1px solid #ebe6d8", padding: "0.35rem" }}>{row.mappedTo}</td>
                <td style={{ borderBottom: "1px solid #ebe6d8", padding: "0.35rem" }}>{row.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p style={{ marginTop: "0.8rem" }}>
          For full workflow execution, continue in <Link to={workspacePath}>Workspace</Link> and inspect outputs in{" "}
          <Link to={fabricationPath}>Fabrication Output</Link>.
        </p>
      </SectionCard>
    </section>
  );
}
