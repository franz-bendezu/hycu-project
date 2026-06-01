import { useEffect } from "react";
import { Outlet, Route, Routes, useParams } from "react-router-dom";

import { AppLayout } from "./components/AppLayout";
import { ProjectNavigation } from "./components/ProjectNavigation";
import { WorkflowProvider } from "./hooks/useVisionWorkflow";
import { useVisionWorkflow } from "./hooks/useVisionWorkflow";
import { FabricationOutputView } from "./views/FabricationOutputView";
import { HomeView } from "./views/HomeView";
import { NotFoundView } from "./views/NotFoundView";
import { ProjectStatusView } from "./views/ProjectStatusView";
import { ProposalCoverageView } from "./views/ProposalCoverageView";
import { WorkspaceView } from "./views/WorkspaceView";

function ProjectScope(): React.JSX.Element {
  const { projectKey = "" } = useParams();
  const { selectProject } = useVisionWorkflow();

  useEffect(() => {
    if (projectKey) {
      selectProject(projectKey);
    }
  }, [projectKey, selectProject]);

  return (
    <>
      <ProjectNavigation />
      <Outlet />
    </>
  );
}

export function App(): React.JSX.Element {
  return (
    <WorkflowProvider>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<HomeView />} />
          <Route path="/projects/:projectKey" element={<ProjectScope />}>
            <Route index element={<ProjectStatusView />} />
            <Route path="workspace" element={<WorkspaceView />} />
            <Route path="proposal-coverage" element={<ProposalCoverageView />} />
            <Route path="fabrication-output" element={<FabricationOutputView />} />
          </Route>
          <Route path="*" element={<NotFoundView />} />
        </Route>
      </Routes>
    </WorkflowProvider>
  );
}
