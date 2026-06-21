import { ProjectModel } from "../../api/backend";
import { HardwareMarker, MAX_RENDER_HARDWARE, PanelMeshModel, SCALE_TO_METERS } from "./models";
import { isFrontLikeComponent } from "./layout";

function hardwareColor(code: string): string {
  const upper = code.toUpperCase();
  if (upper.includes("CAM_LOCK")) {
    return "#2f6e93";
  }
  if (upper.includes("HINGE")) {
    return "#7f4f24";
  }
  if (upper.includes("SLIDE") || upper.includes("TRACK")) {
    return "#495057";
  }
  if (upper.includes("SCREW")) {
    return "#6c757d";
  }
  return "#52796f";
}

function hardwareKind(code: string): HardwareMarker["kind"] {
  const upper = code.toUpperCase();
  if (upper.includes("SCREW") || upper.includes("BOLT")) {
    return "screw";
  }
  if (upper.includes("CAM_LOCK") || upper.includes("CAMLOCK")) {
    return "cam_lock";
  }
  if (upper.includes("SLIDE") || upper.includes("TRACK") || upper.includes("RAIL")) {
    return "slide";
  }
  if (upper.includes("HINGE")) {
    return "hinge";
  }
  return "generic";
}

function hardwareSize(kind: HardwareMarker["kind"]): number {
  if (kind === "screw") {
    return 0.012;
  }
  if (kind === "slide") {
    return 0.016;
  }
  if (kind === "hinge") {
    return 0.015;
  }
  return 0.014;
}

export function buildHardwareMarkers(model: ProjectModel, panels: PanelMeshModel[]): HardwareMarker[] {
  if (!model.hardware || model.hardware.length === 0) {
    return [];
  }

  const panelById = new Map(panels.map((panel) => [panel.id, panel]));
  const componentById = new Map(model.components.map((component) => [component.id, component]));

  const markers: HardwareMarker[] = [];
  for (const line of model.hardware) {
    const qty = Math.max(0, Math.min(line.qty || 0, MAX_RENDER_HARDWARE));
    if (qty === 0) {
      continue;
    }

    const code = line.code || "HARDWARE";
    const mountTargets = line.mount_targets || [];
    if (mountTargets.length === 0) {
      continue;
    }

    const color = hardwareColor(code);
    const kind = hardwareKind(code);
    const size = hardwareSize(kind);

    const targetCount = Math.min(qty, mountTargets.length);
    for (let i = 0; i < targetCount; i += 1) {
      const target = mountTargets[i];
      const targetComponent = componentById.get(target.component_id);
      if (!isFrontLikeComponent(targetComponent)) {
        continue;
      }
      const panel = panelById.get(target.component_id);
      if (!panel) {
        continue;
      }

      const normalOffset = (target.normal_offset_mm ?? 2.0) * SCALE_TO_METERS;
      let normal: [number, number, number] = [0, 0, 1];
      if (target.face === "+x") {
        normal = [1, 0, 0];
      } else if (target.face === "-x") {
        normal = [-1, 0, 0];
      } else if (target.face === "+y") {
        normal = [0, 1, 0];
      } else if (target.face === "-y") {
        normal = [0, -1, 0];
      } else if (target.face === "+z") {
        normal = [0, 0, 1];
      } else if (target.face === "-z") {
        normal = [0, 0, -1];
      }

      const position: [number, number, number] = [
        panel.position[0] + (target.local_x * SCALE_TO_METERS) + normal[0] * normalOffset,
        panel.position[1] + (target.local_y * SCALE_TO_METERS) + normal[1] * normalOffset,
        panel.position[2] + (target.local_z * SCALE_TO_METERS) + normal[2] * normalOffset,
      ];

      markers.push(HardwareMarker.create({
        id: `${line.id || code}-${i}`,
        label: code,
        position,
        kind,
        size,
        color,
      }));
    }
  }

  return markers;
}
