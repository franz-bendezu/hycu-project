import { ProjectModel } from "../../api/backend";
import { PanelMeshModel, PanelTone } from "./models";

function requireFinite(value: number, fieldName: string): number {
  if (!Number.isFinite(value)) {
    throw new Error(`Invalid ${fieldName}: expected finite number`);
  }
  return value;
}

function requirePositive(value: number, fieldName: string): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`Invalid ${fieldName}: expected positive finite number`);
  }
  return value;
}

function toneForKind(kind: ProjectModel["components"][number]["kind"]): PanelTone {
  if (kind === "back_panel") {
    return "back";
  }
  if (kind === "shelf") {
    return "shelf";
  }
  if (kind === "divider_panel") {
    return "divider";
  }
  if (kind === "door_panel") {
    return "door";
  }
  if (kind === "drawer_front") {
    return "drawer";
  }
  return "panel";
}

export function isFrontLikeComponent(component: ProjectModel["components"][number] | undefined): boolean {
  if (!component) {
    return false;
  }
  if (component.category === "front") {
    return true;
  }
  return (
    component.kind === "front_panel"
    || component.kind === "door_panel"
    || component.kind === "drawer_front"
  );
}

export function buildPanelMeshes(model: ProjectModel): PanelMeshModel[] {
  return model.components.map((component) => {
    const size: [number, number, number] = [
      requirePositive(component.width, `${component.id}.width`),
      requirePositive(component.height, `${component.id}.height`),
      requirePositive(component.depth, `${component.id}.depth`),
    ];
    const position: [number, number, number] = [
      requireFinite(component.pos_x, `${component.id}.pos_x`),
      requireFinite(component.pos_y, `${component.id}.pos_y`),
      requireFinite(component.pos_z, `${component.id}.pos_z`),
    ];
    const rotation: [number, number, number] = [
      requireFinite(component.rot_x_deg, `${component.id}.rot_x_deg`),
      requireFinite(component.rot_y_deg, `${component.id}.rot_y_deg`),
      requireFinite(component.rot_z_deg, `${component.id}.rot_z_deg`),
    ];

    return PanelMeshModel.create({
      id: component.id,
      label: component.kind,
      size,
      position,
      rotation,
      tone: toneForKind(component.kind),
    });
  });
}
