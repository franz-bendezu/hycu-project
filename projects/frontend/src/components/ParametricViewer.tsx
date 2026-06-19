import { useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

import { ProjectModel, ValidateResponse } from "../api/backend";

type ParametricViewerProps = {
  model: ProjectModel | null;
  validation: ValidateResponse | null;
};

type PanelMeshModel = {
  id: string;
  label: string;
  size: [number, number, number];
  position: [number, number, number];
  tone: "panel" | "shelf" | "back" | "door" | "drawer" | "divider" | "fallback";
};

enum PanelSemantic {
  LEFT_LEG = "left_leg",
  RIGHT_LEG = "right_leg",
  LEFT_SIDE = "left_side",
  RIGHT_SIDE = "right_side",
  TOP = "top",
  BOTTOM = "bottom",
  BACK = "back",
  DIVIDER = "divider",
  DOOR = "door",
  DRAWER = "drawer",
  FRONT = "front",
  SHELF = "shelf",
  FALLBACK = "fallback",
}

enum ViewerComponentKind {
  LEFT_SIDE = "left_side",
  RIGHT_SIDE = "right_side",
  TOP_PANEL = "top_panel",
  BOTTOM_PANEL = "bottom_panel",
  BACK_PANEL = "back_panel",
  SHELF = "shelf",
  DIVIDER_PANEL = "divider_panel",
  FRONT_PANEL = "front_panel",
  DOOR_PANEL = "door_panel",
  DRAWER_FRONT = "drawer_front",
  LEFT_LEG_FRONT = "left_leg_front",
  RIGHT_LEG_FRONT = "right_leg_front",
  LEFT_LEG_BACK = "left_leg_back",
  RIGHT_LEG_BACK = "right_leg_back",
}

const SCALE_TO_METERS = 0.001;

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function dimensionOrFallback(value: number, fallback: number): number {
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function parseViewerComponentKind(value: string): ViewerComponentKind | null {
  switch (value) {
    case ViewerComponentKind.LEFT_SIDE:
      return ViewerComponentKind.LEFT_SIDE;
    case ViewerComponentKind.RIGHT_SIDE:
      return ViewerComponentKind.RIGHT_SIDE;
    case ViewerComponentKind.TOP_PANEL:
      return ViewerComponentKind.TOP_PANEL;
    case ViewerComponentKind.BOTTOM_PANEL:
      return ViewerComponentKind.BOTTOM_PANEL;
    case ViewerComponentKind.BACK_PANEL:
      return ViewerComponentKind.BACK_PANEL;
    case ViewerComponentKind.SHELF:
      return ViewerComponentKind.SHELF;
    case ViewerComponentKind.DIVIDER_PANEL:
      return ViewerComponentKind.DIVIDER_PANEL;
    case ViewerComponentKind.FRONT_PANEL:
      return ViewerComponentKind.FRONT_PANEL;
    case ViewerComponentKind.DOOR_PANEL:
      return ViewerComponentKind.DOOR_PANEL;
    case ViewerComponentKind.DRAWER_FRONT:
      return ViewerComponentKind.DRAWER_FRONT;
    case ViewerComponentKind.LEFT_LEG_FRONT:
      return ViewerComponentKind.LEFT_LEG_FRONT;
    case ViewerComponentKind.RIGHT_LEG_FRONT:
      return ViewerComponentKind.RIGHT_LEG_FRONT;
    case ViewerComponentKind.LEFT_LEG_BACK:
      return ViewerComponentKind.LEFT_LEG_BACK;
    case ViewerComponentKind.RIGHT_LEG_BACK:
      return ViewerComponentKind.RIGHT_LEG_BACK;
    default:
      return null;
  }
}

function classifySemantic(componentKind: string): PanelSemantic {
  const kind = parseViewerComponentKind(componentKind);

  if (kind === ViewerComponentKind.LEFT_LEG_FRONT || kind === ViewerComponentKind.LEFT_LEG_BACK) {
    return PanelSemantic.LEFT_LEG;
  }
  if (kind === ViewerComponentKind.RIGHT_LEG_FRONT || kind === ViewerComponentKind.RIGHT_LEG_BACK) {
    return PanelSemantic.RIGHT_LEG;
  }
  if (kind === ViewerComponentKind.LEFT_SIDE) {
    return PanelSemantic.LEFT_SIDE;
  }
  if (kind === ViewerComponentKind.RIGHT_SIDE) {
    return PanelSemantic.RIGHT_SIDE;
  }
  if (kind === ViewerComponentKind.TOP_PANEL) {
    return PanelSemantic.TOP;
  }
  if (kind === ViewerComponentKind.BOTTOM_PANEL) {
    return PanelSemantic.BOTTOM;
  }
  if (kind === ViewerComponentKind.BACK_PANEL) {
    return PanelSemantic.BACK;
  }
  if (kind === ViewerComponentKind.SHELF) {
    return PanelSemantic.SHELF;
  }
  if (kind === ViewerComponentKind.DIVIDER_PANEL) {
    return PanelSemantic.DIVIDER;
  }

  if (kind === ViewerComponentKind.DOOR_PANEL) {
    return PanelSemantic.DOOR;
  }
  if (kind === ViewerComponentKind.DRAWER_FRONT) {
    return PanelSemantic.DRAWER;
  }

  if (kind === ViewerComponentKind.FRONT_PANEL) {
    return PanelSemantic.FRONT;
  }

  return PanelSemantic.FALLBACK;
}

function buildPanelMeshes(model: ProjectModel): PanelMeshModel[] {
  const modelWidth = dimensionOrFallback(model.product.target_width, 900);
  const modelHeight = dimensionOrFallback(model.product.target_height, 1200);
  const modelDepth = dimensionOrFallback(model.product.target_depth, 450);
  const thickness = clamp(model.product.material_thickness || 18, 8, 50);
  const frontReveal = Math.max(2, Math.min(6, thickness * 0.2));

  const semanticById = new Map<string, PanelSemantic>();
  for (const component of model.components) {
    const semantic = classifySemantic(component.kind);
    semanticById.set(component.id, semantic);
  }

  const groupBySemantic = (semantic: PanelSemantic): string[] => {
    const group: string[] = [];
    for (const component of model.components) {
      if (semanticById.get(component.id) === semantic) {
        group.push(component.id);
      }
    }
    return group;
  };

  const doorIds = groupBySemantic(PanelSemantic.DOOR);
  const shelfIds = groupBySemantic(PanelSemantic.SHELF);
  const dividerIds = groupBySemantic(PanelSemantic.DIVIDER);
  const drawerIds = groupBySemantic(PanelSemantic.DRAWER);
  const frontIds = groupBySemantic(PanelSemantic.FRONT);

  const indexById = (ids: string[]): Map<string, number> => {
    const map = new Map<string, number>();
    ids.forEach((id, idx) => map.set(id, idx));
    return map;
  };

  const doorIndexById = indexById(doorIds);
  const shelfIndexById = indexById(shelfIds);
  const dividerIndexById = indexById(dividerIds);
  const drawerIndexById = indexById(drawerIds);
  const frontIndexById = indexById(frontIds);

  const jointAccumulator = new Map<string, { x: number; y: number; z: number; count: number }>();
  for (const joint of model.joints || []) {
    const existing = jointAccumulator.get(joint.child_id);
    if (!existing) {
      jointAccumulator.set(joint.child_id, { x: joint.pos_x, y: joint.pos_y, z: joint.pos_z, count: 1 });
      continue;
    }
    existing.x += joint.pos_x;
    existing.y += joint.pos_y;
    existing.z += joint.pos_z;
    existing.count += 1;
  }

  const resolveJointPosition = (
    componentId: string,
    fallback: [number, number, number],
    useX: boolean,
    useY: boolean,
    useZ: boolean,
  ): [number, number, number] => {
    const agg = jointAccumulator.get(componentId);
    if (!agg || agg.count <= 0) {
      return fallback;
    }

    const avgX = (agg.x / agg.count) * SCALE_TO_METERS;
    const avgY = (agg.y / agg.count) * SCALE_TO_METERS;
    const avgZ = (agg.z / agg.count) * SCALE_TO_METERS;

    return [
      useX ? avgX : fallback[0],
      useY ? avgY : fallback[1],
      useZ ? avgZ : fallback[2],
    ];
  };

  const resolveJointMillimeterPosition = (componentId: string): { x: number; y: number; z: number } | null => {
    const agg = jointAccumulator.get(componentId);
    if (!agg || agg.count <= 0) {
      return null;
    }
    return {
      x: agg.x / agg.count,
      y: agg.y / agg.count,
      z: agg.z / agg.count,
    };
  };

  return model.components.map((component, index) => {
    const semantic = semanticById.get(component.id) || classifySemantic(component.kind);
    const parsedKind = parseViewerComponentKind(component.kind);
    const defaultWidth = clamp(modelWidth * 0.5, thickness, modelWidth);
    const defaultHeight = clamp(modelHeight * 0.5, thickness, modelHeight);
    const defaultDepth = clamp(modelDepth * 0.5, thickness, modelDepth);
    const compWidth = clamp(dimensionOrFallback(component.width, defaultWidth), thickness, modelWidth);
    const compHeight = clamp(dimensionOrFallback(component.height, defaultHeight), thickness, modelHeight);
    const compDepth = clamp(dimensionOrFallback(component.depth, defaultDepth), thickness, modelDepth);

    if (semantic === PanelSemantic.LEFT_LEG) {
      const isBackLeg = parsedKind === ViewerComponentKind.LEFT_LEG_BACK;
      return {
        id: component.id,
        label: component.kind,
        size: [compWidth * SCALE_TO_METERS, compHeight * SCALE_TO_METERS, compDepth * SCALE_TO_METERS],
        position: [
          ((-modelWidth / 2) + compWidth / 2 + thickness) * SCALE_TO_METERS,
          ((-modelHeight / 2) + compHeight / 2) * SCALE_TO_METERS,
          (isBackLeg
            ? (-modelDepth / 2) + compDepth / 2 + thickness
            : (modelDepth / 2) - compDepth / 2 - thickness) * SCALE_TO_METERS,
        ],
        tone: "panel",
      };
    }

    if (semantic === PanelSemantic.RIGHT_LEG) {
      const isBackLeg = parsedKind === ViewerComponentKind.RIGHT_LEG_BACK;
      return {
        id: component.id,
        label: component.kind,
        size: [compWidth * SCALE_TO_METERS, compHeight * SCALE_TO_METERS, compDepth * SCALE_TO_METERS],
        position: [
          ((modelWidth / 2) - compWidth / 2 - thickness) * SCALE_TO_METERS,
          ((-modelHeight / 2) + compHeight / 2) * SCALE_TO_METERS,
          (isBackLeg
            ? (-modelDepth / 2) + compDepth / 2 + thickness
            : (modelDepth / 2) - compDepth / 2 - thickness) * SCALE_TO_METERS,
        ],
        tone: "panel",
      };
    }

    if (semantic === PanelSemantic.LEFT_SIDE) {
      return {
        id: component.id,
        label: component.kind,
        size: [thickness * SCALE_TO_METERS, modelHeight * SCALE_TO_METERS, modelDepth * SCALE_TO_METERS],
        position: [
          ((-modelWidth / 2) + thickness / 2) * SCALE_TO_METERS,
          0,
          0,
        ],
        tone: "panel",
      };
    }

    if (semantic === PanelSemantic.RIGHT_SIDE) {
      return {
        id: component.id,
        label: component.kind,
        size: [thickness * SCALE_TO_METERS, modelHeight * SCALE_TO_METERS, modelDepth * SCALE_TO_METERS],
        position: [
          ((modelWidth / 2) - thickness / 2) * SCALE_TO_METERS,
          0,
          0,
        ],
        tone: "panel",
      };
    }

    if (semantic === PanelSemantic.TOP) {
      return {
        id: component.id,
        label: component.kind,
        size: [modelWidth * SCALE_TO_METERS, thickness * SCALE_TO_METERS, modelDepth * SCALE_TO_METERS],
        position: [0, ((modelHeight / 2) - thickness / 2) * SCALE_TO_METERS, 0],
        tone: "panel",
      };
    }

    if (semantic === PanelSemantic.BOTTOM) {
      return {
        id: component.id,
        label: component.kind,
        size: [modelWidth * SCALE_TO_METERS, thickness * SCALE_TO_METERS, modelDepth * SCALE_TO_METERS],
        position: [0, ((-modelHeight / 2) + thickness / 2) * SCALE_TO_METERS, 0],
        tone: "panel",
      };
    }

    if (semantic === PanelSemantic.BACK) {
      return {
        id: component.id,
        label: component.kind,
        size: [
          Math.max(thickness, modelWidth - thickness * 2) * SCALE_TO_METERS,
          Math.max(thickness, modelHeight - thickness * 2) * SCALE_TO_METERS,
          thickness * SCALE_TO_METERS,
        ],
        position: [0, 0, ((-modelDepth / 2) + thickness / 2) * SCALE_TO_METERS],
        tone: "back",
      };
    }

    if (semantic === PanelSemantic.DIVIDER) {
      const dividerIndex = dividerIndexById.get(component.id) || 0;
      const dividerCount = Math.max(dividerIds.length, 1);
      const innerWidth = Math.max(modelWidth - thickness * 2, thickness * 2);
      const spacing = innerWidth / (dividerCount + 1);
      const xPosition = -innerWidth / 2 + spacing * (dividerIndex + 1);

      const fallbackPosition: [number, number, number] = [xPosition * SCALE_TO_METERS, 0, 0];
      return {
        id: component.id,
        label: component.kind,
        size: [
          Math.max(thickness, compWidth) * SCALE_TO_METERS,
          Math.max(thickness, compHeight) * SCALE_TO_METERS,
          Math.max(thickness, modelDepth - thickness * 2) * SCALE_TO_METERS,
        ],
        position: resolveJointPosition(component.id, fallbackPosition, true, true, true),
        tone: "divider",
      };
    }

    if (semantic === PanelSemantic.DOOR) {
      const doorIndex = doorIndexById.get(component.id) || 0;
      const revealGap = 3;
      const doorCount = Math.max(doorIds.length, 1);
      const innerWidth = Math.max(modelWidth - thickness * 2, thickness * 3);
      const fallbackDoorWidth = Math.max(innerWidth / doorCount - revealGap, thickness * 2);
      const doorWidth = Math.max(thickness, Math.min(compWidth - frontReveal, innerWidth));
      const doorHeight = Math.max(thickness * 2, Math.min(compHeight - frontReveal, modelHeight - thickness * 2));
      const xPos =
        (((doorIndex + 0.5) / doorCount) - 0.5) * innerWidth;
      
      const fallbackPosition: [number, number, number] = [xPos * SCALE_TO_METERS, 0, (modelDepth / 2 - Math.max(compDepth, thickness / 2) / 2) * SCALE_TO_METERS];
      const jointMm = resolveJointMillimeterPosition(component.id);
      const isMixedFacade = (model.product.door_count || 0) === 3 && (model.product.drawer_count || 0) === 1;
      let doorTone: PanelMeshModel["tone"] = "door";
      if (isMixedFacade && jointMm) {
        // Match common mixed-facade look: top-left + right doors dark, bottom-left wood tone.
        if (jointMm.x <= 0 && jointMm.y < 0) {
          doorTone = "panel";
        }
      }
      return {
        id: component.id,
        label: component.kind,
        size: [Math.max(doorWidth, fallbackDoorWidth) * SCALE_TO_METERS, doorHeight * SCALE_TO_METERS, Math.max(thickness / 2, compDepth) * SCALE_TO_METERS],
        position: resolveJointPosition(component.id, fallbackPosition, true, true, true),
        tone: doorTone,
      };
    }

    if (semantic === PanelSemantic.DRAWER) {
      const drawerIndex = drawerIndexById.get(component.id) || 0;
      const drawerCount = Math.max(drawerIds.length, 1);
      const frontZoneHeight = modelHeight * 0.45;
      const fallbackDrawerHeight = frontZoneHeight / drawerCount;
      const drawerHeight = Math.max(thickness * 1.5, Math.min(compHeight - frontReveal, modelHeight - thickness * 2));
      const drawerWidth = Math.max(thickness * 2, Math.min(compWidth - frontReveal, modelWidth - thickness * 2));
      const centerFromBottom =
        -modelHeight / 2 + thickness * 1.8 + fallbackDrawerHeight * (drawerIndex + 0.5);

      const fallbackPosition: [number, number, number] = [
        0,
        centerFromBottom * SCALE_TO_METERS,
        ((modelDepth / 2) - Math.max(compDepth, thickness / 2) / 2) * SCALE_TO_METERS,
      ];

      return {
        id: component.id,
        label: component.kind,
        size: [
          drawerWidth * SCALE_TO_METERS,
          drawerHeight * SCALE_TO_METERS,
          Math.max(thickness / 2, compDepth) * SCALE_TO_METERS,
        ],
        position: resolveJointPosition(component.id, fallbackPosition, true, true, true),
        tone: "drawer",
      };
    }

    if (semantic === PanelSemantic.FRONT) {
      const frontIndex = frontIndexById.get(component.id) || 0;
      const frontCount = Math.max(frontIds.length, 1);
      const segmentHeight = (modelHeight - thickness * 2.5) / frontCount;
      const yPosition =
        -modelHeight / 2 + thickness * 1.2 + segmentHeight * (frontIndex + 0.5);

      const fallbackPosition: [number, number, number] = [
        0,
        yPosition * SCALE_TO_METERS,
        ((modelDepth / 2) - Math.max(thickness / 4, compDepth / 2)) * SCALE_TO_METERS,
      ];

      return {
        id: component.id,
        label: component.kind,
        size: [
          Math.max(thickness, compWidth - frontReveal) * SCALE_TO_METERS,
          Math.max(thickness, Math.min(compHeight - frontReveal, segmentHeight - thickness * 0.2)) * SCALE_TO_METERS,
          Math.max(thickness / 2, compDepth) * SCALE_TO_METERS,
        ],
        position: resolveJointPosition(component.id, fallbackPosition, true, true, true),
        tone: "panel",
      };
    }

    if (semantic === PanelSemantic.SHELF) {
      const shelfIndex = shelfIndexById.get(component.id) || 0;
      const shelfCount = Math.max(shelfIds.length, 1);
      const shelfY = ((shelfIndex + 1) / (shelfCount + 1)) * (modelHeight - thickness) - modelHeight / 2;
      const fallbackPosition: [number, number, number] = [0, shelfY * SCALE_TO_METERS, 0];

      return {
        id: component.id,
        label: component.kind,
        size: [
          Math.max(thickness, modelWidth - thickness * 2) * SCALE_TO_METERS,
          Math.max(thickness, compHeight) * SCALE_TO_METERS,
          Math.max(thickness, modelDepth - thickness * 2) * SCALE_TO_METERS,
        ],
        position: resolveJointPosition(component.id, fallbackPosition, true, true, true),
        tone: "shelf",
      };
    }

    const jitterX = ((index % 3) - 1) * thickness * 1.3;
    const jitterY = (Math.floor(index / 3) % 3 - 1) * thickness * 1.3;
    return {
      id: component.id,
      label: component.kind,
      size: [compWidth * SCALE_TO_METERS, compHeight * SCALE_TO_METERS, compDepth * SCALE_TO_METERS],
      position: [jitterX * SCALE_TO_METERS, jitterY * SCALE_TO_METERS, 0],
      tone: "fallback",
    };
  });
}

function toneColor(tone: PanelMeshModel["tone"], selected: boolean): string {
  if (selected) {
    return "#f4cb8f";
  }
  if (tone === "door") {
    return "#7f6740";
  }
  if (tone === "drawer") {
    return "#9e7f4f";
  }
  if (tone === "divider") {
    return "#8d6f42";
  }
  if (tone === "back") {
    return "#cddce8";
  }
  if (tone === "shelf") {
    return "#e8b976";
  }
  if (tone === "fallback") {
    return "#d9b07f";
  }
  return "#eec98c";
}

export function ParametricViewer({ model, validation }: ParametricViewerProps): React.JSX.Element {
  const [selectedId, setSelectedId] = useState<string>("");
  const [autoRotate, setAutoRotate] = useState<boolean>(true);
  const [showGrid, setShowGrid] = useState<boolean>(true);
  const [explode, setExplode] = useState<number>(0);

  const panels = useMemo(() => {
    if (!model) {
      return [];
    }
    return buildPanelMeshes(model);
  }, [model]);

  const selectedPanel = panels.find((panel) => panel.id === selectedId);

  if (!model) {
    return (
      <div className="parametric-viewer" role="img" aria-label="No model loaded">
        <p className="parametric-empty">No model yet. Create and update a project to render workspace geometry.</p>
      </div>
    );
  }

  return (
    <div className="parametric-viewer-shell">
      <div className="viewer-toolbar" aria-label="3D viewer controls">
        <label className="viewer-toolbar-item" htmlFor="auto-rotate-toggle">
          <input
            id="auto-rotate-toggle"
            type="checkbox"
            checked={autoRotate}
            onChange={(event) => setAutoRotate(event.currentTarget.checked)}
          />
          Auto rotate
        </label>
        <label className="viewer-toolbar-item" htmlFor="show-grid-toggle">
          <input
            id="show-grid-toggle"
            type="checkbox"
            checked={showGrid}
            onChange={(event) => setShowGrid(event.currentTarget.checked)}
          />
          Show grid
        </label>
        <label className="viewer-toolbar-item slider" htmlFor="explode-slider">
          Exploded view
          <input
            id="explode-slider"
            type="range"
            min={0}
            max={100}
            step={1}
            value={explode}
            onChange={(event) => setExplode(Number(event.currentTarget.value))}
          />
        </label>
      </div>

      <div className="parametric-viewer" role="img" aria-label="Interactive 3D parametric furniture model">
        <Canvas camera={{ position: [1.5, 1.1, 1.6], fov: 52 }}>
          <color attach="background" args={["#f7fbff"]} />
          <ambientLight intensity={0.55} />
          <directionalLight position={[2.5, 3, 2]} intensity={0.9} />
          <directionalLight position={[-1.5, 1.2, -2]} intensity={0.35} />

          {panels.map((panel) => (
            <mesh
              key={panel.id}
              position={[
                panel.position[0] * (1 + explode / 140),
                panel.position[1] * (1 + explode / 140),
                panel.position[2] + (explode / 2500),
              ]}
              onClick={(event) => {
                event.stopPropagation();
                setSelectedId(panel.id);
              }}
            >
              <boxGeometry args={panel.size} />
              <meshStandardMaterial color={toneColor(panel.tone, panel.id === selectedId)} />
            </mesh>
          ))}

          {showGrid ? <gridHelper args={[3, 24, "#9fb9cc", "#dce7ef"]} position={[0, -0.65, 0]} /> : null}
          <OrbitControls makeDefault enablePan enableZoom enableRotate autoRotate={autoRotate} autoRotateSpeed={0.9} />
        </Canvas>
      </div>

      <div className="viewer-meta-row" aria-live="polite">
        <p>
          Components: <strong>{model.components.length}</strong>
        </p>
        <p>
          Selection: <strong>{selectedPanel?.label || "none"}</strong>
        </p>
        <p>
          Validation: <strong>{validation ? (validation.valid ? "pass" : "fail") : "not run"}</strong>
        </p>
      </div>
      {!validation?.valid && validation ? (
        <p className="viewer-validation-hint">Validation has failing rules. Run adjustments and validate again to clear highlighted issues.</p>
      ) : null}
    </div>
  );
}