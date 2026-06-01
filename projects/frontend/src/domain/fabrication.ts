import { ProjectModel, ValidateResponse } from "../api/backend";

export type NestingPanel = {
  id: string;
  orientation: "original" | "rotated";
  width: number;
  height: number;
  x: number;
  y: number;
};

export type FabricationPackage = {
  generated_at: string;
  project_id: string;
  blueprint: {
    product_name: string;
    dimensions_mm: {
      width: number;
      height: number;
      depth: number;
      shelf_count: number;
    };
    components: Array<{ id: string; kind: string; width: number; height: number; depth: number }>;
  };
  nesting: {
    strategy: string;
    sheet_size_mm: { width: number; height: number };
    estimated_usage_ratio: number;
    panels: NestingPanel[];
  };
  bom: {
    hardware: Array<{ code: string; qty: number }>;
    panel_count: number;
  };
  validation: ValidateResponse | null;
};

const SHEET_WIDTH = 2440;
const SHEET_HEIGHT = 1220;

function area(width: number, height: number): number {
  return width * height;
}

function toNestingPanels(model: ProjectModel): NestingPanel[] {
  const panels: NestingPanel[] = [];
  let cursorX = 0;
  let cursorY = 0;
  let rowHeight = 0;

  model.components.forEach((component) => {
    const originalW = Math.max(component.width, component.depth);
    const originalH = component.height;
    const rotatedW = originalH;
    const rotatedH = Math.max(component.width, component.depth);

    const useRotated = area(rotatedW, rotatedH) < area(originalW, originalH) && rotatedW <= SHEET_WIDTH;
    const width = useRotated ? rotatedW : originalW;
    const height = useRotated ? rotatedH : originalH;

    if (cursorX + width > SHEET_WIDTH) {
      cursorX = 0;
      cursorY += rowHeight;
      rowHeight = 0;
    }

    panels.push({
      id: component.id,
      orientation: useRotated ? "rotated" : "original",
      width,
      height,
      x: cursorX,
      y: cursorY,
    });

    cursorX += width;
    rowHeight = Math.max(rowHeight, height);
  });

  return panels;
}

export function buildFabricationPackage(
  projectId: string,
  model: ProjectModel,
  validation: ValidateResponse | null
): FabricationPackage {
  const panels = toNestingPanels(model);
  const usedArea = panels.reduce((sum, panel) => sum + area(panel.width, panel.height), 0);
  const sheetArea = area(SHEET_WIDTH, SHEET_HEIGHT);

  return {
    generated_at: new Date().toISOString(),
    project_id: projectId,
    blueprint: {
      product_name: model.product.name,
      dimensions_mm: {
        width: model.product.target_width,
        height: model.product.target_height,
        depth: model.product.target_depth,
        shelf_count: model.product.shelf_count,
      },
      components: model.components,
    },
    nesting: {
      strategy: "Bottom-left-fill (frontend preview)",
      sheet_size_mm: { width: SHEET_WIDTH, height: SHEET_HEIGHT },
      estimated_usage_ratio: Number((usedArea / sheetArea).toFixed(3)),
      panels,
    },
    bom: {
      hardware: model.hardware,
      panel_count: model.components.length,
    },
    validation,
  };
}
