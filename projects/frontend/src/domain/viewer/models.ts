export type PanelTone = "panel" | "shelf" | "back" | "door" | "drawer" | "divider" | "fallback";

type UnitSystem = "millimeters" | "meters";
type RotationUnit = "degrees" | "radians";

function convertVectorToMeters(vector: [number, number, number], unit: UnitSystem): [number, number, number] {
  if (unit === "meters") {
    return vector;
  }
  return [
    vector[0] * SCALE_TO_METERS,
    vector[1] * SCALE_TO_METERS,
    vector[2] * SCALE_TO_METERS,
  ];
}

function convertRotationToRadians(rotation: [number, number, number], unit: RotationUnit): [number, number, number] {
  if (unit === "radians") {
    return rotation;
  }
  const DEG_TO_RAD = Math.PI / 180;
  return [
    rotation[0] * DEG_TO_RAD,
    rotation[1] * DEG_TO_RAD,
    rotation[2] * DEG_TO_RAD,
  ];
}

export type PanelMeshProps = {
  id: string;
  label: string;
  size: [number, number, number];
  position: [number, number, number];
  rotation: [number, number, number];
  tone: PanelTone;
};

export class PanelMeshModel {
  readonly id: string;
  readonly label: string;
  readonly size: [number, number, number];
  readonly position: [number, number, number];
  readonly rotation: [number, number, number];
  readonly tone: PanelTone;

  constructor(
    props: PanelMeshProps,
    unit: UnitSystem = "millimeters",
    rotationUnit: RotationUnit = "degrees",
  ) {
    this.id = props.id;
    this.label = props.label;
    this.size = convertVectorToMeters(props.size, unit);
    this.position = convertVectorToMeters(props.position, unit);
    this.rotation = convertRotationToRadians(props.rotation, rotationUnit);
    this.tone = props.tone;
  }

  static create(props: PanelMeshProps): PanelMeshModel {
    return new PanelMeshModel(props);
  }

  static createMeters(props: PanelMeshProps): PanelMeshModel {
    return new PanelMeshModel(props, "meters");
  }

  static createMetersRadians(props: PanelMeshProps): PanelMeshModel {
    return new PanelMeshModel(props, "meters", "radians");
  }
}

export type HardwareKind = "screw" | "cam_lock" | "slide" | "hinge" | "generic";

export type HardwareMarkerProps = {
  id: string;
  label: string;
  position: [number, number, number];
  kind: HardwareKind;
  size: number;
  color: string;
};

export class HardwareMarker {
  readonly id: string;
  readonly label: string;
  readonly position: [number, number, number];
  readonly kind: HardwareKind;
  readonly size: number;
  readonly color: string;

  constructor(props: HardwareMarkerProps) {
    this.id = props.id;
    this.label = props.label;
    this.position = props.position;
    this.kind = props.kind;
    this.size = props.size;
    this.color = props.color;
  }

  static create(props: HardwareMarkerProps): HardwareMarker {
    return new HardwareMarker(props);
  }
}

export const SCALE_TO_METERS = 0.001;
export const MAX_RENDER_HARDWARE = 200;
