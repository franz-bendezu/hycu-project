import { Badge } from "./ui/badge";

type StatusPillProps = {
  label: string;
  tone: "neutral" | "ok" | "warning" | "error";
};

export function StatusPill({ label, tone }: StatusPillProps): React.JSX.Element {
  return (
    <Badge className={`status-pill ${tone}`} variant="secondary">
      <span className="status-pill-dot" aria-hidden="true" />
      {label}
    </Badge>
  );
}
