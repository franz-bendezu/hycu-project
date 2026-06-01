type StatusPillProps = {
  label: string;
  tone: "neutral" | "ok" | "warning" | "error";
};

export function StatusPill({ label, tone }: StatusPillProps): React.JSX.Element {
  return <span className={`status-pill ${tone}`}>{label}</span>;
}
