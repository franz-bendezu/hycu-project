import { ReactNode } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";

type SectionCardProps = {
  title: string;
  subtitle?: string;
  children: ReactNode;
};

export function SectionCard({ title, subtitle, children }: SectionCardProps): React.JSX.Element {
  return (
    <Card className="panel section-card">
      <CardHeader className="section-card-header">
        <CardTitle>{title}</CardTitle>
        {subtitle ? <CardDescription className="section-card-subtitle">{subtitle}</CardDescription> : null}
      </CardHeader>
      <CardContent className="section-card-content">{children}</CardContent>
    </Card>
  );
}
