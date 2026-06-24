import { CheckCircle2 } from "lucide-react";

import { Alert, AlertDescription } from "@/components/ui/alert";

type PlateAlertProps = {
  plate: string | null;
  confidence: number | null;
  emptyMessage?: string;
};

export function PlateAlert({
  plate,
  confidence,
  emptyMessage = "No plate recognised yet.",
}: PlateAlertProps) {
  if (!plate) {
    return <p className="text-sm text-muted-foreground">{emptyMessage}</p>;
  }

  return (
    <Alert variant="success">
      <CheckCircle2 className="h-4 w-4" />
      <AlertDescription className="break-words">
        {plate}{" "}
        {confidence !== null && `(${confidence.toFixed(0)}% confidence)`}
      </AlertDescription>
    </Alert>
  );
}
