import { Loader2 } from "lucide-react";

import { cn } from "@/lib/utils";

type SpinnerProps = {
  className?: string;
};

export function Spinner({ className }: SpinnerProps) {
  return (
    <Loader2
      className={cn("size-8 animate-spin text-primary", className)}
      aria-label="Loading"
    />
  );
}
