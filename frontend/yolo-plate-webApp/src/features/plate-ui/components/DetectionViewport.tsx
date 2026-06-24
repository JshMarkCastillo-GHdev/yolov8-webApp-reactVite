import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

type DetectionViewportProps = {
  children: ReactNode;
  overlay?: ReactNode;
  className?: string;
};

export function DetectionViewport({
  children,
  overlay,
  className,
}: DetectionViewportProps) {
  return (
    <div
      className={cn(
        "relative flex w-full items-center justify-center overflow-hidden rounded-lg bg-black sm:rounded-xl",
        "min-h-[min(42svh,360px)] sm:min-h-[min(50svh,480px)] lg:min-h-[min(65vh,720px)]",
        className,
      )}
    >
      {children}
      {overlay}
    </div>
  );
}
