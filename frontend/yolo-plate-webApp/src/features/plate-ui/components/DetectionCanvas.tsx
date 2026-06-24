import { forwardRef } from "react";

export const DetectionCanvas = forwardRef<HTMLCanvasElement>(
  function DetectionCanvas(_props, ref) {
    return (
      <canvas
        ref={ref}
        className="max-h-full max-w-full object-contain"
      />
    );
  },
);
