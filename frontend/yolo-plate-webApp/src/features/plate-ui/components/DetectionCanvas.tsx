import { forwardRef } from "react";

export const DetectionCanvas = forwardRef<HTMLCanvasElement>(
  function DetectionCanvas(_props, ref) {
    return (
      <canvas
        ref={ref}
        className="w-full h-full object-contain bg-black rounded-lg"
      />
    );
  },
);
