import type { BoundingBox } from "../../../shared/types/plate";
import { getSourceSize } from "../../../shared/lib/imageSource";

export function drawBox(
  ctx: CanvasRenderingContext2D,
  box: BoundingBox,
  text?: string | null,
  ocrConfidence?: number | null,
): void {
  const { x, y, w, h } = box;

  ctx.strokeStyle = "lime";
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, w, h);

  if (text) {
    ctx.fillStyle = "lime";
    ctx.font = "bold 24px Arial";
    ctx.fillText(text, x, y - 12);

    if (ocrConfidence !== null && ocrConfidence !== undefined) {
      ctx.font = "14px Arial";
      ctx.fillText(`conf: ${ocrConfidence.toFixed(0)}%`, x, y - 28);
    }
  }
}

export function drawSourceWithOverlay(
  canvas: HTMLCanvasElement,
  source: CanvasImageSource,
  box: BoundingBox | null,
  text?: string | null,
  ocrConfidence?: number | null,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const { width, height } = getSourceSize(source);
  if (width === 0 || height === 0) return;

  canvas.width = width;
  canvas.height = height;
  ctx.drawImage(source, 0, 0, width, height);

  if (box) {
    drawBox(ctx, box, text, ocrConfidence);
  }
}
