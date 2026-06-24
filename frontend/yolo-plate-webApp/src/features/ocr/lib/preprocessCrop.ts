export function createPreprocessedCropCanvas(
  source: CanvasImageSource,
  x: number,
  y: number,
  w: number,
  h: number,
): HTMLCanvasElement {
  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = w;
  cropCanvas.height = h;
  const cropCtx = cropCanvas.getContext("2d");
  if (!cropCtx) throw new Error("Could not get crop canvas context");

  cropCtx.drawImage(source, x, y, w, h, 0, 0, w, h);

  cropCtx.filter = "grayscale(100%) contrast(1.4) brightness(1.1)";
  cropCtx.drawImage(cropCanvas, 0, 0);

  cropCtx.filter = "contrast(1.2)";
  cropCtx.drawImage(cropCanvas, 0, 0);

  if (w < 180 || h < 60) {
    const temp = document.createElement("canvas");
    temp.width = w * 1.8;
    temp.height = h * 1.8;
    const tCtx = temp.getContext("2d");
    if (!tCtx) throw new Error("Could not get temp canvas context");
    tCtx.drawImage(cropCanvas, 0, 0, temp.width, temp.height);
    cropCanvas.width = temp.width;
    cropCanvas.height = temp.height;
    cropCtx.drawImage(temp, 0, 0);
  }

  return cropCanvas;
}

export function cleanPlateText(text: string): string {
  return text.trim().replace(/[^A-Z0-9- ]/g, "");
}
