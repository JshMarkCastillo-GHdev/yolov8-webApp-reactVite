import type { OrtApi } from "../../../shared/lib/ort";
import { INPUT_SIZE } from "./constants";

export function imageToTensor(
  source: CanvasImageSource,
  ort: OrtApi,
  size: number = INPUT_SIZE,
) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not get canvas context");

  canvas.width = size;
  canvas.height = size;
  ctx.drawImage(source, 0, 0, size, size);

  const imageData = ctx.getImageData(0, 0, size, size).data;
  const input = new Float32Array(size * size * 3);

  for (let i = 0; i < size * size; i++) {
    input[i] = imageData[i * 4] / 255.0;
    input[i + size * size] = imageData[i * 4 + 1] / 255.0;
    input[i + 2 * size * size] = imageData[i * 4 + 2] / 255.0;
  }

  return new ort.Tensor("float32", input, [1, 3, size, size]);
}
