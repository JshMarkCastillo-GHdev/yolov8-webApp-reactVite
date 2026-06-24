import type { BoundingBox } from "../../../shared/types/plate";
import { CONF_THRESHOLD, IOU_THRESHOLD } from "./constants";

export function boxIoU(a: number[], b: number[]): number {
  const [x1, y1, w1, h1] = a;
  const [x2, y2, w2, h2] = b;

  const interW = Math.max(0, Math.min(x1 + w1, x2 + w2) - Math.max(x1, x2));
  const interH = Math.max(0, Math.min(y1 + h1, y2 + h2) - Math.max(y1, y2));

  const interArea = interW * interH;
  const unionArea = w1 * h1 + w2 * h2 - interArea;
  return interArea / unionArea;
}

export function nms(
  boxes: number[][],
  scores: number[],
  iouThreshold: number = IOU_THRESHOLD,
): number[][] {
  const picked: number[][] = [];
  let idxs = scores
    .map((s, i) => [s, i] as const)
    .sort((a, b) => b[0] - a[0])
    .map(([, i]) => i);

  while (idxs.length > 0) {
    const i = idxs.shift()!;
    picked.push(boxes[i]);

    idxs = idxs.filter((j) => boxIoU(boxes[i], boxes[j]) <= iouThreshold);
  }

  return picked;
}

export type RawDetection = {
  box: [number, number, number, number];
  score: number;
};

export function parseYoloOutput(
  outputData: Float32Array,
  dims: number[],
  confThreshold: number = CONF_THRESHOLD,
): RawDetection[] {
  const [, channels, numDets] = dims;
  const detections: RawDetection[] = [];

  for (let det = 0; det < numDets; det++) {
    const row: number[] = [];
    for (let ch = 0; ch < channels; ch++) {
      row.push(outputData[ch * numDets + det]);
    }

    const [cx, cy, w, h, conf] = row;
    if (conf > confThreshold) {
      detections.push({
        box: [cx - w / 2, cy - h / 2, w, h],
        score: conf,
      });
    }
  }

  return detections;
}

export function getBestDetection(
  detections: RawDetection[],
): { box: BoundingBox; score: number } | null {
  if (detections.length === 0) return null;

  const boxes = detections.map((d) => [...d.box]);
  const scores = detections.map((d) => d.score);
  const finalBoxes = nms(boxes, scores);

  if (finalBoxes.length === 0) return null;

  let bestIdx = 0;
  let bestScore = scores[0];
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > bestScore) {
      bestScore = scores[i];
      bestIdx = i;
    }
  }

  const [x, y, w, h] = boxes[bestIdx];
  return { box: { x, y, w, h }, score: bestScore };
}

export function scaleBox(
  box: BoundingBox,
  sourceWidth: number,
  sourceHeight: number,
  inputSize: number,
): BoundingBox {
  const scaleX = sourceWidth / inputSize;
  const scaleY = sourceHeight / inputSize;
  return {
    x: box.x * scaleX,
    y: box.y * scaleY,
    w: box.w * scaleX,
    h: box.h * scaleY,
  };
}
