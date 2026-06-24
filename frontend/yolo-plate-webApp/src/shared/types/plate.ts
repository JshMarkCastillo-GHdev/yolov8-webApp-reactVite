export type InputMode = "camera" | "upload" | "sample";

export type BoundingBox = {
  x: number;
  y: number;
  w: number;
  h: number;
};

export type PlateDetection = {
  box: BoundingBox;
  detectionScore: number;
  text: string | null;
  ocrConfidence: number | null;
};
