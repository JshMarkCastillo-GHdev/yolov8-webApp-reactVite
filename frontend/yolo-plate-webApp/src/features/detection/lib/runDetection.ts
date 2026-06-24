import type Tesseract from "tesseract.js";
import { getSourceSize } from "../../../shared/lib/imageSource";
import type { OrtApi, OrtInferenceSession } from "../../../shared/lib/ort";
import type { PlateDetection } from "../../../shared/types/plate";
import { INPUT_SIZE, OCR_MIN_CONFIDENCE, OCR_MIN_TEXT_LENGTH } from "./constants";
import { drawSourceWithOverlay } from "./drawOverlay";
import { getBestDetection, parseYoloOutput, scaleBox } from "./postprocess";
import { imageToTensor } from "./preprocess";
import {
  cleanPlateText,
  createPreprocessedCropCanvas,
} from "../../ocr/lib/preprocessCrop";

export type RunDetectionParams = {
  source: CanvasImageSource;
  session: OrtInferenceSession;
  ort: OrtApi;
  worker?: Tesseract.Worker | null;
  canvas?: HTMLCanvasElement | null;
};

export async function runDetection({
  source,
  session,
  ort,
  worker,
  canvas,
}: RunDetectionParams): Promise<PlateDetection | null> {
  const { width, height } = getSourceSize(source);
  if (width === 0 || height === 0) return null;

  const tensor = imageToTensor(source, ort, INPUT_SIZE);
  const feeds = { [session.inputNames[0]]: tensor };
  const results = await session.run(feeds);
  const outputTensor = results[session.outputNames[0]];

  const rawDetections = parseYoloOutput(outputTensor.data, outputTensor.dims);
  const best = getBestDetection(rawDetections);
  if (!best) {
    if (canvas) {
      drawSourceWithOverlay(canvas, source, null);
    }
    return null;
  }

  const scaledBox = scaleBox(best.box, width, height, INPUT_SIZE);

  let text: string | null = null;
  let ocrConfidence: number | null = null;

  if (worker) {
    const cropCanvas = createPreprocessedCropCanvas(
      source,
      scaledBox.x,
      scaledBox.y,
      scaledBox.w,
      scaledBox.h,
    );

    try {
      const {
        data: { text: rawText, confidence },
      } = await worker.recognize(cropCanvas);
      const cleanText = cleanPlateText(rawText);

      if (
        cleanText.length >= OCR_MIN_TEXT_LENGTH &&
        confidence >= OCR_MIN_CONFIDENCE
      ) {
        text = cleanText;
        ocrConfidence = confidence;
      }
    } catch (err) {
      console.error("OCR failed:", err);
    }
  }

  if (canvas) {
    drawSourceWithOverlay(canvas, source, scaledBox, text, ocrConfidence);
  }

  return {
    box: scaledBox,
    detectionScore: best.score,
    text,
    ocrConfidence,
  };
}
