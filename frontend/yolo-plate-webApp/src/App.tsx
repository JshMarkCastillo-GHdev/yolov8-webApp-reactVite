import { useRef, useEffect } from "react";
import Tesseract, { PSM } from "tesseract.js";

type OrtSession = any; // from CDN

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const sessionRef = useRef<OrtSession | null>(null);
  const workerRef = useRef<Tesseract.Worker | null>(null);

  const lastPlateTextRef = useRef<string | null>(null);
  const lastConfidenceRef = useRef<number | null>(null);
  const lastBoxRef = useRef<{
    x: number;
    y: number;
    w: number;
    h: number;
  } | null>(null);

  const lastInferenceTime = useRef(0);
  const INFERENCE_INTERVAL_MS = 1200; // tune: 600 = faster response, 1200 = smoother (less CPU/GPU load)

  // @ts-ignore
  const ort = (window as any).ort;

  if (ort) ort.env.wasm.numThreads = 1; // mobile-safe

  // Load YOLO model once
  const loadModel = async () => {
    if (!sessionRef.current) {
      try {
        sessionRef.current =
          await ort.InferenceSession.create("/models/best.onnx");
        console.log("Model loaded");
        console.log("Input names:", sessionRef.current.inputNames);
      } catch (err) {
        console.error("MODEL LOAD ERROR:", err);
      }
    }
  };

  // Start camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (err) {
      console.error("Camera error:", err);
    }
  };

  // Capture frame and convert to tensor
  const captureFrame = (video: HTMLVideoElement) => {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d")!;
    const size = 640;

    canvas.width = size;
    canvas.height = size;
    ctx.drawImage(video, 0, 0, size, size);

    const imageData = ctx.getImageData(0, 0, size, size).data;
    const input = new Float32Array(size * size * 3);

    for (let i = 0; i < size * size; i++) {
      input[i] = imageData[i * 4] / 255.0;
      input[i + size * size] = imageData[i * 4 + 1] / 255.0;
      input[i + 2 * size * size] = imageData[i * 4 + 2] / 255.0;
    }

    return new ort.Tensor("float32", input, [1, 3, size, size]);
  };

  // Run inference continuously
  const runLiveInference = () => {
    if (!videoRef.current) {
      requestAnimationFrame(runLiveInference);
      return;
    }

    // Always keep video smooth: redraw every frame
    if (canvasRef.current && videoRef.current.videoWidth > 0) {
      const canvas = canvasRef.current;
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

        // Re-draw last known good overlay (very cheap)
        if (lastBoxRef.current && lastPlateTextRef.current) {
          const { x, y, w, h } = lastBoxRef.current;
          ctx.strokeStyle = "lime";
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, w, h);

          ctx.fillStyle = "lime";
          ctx.font = "bold 24px Arial";
          ctx.fillText(lastPlateTextRef.current, x, y - 12);

          if (lastConfidenceRef.current !== null) {
            ctx.font = "14px Arial";
            ctx.fillText(
              `conf: ${lastConfidenceRef.current.toFixed(0)}%`,
              x,
              y - 28,
            );
          }
        }
      }
    }

    // Only run heavy stuff occasionally, 1200 MS = Smooth, 800 = faster response but more CPU/GPU load.
    const now = performance.now();
    if (now - lastInferenceTime.current >= INFERENCE_INTERVAL_MS) {
      lastInferenceTime.current = now;

      if (sessionRef.current && videoRef.current) {
        // Only capture frame when we actually need it
        const tensor = captureFrame(videoRef.current);

        const feeds = { [sessionRef.current.inputNames[0]]: tensor };
        const video = videoRef.current; // capture current video ref for closure

        sessionRef.current
          .run(feeds)
          .then((results: Record<string, any>) => {
            const outputTensor = results[sessionRef.current.outputNames[0]];
            // Reuse existing drawBoxes logic, but it now only updates refs
            drawBoxes(video, outputTensor);
          })
          .catch((err: any) => {
            console.error("Inference failed:", err);
          });
      }
    }

    requestAnimationFrame(runLiveInference);
  };

  // Draw bounding boxes with proper post-processing and NMS
  // Draw bounding boxes with post-processing, NMS, single best box + OCR
  const drawBoxes = (video: HTMLVideoElement, outputTensor: any) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Match canvas to video size and draw the current video frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const inputSize = 640;
    const scaleX = canvas.width / inputSize;
    const scaleY = canvas.height / inputSize;

    // --- Parse YOLOv8 output ---
    const [, channels, numDets] = outputTensor.dims; // e.g. [1, 5, 8400]
    const outputData = outputTensor.data as Float32Array;

    // Transpose to [numDets, channels]
    const predictions: number[][] = [];
    for (let det = 0; det < numDets; det++) {
      const row: number[] = [];
      for (let ch = 0; ch < channels; ch++) {
        row.push(outputData[ch * numDets + det]);
      }
      predictions.push(row);
    }

    const boxes: number[][] = [];
    const scores: number[] = [];
    const confThreshold = 0.35; // tune: 0.25–0.5 depending on your model/false positives

    for (const pred of predictions) {
      const [cx, cy, w, h, conf] = pred; // single class → conf is objectness × class prob

      if (conf > confThreshold) {
        const x = cx - w / 2; // top-left
        const y = cy - h / 2;
        boxes.push([x, y, w, h]);
        scores.push(conf);
      }
    }

    // Apply NMS (your existing function)
    const iouThreshold = 0.45;
    const finalBoxes = nms(boxes, scores, iouThreshold);

    // Only proceed if we have at least one good detection
    if (finalBoxes.length > 0) {
      // Sort by descending score (highest confidence first)
      const sortedIndices = scores
        .map((s, i) => ({ score: s, index: i }))
        .sort((a, b) => b.score - a.score)
        .map((item) => item.index);

      const bestIdx = sortedIndices[0]; // highest score
      const [x, y, w, h] = boxes[bestIdx];
      const scaledX = x * scaleX;
      const scaledY = y * scaleY;
      const scaledW = w * scaleX;
      const scaledH = h * scaleY;

      // Draw the lime box
      ctx.strokeStyle = "lime";
      ctx.lineWidth = 3;
      ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);

      // --- Optional: Simple preprocessing + OCR ---
      if (workerRef.current) {
        // Create crop canvas
        const cropCanvas = document.createElement("canvas");
        cropCanvas.width = scaledW;
        cropCanvas.height = scaledH;
        const cropCtx = cropCanvas.getContext("2d")!;

        // Draw cropped region from video
        cropCtx.drawImage(
          video,
          scaledX,
          scaledY,
          scaledW,
          scaledH,
          0,
          0,
          scaledW,
          scaledH,
        );

        // --- Simple but effective preprocessing ---
        // 1. Grayscale + contrast/brightness boost
        cropCtx.filter = "grayscale(100%) contrast(1.4) brightness(1.1)";
        cropCtx.drawImage(cropCanvas, 0, 0);

        // 2. Optional: slight sharpen (unsharp mask simulation)
        cropCtx.filter = "contrast(1.2)"; // extra contrast after grayscale
        cropCtx.drawImage(cropCanvas, 0, 0);

        // Optional resize if crop is tiny (Tesseract likes ~200–500px width)
        if (scaledW < 180 || scaledH < 60) {
          const temp = document.createElement("canvas");
          temp.width = scaledW * 1.8;
          temp.height = scaledH * 1.8;
          const tCtx = temp.getContext("2d")!;
          tCtx.drawImage(cropCanvas, 0, 0, temp.width, temp.height);
          cropCanvas.width = temp.width;
          cropCanvas.height = temp.height;
          cropCtx.drawImage(temp, 0, 0);
        }

        // --- Run OCR ---
        (async () => {
          try {
            const worker = workerRef.current!;
            const {
              data: { text, confidence },
            } = await worker.recognize(cropCanvas);

            const cleanText = text.trim().replace(/[^A-Z0-9- ]/g, ""); // allow space

            console.log("OCR raw result:", { text, confidence });
            console.log("Cleaned text:", cleanText);

            if (cleanText.length >= 5 && confidence >= 30) {
              console.log("LP: " + cleanText);

              // Save to refs (for persistence across frames)
              lastPlateTextRef.current = cleanText;
              lastConfidenceRef.current = confidence;
              lastBoxRef.current = {
                x: scaledX,
                y: scaledY,
                w: scaledW,
                h: scaledH,
              };

              ctx.save();
              ctx.fillStyle = "lime";
              ctx.font = "bold 24px Arial";
              ctx.fillText(cleanText, scaledX, scaledY - 12);

              ctx.font = "14px Arial";
              ctx.fillText(
                `conf: ${confidence.toFixed(0)}%`,
                scaledX,
                scaledY - 28,
              );
              ctx.restore();
            }
          } catch (err) {
            console.error("OCR failed:", err);
          }
        })();
      }
    }

    // Persistent overlay — this is what actually keeps text visible in live video
    if (lastBoxRef.current && lastPlateTextRef.current) {
      const { x, y } = lastBoxRef.current;

      // Draw text
      ctx.fillStyle = "lime";
      ctx.font = "bold 24px Arial";
      ctx.fillText(lastPlateTextRef.current, x, y - 12);

      if (lastConfidenceRef.current !== null) {
        ctx.font = "14px Arial";
        ctx.fillText(
          `conf: ${lastConfidenceRef.current.toFixed(0)}%`,
          x,
          y - 28,
        );
      }
    }

    // If no detection, just keep the clean video frame (already drawn)
  };

  // Non-Max Suppression, for filtering overlapping boxes based on IoU and confidence scores
  const nms = (
    boxes: number[][],
    scores: number[],
    iouThreshold: number = 0.45,
  ): number[][] => {
    const picked: number[][] = [];
    let idxs = scores
      .map((s, i) => [s, i] as const)
      .sort((a, b) => b[0] - a[0])
      .map(([, i]) => i);

    while (idxs.length > 0) {
      const i = idxs.shift()!;
      picked.push(boxes[i]);

      idxs = idxs.filter((j) => {
        const iou = boxIoU(boxes[i], boxes[j]);
        return iou <= iouThreshold; // keep if IoU is low (not too overlapping)
      });
    }

    return picked;
  };

  // boxIoU remains the same

  const boxIoU = (a: number[], b: number[]) => {
    const [x1, y1, w1, h1] = a;
    const [x2, y2, w2, h2] = b;

    const interW = Math.max(0, Math.min(x1 + w1, x2 + w2) - Math.max(x1, x2));
    const interH = Math.max(0, Math.min(y1 + h1, y2 + h2) - Math.max(y1, y2));

    const interArea = interW * interH;
    const unionArea = w1 * h1 + w2 * h2 - interArea;
    return interArea / unionArea;
  };

  // Start camera and inference on mount
  useEffect(() => {
    startCamera().then(() => loadModel().then(runLiveInference));

    (async () => {
      try {
        workerRef.current = await Tesseract.createWorker("eng", 1, {
          workerPath:
            "https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/worker.min.js",
          langPath:
            "https://cdn.jsdelivr.net/npm/@tesseract.js-data/eng@1.0.0/4.0.0_best_int",
          corePath:
            "https://cdn.jsdelivr.net/npm/tesseract.js-core@5/tesseract-core.wasm.js",
          logger: (m) => console.log(m), // optional progress logging
        });

        // Set global/default parameters once here (recommended for performance)
        await workerRef.current.setParameters({
          tessedit_char_whitelist: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -",
          tessedit_pageseg_mode: PSM.SINGLE_WORD, // PSM.SINGLE_WORD = 8 → best for license plates e.g NBC1234.
          // tessedit_pageseg_mode: '7', // PSM.SINGLE_LINE if plates have spaces/sections e.g NBC 1234
          preserve_interword_spaces: "0", // usually good for plates
        });

        console.log(
          "Tesseract worker ready with parameters set, using jsDelivr",
        );
      } catch (err) {
        console.error("Worker init failed:", err);
      }
    })();

    return () => {
      workerRef.current?.terminate();
    };
  }, []);

  return (
    <div className="min-h-screen bg-base-200 flex flex-col items-center justify-center p-5 gap-4">
      <h1 className="text-2xl font-bold">Live YOLO License Plate Detector</h1>

      <div className="relative w-full max-w-md">
        <video ref={videoRef} className="w-full rounded-lg" playsInline muted />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
      </div>
    </div>
  );
}
