import { useEffect, useRef } from "react";
import type Tesseract from "tesseract.js";
import { getOrt, type OrtInferenceSession } from "../../../shared/lib/ort";
import type { BoundingBox, PlateDetection } from "../../../shared/types/plate";
import { INFERENCE_INTERVAL_MS } from "../../detection/lib/constants";
import { drawBox } from "../../detection/lib/drawOverlay";
import { runDetection } from "../../detection/lib/runDetection";

type LiveCameraViewProps = {
  session: OrtInferenceSession | null;
  worker: Tesseract.Worker | null;
  active: boolean;
  onStatus: (status: string) => void;
  onDetection: (detection: PlateDetection | null) => void;
};

export function LiveCameraView({
  session,
  worker,
  active,
  onStatus,
  onDetection,
}: LiveCameraViewProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastInferenceTime = useRef(0);

  const lastPlateTextRef = useRef<string | null>(null);
  const lastConfidenceRef = useRef<number | null>(null);
  const lastBoxRef = useRef<BoundingBox | null>(null);

  useEffect(() => {
    if (!active) return;

    let cancelled = false;
    const video = videoRef.current;

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        onStatus("camera started");
      } catch (err) {
        console.error("Camera error:", err);
        onStatus("camera error");
      }
    };

    const runLiveInference = () => {
      if (cancelled || !active) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (video && canvas && video.videoWidth > 0) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          if (lastBoxRef.current && lastPlateTextRef.current) {
            drawBox(
              ctx,
              lastBoxRef.current,
              lastPlateTextRef.current,
              lastConfidenceRef.current,
            );
          }
        }

        const now = performance.now();
        const ort = getOrt();
        if (
          session &&
          ort &&
          now - lastInferenceTime.current >= INFERENCE_INTERVAL_MS
        ) {
          lastInferenceTime.current = now;

          runDetection({
            source: video,
            session,
            ort,
            worker,
            canvas: null,
          })
            .then((result) => {
              if (cancelled || !canvas) return;
              const drawCtx = canvas.getContext("2d");
              if (!drawCtx) return;

              canvas.width = video.videoWidth;
              canvas.height = video.videoHeight;
              drawCtx.drawImage(video, 0, 0, canvas.width, canvas.height);

              if (result?.text) {
                lastPlateTextRef.current = result.text;
                lastConfidenceRef.current = result.ocrConfidence;
                lastBoxRef.current = result.box;
                onDetection(result);
                drawBox(
                  drawCtx,
                  result.box,
                  result.text,
                  result.ocrConfidence,
                );
              } else if (lastBoxRef.current && lastPlateTextRef.current) {
                drawBox(
                  drawCtx,
                  lastBoxRef.current,
                  lastPlateTextRef.current,
                  lastConfidenceRef.current,
                );
              }
            })
            .catch((err) => {
              console.error("Inference failed:", err);
            });
        }
      }

      rafRef.current = requestAnimationFrame(runLiveInference);
    };

    startCamera().then(() => {
      if (!cancelled) runLiveInference();
    });

    return () => {
      cancelled = true;
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      if (video) {
        video.srcObject = null;
      }
      lastPlateTextRef.current = null;
      lastConfidenceRef.current = null;
      lastBoxRef.current = null;
    };
  }, [active, session, worker, onStatus, onDetection]);

  return (
    <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
      <video
        ref={videoRef}
        className="w-full h-full object-cover"
        playsInline
        muted
      />
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 w-full h-full pointer-events-none"
      />
    </div>
  );
}
