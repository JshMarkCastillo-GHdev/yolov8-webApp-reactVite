import { useEffect, useRef, useState } from "react";
import type Tesseract from "tesseract.js";
import { getOrt, type OrtInferenceSession } from "../../../shared/lib/ort";
import type { PlateDetection } from "../../../shared/types/plate";
import { runDetection } from "../../detection/lib/runDetection";
import { DetectionCanvas } from "../../plate-ui/components/DetectionCanvas";

type UploadPanelProps = {
  session: OrtInferenceSession | null;
  worker: Tesseract.Worker | null;
  ready: boolean;
  onStatus: (status: string) => void;
  onDetection: (detection: PlateDetection | null) => void;
};

export function UploadPanel({
  session,
  worker,
  ready,
  onStatus,
  onDetection,
}: UploadPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const [analysing, setAnalysing] = useState(false);

  const revokeObjectUrl = () => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  };

  useEffect(() => {
    return () => revokeObjectUrl();
  }, []);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    revokeObjectUrl();
    onDetection(null);
    setAnalysing(true);
    onStatus("loading image…");

    const ort = getOrt();
    if (!session || !ort || !ready) {
      onStatus("model not ready");
      setAnalysing(false);
      return;
    }

    try {
      const objectUrl = URL.createObjectURL(file);
      objectUrlRef.current = objectUrl;

      const img = new Image();
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error("Failed to load image"));
        img.src = objectUrl;
      });

      onStatus("analysing…");
      const result = await runDetection({
        source: img,
        session,
        ort,
        worker,
        canvas: canvasRef.current,
      });

      onDetection(result);
      onStatus(result?.text ? "done" : "no plate detected");
    } catch (err) {
      console.error("Upload analysis failed:", err);
      onStatus("analysis error");
      onDetection(null);
    } finally {
      setAnalysing(false);
      e.target.value = "";
    }
  };

  return (
    <div className="space-y-4">
      <p className="text-sm text-base-content/70">
        Your image is processed in your browser and is not uploaded to any
        server.
      </p>

      <input
        type="file"
        accept="image/*"
        className="file-input file-input-bordered w-full max-w-md"
        onChange={handleFileChange}
        disabled={!ready || analysing}
      />

      <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden flex items-center justify-center">
        <DetectionCanvas ref={canvasRef} />
        {!analysing && !canvasRef.current?.width && (
          <p className="absolute text-sm text-gray-400">
            Select an image to analyse
          </p>
        )}
        {analysing && (
          <span className="absolute loading loading-spinner loading-lg text-primary" />
        )}
      </div>
    </div>
  );
}
