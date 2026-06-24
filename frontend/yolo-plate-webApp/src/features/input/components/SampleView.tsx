import { useEffect, useRef, useState } from "react";
import type Tesseract from "tesseract.js";

import { Spinner } from "@/components/ui/spinner";
import { loadImageFromUrl } from "@/shared/lib/imageSource";
import { getOrt, type OrtInferenceSession } from "@/shared/lib/ort";
import type { PlateDetection } from "@/shared/types/plate";
import type { SampleEntry } from "@/shared/types/sample";
import { runDetection } from "@/features/detection/lib/runDetection";
import { DetectionCanvas } from "@/features/plate-ui/components/DetectionCanvas";
import { DetectionViewport } from "@/features/plate-ui/components/DetectionViewport";

type SampleViewProps = {
  session: OrtInferenceSession | null;
  worker: Tesseract.Worker | null;
  ready: boolean;
  activeSample: SampleEntry | null;
  onStatus: (status: string) => void;
  onDetection: (detection: PlateDetection | null) => void;
};

export function SampleView({
  session,
  worker,
  ready,
  activeSample,
  onStatus,
  onDetection,
}: SampleViewProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [analysing, setAnalysing] = useState(false);
  const activeSampleRef = useRef(activeSample);
  activeSampleRef.current = activeSample;

  useEffect(() => {
    const sample = activeSampleRef.current;
    if (!sample) return;

    let cancelled = false;

    (async () => {
      const ort = getOrt();
      if (!session || !ort || !ready) {
        onStatus("model not ready");
        return;
      }

      setAnalysing(true);
      onDetection(null);
      onStatus(`loading ${sample.title}…`);

      try {
        const img = await loadImageFromUrl(sample.src);
        if (cancelled) return;

        onStatus("analysing…");
        const result = await runDetection({
          source: img,
          session,
          ort,
          worker,
          canvas: canvasRef.current,
        });

        if (cancelled) return;
        onDetection(result);
        onStatus(result?.text ? "done" : "no plate detected");
      } catch (err) {
        console.error("Sample analysis failed:", err);
        if (!cancelled) {
          onStatus("sample load error — check image file exists");
          onDetection(null);
        }
      } finally {
        if (!cancelled) setAnalysing(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [activeSample?.id, session, worker, ready, onStatus, onDetection]);

  return (
    <DetectionViewport
      overlay={
        <>
          {analysing && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/40">
              <Spinner />
            </div>
          )}
          {!analysing && !activeSample && (
            <p className="absolute px-4 text-center text-sm text-muted-foreground">
              Pick a sample below
            </p>
          )}
        </>
      }
    >
      <DetectionCanvas ref={canvasRef} />
    </DetectionViewport>
  );
}
