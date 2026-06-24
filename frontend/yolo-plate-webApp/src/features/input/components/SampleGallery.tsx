import { useEffect, useRef, useState } from "react";
import type Tesseract from "tesseract.js";
import { loadImageFromUrl } from "../../../shared/lib/imageSource";
import { getOrt, type OrtInferenceSession } from "../../../shared/lib/ort";
import type { PlateDetection } from "../../../shared/types/plate";
import type { SampleEntry } from "../../../shared/types/sample";
import { runDetection } from "../../detection/lib/runDetection";
import { DetectionCanvas } from "../../plate-ui/components/DetectionCanvas";

type SampleGalleryProps = {
  session: OrtInferenceSession | null;
  worker: Tesseract.Worker | null;
  ready: boolean;
  onStatus: (status: string) => void;
  onDetection: (detection: PlateDetection | null) => void;
};

export function SampleGallery({
  session,
  worker,
  ready,
  onStatus,
  onDetection,
}: SampleGalleryProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [samples, setSamples] = useState<SampleEntry[]>([]);
  const [loadingManifest, setLoadingManifest] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [analysing, setAnalysing] = useState(false);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const res = await fetch("/samples/samples.json");
        if (!res.ok) throw new Error("Manifest not found");
        const data = (await res.json()) as SampleEntry[];
        if (!cancelled) setSamples(Array.isArray(data) ? data : []);
      } catch (err) {
        console.error("Failed to load samples manifest:", err);
        if (!cancelled) setSamples([]);
      } finally {
        if (!cancelled) setLoadingManifest(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const handleSelect = async (sample: SampleEntry) => {
    const ort = getOrt();
    if (!session || !ort || !ready) {
      onStatus("model not ready");
      return;
    }

    setSelectedId(sample.id);
    setAnalysing(true);
    onDetection(null);
    onStatus(`loading ${sample.title}…`);

    try {
      const img = await loadImageFromUrl(sample.src);
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
      console.error("Sample analysis failed:", err);
      onStatus("sample load error — check image file exists");
      onDetection(null);
    } finally {
      setAnalysing(false);
    }
  };

  if (loadingManifest) {
    return <p className="text-sm text-gray-500">Loading samples…</p>;
  }

  if (samples.length === 0) {
    return (
      <div className="space-y-4">
        <p className="text-sm text-gray-500">
          No samples yet. Add images to{" "}
          <code className="text-xs">public/samples/</code> and entries to{" "}
          <code className="text-xs">samples.json</code>.
        </p>
        <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden flex items-center justify-center">
          <DetectionCanvas ref={canvasRef} />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-base-content/70">
        Curated demo images — click a thumbnail to run detection.
      </p>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {samples.map((sample) => (
          <button
            key={sample.id}
            type="button"
            className={`card card-compact bg-base-200 cursor-pointer hover:bg-base-300 transition-colors ${
              selectedId === sample.id ? "ring-2 ring-primary" : ""
            }`}
            onClick={() => handleSelect(sample)}
            disabled={!ready || analysing}
          >
            <figure className="aspect-video bg-black">
              <img
                src={sample.src}
                alt={sample.title}
                className="w-full h-full object-cover"
              />
            </figure>
            <div className="card-body p-2">
              <p className="text-xs font-medium truncate">{sample.title}</p>
              {sample.expectedPlate && (
                <p className="text-xs text-base-content/60">
                  Expected: {sample.expectedPlate}
                </p>
              )}
            </div>
          </button>
        ))}
      </div>

      <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden flex items-center justify-center">
        <DetectionCanvas ref={canvasRef} />
        {analysing && (
          <span className="absolute loading loading-spinner loading-lg text-primary" />
        )}
        {!analysing && selectedId === null && (
          <p className="absolute text-sm text-gray-400">
            Select a sample above
          </p>
        )}
      </div>
    </div>
  );
}
