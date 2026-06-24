import { useEffect, useState } from "react";
import { configureOrt, getOrt } from "../../../shared/lib/ort";
import type { OrtInferenceSession } from "../../../shared/lib/ort";

function initialStatus(): string {
  configureOrt();
  return getOrt() ? "initialising…" : "ONNX runtime not loaded";
}

export function useYoloSession() {
  const [session, setSession] = useState<OrtInferenceSession | null>(null);
  const [status, setStatus] = useState(initialStatus);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const ort = getOrt();
    if (!ort) return;

    let cancelled = false;

    (async () => {
      try {
        setStatus("loading model…");
        const loaded = await ort.InferenceSession.create("/models/best.onnx");
        if (cancelled) return;
        setSession(loaded);
        setStatus("model loaded");
        setReady(true);
      } catch (err) {
        console.error("MODEL LOAD ERROR:", err);
        if (!cancelled) setStatus("model load error");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  return { session, status, setStatus, ready };
}
