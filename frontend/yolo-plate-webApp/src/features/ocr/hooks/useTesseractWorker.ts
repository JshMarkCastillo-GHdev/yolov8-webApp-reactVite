import { useEffect, useState } from "react";
import Tesseract, { PSM } from "tesseract.js";

export function useTesseractWorker() {
  const [worker, setWorker] = useState<Tesseract.Worker | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let activeWorker: Tesseract.Worker | null = null;

    (async () => {
      try {
        activeWorker = await Tesseract.createWorker("eng");
        if (cancelled) {
          await activeWorker.terminate();
          return;
        }

        await activeWorker.setParameters({
          tessedit_char_whitelist: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -",
          tessedit_pageseg_mode: PSM.SINGLE_WORD,
          preserve_interword_spaces: "0",
        });

        setWorker(activeWorker);
        setReady(true);
      } catch (err) {
        console.error("Worker init failed:", err);
      }
    })();

    return () => {
      cancelled = true;
      activeWorker?.terminate();
    };
  }, []);

  return { worker, ready };
}
