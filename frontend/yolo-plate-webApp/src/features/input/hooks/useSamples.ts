import { useEffect, useState } from "react";

import type { SampleEntry } from "@/shared/types/sample";

export function useSamples() {
  const [samples, setSamples] = useState<SampleEntry[]>([]);
  const [loading, setLoading] = useState(true);

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
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  return { samples, loading };
}
