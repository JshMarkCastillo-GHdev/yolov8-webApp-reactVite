import { useCallback, useState } from "react";
import logo from "@/assets/Ultralytics YOLOv8.png";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { LiveCameraView } from "@/features/camera/components/LiveCameraView";
import { useYoloSession } from "@/features/detection/hooks/useYoloSession";
import { SampleSidebar } from "@/features/input/components/SampleSidebar";
import { SampleView } from "@/features/input/components/SampleView";
import { UploadPanel } from "@/features/input/components/UploadPanel";
import { useSamples } from "@/features/input/hooks/useSamples";
import { useTesseractWorker } from "@/features/ocr/hooks/useTesseractWorker";
import { ModeTabs } from "@/features/plate-ui/components/ModeTabs";
import { PlateAlert } from "@/features/plate-ui/components/PlateAlert";
import { ThemeToggle } from "@/features/plate-ui/components/ThemeToggle";
import type { InputMode, PlateDetection } from "@/shared/types/plate";
import type { SampleEntry } from "@/shared/types/sample";

const MODE_TITLES: Record<InputMode, string> = {
  camera: "Live detection",
  upload: "Upload image",
  sample: "Sample detection",
};

export default function App() {
  const [inputMode, setInputMode] = useState<InputMode>("camera");
  const [status, setStatus] = useState("initialising…");
  const [detectedPlate, setDetectedPlate] = useState<string | null>(null);
  const [detectedConf, setDetectedConf] = useState<number | null>(null);
  const [activeSample, setActiveSample] = useState<SampleEntry | null>(null);
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);

  const { samples, loading: samplesLoading } = useSamples();
  const {
    session,
    status: modelStatus,
    setStatus: setModelStatus,
    ready: modelReady,
  } = useYoloSession();
  const { worker, ready: ocrReady } = useTesseractWorker();

  const ready = modelReady && ocrReady;

  const handleDetection = useCallback((detection: PlateDetection | null) => {
    setDetectedPlate(detection?.text ?? null);
    setDetectedConf(detection?.ocrConfidence ?? null);
  }, []);

  const handleModeChange = (mode: InputMode) => {
    if (mode === "sample") return;
    setInputMode(mode);
    setDetectedPlate(null);
    setDetectedConf(null);
    setActiveSample(null);
    setSelectedSampleId(null);

    if (mode === "camera") {
      setStatus(modelStatus);
    } else if (!ready) {
      setStatus("waiting for model…");
    } else {
      setStatus("ready");
    }
  };

  const handleSampleSelect = (sample: SampleEntry) => {
    setInputMode("sample");
    setSelectedSampleId(sample.id);
    setActiveSample(sample);
    setDetectedPlate(null);
    setDetectedConf(null);
  };

  const handleCameraStatus = useCallback(
    (cameraStatus: string) => {
      setStatus(cameraStatus);
      setModelStatus(cameraStatus);
    },
    [setModelStatus],
  );

  const emptyMessages: Record<InputMode, string> = {
    camera: "No plate recognised yet.",
    upload: "Upload an image to detect a plate.",
    sample: "Select a demo sample below.",
  };

  const statusLine = inputMode === "camera" ? modelStatus : status;

  return (
    <div className="flex min-h-[100dvh] flex-col bg-muted/40">
      <header className="sticky top-0 z-10 border-b bg-background/95 pt-[env(safe-area-inset-top)] shadow-sm backdrop-blur supports-[backdrop-filter]:bg-background/80">
        <div className="flex h-12 items-center justify-between gap-2 px-3 sm:h-14 sm:px-4 lg:px-6">
          <span className="truncate text-base font-bold sm:text-lg">
            <span className="sm:hidden">Plate Detector</span>
            <span className="hidden sm:inline">YOLO License Plate Detector</span>
          </span>
          <div className="flex shrink-0 items-center gap-0.5 sm:gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="sm:hidden"
              onClick={() =>
                window.open("https://github.com/JshMarkCastillo-GHdev", "_blank")
              }
              aria-label="GitHub"
            >
              <span className="text-xs font-semibold">GH</span>
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="hidden sm:inline-flex"
              onClick={() =>
                window.open("https://github.com/JshMarkCastillo-GHdev", "_blank")
              }
            >
              GitHub
            </Button>
            <ThemeToggle />
            <Button
              variant="ghost"
              size="icon"
              className="sm:hidden"
              onClick={() => window.location.reload()}
              aria-label="Reset"
            >
              <span className="text-xs">↺</span>
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="hidden sm:inline-flex"
              onClick={() => window.location.reload()}
            >
              Reset
            </Button>
          </div>
        </div>
      </header>

      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <main className="order-1 flex min-h-0 flex-1 flex-col p-3 sm:p-4 lg:order-2 lg:p-6">
          <Card className="flex min-h-0 flex-1 flex-col shadow-md">
            <CardHeader className="shrink-0 space-y-3 p-4 sm:space-y-4 sm:p-6">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between sm:gap-4">
                <div className="min-w-0">
                  <CardTitle className="text-lg sm:text-xl">
                    {MODE_TITLES[inputMode]}
                  </CardTitle>
                  <CardDescription className="truncate">
                    {statusLine}
                  </CardDescription>
                </div>
                <ModeTabs mode={inputMode} onChange={handleModeChange} />
              </div>
            </CardHeader>
            <CardContent className="flex min-h-0 flex-1 flex-col gap-3 p-4 pt-0 sm:gap-4 sm:p-6 sm:pt-0">
              {inputMode === "camera" && (
                <LiveCameraView
                  session={session}
                  worker={worker}
                  active={inputMode === "camera"}
                  onStatus={handleCameraStatus}
                  onDetection={handleDetection}
                />
              )}

              {inputMode === "upload" && (
                <UploadPanel
                  session={session}
                  worker={worker}
                  ready={ready}
                  onStatus={setStatus}
                  onDetection={handleDetection}
                />
              )}

              {inputMode === "sample" && (
                <SampleView
                  session={session}
                  worker={worker}
                  ready={ready}
                  activeSample={activeSample}
                  onStatus={setStatus}
                  onDetection={handleDetection}
                />
              )}

              <PlateAlert
                plate={detectedPlate}
                confidence={detectedConf}
                emptyMessage={emptyMessages[inputMode]}
              />
            </CardContent>
          </Card>
        </main>

        <SampleSidebar
          samples={samples}
          loading={samplesLoading}
          selectedId={selectedSampleId}
          disabled={!ready}
          onSelect={handleSampleSelect}
        />
      </div>

      <footer className="hidden border-t bg-background p-3 sm:block sm:p-4">
        <div className="flex items-center gap-2 px-2 lg:px-4">
          <p className="text-base font-semibold sm:text-lg">Powered By</p>
          <Button variant="ghost" size="sm" className="bg-white" asChild>
            <a
              href="https://yolov8.com/"
              target="_blank"
              rel="noopener noreferrer"
            >
              <img src={logo} alt="Ultralytics YOLOv8" className="h-6 w-20" />
            </a>
          </Button>
        </div>
      </footer>
    </div>
  );
}
