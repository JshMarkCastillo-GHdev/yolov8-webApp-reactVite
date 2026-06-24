import { useCallback, useEffect, useState } from "react";
import logo from "./assets/Ultralytics YOLOv8.png";
import { LiveCameraView } from "./features/camera/components/LiveCameraView";
import { useYoloSession } from "./features/detection/hooks/useYoloSession";
import { SampleGallery } from "./features/input/components/SampleGallery";
import { UploadPanel } from "./features/input/components/UploadPanel";
import { useTesseractWorker } from "./features/ocr/hooks/useTesseractWorker";
import { ModeTabs } from "./features/plate-ui/components/ModeTabs";
import { PlateAlert } from "./features/plate-ui/components/PlateAlert";
import type { InputMode, PlateDetection } from "./shared/types/plate";

const MODE_TITLES: Record<InputMode, string> = {
  camera: "Live detection",
  upload: "Upload image",
  sample: "Sample gallery",
};

export default function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [inputMode, setInputMode] = useState<InputMode>("camera");
  const [status, setStatus] = useState("initialising…");
  const [detectedPlate, setDetectedPlate] = useState<string | null>(null);
  const [detectedConf, setDetectedConf] = useState<number | null>(null);

  const { session, status: modelStatus, setStatus: setModelStatus, ready: modelReady } =
    useYoloSession();
  const { worker, ready: ocrReady } = useTesseractWorker();

  const ready = modelReady && ocrReady;

  useEffect(() => {
    document.documentElement.setAttribute(
      "data-theme",
      darkMode ? "dark" : "light",
    );
  }, [darkMode]);

  const handleDetection = useCallback((detection: PlateDetection | null) => {
    setDetectedPlate(detection?.text ?? null);
    setDetectedConf(detection?.ocrConfidence ?? null);
  }, []);

  const handleModeChange = (mode: InputMode) => {
    setInputMode(mode);
    setDetectedPlate(null);
    setDetectedConf(null);

    if (mode === "camera") {
      setStatus(modelStatus);
    } else if (!ready) {
      setStatus("waiting for model…");
    } else {
      setStatus("ready");
    }
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
    sample: "Select a sample image to detect a plate.",
  };

  return (
    <div className="min-h-screen flex flex-col bg-base-200">
      <header className="navbar bg-base-100 shadow-md mb-4">
        <div className="flex-1 px-2 mx-2">
          <span className="text-lg font-bold">YOLO License Plate Detector</span>
        </div>
        <div className="flex-none space-x-2">
          <button
            className="btn btn-ghost btn-sm"
            onClick={() =>
              window.open("https://github.com/JshMarkCastillo-GHdev", "_blank")
            }
          >
            GitHub
          </button>

          <button
            className="btn btn-ghost btn-sm"
            onClick={() => setDarkMode((d) => !d)}
            title="Toggle dark/light mode"
          >
            {darkMode ? (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M12 3v1m0 16v1m8.66-9h-1M4.34 12h-1m13.06-6.06l-.7.7M6.34 17.66l-.7.7m13.06 0l-.7-.7M6.34 6.34l-.7-.7M12 5a7 7 0 100 14 7 7 0 000-14z"
                />
              </svg>
            ) : (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"
                />
              </svg>
            )}
          </button>

          <button
            className="btn btn-ghost btn-sm"
            onClick={() => window.location.reload()}
          >
            Reset
          </button>
        </div>
      </header>

      <main className="grow container mx-auto px-4">
        <div className="card bg-base-100 shadow-xl w-full max-w-3xl mx-auto">
          <div className="card-body">
            <h2 className="card-title">{MODE_TITLES[inputMode]}</h2>
            <p className="text-sm text-gray-500 mb-2">
              {inputMode === "camera" ? modelStatus : status}
            </p>

            <ModeTabs mode={inputMode} onChange={handleModeChange} />

            <div className="mt-4">
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
                <SampleGallery
                  session={session}
                  worker={worker}
                  ready={ready}
                  onStatus={setStatus}
                  onDetection={handleDetection}
                />
              )}
            </div>

            <div className="mt-4">
              <PlateAlert
                plate={detectedPlate}
                confidence={detectedConf}
                emptyMessage={emptyMessages[inputMode]}
              />
            </div>
          </div>
        </div>
      </main>

      <footer className="footer p-4 bg-base-100 text-base-content">
        <div className="items-center grid-flow-col">
          <p className="ml-2 text-lg font-semibold">Powered By</p>

          <a
            href="https://yolov8.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-ghost btn-sm ml-2 bg-white"
          >
            <img src={logo} alt="Ultralytics YOLOv8" className="w-20 h-6" />
          </a>
        </div>
      </footer>
    </div>
  );
}
