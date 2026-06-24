import type { InputMode } from "../../../shared/types/plate";

type ModeTabsProps = {
  mode: InputMode;
  onChange: (mode: InputMode) => void;
};

const MODES: { id: InputMode; label: string }[] = [
  { id: "camera", label: "Camera" },
  { id: "upload", label: "Upload" },
  { id: "sample", label: "Samples" },
];

export function ModeTabs({ mode, onChange }: ModeTabsProps) {
  return (
    <div role="tablist" className="tabs tabs-boxed w-full max-w-md">
      {MODES.map(({ id, label }) => (
        <button
          key={id}
          role="tab"
          type="button"
          className={`tab flex-1 ${mode === id ? "tab-active" : ""}`}
          onClick={() => onChange(id)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
