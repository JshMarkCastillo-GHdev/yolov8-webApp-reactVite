import type { InputMode } from "@/shared/types/plate";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

type ModeTabsProps = {
  mode: InputMode;
  onChange: (mode: InputMode) => void;
};

const MODES: { id: Exclude<InputMode, "sample">; label: string }[] = [
  { id: "camera", label: "Camera" },
  { id: "upload", label: "Upload" },
];

export function ModeTabs({ mode, onChange }: ModeTabsProps) {
  const tabValue = mode === "sample" ? "camera" : mode;

  return (
    <Tabs
      value={tabValue}
      onValueChange={(value) => onChange(value as InputMode)}
      className="w-full sm:w-auto"
    >
      <TabsList className="grid h-10 w-full grid-cols-2 sm:inline-flex sm:w-auto">
        {MODES.map(({ id, label }) => (
          <TabsTrigger key={id} value={id}>
            {label}
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
}
