import { cn } from "@/lib/utils";
import type { SampleEntry } from "@/shared/types/sample";

type SampleSidebarProps = {
  samples: SampleEntry[];
  loading: boolean;
  selectedId: string | null;
  disabled?: boolean;
  onSelect: (sample: SampleEntry) => void;
};

export function SampleSidebar({
  samples,
  loading,
  selectedId,
  disabled,
  onSelect,
}: SampleSidebarProps) {
  return (
    <aside
      className={cn(
        "order-2 flex w-full shrink-0 flex-col bg-background",
        "border-t lg:order-1 lg:w-52 lg:min-h-0 lg:border-t-0 lg:border-r xl:w-60",
      )}
    >
      <div className="flex items-center justify-between gap-2 border-b px-3 py-2 sm:px-4 sm:py-3 lg:block">
        <div>
          <h2 className="text-sm font-semibold">Demo samples</h2>
          <p className="hidden text-xs text-muted-foreground sm:block lg:block">
            Scroll and tap a thumbnail
          </p>
        </div>
        {!loading && samples.length > 0 && (
          <span className="text-[10px] text-muted-foreground sm:hidden">
            Swipe →
          </span>
        )}
      </div>

      <div className="p-2 sm:p-3">
        {loading ? (
          <p className="px-1 text-xs text-muted-foreground">Loading samples…</p>
        ) : samples.length === 0 ? (
          <p className="px-1 text-xs text-muted-foreground">
            Add images to <code className="text-[10px]">public/samples/</code>
          </p>
        ) : (
          <div
            className={cn(
              "flex gap-2.5 overflow-x-auto pb-1 scrollbar-none snap-x snap-mandatory touch-pan-x",
              "sm:gap-3",
              "lg:max-h-[calc(100dvh-12rem)] lg:flex-col lg:overflow-x-hidden lg:overflow-y-auto lg:snap-none",
            )}
          >
            {samples.map((sample) => (
              <button
                key={sample.id}
                type="button"
                disabled={disabled}
                onClick={() => onSelect(sample)}
                className={cn(
                  "w-32 shrink-0 snap-start overflow-hidden rounded-lg border bg-card text-left transition-colors",
                  "touch-manipulation active:scale-[0.98] hover:bg-accent/50 disabled:opacity-50",
                  "sm:w-36 lg:w-full",
                  selectedId === sample.id && "ring-2 ring-primary",
                )}
              >
                <div className="aspect-video bg-black">
                  <img
                    src={sample.src}
                    alt={sample.title}
                    className="h-full w-full object-cover"
                    loading="lazy"
                    draggable={false}
                  />
                </div>
                <div className="space-y-0.5 p-2">
                  <p className="truncate text-xs font-medium">{sample.title}</p>
                  {sample.expectedPlate && (
                    <p className="truncate text-[10px] text-muted-foreground">
                      {sample.expectedPlate}
                    </p>
                  )}
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}
