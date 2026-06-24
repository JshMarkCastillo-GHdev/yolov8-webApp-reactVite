type PlateAlertProps = {
  plate: string | null;
  confidence: number | null;
  emptyMessage?: string;
};

export function PlateAlert({
  plate,
  confidence,
  emptyMessage = "No plate recognised yet.",
}: PlateAlertProps) {
  if (!plate) {
    return <p className="text-sm text-gray-500">{emptyMessage}</p>;
  }

  return (
    <div className="alert alert-success shadow-lg">
      <div>
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="stroke-current shrink-0 h-6 w-6"
          fill="none"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M5 13l4 4L19 7"
          />
        </svg>
        <span>
          {plate}{" "}
          {confidence !== null && `(${confidence.toFixed(0)}% confidence)`}
        </span>
      </div>
    </div>
  );
}
