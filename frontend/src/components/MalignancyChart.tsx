import { AnalysisResult } from "@/types/patient";

interface MalignancyChartProps {
  result: AnalysisResult;
}

export const MalignancyChart = ({ result }: MalignancyChartProps) => {
  const getRiskLevel = (score: number) => {
    if (score < 30)
      return { label: "Low Risk", color: "text-success", bg: "bg-success" };
    if (score < 60)
      return {
        label: "Moderate Risk",
        color: "text-warning",
        bg: "bg-warning",
      };
    return {
      label: "High Risk",
      color: "text-destructive",
      bg: "bg-destructive",
    };
  };

  const risk = getRiskLevel(result.malignancyScore);

  return (
    <div className="space-y-6">
      {/* Main Score */}
      <div className="text-center p-6 bg-card rounded-lg border border-border">
        <p className="text-sm text-muted-foreground mb-2">Malignancy Score</p>
        <div className="relative inline-flex items-center justify-center">
          <svg className="w-32 h-32 transform -rotate-90">
            <circle
              cx="64"
              cy="64"
              r="56"
              stroke="currentColor"
              strokeWidth="8"
              fill="none"
              className="text-muted"
            />
            <circle
              cx="64"
              cy="64"
              r="56"
              stroke="currentColor"
              strokeWidth="8"
              fill="none"
              strokeDasharray={`${result.malignancyScore * 3.52} 352`}
              className={risk.color}
              strokeLinecap="round"
            />
          </svg>
          <div className="absolute flex flex-col items-center">
            <span className={`text-3xl font-bold ${risk.color}`}>
              {result.malignancyScore}%
            </span>
            <span className="text-xs text-muted-foreground">{risk.label}</span>
          </div>
        </div>
      </div>

      {/* Confidence Metrics */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-card rounded-lg border border-border">
          <p className="text-sm text-muted-foreground mb-2">Model Confidence</p>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-foreground font-medium">
                {result.confidence}%
              </span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-info rounded-full transition-all duration-500"
                style={{ width: `${result.confidence}%` }}
              />
            </div>
          </div>
        </div>

        <div className="p-4 bg-card rounded-lg border border-border">
          <p className="text-sm text-muted-foreground mb-2">Nodules Detected</p>
          <p className="text-2xl font-bold text-foreground">
            {result.noduleCount}
          </p>
        </div>
      </div>
    </div>
  );
};
