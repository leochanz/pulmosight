import { AnalysisResult } from "@/types/patient";

interface MalignancyChartProps {
  result: AnalysisResult;
}

export const MalignancyChart = ({ result }: MalignancyChartProps) => {
  return (
    <div className="grid grid-cols-1 gap-4">
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
