import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";

interface HistoryItem {
  jobId: string;
  patientName: string;
  patientId: string;
  scanDate: string;
  malignancyScore: number | null;
  isCancer: boolean;
  confidence?: number | null;
}

const History = () => {
  const navigate = useNavigate();
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [sortedHighToLow, setSortedHighToLow] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const res = await fetch(
          `${import.meta.env.VITE_API_BASE_URL}/api/history`,
        );
        if (!res.ok) throw new Error(`History load failed: ${res.status}`);

        const data: HistoryItem[] = await res.json();
        setItems(data);
      } catch (error) {
        console.error("[History] failed to load history:", error);
        setItems([]);
      } finally {
        setLoading(false);
      }
    };

    loadHistory();
  }, []);

  const visibleItems = useMemo(() => {
    const next = [...items];

    if (sortedHighToLow) {
      next.sort((a, b) => {
        const scoreA = Number(a.malignancyScore ?? -1);
        const scoreB = Number(b.malignancyScore ?? -1);

        const aHasScore = Number.isFinite(scoreA);
        const bHasScore = Number.isFinite(scoreB);

        if (aHasScore && bHasScore) {
          return scoreB - scoreA;
        }
        if (aHasScore) return -1;
        if (bHasScore) return 1;
        return 0;
      });
    } else {
      next.sort((a, b) => {
        const dateA = a.scanDate ? new Date(a.scanDate).getTime() : 0;
        const dateB = b.scanDate ? new Date(b.scanDate).getTime() : 0;
        return dateB - dateA;
      });
    }

    return next;
  }, [items, sortedHighToLow]);

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-7xl">
        <div className="mb-8 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-muted-foreground">
              History
            </p>
            <h1 className="mt-2 text-3xl font-semibold text-foreground">
              CT Scan History
            </h1>
          </div>

          <div className="flex items-center gap-3">
            <Button variant="outline" onClick={() => navigate("/dashboard")}>
              Back to Dashboard
            </Button>
            <Button
              variant={sortedHighToLow ? "default" : "outline"}
              onClick={() => setSortedHighToLow((prev) => !prev)}
            >
              {sortedHighToLow
                ? "Reset to Date Order"
                : "Rank by Malignancy Score"}
            </Button>
          </div>
        </div>

        {loading ? (
          <div className="rounded-xl border border-border bg-card p-8 text-muted-foreground">
            Loading scan history...
          </div>
        ) : visibleItems.length === 0 ? (
          <div className="rounded-xl border border-dashed border-border bg-card p-12 text-center text-muted-foreground">
            No CT scans found in history.
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-3">
            {visibleItems.map((job) => {
              const score =
                typeof job.malignancyScore === "number"
                  ? job.malignancyScore
                  : 0;
              const borderClass = job.isCancer
                ? "border-red-500/70 bg-red-500/5"
                : "border-emerald-500/70 bg-emerald-500/5";

              return (
                <button
                  key={job.jobId}
                  type="button"
                  onClick={() =>
                    navigate(`/analysis/${job.jobId}`, {
                      state: {
                        fromHistory: true,
                        jobId: job.jobId,
                        patient: {
                          id: job.patientId,
                          name: job.patientName,
                        },
                      },
                    })
                  }
                  className={`rounded-2xl border p-5 text-left shadow-sm transition hover:-translate-y-0.5 hover:shadow-md ${borderClass}`}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-sm text-muted-foreground">Patient</p>
                      <h2 className="mt-1 text-xl font-semibold text-foreground">
                        {job.patientName || "Unknown Patient"}
                      </h2>
                    </div>

                    <span
                      className={`rounded-full px-2 py-1 text-xs font-medium ${
                        job.isCancer
                          ? "bg-red-500/10 text-red-400"
                          : "bg-emerald-500/10 text-emerald-400"
                      }`}
                    >
                      {job.isCancer ? "Cancer Detected" : "No Cancer"}
                    </span>
                  </div>

                  <div className="mt-5 space-y-2 text-sm text-muted-foreground">
                    <div className="flex items-center justify-between">
                      <span>Scan Date</span>
                      <span className="text-foreground">
                        {job.scanDate
                          ? new Date(job.scanDate).toLocaleDateString()
                          : "—"}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Malignancy</span>
                      <span className="text-foreground font-medium">
                        {score.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Confidence</span>
                      <span className="text-foreground">
                        {typeof job.confidence === "number"
                          ? `${job.confidence.toFixed(2)}%`
                          : "—"}
                      </span>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default History;
