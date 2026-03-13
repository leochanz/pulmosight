import { useNavigate, useLocation } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { CTScanViewer } from "@/components/CTScanViewer";
import { MalignancyChart } from "@/components/MalignancyChart";
import { Button } from "@/components/ui/button";
import { AnalysisResult, Patient } from "@/types/patient";
import { ArrowLeft, Download, FileText, Layers } from "lucide-react";
import { useState, useEffect, useMemo, useRef } from "react";
import { toast } from "@/hooks/use-toast";

const emptyAnalysisResult: AnalysisResult = {
  malignancyScore: 0,
  confidence: 0,
  noduleCount: 0,
  coordinates: [],
  segmentationData: "",
  originalScan: "",
  findings: [],
};

type PipelineStage =
  | "idle"
  | "queued"
  | "classifying"
  | "segmenting"
  | "complete"
  | "error"
  | "cancelled";

type ClassificationResult = {
  has_cancer: boolean;
  confidence: number;
  label?: string;
};

type ApiAnalysisResponse = {
  status?: "pending" | "running" | "completed" | "failed";
  stage?: "queued" | "classification" | "segmentation" | "completed";
  error?: string;
  warnings?: string[];
  requestId?: string;
  classification?: {
    has_cancer?: boolean;
    confidence?: number;
    label?: string;
    threshold?: number;
    processing_time?: number;
  };
  segmentation?: {
    processing_time?: number;
    shape?: number[];
    maskUrl?: string;
    overlayUrl?: string;
    error?: string;
    failed?: boolean;
  } | null;
  malignancyScore?: number;
  confidence?: number;
  noduleCount?: number;
  coordinates?: Array<{ x: number; y: number; label: string }>;
  findings?: string[];
  originalScan?: string;
  segmentationImages?: string[];
};

const Analysis = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const patient = location.state?.patient as Patient | undefined;
  const routeJobId = location.state?.jobId as string | undefined;
  const [showSegmentation, setShowSegmentation] = useState(false);

  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [segmentImages, setSegmentImages] = useState<string[]>([]);
  const [classification, setClassification] =
    useState<ClassificationResult | null>(null);
  const [pipelineStage, setPipelineStage] = useState<PipelineStage>("idle");
  const [stageMessage, setStageMessage] = useState(
    "Waiting to start analysis...",
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [segmentationError, setSegmentationError] = useState<string | null>(
    null,
  );
  const [pollTick, setPollTick] = useState(0);
  const [pollingCancelled, setPollingCancelled] = useState(false);
  const activeController = useRef<AbortController | null>(null);

  const hasSegmentation = segmentImages.length > 0;

  const updateStageUI = (
    stage?: ApiAnalysisResponse["stage"],
    status?: ApiAnalysisResponse["status"],
  ) => {
    if (status === "completed") {
      setPipelineStage("complete");
      setStageMessage("Analysis completed.");
      return;
    }
    if (status === "failed") {
      setPipelineStage("error");
      setStageMessage("Analysis failed.");
      return;
    }
    if (stage === "classification") {
      setPipelineStage("classifying");
      setStageMessage("Analyzing scan for cancer indicators...");
      return;
    }
    if (stage === "segmentation") {
      setPipelineStage("segmenting");
      setStageMessage(
        "Cancer detected. Running detailed segmentation analysis...",
      );
      return;
    }
    setPipelineStage("queued");
    setStageMessage("Scan uploaded. Preparing analysis job...");
  };

  const jobId = useMemo(() => {
    if (routeJobId) return routeJobId;
    const raw = sessionStorage.getItem("analysisContext");
    if (!raw) return undefined;
    try {
      const parsed = JSON.parse(raw) as { jobId?: string };
      return parsed.jobId;
    } catch {
      return undefined;
    }
  }, [routeJobId]);

  useEffect(() => {
    if (!jobId) return;
    if (pollingCancelled) return;

    let cancelled = false;
    let timer: number | null = null;

    const fetchResult = async () => {
      activeController.current?.abort();
      const controller = new AbortController();
      activeController.current = controller;

      try {
        const res = await fetch(
          `${import.meta.env.VITE_API_BASE_URL}/api/analysis/result/${jobId}`,
          { signal: controller.signal },
        );
        if (!res.ok) throw new Error(`Failed to fetch result (${res.status})`);
        const data: ApiAnalysisResponse = await res.json();

        console.log("[Analysis] raw /result payload:", data);

        if (cancelled) return;
        setErrorMessage(null);

        updateStageUI(data.stage, data.status);

        if (data.classification) {
          setClassification({
            has_cancer: Boolean(data.classification.has_cancer),
            confidence: Number(data.classification.confidence ?? 0),
            label: data.classification.label,
          });
        }

        if (data.status === "failed") {
          throw new Error(data.error || "Analysis failed");
        }

        if (data.status !== "completed") {
          timer = window.setTimeout(fetchResult, 2500);
          return;
        }

        const mappedResult: AnalysisResult = {
          malignancyScore: data.malignancyScore ?? 0,
          confidence: data.confidence ?? 0,
          noduleCount: data.noduleCount ?? 0,
          coordinates: data.coordinates ?? [],
          segmentationData: "",
          originalScan: data.originalScan ?? "",
          findings: data.findings ?? [],
        };

        setResult(mappedResult);

        const nextSegmentImages = data.segmentationImages ?? [];
        setSegmentImages(nextSegmentImages);

        if (data.segmentation?.failed) {
          setSegmentationError(
            data.segmentation.error ||
              "Segmentation failed after positive classification.",
          );
        } else {
          setSegmentationError(null);
        }

        if (nextSegmentImages.length === 0) {
          setShowSegmentation(false);
        }
      } catch (e) {
        if (cancelled) return;
        if (e instanceof DOMException && e.name === "AbortError") return;

        console.error("[Analysis] polling error:", e);
        const message =
          e instanceof Error ? e.message : "Failed to fetch analysis result";
        setErrorMessage(message);
        setPipelineStage("error");
        setStageMessage("Unable to retrieve analysis status.");
      }
    };

    fetchResult();
    return () => {
      cancelled = true;
      activeController.current?.abort();
      if (timer) window.clearTimeout(timer);
    };
  }, [jobId, pollTick, pollingCancelled]);

  const handleCancelPolling = () => {
    setPollingCancelled(true);
    activeController.current?.abort();
    setPipelineStage("cancelled");
    setStageMessage("Analysis polling cancelled. You can retry anytime.");
  };

  const handleRetryPolling = () => {
    setPollingCancelled(false);
    setErrorMessage(null);
    setPipelineStage("queued");
    setPollTick((prev) => prev + 1);
  };

  const handleDownloadPDF = () => {
    toast({
      title: "Download Started",
      description: "Your analysis report is being generated...",
    });

    // Simulate PDF generation
    setTimeout(() => {
      toast({
        title: "Download Complete",
        description: "Analysis report has been saved.",
      });
    }, 2000);
  };

  const classificationText = classification
    ? classification.has_cancer
      ? "Cancer indicators detected"
      : "No cancer indicators detected"
    : "Classification pending";

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <Logo />

          <div className="flex items-center gap-4">
            {patient && (
              <div className="text-sm text-muted-foreground">
                Patient:{" "}
                <span className="text-foreground font-medium">
                  {patient.name}
                </span>
                <span className="text-border mx-2">|</span>
                ID:{" "}
                <span className="text-foreground font-medium">
                  {patient.id}
                </span>
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Back Button & Title */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Button
              variant="outline"
              size="sm"
              onClick={() => navigate("/dashboard")}
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Dashboard
            </Button>

            <div>
              <h1 className="text-2xl font-semibold text-foreground">
                CT Scan Analysis Results
              </h1>
              <p className="text-muted-foreground mt-1">
                Analysis completed •{" "}
                {new Date().toLocaleDateString("en-US", {
                  year: "numeric",
                  month: "long",
                  day: "numeric",
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {(pipelineStage === "queued" ||
              pipelineStage === "classifying" ||
              pipelineStage === "segmenting") && (
              <Button variant="outline" onClick={handleCancelPolling}>
                Cancel Processing
              </Button>
            )}
            <Button variant="glow" onClick={handleDownloadPDF}>
              <Download className="w-4 h-4" />
              Download Results as PDF
            </Button>
          </div>
        </div>

        <div className="mb-6 rounded-lg border border-border bg-card p-4">
          <p className="text-sm text-muted-foreground">Pipeline Stage</p>
          <p className="text-base font-medium text-foreground">
            {stageMessage}
          </p>
          <p className="text-sm mt-1 text-muted-foreground">
            {classificationText}
          </p>
          {classification && (
            <p className="text-sm text-foreground mt-1">
              Confidence: {Math.round(classification.confidence * 100)}%
            </p>
          )}
          {errorMessage && (
            <div className="mt-3">
              <p className="text-sm text-destructive">{errorMessage}</p>
              <div className="mt-2 flex gap-2">
                <Button size="sm" onClick={handleRetryPolling}>
                  Retry
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => navigate("/dashboard")}
                >
                  Restart
                </Button>
              </div>
            </div>
          )}
          {segmentationError && (
            <p className="text-sm text-warning mt-2">{segmentationError}</p>
          )}
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* CT Scan Viewers */}
          <div className="space-y-6">
            {/* Toggle */}
            <div className="flex items-center gap-4">
              <Button
                variant={!showSegmentation ? "default" : "outline"}
                size="sm"
                onClick={() => setShowSegmentation(false)}
              >
                <FileText className="w-4 h-4" />
                Original Scan
              </Button>
              <Button
                variant={showSegmentation ? "default" : "outline"}
                size="sm"
                onClick={() => setShowSegmentation(true)}
                disabled={!hasSegmentation}
              >
                <Layers className="w-4 h-4" />
                Segmentation View
              </Button>
            </div>

            {/* CT Scan Viewer */}
            <CTScanViewer
              result={result ?? emptyAnalysisResult}
              showSegmentation={showSegmentation}
              segmentImages={segmentImages}
            />

            {!hasSegmentation && classification?.has_cancer === false && (
              <div className="bg-card border border-border rounded-lg p-4 text-sm text-muted-foreground">
                Segmentation was skipped because classification was negative.
              </div>
            )}

            {/* Legend */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="text-sm font-medium text-foreground mb-3">
                Coordinate Legend
              </h3>
              <div className="space-y-2">
                {(result?.coordinates ?? []).map((coord, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between text-sm"
                  >
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 bg-destructive text-destructive-foreground text-xs rounded">
                        {coord.label}
                      </span>
                      <span className="text-muted-foreground">
                        Nodule {index + 1}
                      </span>
                    </div>
                    <span className="text-foreground font-mono">
                      ({coord.x}%, {coord.y}%)
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Analysis Results */}
          <div className="space-y-6">
            <div>
              <h2 className="text-lg font-semibold text-foreground mb-4">
                Analysis Metrics
              </h2>
              <MalignancyChart result={result ?? emptyAnalysisResult} />
            </div>

            {classification && (
              <div className="bg-card border border-border rounded-lg p-4">
                <h3 className="text-sm font-medium text-foreground mb-2">
                  Classification Result
                </h3>
                <p className="text-sm text-muted-foreground">
                  {classification.has_cancer
                    ? "Cancer detected by classification model."
                    : "No cancer detected by classification model."}
                </p>
                <p className="text-sm text-foreground mt-1">
                  Confidence: {Math.round(classification.confidence * 100)}%
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Analysis;
