import { AnalysisResult } from "@/types/patient";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useEffect, useState } from "react";

interface CTScanViewerProps {
  result: AnalysisResult;
  showSegmentation?: boolean;
  segmentImages?: string[];
}

export const CTScanViewer = ({
  result,
  showSegmentation = false,
  segmentImages = [],
}: CTScanViewerProps) => {
  const [currentSegmentIndex, setCurrentSegmentIndex] = useState(0);

  useEffect(() => {
    if (currentSegmentIndex >= segmentImages.length) {
      setCurrentSegmentIndex(0);
    }
  }, [currentSegmentIndex, segmentImages.length]);

  const handleNextSegment = () => {
    setCurrentSegmentIndex((prev) => (prev + 1) % segmentImages.length);
  };

  const handlePrevSegment = () => {
    setCurrentSegmentIndex(
      (prev) => (prev - 1 + segmentImages.length) % segmentImages.length,
    );
  };

  return (
    <div className="relative bg-card rounded-lg border border-border overflow-hidden">
      <div className="aspect-square bg-background relative">
        {/* Display actual CT scan images */}
        {!showSegmentation && result.originalScan && (
          <img
            src={result.originalScan}
            alt="CT Scan"
            className="absolute inset-0 w-full h-full object-cover"
          />
        )}

        {/* Display segmentation images with slider */}
        {showSegmentation && segmentImages.length > 0 && (
          <>
            <img
              src={segmentImages[currentSegmentIndex]}
              alt={`Segmentation ${currentSegmentIndex + 1}`}
              className="absolute inset-0 w-full h-full object-cover"
            />

            {/* Navigation buttons */}
            <div className="absolute inset-x-0 top-1/2 transform -translate-y-1/2 flex items-center justify-between px-4 pointer-events-none">
              <Button
                variant="secondary"
                size="icon"
                onClick={handlePrevSegment}
                className="pointer-events-auto shadow-lg"
                disabled={segmentImages.length <= 1}
              >
                <ChevronLeft className="w-4 h-4" />
              </Button>
              <Button
                variant="secondary"
                size="icon"
                onClick={handleNextSegment}
                className="pointer-events-auto shadow-lg"
                disabled={segmentImages.length <= 1}
              >
                <ChevronRight className="w-4 h-4" />
              </Button>
            </div>

            {/* Segment indicator */}
            <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex items-center gap-2 bg-background/80 backdrop-blur-sm px-3 py-1.5 rounded-full">
              {segmentImages.map((_, index) => (
                <div
                  key={index}
                  className={`w-2 h-2 rounded-full transition-colors ${
                    index === currentSegmentIndex
                      ? "bg-primary"
                      : "bg-muted-foreground/50"
                  }`}
                />
              ))}
              <span className="text-xs text-foreground ml-1">
                {currentSegmentIndex + 1} / {segmentImages.length}
              </span>
            </div>
          </>
        )}

        {showSegmentation && segmentImages.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/80">
            <p className="text-sm text-muted-foreground">
              Segmentation output is not available for this scan.
            </p>
          </div>
        )}

        {/* Fallback gradient if no images */}
        {!result.originalScan && !showSegmentation && (
          <div className="absolute inset-0 bg-gradient-radial from-muted to-background" />
        )}

        {/* Grid overlay */}
        <div
          className="absolute inset-0 opacity-30 pointer-events-none"
          style={{
            backgroundImage:
              "linear-gradient(to right, hsl(var(--border)) 1px, transparent 1px), linear-gradient(to bottom, hsl(var(--border)) 1px, transparent 1px)",
            backgroundSize: "40px 40px",
          }}
        />

        {/* Coordinate markers - only show in original view */}
        {!showSegmentation &&
          result.coordinates.map((coord, index) => (
            <div
              key={index}
              className="absolute transform -translate-x-1/2 -translate-y-1/2"
              style={{ left: `${coord.x}%`, top: `${coord.y}%` }}
            >
              <div className="relative">
                <div className="w-12 h-12 border-3 border-destructive rounded-full animate-pulse" />
                <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 whitespace-nowrap">
                  <span className="px-3 py-1 bg-destructive text-destructive-foreground text-sm font-medium rounded">
                    {coord.label}
                  </span>
                </div>
              </div>
            </div>
          ))}
      </div>

      {/* Axis labels */}
      <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 text-xs text-muted-foreground">
        X-Axis (mm)
      </div>
      <div className="absolute left-2 top-1/2 transform -translate-y-1/2 -rotate-90 text-xs text-muted-foreground">
        Y-Axis (mm)
      </div>
    </div>
  );
};
