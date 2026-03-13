import { useState, useCallback } from "react";
import { Upload, FileImage, X, CheckCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
}

export const FileUpload = ({
  onFileSelect,
  selectedFile,
  onClear,
}: FileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragging(true);
    } else if (e.type === "dragleave") {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        const file = e.dataTransfer.files[0];
        if (file.type.startsWith("image/") || file.name.endsWith(".dcm")) {
          onFileSelect(file);
        }
      }
    },
    [onFileSelect],
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  if (selectedFile) {
    return (
      <div className="border border-primary/50 rounded-lg p-4 bg-primary/5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-primary" />
            </div>
            <div>
              <p className="font-medium text-foreground">{selectedFile.name}</p>
              <p className="text-sm text-muted-foreground">
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
              </p>
            </div>
          </div>
          <button
            onClick={onClear}
            className="p-2 hover:bg-destructive/20 rounded-lg transition-colors group"
          >
            <X className="w-5 h-5 text-muted-foreground group-hover:text-destructive" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      className={cn(
        "relative border-2 border-dashed rounded-lg p-8 transition-all duration-200 cursor-pointer",
        isDragging
          ? "border-primary bg-primary/10"
          : "border-border hover:border-primary/50 hover:bg-secondary/50",
      )}
    >
      <input
        type="file"
        accept="image/*,.dcm"
        onChange={handleChange}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
      />
      <div className="flex flex-col items-center gap-3 text-center">
        <div
          className={cn(
            "w-14 h-14 rounded-full flex items-center justify-center transition-colors",
            isDragging ? "bg-primary/20" : "bg-muted",
          )}
        >
          {isDragging ? (
            <FileImage className="w-7 h-7 text-primary" />
          ) : (
            <Upload className="w-7 h-7 text-muted-foreground" />
          )}
        </div>
        <div>
          <p className="font-medium text-foreground">
            {isDragging ? "Drop CT scan here" : "Drag & drop CT scan"}
          </p>
          <p className="text-sm text-muted-foreground mt-1">
            or click to browse files (DICOM, PNG, JPG)
          </p>
        </div>
      </div>
    </div>
  );
};
