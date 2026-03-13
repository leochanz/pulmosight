import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { PatientCard } from "@/components/PatientCard";
import { FileUpload } from "@/components/FileUpload";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { mockPatients } from "@/data/mockPatients";
import { Patient } from "@/types/patient";
import { Search, LogOut, Play, User, Calendar, FileText } from "lucide-react";

const Dashboard = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);

  const navigate = useNavigate();

  const filteredPatients = mockPatients.filter(
    (patient) =>
      patient.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      patient.id.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  // const handleStartAnalysis = () => {
  //   if (selectedFile && selectedPatient) {
  //     setIsAnalyzing(true);
  //   }
  // };
  const handleStartAnalysis = async () => {
    if (!selectedFile || !selectedPatient) return;

    setAnalysisError(null);
    setIsStarting(true);

    try {
      const allowed = [".dcm", ".zip", ".nii", ".nii.gz"];
      const lowerName = selectedFile.name.toLowerCase();
      const isAllowed = allowed.some((ext) => lowerName.endsWith(ext));
      if (!isAllowed) {
        throw new Error(
          "Unsupported file type. Please upload DICOM/ZIP/NIfTI.",
        );
      }

      const formData = new FormData();
      formData.append("ctScan", selectedFile);
      formData.append("patientId", selectedPatient.id);
      formData.append("patientName", selectedPatient.name);
      formData.append("age", String(selectedPatient.age));
      formData.append("gender", selectedPatient.gender);

      const res = await fetch(
        `${import.meta.env.VITE_API_BASE_URL}/api/analysis/start`,
        {
          method: "POST",
          body: formData,
        },
      );

      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || `Upload failed (${res.status})`);
      }

      const data: { jobId?: string } = await res.json();
      console.log("[Dashboard] /start response:", data);
      console.log("[Dashboard] selected file:", {
        name: selectedFile.name,
        type: selectedFile.type,
        size: selectedFile.size,
      });
      if (!data.jobId) throw new Error("Missing jobId from server response.");

      sessionStorage.setItem(
        "analysisContext",
        JSON.stringify({
          jobId: data.jobId,
          patient: selectedPatient,
          fileName: selectedFile.name,
          startedAt: Date.now(),
        }),
      );
      navigate("/analysis", {
        state: { patient: selectedPatient, jobId: data.jobId },
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to start analysis";
      setAnalysisError(message);
    } finally {
      setIsStarting(false);
    }
  };

  const handleLogout = () => {
    navigate("/");
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <Logo />

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <User className="w-4 h-4" />
              <span>Dr. Smith</span>
            </div>
            <Button variant="ghost" size="sm" onClick={handleLogout}>
              <LogOut className="w-4 h-4" />
              <span>Logout</span>
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Patient List */}
          <div className="lg:col-span-2 space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-semibold text-foreground">
                  Patients
                </h1>
                <p className="text-muted-foreground mt-1">
                  {filteredPatients.length} patients registered
                </p>
              </div>

              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Calendar className="w-4 h-4" />
                <span>
                  {new Date().toLocaleDateString("en-US", {
                    weekday: "long",
                    year: "numeric",
                    month: "long",
                    day: "numeric",
                  })}
                </span>
              </div>
            </div>

            {/* Search Bar */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground" />
              <Input
                type="text"
                placeholder="Search by patient name or ID..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>

            {/* Patient Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {filteredPatients.map((patient) => (
                <PatientCard
                  key={patient.id}
                  patient={patient}
                  isSelected={selectedPatient?.id === patient.id}
                  onClick={() => setSelectedPatient(patient)}
                />
              ))}
            </div>

            {filteredPatients.length === 0 && (
              <div className="text-center py-12">
                <FileText className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">
                  No patients found matching your search.
                </p>
              </div>
            )}
          </div>

          {/* Patient Details Pane */}
          <div className="lg:col-span-1">
            <div className="sticky top-24 space-y-6">
              {selectedPatient ? (
                <div className="bg-card border border-border rounded-lg p-6 space-y-6 animate-fade-in">
                  <div>
                    <h2 className="text-lg font-semibold text-foreground">
                      {selectedPatient.name}
                    </h2>
                    <p className="text-sm text-muted-foreground">
                      Patient ID: {selectedPatient.id}
                    </p>
                  </div>

                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-muted-foreground">Age</p>
                      <p className="font-medium text-foreground">
                        {selectedPatient.age} years
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Gender</p>
                      <p className="font-medium text-foreground">
                        {selectedPatient.gender}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Date of Birth</p>
                      <p className="font-medium text-foreground">
                        {new Date(
                          selectedPatient.dateOfBirth,
                        ).toLocaleDateString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Last Visit</p>
                      <p className="font-medium text-foreground">
                        {new Date(
                          selectedPatient.lastVisit,
                        ).toLocaleDateString()}
                      </p>
                    </div>
                  </div>

                  <div className="border-t border-border pt-6">
                    <h3 className="text-sm font-medium text-foreground mb-4">
                      Upload CT Scan for Analysis
                    </h3>

                    {analysisError && (
                      <p className="text-sm text-red-500 mb-3">
                        {analysisError}
                      </p>
                    )}

                    <>
                      <FileUpload
                        onFileSelect={setSelectedFile}
                        selectedFile={selectedFile}
                        onClear={() => setSelectedFile(null)}
                      />

                      <Button
                        variant="glow"
                        size="lg"
                        className="w-full mt-4"
                        disabled={!selectedFile || isStarting}
                        onClick={handleStartAnalysis}
                      >
                        <Play className="w-4 h-4" />
                        {isStarting ? "Starting..." : "Start Analysis"}
                      </Button>
                    </>
                  </div>
                </div>
              ) : (
                <div className="bg-card border border-border rounded-lg p-8 text-center">
                  <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto mb-4">
                    <User className="w-8 h-8 text-muted-foreground" />
                  </div>
                  <h3 className="text-lg font-medium text-foreground mb-2">
                    Select a Patient
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    Choose a patient from the list to view details and upload CT
                    scans for analysis.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
