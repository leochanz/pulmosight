export interface Patient {
  id: string;
  name: string;
  age: number;
  gender: 'Male' | 'Female' | 'Other';
  dateOfBirth: string;
  lastVisit: string;
  status: 'Active' | 'Pending' | 'Completed';
}

export interface AnalysisResult {
  malignancyScore: number;
  confidence: number;
  noduleCount: number;
  coordinates: { x: number; y: number; label: string }[];
  segmentationData: string;
  originalScan: string;
  findings: string[];
}
