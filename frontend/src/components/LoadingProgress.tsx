import { useEffect, useState } from 'react';

interface LoadingProgressProps {
  onComplete: () => void;
}

export const LoadingProgress = ({ onComplete }: LoadingProgressProps) => {
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState('Initializing analysis...');

  useEffect(() => {
    const stages = [
      { at: 0, text: 'Initializing analysis...' },
      { at: 20, text: 'Loading CT scan data...' },
      { at: 40, text: 'Preprocessing images...' },
      { at: 60, text: 'Running AI detection model...' },
      { at: 80, text: 'Generating segmentation...' },
      { at: 95, text: 'Finalizing results...' },
    ];

    const interval = setInterval(() => {
      setProgress((prev) => {
        const next = prev + Math.random() * 3 + 1;
        if (next >= 100) {
          clearInterval(interval);
          setTimeout(onComplete, 500);
          return 100;
        }
        
        const currentStage = stages.filter(s => s.at <= next).pop();
        if (currentStage) {
          setStage(currentStage.text);
        }
        
        return next;
      });
    }, 100);

    return () => clearInterval(interval);
  }, [onComplete]);

  return (
    <div className="space-y-6">
      <div className="text-center space-y-2">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/20 animate-pulse-glow">
          <div className="w-8 h-8 border-3 border-primary border-t-transparent rounded-full animate-spin" />
        </div>
        <h3 className="text-lg font-medium text-foreground">Analyzing CT Scan</h3>
        <p className="text-sm text-muted-foreground">{stage}</p>
      </div>
      
      <div className="space-y-2">
        <div className="h-2 bg-muted rounded-full overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-300 rounded-full"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-muted-foreground">Progress</span>
          <span className="text-primary font-medium">{Math.round(progress)}%</span>
        </div>
      </div>
    </div>
  );
};
