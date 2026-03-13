import { Patient } from '@/types/patient';
import { User, Calendar, Activity } from 'lucide-react';
import { cn } from '@/lib/utils';

interface PatientCardProps {
  patient: Patient;
  isSelected: boolean;
  onClick: () => void;
}

export const PatientCard = ({ patient, isSelected, onClick }: PatientCardProps) => {
  const statusColors = {
    Active: 'bg-success/20 text-success border-success/30',
    Pending: 'bg-warning/20 text-warning border-warning/30',
    Completed: 'bg-info/20 text-info border-info/30',
  };

  return (
    <div
      onClick={onClick}
      className={cn(
        "p-4 rounded-lg border cursor-pointer transition-all duration-200",
        "hover:border-primary/50 hover:bg-secondary/50",
        isSelected 
          ? "border-primary bg-secondary glow-border" 
          : "border-border bg-card"
      )}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-muted flex items-center justify-center">
            <User className="w-5 h-5 text-muted-foreground" />
          </div>
          <div>
            <h3 className="font-medium text-foreground">{patient.name}</h3>
            <p className="text-sm text-muted-foreground">ID: {patient.id}</p>
          </div>
        </div>
        <span className={cn(
          "px-2 py-1 text-xs rounded-full border",
          statusColors[patient.status]
        )}>
          {patient.status}
        </span>
      </div>
      
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Calendar className="w-4 h-4" />
          <span>{patient.age} yrs</span>
        </div>
        <div className="flex items-center gap-2 text-muted-foreground">
          <Activity className="w-4 h-4" />
          <span>{patient.gender}</span>
        </div>
      </div>
      
      <div className="mt-3 pt-3 border-t border-border">
        <p className="text-xs text-muted-foreground">
          Last Visit: {new Date(patient.lastVisit).toLocaleDateString()}
        </p>
      </div>
    </div>
  );
};
