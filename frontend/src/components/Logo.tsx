import { Shield } from 'lucide-react';

export const Logo = () => {
  return (
    <div className="flex items-center gap-3">
      <div className="relative">
        <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-primary to-accent flex items-center justify-center shadow-lg shadow-primary/20">
          <Shield className="w-7 h-7 text-primary-foreground" />
        </div>
        <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-success rounded-full border-2 border-background" />
      </div>
      <div className="flex flex-col">
        <span className="text-xl font-semibold tracking-tight text-foreground">PulmoSight</span>
        <span className="text-xs text-muted-foreground tracking-wide uppercase">Healthcare System Portal</span>
      </div>
    </div>
  );
};
