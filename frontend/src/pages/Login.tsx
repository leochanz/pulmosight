import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Logo } from '@/components/Logo';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Lock, User, ArrowRight, Shield } from 'lucide-react';

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    // Simulate login
    setTimeout(() => {
      setIsLoading(false);
      navigate('/dashboard');
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-background flex">
      {/* Left side - Branding */}
      <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-background to-background" />
        <div 
          className="absolute inset-0 opacity-20"
          style={{
            backgroundImage: 'linear-gradient(to right, hsl(var(--border)) 1px, transparent 1px), linear-gradient(to bottom, hsl(var(--border)) 1px, transparent 1px)',
            backgroundSize: '60px 60px',
          }}
        />
        
        {/* Glow effect */}
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-64 h-64 bg-accent/20 rounded-full blur-3xl" />
        
        <div className="relative z-10 flex flex-col justify-center items-start p-16">
          <Logo />
          
          <h1 className="mt-12 text-4xl font-semibold text-foreground leading-tight max-w-md">
            Advanced CT Scan Analysis Portal
          </h1>
          
          <p className="mt-4 text-lg text-muted-foreground max-w-md">
            Secure access for authorized healthcare professionals. AI-powered diagnostics at your fingertips.
          </p>
          
          <div className="mt-12 space-y-4">
            {[
              'AI-Powered Lung Nodule Detection',
              'Secure Patient Data Management',
              'Real-time Analysis Results',
            ].map((feature, index) => (
              <div key={index} className="flex items-center gap-3 text-muted-foreground animate-fade-in" style={{ animationDelay: `${index * 100}ms` }}>
                <div className="w-2 h-2 rounded-full bg-primary" />
                <span>{feature}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      {/* Right side - Login Form */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-8">
        <div className="w-full max-w-md space-y-8 animate-slide-up">
          <div className="lg:hidden mb-8">
            <Logo />
          </div>
          
          <div>
            <h2 className="text-2xl font-semibold text-foreground">Doctor Login</h2>
            <p className="mt-2 text-muted-foreground">
              Enter your credentials to access the portal
            </p>
          </div>
          
          <form onSubmit={handleLogin} className="space-y-6">
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Username</label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                  <Input
                    type="text"
                    placeholder="Enter your username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    className="pl-10"
                    required
                  />
                </div>
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Password</label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                  <Input
                    type="password"
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="pl-10"
                    required
                  />
                </div>
              </div>
            </div>
            
            <Button
              type="submit"
              variant="glow"
              size="xl"
              className="w-full"
              disabled={isLoading}
            >
              {isLoading ? (
                <div className="w-5 h-5 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin" />
              ) : (
                <>
                  Sign In
                  <ArrowRight className="w-5 h-5" />
                </>
              )}
            </Button>
          </form>
          
          <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
            <Shield className="w-4 h-4" />
            <span>Secured by 256-bit encryption</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
