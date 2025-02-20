'use client';

import React from 'react';
import { useRouter } from 'next/navigation';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Scale } from 'lucide-react';
import { FcGoogle } from 'react-icons/fc';
import { useAuth } from '@/contexts/AuthContext';

const LoginPage = () => {
  const { signInWithGoogle } = useAuth();
  const router = useRouter();

  const handleGoogleLogin = async () => {
    try {
      await signInWithGoogle();
      router.push('/search'); // Redirect to search page after successful login
    } catch (error) {
      console.error('Login failed:', error);
      // Handle error (show toast notification, etc.)
    }
  };

  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-gradient-to-b from-background to-background/95">
      <div className="w-full max-w-5xl h-[600px] flex rounded-lg overflow-hidden">
        {/* Left side - Login form */}
        <Card className="w-full md:w-1/2 flex flex-col items-center justify-center p-8 bg-background border-0 shadow-none">
          <div className="w-full max-w-sm space-y-8">
            {/* Logo and Welcome */}
            <div className="space-y-2 text-center">
              <div className="flex justify-center items-center gap-2 mb-6">
                <Scale className="h-8 w-8 text-primary" />
                <span className="text-2xl font-semibold">Legal AI Platform</span>
              </div>
              <h1 className="text-2xl font-semibold tracking-tight">Welcome back</h1>
              <p className="text-sm text-muted-foreground">
                Please login to continue or view our demo <a href="#"
                  className="text-primary hover:underline">here</a>
              </p>
            </div>

            {/* Google Login Button */}
            <Button 
              variant="outline" 
              className="w-full h-12 space-x-3"
              onClick={handleGoogleLogin}
            >
              <FcGoogle className="h-5 w-5" />
              <span>Continue with Google</span>
            </Button>
          </div>
        </Card>

        {/* Right side - Image */}
        <div className="hidden md:block md:w-1/2 relative bg-muted">
          <div className="absolute inset-0 grid grid-cols-3 grid-rows-3">
            {Array.from({ length: 9 }).map((_, i) => (
              <div 
                key={i} 
                className="border-[0.5px] border-muted-foreground/10 backdrop-blur-sm"
              >
                <div className="w-full h-full bg-primary/5" />
              </div>
            ))}
          </div>
          
          <div className="absolute inset-0 flex items-center justify-center">
            <img
              src="https://4legalleads.com/wp-content/uploads/2020/05/4legalleads-building-successful-attorney-client-relationships.jpg"
              alt="Login decoration"
              className="object-cover w-full h-full opacity-100"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;