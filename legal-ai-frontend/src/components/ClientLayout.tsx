'use client';

import React, { useEffect, useState, useRef } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { cn } from '@/lib/utils';
import {
  Search,
  Scale,
  TrendingUp,
  Database,
  Menu,
  X,
  LucideIcon,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ThemeToggle } from '@/components/theme-toggle';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Separator } from '@/components/ui/separator';
import { useAuth } from '@/contexts/AuthContext';

interface NavigationItem {
  name: string;
  href: string;
  icon: LucideIcon;
  description: string;
}

const navigation: NavigationItem[] = [
  { 
    name: 'Case Search', 
    href: '/search', 
    icon: Search,
    description: 'Search and explore legal cases'
  },
  { 
    name: 'Case Analysis', 
    href: '/analysis', 
    icon: Scale,
    description: 'Analyze case content and patterns'
  },
  { 
    name: 'Legal Insights', 
    href: '/insights', 
    icon: TrendingUp,
    description: 'Discover trends and insights'
  },
  { 
    name: 'Patent Dashboard', 
    href: '/patents', 
    icon: Database,
    description: 'Patent analysis and metrics'
  },
];

function UserProfile({ className }: { className?: string }) {
  const { user, signOut } = useAuth();
  const router = useRouter();
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Always call hooks before any conditional returns.
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () =>
      document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Now we can conditionally render if no user.
  if (!user) return null;

  // Normalize the default Google avatar.
  if (
    user.photoURL === null ||
    user.photoURL === "https://lh3.googleusercontent.com/a/ACg8ocK7JndtInn8UQuKE4AsAFJfhZNowcp-TAZ4ux6GOdIyBeaZmm0=s96-c"
  ) {
    user.photoURL = null;
  }

  const displayName = user.displayName || 'User';
  const email = user.email || '';
  const initials = displayName.split(' ').map((n) => n[0]).join('');

  const handleLogout = async () => {
    try {
      await signOut();
      router.push('/login');
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-4 px-4 py-3 w-full",
        className
      )}
      ref={dropdownRef}
    >
      <div className="flex items-center gap-3">
        <Avatar className="h-10 w-10">
          {user.photoURL ? (
            <AvatarImage src={user.photoURL} alt={displayName} />
          ) : (
            <AvatarFallback className="bg-primary/10">
              {initials}
            </AvatarFallback>
          )}
        </Avatar>
        <div className="flex flex-col">
          <span className="font-medium">{displayName}</span>
          <span className="text-sm text-muted-foreground">{email}</span>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <ThemeToggle />
        <Button variant="outline" size="sm" onClick={handleLogout}>
          Logout
        </Button>
      </div>
    </div>
  );
}

export function ClientLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, loading } = useAuth();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Always call hooks first.
  useEffect(() => {
    if (!loading && !user && pathname !== '/login') {
      router.push('/login');
    }
  }, [loading, user, pathname, router]);

  // Instead of returning early (which can change the number of hooks called),
  // determine what content to render after all hooks have been executed.
  let content;

  if (loading) {
    content = (
      <div className="flex items-center justify-center min-h-screen">
        Loading...
      </div>
    );
  } else if (pathname === '/login') {
    content = <>{children}</>;
  } else {
    content = (
      <div className="min-h-screen bg-gradient-to-b from-background to-background/95">
        {/* Mobile Navigation */}
        <div className="lg:hidden">
          <div className="flex items-center justify-between p-4 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <Menu className="h-6 w-6" />
            </Button>
          </div>
          {sidebarOpen && (
            <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm">
              <div className="fixed inset-y-0 left-0 w-80 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-r">
                <div className="flex h-full flex-col">
                  <div className="flex items-center justify-between p-6">
                    <div className="flex items-center gap-2">
                      <Scale className="h-6 w-6 text-primary" />
                      <span className="text-lg font-semibold">
                        Legal AI Platform
                      </span>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => setSidebarOpen(false)}
                    >
                      <X className="h-6 w-6" />
                    </Button>
                  </div>
                  <nav className="flex-1 space-y-2 p-6">
                    {navigation.map((item) => (
                      <Link
                        key={item.name}
                        href={item.href}
                        className={cn(
                          "flex flex-col space-y-1 px-4 py-3 rounded-lg transition-colors",
                          pathname === item.href
                            ? "bg-primary text-primary-foreground"
                            : "hover:bg-muted"
                        )}
                        onClick={() => setSidebarOpen(false)}
                      >
                        <div className="flex items-center gap-3">
                          <item.icon className="h-5 w-5" />
                          <span className="font-medium">{item.name}</span>
                        </div>
                        <span className="text-sm opacity-70">
                          {item.description}
                        </span>
                      </Link>
                    ))}
                  </nav>
                  <div className="mt-auto pb-6">
                    <Separator className="mb-6" />
                    <UserProfile />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Desktop Navigation */}
        <div className="hidden lg:fixed lg:inset-y-0 lg:flex lg:w-80 lg:flex-col">
          <div className="flex grow flex-col gap-y-6 overflow-y-auto border-r bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-6">
            <div className="flex h-16 items-center gap-2 mt-4">
              <Scale className="h-6 w-6 text-primary" />
              <span className="text-lg font-semibold">
                Legal AI Platform
              </span>
            </div>
            <nav className="flex flex-1 flex-col">
              <ul role="list" className="flex flex-1 flex-col gap-y-4">
                {navigation.map((item) => (
                  <li key={item.name}>
                    <Link
                      href={item.href}
                      className={cn(
                        "group flex flex-col space-y-1 rounded-lg px-4 py-3 transition-colors",
                        pathname === item.href
                          ? "bg-primary text-primary-foreground"
                          : "hover:bg-muted"
                      )}
                    >
                      <div className="flex items-center gap-3">
                        <item.icon className="h-5 w-5" />
                        <span className="font-medium">{item.name}</span>
                      </div>
                      <span className="text-sm opacity-70">
                        {item.description}
                      </span>
                    </Link>
                  </li>
                ))}
              </ul>
            </nav>
            <div className="mt-auto pb-6">
              <Separator className="mb-6" />
              <UserProfile />
            </div>
          </div>
        </div>

        {/* Main Content */}
        <main className="lg:pl-80">
          <div className="px-6 py-8 md:px-8 lg:px-10">{children}</div>
        </main>
      </div>
    );
  }
  return content;
}
