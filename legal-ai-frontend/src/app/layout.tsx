// app/layout.tsx
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { ClientLayout } from '@/components/ClientLayout';
import { Providers } from '@/components/Providers';
import { AuthProvider } from '@/contexts/AuthContext';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Legal AI Platform',
  description: 'Advanced legal case analysis and search platform',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <AuthProvider>
          <Providers>
            <ClientLayout>{children}</ClientLayout>
          </Providers>
        </AuthProvider>
      </body>
    </html>
  );
}