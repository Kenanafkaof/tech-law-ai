// lib/query-client.ts
import { QueryClient } from '@tanstack/react-query';

// cacheTime: 30 minutes

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: Infinity, // Keep data fresh indefinitely
      cacheTime: 1000 * 60 * 30, // Cache for 30 minutes
      refetchOnWindowFocus: false, // Don't refetch when window regains focus
      refetchOnMount: false, // Don't refetch when component remounts
      retry: 1, // Only retry failed requests once
    },
  },
});