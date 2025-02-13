'use client';

import { useState, useEffect } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Search as SearchIcon, Loader2, History, Clock, ChevronDown } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { searchCases } from '@/lib/api';

interface QueryHistoryItem {
  text: string;
  timestamp: string;
}

export default function SearchPage() {
  const queryClient = useQueryClient();
  const [query, setQuery] = useState('');
  const [queryHistory, setQueryHistory] = useState<QueryHistoryItem[]>([]);

  // Load query history from session storage
  useEffect(() => {
    const savedHistory = sessionStorage.getItem('queryHistory');
    if (savedHistory) {
      setQueryHistory(JSON.parse(savedHistory));
    }
  }, []);

  const { data, isLoading, error } = useQuery({
    queryKey: ['search', query],
    queryFn: () => searchCases(query),
    enabled: !!query,
    staleTime: Infinity,
    cacheTime: Infinity,
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      // Add to query history
      const newHistoryItem: QueryHistoryItem = {
        text: query.trim(),
        timestamp: new Date().toISOString()
      };
      
      const updatedHistory = [
        newHistoryItem,
        ...queryHistory.filter(item => item.text !== query.trim()).slice(0, 9)
      ];
      setQueryHistory(updatedHistory);
      sessionStorage.setItem('queryHistory', JSON.stringify(updatedHistory));
      
      // Force refetch
      queryClient.invalidateQueries(['search', query]);
    }
  };

  const loadFromHistory = (historyItem: QueryHistoryItem) => {
    setQuery(historyItem.text);
    queryClient.invalidateQueries(['search', historyItem.text]);
  };

  const clearHistory = () => {
    setQueryHistory([]);
    sessionStorage.removeItem('queryHistory');
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-4xl font-bold mb-2">Case Search</h1>
          <p className="text-muted-foreground">
            Search and explore legal cases using advanced AI analysis
          </p>
        </div>
        {queryHistory.length > 0 && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="flex items-center gap-2">
                <History className="h-4 w-4" />
                History
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-[300px]">
              <DropdownMenuLabel>Recent Searches</DropdownMenuLabel>
              <DropdownMenuSeparator />
              {queryHistory.map((item, index) => (
                <DropdownMenuItem 
                  key={index}
                  onClick={() => loadFromHistory(item)}
                  className="flex flex-col items-start py-2 gap-1"
                >
                  <span className="text-sm font-medium">
                    {item.text}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {new Date(item.timestamp).toLocaleString()}
                  </span>
                </DropdownMenuItem>
              ))}
              <DropdownMenuSeparator />
              <DropdownMenuItem 
                onClick={clearHistory}
                className="text-destructive"
              >
                Clear History
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Search Legal Cases</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSearch} className="flex gap-4">
            <div className="relative flex-1">
              <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Enter your search query..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            <Button type="submit" disabled={isLoading}>
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Searching
                </>
              ) : (
                'Search'
              )}
            </Button>
          </form>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <p className="text-destructive">Error performing search. Please try again.</p>
          </CardContent>
        </Card>
      )}

      {data?.results && (
        <div className="space-y-4">
          {data.results.map((result, index) => (
            <Card key={index} className="transition-all hover:shadow-lg">
              <CardContent className="pt-6">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-xl font-semibold mb-1">{result.case_name}</h3>
                    <p className="text-sm text-muted-foreground">
                      {result.court} • {result.date_filed}
                    </p>
                  </div>
                  <div className="text-sm bg-primary/10 px-3 py-1 rounded-full">
                    Score: {(result.similarity_score * 100).toFixed(1)}%
                  </div>
                </div>
                
                <p className="text-sm mb-4">{result.excerpt}</p>
                
                <div className="flex flex-wrap gap-2">
                  {result.tech_keywords.split(',').map((keyword, i) => (
                    <span
                      key={i}
                      className="px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded-full text-xs"
                    >
                      {keyword.trim().replace(/[\[\]']/g, '')}
                    </span>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}