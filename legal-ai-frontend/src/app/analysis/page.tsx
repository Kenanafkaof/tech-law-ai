'use client';

import { useState, useEffect } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import {
  AlertCircle,
  ArrowRight,
  Book,
  Scale,
  TrendingUp,
  History,
  RefreshCw,
  Clock,
  ChevronDown,
} from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

// Import the analyzeStrategy function from your API file.
import { analyzeStrategy } from '@/lib/api';

interface QueryHistoryItem {
  text: string;
  timestamp: string;
  preview: string;
}

interface AnalysisResponse {
  key_reasoning: Array<{
    context: string;
    text: string;
  }>;
  winning_arguments: Array<{
    context: string;
    text: string;
    time_period: string;
  }>;
  similar_cases: Array<{
    similarity_breakdown: {
      citation_overlap: number;
      issue_overlap: number;
      temporal_score: number;
      text_similarity: number;
    };
    similarity_score: number;
    time_period: string;
  }>;
  temporal_distribution: Record<string, number>;
  metadata: {
    analysis_version: string;
    timestamp: string;
    total_similar_cases: number;
  };
}

export default function CaseAnalysis() {
  const queryClient = useQueryClient();
  const [caseText, setCaseText] = useState('');
  const [queryHistory, setQueryHistory] = useState<QueryHistoryItem[]>([]);

  // Get the last successful query data from the cache
  const cachedData = queryClient.getQueryData(['analyze-strategy']);

  useEffect(() => {
    const savedHistory = sessionStorage.getItem('queryHistory');
    if (savedHistory) {
      setQueryHistory(JSON.parse(savedHistory));
    }
  }, []);

  // Restore last analyzed text from session storage
  useEffect(() => {
    const lastText = sessionStorage.getItem('lastAnalyzedCase');
    if (lastText) {
      setCaseText(lastText);
    }
  }, []);

  const { data, isLoading, error, refetch } = useQuery<AnalysisResponse>({
    queryKey: ['analyze-strategy', caseText],
    queryFn: () => analyzeStrategy(caseText),
    enabled: false,
    staleTime: Infinity,
    cacheTime: Infinity,
  });

  const clearAnalysis = () => {
    setCaseText('');
    sessionStorage.removeItem('lastAnalyzedCase');
    queryClient.removeQueries(['analyze-strategy']);
  };

  const handleAnalyze = () => {
    if (caseText.trim()) {
      // Add to history before analyzing
      const newHistoryItem: QueryHistoryItem = {
        text: caseText,
        timestamp: new Date().toISOString(),
        preview: caseText.slice(0, 100) + (caseText.length > 100 ? '...' : ''),
      };

      const updatedHistory = [newHistoryItem, ...queryHistory.slice(0, 9)]; // Keep last 10 queries
      setQueryHistory(updatedHistory);
      sessionStorage.setItem('queryHistory', JSON.stringify(updatedHistory));

      refetch();
    }
  };

  const handleReset = () => {
    setCaseText('');
    queryClient.removeQueries(['analyze-strategy']);
  };

  const loadFromHistory = (historyItem: QueryHistoryItem) => {
    setCaseText(historyItem.text);
  };

  const clearHistory = () => {
    setQueryHistory([]);
    sessionStorage.removeItem('queryHistory');
  };

  const similarityMetrics = data?.similar_cases[0]?.similarity_breakdown
    ? [
        {
          name: 'Citation Overlap',
          value: data.similar_cases[0].similarity_breakdown.citation_overlap * 100,
        },
        {
          name: 'Issue Overlap',
          value: data.similar_cases[0].similarity_breakdown.issue_overlap * 100,
        },
        {
          name: 'Temporal Score',
          value: data.similar_cases[0].similarity_breakdown.temporal_score * 100,
        },
        {
          name: 'Text Similarity',
          value: data.similar_cases[0].similarity_breakdown.text_similarity * 100,
        },
      ]
    : [];

  const temporalData = data?.temporal_distribution
    ? Object.entries(data.temporal_distribution).map(([period, count]) => ({
        period: period.replace('_', '-'),
        count,
      }))
    : [];

  return (
    <div className="space-y-8 p-8">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-4xl font-bold mb-2">Case Analysis</h1>
          <p className="text-muted-foreground">
            Analyze legal cases using advanced AI to identify key arguments and patterns
          </p>
        </div>
        <div className="flex items-center gap-4">
          {queryHistory.length > 0 && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm" className="flex items-center gap-2">
                  <Clock className="h-4 w-4" />
                  Query History
                  <ChevronDown className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-[400px]">
                <DropdownMenuLabel>Recent Queries</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {queryHistory.map((item, index) => (
                  <DropdownMenuItem
                    key={index}
                    onClick={() => loadFromHistory(item)}
                    className="flex flex-col items-start py-2 gap-1"
                  >
                    <span className="text-sm font-medium truncate w-full">
                      {item.preview}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {new Date(item.timestamp).toLocaleString()}
                    </span>
                  </DropdownMenuItem>
                ))}
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={clearHistory} className="text-destructive">
                  Clear History
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}
          {(data || cachedData) && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleReset}
              className="flex items-center gap-2"
            >
              <RefreshCw className="h-4 w-4" />
              New Analysis
            </Button>
          )}
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Enter Case Text</CardTitle>
          <CardDescription>Paste your case text below for analysis</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder="Enter case text here..."
            className="min-h-[200px]"
            value={caseText}
            onChange={(e) => setCaseText(e.target.value)}
          />
          <Button onClick={handleAnalyze} disabled={isLoading || !caseText.trim()} className="w-full">
            {isLoading ? 'Analyzing...' : 'Analyze Case'}
          </Button>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 text-destructive">
              <AlertCircle className="h-4 w-4" />
              <p>Error analyzing case. Please try again.</p>
            </div>
          </CardContent>
        </Card>
      )}

      {(data || cachedData) && (
        <div className="grid gap-6">
          {/* Winning Arguments */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Scale className="h-5 w-5" />
                Winning Arguments
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {(data || cachedData)?.winning_arguments?.map((arg, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 mt-1 flex-shrink-0 text-muted-foreground" />
                    <div>
                      <p className="font-medium">{arg.text}</p>
                      <p className="text-sm text-muted-foreground">{arg.context}</p>
                      <span className="text-xs bg-primary/10 px-2 py-1 rounded-full">
                        {arg.time_period.replace('_', '-')}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Similarity Analysis */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Book className="h-5 w-5" />
                Similar Cases Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={similarityMetrics}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis
                      dataKey="name"
                      tick={{ fill: 'hsl(var(--foreground))', fontSize: 12 }}
                      angle={-45}
                      textAnchor="end"
                      height={80}
                    />
                    <YAxis tick={{ fill: 'hsl(var(--foreground))' }} domain={[0, 100]} />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          return (
                            <div className="rounded-lg border bg-background p-2 shadow-sm">
                              <div className="flex flex-col">
                                <span className="text-sm font-medium">
                                  {payload[0].payload.name}
                                </span>
                                <span className="text-sm text-muted-foreground">
                                  {payload[0].value?.toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Bar dataKey="value" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Temporal Distribution */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Temporal Distribution
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={temporalData}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="period" tick={{ fill: 'hsl(var(--foreground))' }} />
                    <YAxis tick={{ fill: 'hsl(var(--foreground))' }} />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          return (
                            <div className="rounded-lg border bg-background p-2 shadow-sm">
                              <div className="flex flex-col">
                                <span className="text-sm font-medium">
                                  {payload[0].payload.period}
                                </span>
                                <span className="text-sm text-muted-foreground">
                                  {payload[0].value} cases
                                </span>
                              </div>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
