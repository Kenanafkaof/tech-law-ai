'use client';
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp, BarChart2, ChevronDown } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import KeywordSummaryChart from '@/components/charts/KeywordSummaryChart';
import KeywordTrendsChart from '@/components/charts/KeywordsTrendChart';

// Import API helpers
import { getAvailableKeywords, analyzeTrends, getKeywordSummary } from '@/lib/api';

const KeywordSelector = ({ 
  keywords = [], 
  selectedKeywords, 
  setSelectedKeywords,
  maxSelections = 5 
}) => {
  const [searchTerm, setSearchTerm] = useState('');

  const filteredKeywords = keywords.filter(keyword =>
    keyword.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="w-72">
      <div className="p-2">
        <Input
          type="search"
          placeholder="Search keywords..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="h-8"
        />
      </div>
      
      {selectedKeywords.length > 0 && (
        <div className="px-2 pb-2 flex flex-wrap gap-1">
          {selectedKeywords.map(keyword => (
            <Badge
              key={keyword}
              variant="secondary"
              className="cursor-pointer"
              onClick={() => setSelectedKeywords(
                selectedKeywords.filter(k => k !== keyword)
              )}
            >
              {keyword} ×
            </Badge>
          ))}
        </div>
      )}
      
      <ScrollArea className="h-72 px-2">
        <div className="space-y-1">
          {filteredKeywords.map((keyword) => (
            <div
              key={keyword}
              onClick={() => {
                if (selectedKeywords.includes(keyword)) {
                  setSelectedKeywords(selectedKeywords.filter(k => k !== keyword));
                } else if (selectedKeywords.length < maxSelections) {
                  setSelectedKeywords([...selectedKeywords, keyword]);
                }
              }}
              className={`
                flex items-center gap-2 px-2 py-1.5 text-sm rounded-md
                ${selectedKeywords.includes(keyword) 
                  ? 'bg-primary/10 text-primary' 
                  : 'hover:bg-muted cursor-pointer'
                }
                ${selectedKeywords.length >= maxSelections && !selectedKeywords.includes(keyword)
                  ? 'opacity-50 cursor-not-allowed'
                  : ''
                }
              `}
            >
              <div className={`w-2 h-2 rounded-full ${
                selectedKeywords.includes(keyword) 
                  ? 'bg-primary' 
                  : 'bg-muted-foreground'
              }`} />
              {keyword}
            </div>
          ))}
        </div>
      </ScrollArea>
      
      <div className="p-2 border-t">
        <p className="text-xs text-muted-foreground">
          Selected {selectedKeywords.length} of {maxSelections} keywords
        </p>
      </div>
    </div>
  );
};

const getColorForKeyword = (keyword: string, index: number) => {
  const colors: Record<string, string> = {
    technology: 'hsl(221, 83%, 53%)',  // Blue
    patent: 'hsl(142, 76%, 36%)',      // Green
    software: 'hsl(334, 86%, 48%)'     // Pink
  };
  const defaultColors = [
    'hsl(221, 83%, 53%)',
    'hsl(142, 76%, 36%)',
    'hsl(334, 86%, 48%)',
    'hsl(271, 91%, 65%)',
    'hsl(31, 95%, 44%)'
  ];
  return colors[keyword] || defaultColors[index % defaultColors.length];
};

export default function TechLawInsights() {
  const [selectedKeywords, setSelectedKeywords] = useState<string[]>(['technology', 'patent', 'software']);
  const [availableKeywords, setAvailableKeywords] = useState<string[]>([]);
  const [trendsData, setTrendsData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [summaryData, setSummaryData] = useState<any>(null);
  const [isSummaryLoading, setIsSummaryLoading] = useState(true);

  // Fetch available keywords
  useEffect(() => {
    const fetchKeywords = async () => {
      try {
        const data = await getAvailableKeywords();
        setAvailableKeywords(data.available_keywords);
      } catch (err: any) {
        setError(err.message);
      }
    };
    fetchKeywords();
  }, []);

  // Fetch trends data when keywords change
  useEffect(() => {
    const fetchTrends = async () => {
      setIsLoading(true);
      try {
        const data = await analyzeTrends(selectedKeywords);
        setTrendsData(data);
        setError(null);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };

    if (selectedKeywords.length > 0) {
      fetchTrends();
    }
  }, [selectedKeywords]);

  // Fetch keyword summary data
  useEffect(() => {
    const fetchSummary = async () => {
      setIsSummaryLoading(true);
      try {
        const data = await getKeywordSummary();
        setSummaryData(data);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setIsSummaryLoading(false);
      }
    };
  
    fetchSummary();
  }, []);

  if (error) {
    return (
      <Card className="p-6">
        <div className="text-destructive">Error: {error}</div>
      </Card>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header with Keyword Selector */}
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Tech Law Insights</h1>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Select Keywords ({selectedKeywords.length})
              <ChevronDown className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-72 p-0">
            <KeywordSelector
              keywords={availableKeywords}
              selectedKeywords={selectedKeywords}
              setSelectedKeywords={setSelectedKeywords}
              maxSelections={5}
            />
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Loading State */}
      {isLoading && (
        <Card className="p-6">
          <div className="flex justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        </Card>
      )}
      
      {/* Trends Chart */}
      {!isLoading && trendsData && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Keyword Trends Over Time
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div style={{ width: '100%', height: 500 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[2000, 2030]} dataKey="year" />
                  <YAxis />
                  <Tooltip 
                    content={({ active, payload, label }) => {
                      if (active && payload && payload.length) {
                        return (
                          <div className="bg-background/95 p-2 border rounded-lg shadow-sm">
                            <p className="text-sm font-medium">Year: {label}</p>
                            {payload.map((entry) => (
                              <p key={entry.name} style={{ color: entry.color }} className="text-sm">
                                {entry.name}: {entry.value.toFixed(1)}
                              </p>
                            ))}
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  {Object.entries(trendsData.data).map(([keyword, data]: [string, any], index) => {
                    const hasEnoughData = data.historical_data.length > 2;
                    return hasEnoughData && (
                      <Line
                        key={keyword}
                        data={data.historical_data}
                        type="monotone"
                        dataKey="value"
                        name={keyword}
                        stroke={getColorForKeyword(keyword, index)}
                        dot={{ r: 2 }}
                      />
                    );
                  })}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Keyword Summary Chart */}
      {!isSummaryLoading && summaryData && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Top Keywords by Mentions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="w-full h-[500px]">
              <KeywordSummaryChart data={summaryData} />
            </div>
          </CardContent>
        </Card>
      )}  

      {/* Metrics Grid */}
      {!isLoading && trendsData && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(trendsData.data).map(([keyword, data], index) => (
            <Card key={keyword}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart2 className="h-5 w-5" />
                  {keyword.charAt(0).toUpperCase() + keyword.slice(1)}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <dl className="space-y-2">
                  <div>
                    <dt className="text-sm text-muted-foreground">Total Mentions</dt>
                    <dd className="text-2xl font-bold">{data.metrics.totalMentions}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-muted-foreground">Average Mentions</dt>
                    <dd className="text-lg">{data.metrics.meanCount.toFixed(1)}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-muted-foreground">Trend</dt>
                    <dd className="text-lg">{data.metrics.trendInterpretation}</dd>
                  </div>
                </dl>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
