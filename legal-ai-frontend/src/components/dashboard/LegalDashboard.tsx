import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Scale, Book, Search, TrendingUp, FileText } from 'lucide-react';

interface LegalDashboardProps {}

const LegalDashboard = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [caseText, setCaseText] = useState('');

  // Query hooks for different endpoints
  const { data: classificationData, isLoading: classLoading } = useQuery({
    queryKey: ['classify', caseText],
    queryFn: async () => {
      const response = await fetch('http://localhost:5000/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: caseText })
      });
      return response.json();
    },
    enabled: !!caseText
  });

  const { data: searchResults, isLoading: searchLoading } = useQuery({
    queryKey: ['search', searchQuery],
    queryFn: async () => {
      const response = await fetch('http://localhost:5000/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery })
      });
      return response.json();
    },
    enabled: !!searchQuery
  });

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="container mx-auto p-6">
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">Legal AI Analysis Platform</h1>
          <p className="text-gray-600 dark:text-gray-400">Advanced legal case analysis and search</p>
        </header>

        <Tabs defaultValue="search" className="space-y-6">
          <TabsList className="grid grid-cols-3 gap-4 bg-transparent">
            <TabsTrigger value="search" className="flex items-center gap-2">
              <Search className="w-4 h-4" />
              Case Search
            </TabsTrigger>
            <TabsTrigger value="analysis" className="flex items-center gap-2">
              <Scale className="w-4 h-4" />
              Case Analysis
            </TabsTrigger>
            <TabsTrigger value="insights" className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Insights
            </TabsTrigger>
          </TabsList>

          <TabsContent value="search" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Search Legal Cases</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex gap-4">
                  <Input 
                    placeholder="Enter your search query..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="flex-1"
                  />
                  <Button>Search</Button>
                </div>
              </CardContent>
            </Card>

            {searchResults && (
              <div className="grid gap-4">
                {searchResults.results.map((result, index) => (
                  <Card key={index}>
                    <CardContent className="pt-6">
                      <h3 className="text-lg font-semibold mb-2">{result.case_name}</h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        {result.court} • {result.date_filed}
                      </p>
                      <p className="text-sm">{result.excerpt}</p>
                      <div className="mt-4 flex gap-2">
                        {result.tech_keywords.split(',').map((keyword, i) => (
                          <span key={i} className="px-2 py-1 bg-blue-100 dark:bg-blue-900 rounded-full text-xs">
                            {keyword.replace(/[\[\]']/g, '')}
                          </span>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="analysis" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Case Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <textarea
                  className="w-full h-32 p-2 border rounded-md"
                  placeholder="Enter case text for analysis..."
                  value={caseText}
                  onChange={(e) => setCaseText(e.target.value)}
                />
                <Button className="mt-4">Analyze</Button>
              </CardContent>
            </Card>

            {classificationData && (
              <div className="grid gap-4 md:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle>Classification Results</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {classificationData.predictions.map((prediction, index) => (
                        <div key={index} className="flex justify-between items-center">
                          <span className="capitalize">{prediction}</span>
                          <span className="text-sm bg-green-100 dark:bg-green-900 px-2 py-1 rounded-full">
                            {(classificationData.confidence_scores[prediction] * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Detected Patterns</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-2">
                      {classificationData.detected_patterns.map((pattern, index) => (
                        <span key={index} className="px-3 py-1 bg-purple-100 dark:bg-purple-900 rounded-full">
                          {pattern}
                        </span>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </TabsContent>

          <TabsContent value="insights" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Time Period Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  {classificationData?.time_period && (
                    <Alert>
                      <AlertDescription>
                        This case falls within the {classificationData.time_period.replace('_', '-')} period
                      </AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Enhanced Features</CardTitle>
                </CardHeader>
                <CardContent>
                  {classificationData?.enhanced_features && (
                    <div className="space-y-2">
                      {Object.entries(classificationData.enhanced_features).map(([key, value]) => (
                        <div key={key} className="flex justify-between items-center">
                          <span className="capitalize">{key.replace(/_/g, ' ')}</span>
                          <span>{(value as number).toFixed(3)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default LegalDashboard;