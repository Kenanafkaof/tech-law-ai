'use client';

import { useQuery } from '@tanstack/react-query';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from '@/components/ui/card';
import { FileText, AlertTriangle, TrendingUp, Info } from 'lucide-react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { analyzeTechCenter } from '@/lib/api';

// Types for the API response
interface PatentData {
  analysis: {
    temporal_analysis: {
      total_applications: number;
      yearly_trends: Record<string, number>;
    };
    rejection_analysis: {
      counts: Record<string, number>;
      most_common: [string, number];
      total_rejections: number;
    };
    claim_analysis: {
      most_rejected_claims: [number, number][];
      avg_claims_per_rejection: number;
    };
    citation_analysis: {
      total_citations: number;
      top_citations: [string, number][];
    };
  };
  metadata: {
    timestamp: string;
    tech_center: string;
    analysis_version: string;
  };
}

const rejectionDescriptions = {
  '101': 'Subject Matter Eligibility - Is the invention eligible for patent protection?',
  '102': 'Novelty - Is the invention new and not previously disclosed?',
  '103': 'Obviousness - Would the invention be obvious to someone skilled in the field?',
  '112': 'Written Description/Enablement - Is the invention clearly described?',
};

export default function PatentsPage() {
  const { data, isLoading, error } = useQuery<PatentData>({
    queryKey: ['patents'],
    queryFn: async () => {
      // Use our API call that automatically adds the auth headers
      return await analyzeTechCenter('2400');
    },
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-4">
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <p className="text-destructive">
              Error loading patent data. Please try again later.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Calculate success rate
  const successRate = Math.round(
    (1 -
      data.analysis.rejection_analysis.total_rejections /
        data.analysis.temporal_analysis.total_applications) *
      100
  );

  // Prepare chart data
  const rejectionChartData = Object.entries(
    data.analysis.rejection_analysis.counts
  ).map(([type, count]) => ({
    type: `§${type}`,
    count,
    description:
      rejectionDescriptions[type as keyof typeof rejectionDescriptions],
  }));

  const trendsChartData = Object.entries(
    data.analysis.temporal_analysis.yearly_trends
  )
    .map(([year, count]) => ({
      year,
      applications: count,
    }))
    .sort((a, b) => parseInt(a.year) - parseInt(b.year));

  return (
    <div className="space-y-8 p-8">
      <div>
        <h1 className="text-4xl font-bold mb-2">
          Patent Analytics Dashboard
        </h1>
        <p className="text-muted-foreground">
          Comprehensive analysis of patent applications and examination outcomes
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Total Applications
            </CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {data.analysis.temporal_analysis.total_applications.toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground">
              Total patent applications analyzed
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Total Rejections
            </CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {data.analysis.rejection_analysis.total_rejections.toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground">
              Total rejection instances
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Success Rate
            </CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{successRate}%</div>
            <p className="text-xs text-muted-foreground">
              Applications without rejections
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid gap-8">
        {/* Rejection Analysis Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Rejection Analysis</CardTitle>
            <CardDescription>
              Distribution of rejection types across patent applications
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={rejectionChartData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis
                    dataKey="type"
                    tick={{ fill: 'hsl(var(--foreground))' }}
                  />
                  <YAxis tick={{ fill: 'hsl(var(--foreground))' }} />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="rounded-lg border bg-background p-2 shadow-sm">
                            <div className="grid grid-cols-2 gap-2">
                              <div className="flex flex-col">
                                <span className="text-[0.70rem] uppercase text-muted-foreground">
                                  {data.type}
                                </span>
                                <span className="font-bold">
                                  {data.count.toLocaleString()} rejections
                                </span>
                              </div>
                            </div>
                            <div className="mt-2 text-xs text-muted-foreground">
                              {data.description}
                            </div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar
                    dataKey="count"
                    fill="hsl(var(--primary))"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Trends Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Application Trends</CardTitle>
            <CardDescription>
              Year-over-year patent application patterns
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trendsChartData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis
                    dataKey="year"
                    tick={{ fill: 'hsl(var(--foreground))' }}
                  />
                  <YAxis tick={{ fill: 'hsl(var(--foreground))' }} />
                  <Tooltip
                    content={({ active, payload, label }) => {
                      if (active && payload && payload.length) {
                        return (
                          <div className="rounded-lg border bg-background p-2 shadow-sm">
                            <div className="grid grid-cols-2 gap-2">
                              <div className="flex flex-col">
                                <span className="text-[0.70rem] uppercase text-muted-foreground">
                                  Year
                                </span>
                                <span className="font-bold">{label}</span>
                              </div>
                              <div className="flex flex-col">
                                <span className="text-[0.70rem] uppercase text-muted-foreground">
                                  Applications
                                </span>
                                <span className="font-bold">
                                  {payload[0].value?.toLocaleString()}
                                </span>
                              </div>
                            </div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="applications"
                    stroke="hsl(var(--primary))"
                    strokeWidth={2}
                    dot={{ fill: 'hsl(var(--primary))', strokeWidth: 2 }}
                    activeDot={{ r: 6, fill: 'hsl(var(--primary))' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Citations Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Info className="h-4 w-4" />
              Top Patent Citations
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {data.analysis.citation_analysis.top_citations.slice(0, 5).map(([patent, count]) => (
                <div key={patent} className="flex items-center justify-between">
                  <span className="font-medium">{patent}</span>
                  <span className="text-sm text-muted-foreground">
                    {count} citations
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
