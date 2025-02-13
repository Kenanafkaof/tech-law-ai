import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const KeywordSummaryChart = ({ data }) => {
  console.log('KeywordSummaryChart received data:', data);

  if (!data?.data?.relevantKeywords) {
    console.log('No relevant keywords found in data');
    return null;
  }

  // Get top 5 keywords by total mentions
  const chartData = data.data.relevantKeywords
    .sort((a, b) => b.total_mentions - a.total_mentions)
    .slice(0, 5)
    .map(keyword => ({
      name: keyword.keyword,
      total: keyword.total_mentions
    }));

  console.log('Processed chart data:', chartData);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={chartData}
        margin={{
          top: 20,
          right: 30,
          left: 20,
          bottom: 40,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="name"
          angle={-45}
          textAnchor="end"
          height={60}
        />
        <YAxis />
        <Tooltip />
        <Bar dataKey="total" fill="#3b82f6" />
      </BarChart>
    </ResponsiveContainer>
  );
};

export default KeywordSummaryChart;