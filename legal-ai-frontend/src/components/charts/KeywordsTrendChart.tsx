import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp } from 'lucide-react';

const KeywordTrendsChart = ({ data }) => {
  // Validate and process the data
  const processChartData = (trendsData) => {
    if (!trendsData || !trendsData.data) return [];
    
    // Get all unique years across all keywords
    const years = new Set();
    Object.values(trendsData.data).forEach(keywordData => {
      keywordData.historical_data.forEach(point => years.add(point.year));
      if (keywordData.forecast) {
        keywordData.forecast.forEach(point => years.add(point.year));
      }
    });
    
    // Create a sorted array of years
    const sortedYears = Array.from(years).sort((a, b) => a - b);
    
    // Create data points for each year
    return sortedYears.map(year => {
      const dataPoint = { year };
      
      // Add data for each keyword
      Object.entries(trendsData.data).forEach(([keyword, keywordData]) => {
        // Find historical data point
        const historicalPoint = keywordData.historical_data.find(p => p.year === year);
        if (historicalPoint) {
          dataPoint[keyword] = historicalPoint.value;
        }
        
        // Find forecast point
        if (keywordData.forecast) {
          const forecastPoint = keywordData.forecast.find(p => p.year === year);
          if (forecastPoint) {
            dataPoint[`${keyword}_forecast`] = forecastPoint.value;
            dataPoint[`${keyword}_lower`] = forecastPoint.lowerBound;
            dataPoint[`${keyword}_upper`] = forecastPoint.upperBound;
          }
        }
      });
      
      return dataPoint;
    });
  };

  const chartData = processChartData(data);
  const keywords = Object.keys(data?.data || {});
  
  const getColorForKeyword = (keyword, index) => {
    const colors = {
      technology: 'hsl(221, 83%, 53%)',  // Blue
      patent: 'hsl(142, 76%, 36%)',      // Green
      software: 'hsl(334, 86%, 48%)',    // Pink
      'artificial intelligence': 'hsl(271, 91%, 65%)' // Purple
    };
    return colors[keyword] || colors[Object.keys(colors)[index % Object.keys(colors).length]];
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5" />
          Keyword Trends Over Time
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[500px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                type="number"
                domain={['dataMin', 'dataMax']}
                dataKey="year"
                tickCount={10}
              />
              <YAxis />
              <Tooltip
                formatter={(value, name) => {
                  if (name.endsWith('_forecast')) return [`${value.toFixed(1)} (forecast)`, name.replace('_forecast', '')];
                  if (name.endsWith('_lower') || name.endsWith('_upper')) return null;
                  return [value, name];
                }}
              />
              
              {keywords.map((keyword, index) => (
                <React.Fragment key={keyword}>
                  {/* Historical data line */}
                  <Line
                    type="monotone"
                    dataKey={keyword}
                    stroke={getColorForKeyword(keyword, index)}
                    strokeWidth={2}
                    dot={{ r: 2 }}
                    name={keyword}
                  />
                  
                  {/* Forecast line (dashed) */}
                  <Line
                    type="monotone"
                    dataKey={`${keyword}_forecast`}
                    stroke={getColorForKeyword(keyword, index)}
                    strokeDasharray="5 5"
                    strokeWidth={2}
                    dot={{ r: 2 }}
                    name={`${keyword}_forecast`}
                  />
                </React.Fragment>
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export default KeywordTrendsChart;