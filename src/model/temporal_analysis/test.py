import pandas as pd
import numpy as np
from collections import Counter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression

class TechLawData:
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path)
        self._process_data()
    
    def _process_data(self):
        # Convert date_filed to datetime
        self.df['date_filed'] = pd.to_datetime(self.df['date_filed'], errors='coerce')
        self.df['year_filed'] = self.df['date_filed'].dt.year
        
        # Convert tech_keywords_found from string to list and remove prefixes
        self.df['tech_keywords_found'] = self.df['tech_keywords_found'].apply(
            lambda x: [keyword.split(":", 1)[-1] if ":" in keyword else keyword for keyword in eval(x)] if isinstance(x, str) else [])
        
        # Extract ALL keyword trends, not just the top 5
        self.keyword_trends = []
        keyword_counts = self.df.explode('tech_keywords_found').groupby(['year_filed', 'tech_keywords_found']).size().reset_index(name='Count')
        for _, row in keyword_counts.iterrows():
            self.keyword_trends.append({'Year': row['year_filed'], 'Keyword': row['tech_keywords_found'], 'Count': row['Count']})
        self.keyword_df = pd.DataFrame(self.keyword_trends)
    
    def forecast_keyword_trends(self, keyword: str, periods: int = 5):
        keyword_data = self.keyword_df[self.keyword_df['Keyword'] == keyword][['Year', 'Count']]
        
        if keyword_data.empty:
            return {"error": f"No historical data available for '{keyword}'."}
        
        # Convert Year column to datetime
        keyword_data['Year'] = pd.to_datetime(keyword_data['Year'], format='%Y')
        keyword_data = keyword_data.set_index('Year').asfreq('YE')
        keyword_data['Count'] = keyword_data['Count'].fillna(0)  # Fill missing values with 0
        
        # If all values are zero, assume a small increasing trend
        if keyword_data['Count'].sum() == 0:
            future_years = [keyword_data.index.max().year + i for i in range(1, periods + 1)]
            forecast_results = [{'Year': year, 'Predicted_Count': i * 0.5} for i, year in enumerate(future_years, 1)]
            return {"forecast": forecast_results, "historical_data": keyword_data.to_dict(orient='records')}
        
        # Use Linear Regression if data is sparse (less than 5 years of data)
        if len(keyword_data) < 5:
            X = np.array(keyword_data.index.year).reshape(-1, 1)
            y = np.array(keyword_data['Count'])
            model = LinearRegression()
            model.fit(X, y)
            future_years = np.array([keyword_data.index.max().year + i for i in range(1, periods + 1)]).reshape(-1, 1)
            predictions = model.predict(future_years)
            
            forecast_results = [{'Year': year, 'Predicted_Count': max(1, count)} for year, count in zip(future_years.flatten(), predictions)]
            return {"forecast": forecast_results, "historical_data": keyword_data.to_dict(orient='records')}
        
        # Fit Exponential Smoothing model for trend-based forecasting
        model = ExponentialSmoothing(keyword_data['Count'], trend='add', seasonal=None, damped_trend=True)
        fit_model = model.fit()
        
        # Forecast future values
        future_dates = pd.date_range(start=keyword_data.index.max(), periods=periods+1, freq='YE')[1:]
        predictions = fit_model.forecast(periods)
        
        # Override zero forecasts with a small increase
        forecast_results = []
        for year, count in zip(future_dates, predictions):
            if count <= 0:
                count = max(1, keyword_data['Count'].mean() * 0.1)  # Ensure a small growth
            forecast_results.append({'Year': year.year, 'Predicted_Count': count})
        
        return {"forecast": forecast_results, "historical_data": keyword_data.to_dict(orient='records')}

# Example usage
data_provider = TechLawData('../../datasets/test_all_tech_cases_fixed.csv')

keyword = 'technology'
periods = 10
print(data_provider.forecast_keyword_trends(keyword, periods))
