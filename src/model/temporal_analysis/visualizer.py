import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
from dataclasses import dataclass
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import logging

@dataclass
class ForecastResult:
    forecast: List[Dict[str, Union[int, float]]]
    historical_data: List[Dict[str, Union[int, float]]]
    confidence_intervals: Optional[List[Dict[str, float]]] = None
    metrics: Optional[Dict[str, float]] = None

class TechLawData:
    def __init__(self, file_path: str):
        """
        Initialize the TechLawData class with improved error handling and logging.
        
        Args:
            file_path (str): Path to the CSV file containing tech law data
        """
        self.logger = self._setup_logger()
        try:
            self.df = pd.read_csv(file_path)
            self._process_data()
            print(self.keyword_df.describe())  # Summarizes numerical columns
            print(self.keyword_df['Count'].value_counts())  # Check frequency of different values
            print(self.keyword_df[self.keyword_df['Count'] > 0])  # Show only rows with non-zero count

        except Exception as e:
            self.logger.error(f"Error initializing TechLawData: {str(e)}")
            raise

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('TechLawData')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _process_data(self) -> None:
        """
        Process and clean the raw data with improved aggregation and validation.
        """
        try:
            # Convert date_filed to datetime
            self.df['date_filed'] = pd.to_datetime(self.df['date_filed'], errors='coerce')
            self.df['year_filed'] = self.df['date_filed'].dt.year
            
            # Drop rows with invalid dates
            self.df = self.df.dropna(subset=['date_filed'])
            
            # Process tech keywords
            self.df['tech_keywords_found'] = self.df['tech_keywords_found'].apply(self._process_keywords)
            
            # Explode the keywords list to get one row per keyword
            exploded_df = self.df.explode('tech_keywords_found')
            exploded_df = exploded_df[exploded_df['tech_keywords_found'].notna()]
            
            # Group by year and keyword to get counts
            self.keyword_df = (exploded_df
                .groupby(['year_filed', 'tech_keywords_found'])
                .size()
                .reset_index(name='Count'))
            
            # Rename columns for clarity
            self.keyword_df.columns = ['Year', 'Keyword', 'Count']
            
            # Store keyword trends data
            self.keyword_trends = self.keyword_df.to_dict('records')
            
            print("\nData Processing Summary:")
            print(f"Total records: {len(self.df)}")
            print(f"Unique keywords: {len(self.keyword_df['Keyword'].unique())}")
            print(f"Year range: {self.keyword_df['Year'].min()} - {self.keyword_df['Year'].max()}")
            
            self.logger.info("Data processing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in data processing: {str(e)}")
            raise

    def _get_keyword_data(self, keyword: str) -> pd.DataFrame:
        """
        Get historical data for a specific keyword with improved aggregation.
        """
        try:
            # Get data for the specific keyword
            keyword_data = self.keyword_df[self.keyword_df['Keyword'] == keyword].copy()
            
            if keyword_data.empty:
                return pd.DataFrame()
                
            # Convert year to datetime and set as index
            keyword_data['Year'] = pd.to_datetime(keyword_data['Year'].astype(str), format='%Y')
            keyword_data = keyword_data.set_index('Year')[['Count']]
            
            # Get full date range
            date_range = pd.date_range(
                start=keyword_data.index.min(),
                end=keyword_data.index.max(),
                freq='YE'
            )
            
            # Reindex to get all years, fill missing with 0
            keyword_data = keyword_data.reindex(date_range, fill_value=0)
            
            print(f"\nKeyword data for {keyword}:")
            print(f"Years with data: {len(keyword_data[keyword_data['Count'] > 0])}")
            print(f"Total mentions: {keyword_data['Count'].sum()}")
            print(f"Average mentions per year: {keyword_data['Count'].mean():.2f}")
            
            return keyword_data
            
        except Exception as e:
            self.logger.error(f"Error in _get_keyword_data for {keyword}: {str(e)}")
            return pd.DataFrame()


    def _process_keywords(self, keywords_str: str) -> List[str]:
        """
        Process and normalize keywords from string format.
        
        Args:
            keywords_str (str): String representation of keywords list
        
        Returns:
            List[str]: Normalized list of keywords
        """
        try:
            if pd.isna(keywords_str):
                return []
            
            # Convert string representation to list
            if isinstance(keywords_str, str):
                keywords = eval(keywords_str)
            else:
                keywords = []
                
            # Normalize keywords
            normalized = []
            for keyword in keywords:
                # Remove prefix if present
                if ":" in keyword:
                    keyword = keyword.split(":", 1)[-1]
                # Clean and normalize
                keyword = keyword.lower().strip()
                if keyword:
                    normalized.append(keyword)
                    
            return normalized
        except Exception:
            return []

    def _create_keyword_trends(self) -> None:
        """
        Create and store keyword trends data.
        """
        self.keyword_trends = []
        keyword_counts = (self.df.explode('tech_keywords_found')
                         .groupby(['year_filed', 'tech_keywords_found'])
                         .size()
                         .reset_index(name='Count'))
        
        for _, row in keyword_counts.iterrows():
            if pd.notna(row['tech_keywords_found']):
                self.keyword_trends.append({
                    'Year': row['year_filed'],
                    'Keyword': row['tech_keywords_found'],
                    'Count': row['Count']
                })
        
        self.keyword_df = pd.DataFrame(self.keyword_trends)

    def validate_forecast(self, keyword: str, min_samples_per_split: int = 2) -> Dict[str, float]:
        """
        Validate forecast accuracy using adaptive time series cross-validation.
        
        Args:
            keyword (str): Keyword to validate
            min_samples_per_split (int): Minimum number of samples required per split
        
        Returns:
            Dict[str, float]: Dictionary containing error metrics
        """
        keyword_data = self._get_keyword_data(keyword)
        if keyword_data.empty:
            return {}

        # Calculate maximum possible splits based on data size
        n_samples = len(keyword_data)
        if n_samples < min_samples_per_split * 2:  # Need at least enough for train and test
            # For very small datasets, use a simple train-test split
            train_size = max(1, n_samples - min_samples_per_split)
            metrics = self._calculate_single_split_metrics(keyword_data, train_size)
        else:
            # Calculate appropriate number of splits
            max_splits = (n_samples // min_samples_per_split) - 1
            n_splits = min(3, max_splits)  # Cap at 3 splits
            metrics = self._calculate_cross_val_metrics(keyword_data, n_splits)
        
        return metrics

    def _calculate_single_split_metrics(self, data: pd.DataFrame, train_size: int) -> Dict[str, float]:
        """Calculate metrics using a single train-test split."""
        try:
            train = data.iloc[:train_size]
            test = data.iloc[train_size:]
            
            # Fit model
            model = ExponentialSmoothing(
                train['Count'],
                trend='add' if len(train) > 2 else None,  # Only use trend if enough data
                seasonal=None,
                damped_trend=True
            )
            fit_model = model.fit()
            
            # Make predictions
            predictions = fit_model.forecast(len(test))
            
            return {
                'mse': mean_squared_error(test['Count'], predictions),
                'mae': mean_absolute_error(test['Count'], predictions),
                'rmse': np.sqrt(mean_squared_error(test['Count'], predictions))
            }
        except Exception as e:
            self.logger.warning(f"Error in single split validation: {str(e)}")
            return {}

    def _calculate_cross_val_metrics(self, data: pd.DataFrame, n_splits: int) -> Dict[str, float]:
        """Calculate metrics using time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = {
            'mse': [],
            'mae': [],
            'rmse': []
        }

        for train_idx, test_idx in tscv.split(data):
            train = data.iloc[train_idx]
            test = data.iloc[test_idx]
            
            # Only proceed if we have enough data in both sets
            if len(train) > 0 and len(test) > 0:
                # Fit model
                model = ExponentialSmoothing(
                    train['Count'],
                    trend='add' if len(train) > 2 else None,
                    seasonal=None,
                    damped_trend=True
                )
                fit_model = model.fit()
                
                # Make predictions
                predictions = fit_model.forecast(len(test))
                
                # Calculate metrics
                metrics['mse'].append(mean_squared_error(test['Count'], predictions))
                metrics['mae'].append(mean_absolute_error(test['Count'], predictions))
                metrics['rmse'].append(np.sqrt(metrics['mse'][-1]))

        # Average metrics across folds if we have any valid results
        if any(len(m) > 0 for m in metrics.values()):
            return {
                'mse': np.mean(metrics['mse']) if metrics['mse'] else None,
                'mae': np.mean(metrics['mae']) if metrics['mae'] else None,
                'rmse': np.mean(metrics['rmse']) if metrics['rmse'] else None
            }
        return {}

    def _calculate_consecutive_years(self, data: pd.DataFrame) -> int:
        """Calculate the number of consecutive years with mentions."""
        current_streak = 0
        max_streak = 0
        
        for count in data['Count'].values:
            if count > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
                
        return max_streak


    def _get_keyword_data(self, keyword: str) -> pd.DataFrame:
        """
        Get historical data for a specific keyword with proper date handling.
        
        Args:
            keyword (str): Keyword to get data for
        
        Returns:
            pd.DataFrame: DataFrame containing keyword data with proper date indexing
        """
        try:
            # Get raw data for the keyword
            keyword_data = self.keyword_df[self.keyword_df['Keyword'] == keyword].copy()
            
            if keyword_data.empty:
                return pd.DataFrame()

            # Convert year to datetime and set as index
            keyword_data['Year'] = pd.to_datetime(keyword_data['Year'].astype(str), format='%Y')
            keyword_data = keyword_data.set_index('Year')[['Count']]
            
            # Sort by date
            keyword_data = keyword_data.sort_index()
            
            # Get full date range
            date_range = pd.date_range(
                start=keyword_data.index.min(),
                end=keyword_data.index.max(),
                freq='YE'
            )
            
            # Reindex to get all years, fill missing with 0
            keyword_data = keyword_data.reindex(date_range, fill_value=0)
            
            return keyword_data

        except Exception as e:
            self.logger.error(f"Error in _get_keyword_data for {keyword}: {str(e)}")
            return pd.DataFrame()


    def _get_keyword_data(self, keyword: str) -> pd.DataFrame:
        """
        Get historical data for a specific keyword with proper date handling.
        
        Args:
            keyword (str): Keyword to get data for
        
        Returns:
            pd.DataFrame: DataFrame containing keyword data with proper date indexing
        """
        try:
            # Get raw data for the keyword
            keyword_data = self.keyword_df[self.keyword_df['Keyword'] == keyword].copy()
            
            if keyword_data.empty:
                return pd.DataFrame()

            # Convert year to datetime and set as index
            keyword_data['Year'] = pd.to_datetime(keyword_data['Year'].astype(str), format='%Y')
            keyword_data = keyword_data.set_index('Year')[['Count']]
            
            # Sort by date
            keyword_data = keyword_data.sort_index()
            
            # Get full date range
            date_range = pd.date_range(
                start=keyword_data.index.min(),
                end=keyword_data.index.max(),
                freq='YE'
            )
            
            # Reindex to get all years, fill missing with 0
            keyword_data = keyword_data.reindex(date_range, fill_value=0)
            
            return keyword_data

        except Exception as e:
            self.logger.error(f"Error in _get_keyword_data for {keyword}: {str(e)}")
            return pd.DataFrame()


    def _simple_projection_forecast(self, data: pd.DataFrame, periods: int) -> ForecastResult:
        """Handle forecasting for very sparse datasets using simple projection with floor preservation."""
        try:
            historical_data = [
                {
                    'Year': index.year,
                    'Count': count
                }
                for index, count in data.itertuples()
            ]
            
            # Calculate trend considering the data floor
            values = data['Count'].values
            min_value = np.min(values) if len(values) > 0 else 0
            
            if len(values) > 1:
                # Calculate trend using values above the floor
                values_above_floor = values - min_value
                if np.sum(values_above_floor) > 0:
                    avg_change = (values_above_floor[-1] - values_above_floor[0]) / (len(values) - 1)
                else:
                    avg_change = 0
            else:
                avg_change = 0
            
            # Project future values
            last_year = data.index.max().year
            last_value = values[-1] if len(values) > 0 else 0
            base_value = max(min_value, last_value)  # Don't go below the minimum observed value
            
            forecast_results = []
            for i in range(periods):
                year = last_year + i + 1
                predicted_value = max(base_value, base_value + avg_change * (i + 1))
                forecast_results.append({
                    'Year': year,
                    'Predicted_Count': predicted_value
                })
            
            # Calculate confidence intervals with floor preservation
            confidence_intervals = []
            for result in forecast_results:
                year = result['Year']
                pred = result['Predicted_Count']
                confidence_intervals.append({
                    'Year': year,
                    'Lower': max(min_value, pred * 0.8),  # Don't go below historical minimum
                    'Upper': pred * 1.2
                })
            
            metrics = {
                'warning': 'Limited historical data - using conservative projection',
                'min_historical_value': min_value,
                'max_historical_value': np.max(values) if len(values) > 0 else 0,
                'trend': avg_change
            }
            
            return ForecastResult(
                forecast=forecast_results,
                historical_data=historical_data,
                confidence_intervals=confidence_intervals,
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error in simple projection forecast: {str(e)}")
            raise


    def _simple_smoothing_forecast(self, data: pd.DataFrame, periods: int) -> ForecastResult:
        """Handle forecasting for moderate-sized datasets using simple exponential smoothing."""
        try:
            # Use simple exponential smoothing without trend or seasonal components
            model = ExponentialSmoothing(
                data['Count'],
                trend=None,
                seasonal=None
            )
            
            fit_model = model.fit(optimized=True, remove_bias=True)
            predictions = fit_model.forecast(periods)
            
            # Prepare results
            historical_data = [
                {
                    'Year': index.year,
                    'Count': count
                }
                for index, count in data.itertuples()
            ]
            
            forecast_results = [
                {
                    'Year': year.year,
                    'Predicted_Count': max(0, pred)
                }
                for year, pred in zip(
                    pd.date_range(start=data.index.max(), periods=periods+1, freq='YE')[1:],
                    predictions
                )
            ]
            
            # Calculate confidence intervals using prediction standard errors
            std_err = np.std(data['Count']) if len(data) > 1 else data['Count'].iloc[0] * 0.1
            confidence_intervals = [
                {
                    'Year': result['Year'],
                    'Lower': max(0, result['Predicted_Count'] - 1.96 * std_err),
                    'Upper': result['Predicted_Count'] + 1.96 * std_err
                }
                for result in forecast_results
            ]
            
            return ForecastResult(
                forecast=forecast_results,
                historical_data=historical_data,
                confidence_intervals=confidence_intervals,
                metrics={'warning': 'Limited historical data - using simple smoothing'}
            )
            
        except Exception as e:
            self.logger.error(f"Error in simple smoothing forecast: {str(e)}")
            raise

    def _sarima_forecast(self, data: pd.DataFrame, periods: int) -> ForecastResult:
        """Handle forecasting for larger datasets using SARIMA model."""
        try:
            model = SARIMAX(
                data['Count'],
                order=(1, 0, 1),
                enforce_stationarity=False
            )
            
            fit_model = model.fit(disp=False)
            forecast = fit_model.get_forecast(periods)
            predictions = forecast.predicted_mean
            conf_int = forecast.conf_int()
            
            historical_data = [
                {
                    'Year': index.year,
                    'Count': count
                }
                for index, count in data.itertuples()
            ]
            
            forecast_results = [
                {
                    'Year': year.year,
                    'Predicted_Count': max(0, pred)
                }
                for year, pred in zip(
                    pd.date_range(start=data.index.max(), periods=periods+1, freq='YE')[1:],
                    predictions
                )
            ]
            
            confidence_intervals = [
                {
                    'Year': date.year,
                    'Lower': max(0, lower),
                    'Upper': upper
                }
                for date, (lower, upper) in zip(
                    pd.date_range(start=data.index.max(), periods=periods+1, freq='YE')[1:],
                    conf_int.values
                )
            ]
            
            # Calculate basic metrics if possible
            metrics = self.validate_forecast(data['Count'].values)
            
            return ForecastResult(
                forecast=forecast_results,
                historical_data=historical_data,
                confidence_intervals=confidence_intervals,
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error in SARIMA forecast: {str(e)}")
            raise

    def validate_forecast(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate basic forecast validation metrics."""
        if len(values) < 3:
            return {}
            
        try:
            # Calculate simple metrics based on one-step-ahead predictions
            predictions = values[:-1]  # Use previous values as predictions
            actuals = values[1:]  # Use next values as actuals
            
            return {
                'mse': mean_squared_error(actuals, predictions),
                'mae': mean_absolute_error(actuals, predictions),
                'rmse': np.sqrt(mean_squared_error(actuals, predictions))
            }
        except Exception:
            return {}


    def get_cases_over_time(self) -> List[Dict[str, Union[int, float]]]:
        """Get the number of cases per year."""
        cases_per_year = (self.df.groupby('year_filed')
                         .size()
                         .reset_index(name='count'))
        return cases_per_year.to_dict(orient='records')

    def get_keyword_trends(self) -> List[Dict[str, Union[str, int, float]]]:
        """Get historical keyword trends."""
        return self.keyword_trends

    def get_available_keywords(self) -> Dict[str, List[str]]:
        """Get list of available keywords."""
        unique_keywords = sorted(self.keyword_df['Keyword'].unique())
        return {"available_keywords": unique_keywords}

    def get_citations_over_time(self) -> List[Dict[str, Union[int, float]]]:
        """Get citation counts over time."""
        citation_trend = (self.df.groupby('year_filed')['citation_count']
                        .sum()
                        .reset_index())
        return citation_trend.to_dict(orient='records')
    
    def _convert_to_serializable(self, value):
        """Convert NumPy/Pandas values to JSON-serializable types."""
        if isinstance(value, (np.int32, np.int64)):
            return int(value)
        elif isinstance(value, (np.float32, np.float64)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return [self._convert_to_serializable(x) for x in value]
        elif isinstance(value, pd.Timestamp):
            return value.isoformat()
        return value


    def analyze_keyword_trends(self) -> Dict:
        """Analyze trends for all keywords with proper serialization."""
        keyword_stats = {}
        
        for keyword in self.keyword_df['Keyword'].unique():
            keyword_data = self.keyword_df[self.keyword_df['Keyword'] == keyword]
            
            # Calculate statistics
            stats = {
                'total_mentions': self._convert_to_serializable(keyword_data['Count'].sum()),
                'years_present': self._convert_to_serializable(len(keyword_data)),
                'first_year': self._convert_to_serializable(keyword_data['Year'].min()),
                'last_year': self._convert_to_serializable(keyword_data['Year'].max()),
                'max_count': self._convert_to_serializable(keyword_data['Count'].max()),
                'mean_count': self._convert_to_serializable(keyword_data['Count'].mean()),
                'has_recent_data': keyword_data['Year'].max() >= 2020
            }
            
            # Calculate trend
            if len(keyword_data) > 1:
                years = keyword_data['Year'].values
                counts = keyword_data['Count'].values
                z = np.polyfit(years, counts, 1)
                stats['trend'] = self._convert_to_serializable(z[0])
            else:
                stats['trend'] = 0.0
                
            keyword_stats[keyword] = stats
        
        # Find relevant keywords
        relevant_keywords = []
        for keyword, stats in keyword_stats.items():
            if (stats['total_mentions'] > 0 and 
                stats['has_recent_data'] and 
                stats['years_present'] >= 3):
                relevant_keywords.append({
                    'keyword': keyword,
                    **{k: self._convert_to_serializable(v) 
                    for k, v in stats.items()}
                })
        
        # Sort by total mentions
        relevant_keywords.sort(
            key=lambda x: x['total_mentions'], 
            reverse=True
        )
        
        return {
            'all_stats': keyword_stats,
            'relevant_keywords': relevant_keywords
        }


    def get_top_keywords(self, n: int = 10) -> List[Dict]:
        """Get the top n keywords by total mentions with meaningful trends."""
        analysis = self.analyze_keyword_trends()
        return analysis['relevant_keywords'][:n]
    
    def forecast_keyword_trends(self, keyword: str, periods: int = 5) -> ForecastResult:
        """
        Forecast future trends with improved handling of cyclical patterns.
        """
        try:
            # Get data directly from keyword_df
            keyword_data = self.keyword_df[self.keyword_df['Keyword'] == keyword].copy()
            keyword_data = keyword_data.sort_values('Year')
            
            if keyword_data.empty:
                return ForecastResult(
                    forecast=[],
                    historical_data=[],
                    confidence_intervals=[],
                    metrics={'warning': 'No data available'}
                )

            # Create historical data
            historical_data = [
                {
                    'Year': int(row['Year']),
                    'Count': float(row['Count'])
                }
                for _, row in keyword_data.iterrows()
            ]
            
            # Calculate base metrics
            metrics = {
                'total_mentions': int(keyword_data['Count'].sum()),
                'mean_count': float(keyword_data['Count'].mean()),
                'max_count': float(keyword_data['Count'].max()),
                'years_with_mentions': len(keyword_data),
                'year_range': f"{keyword_data['Year'].min()} - {keyword_data['Year'].max()}"
            }
            
            # Calculate long-term and recent trends
            years = np.array(keyword_data['Year'])
            counts = np.array(keyword_data['Count'])
            
            # Long-term trend using all data
            z_long = np.polyfit(years, counts, 1)
            long_term_trend = z_long[0]
            
            # Recent trend using last 5 years
            recent_data = keyword_data.tail(5)
            z_recent = np.polyfit(recent_data['Year'], recent_data['Count'], 1)
            recent_trend = z_recent[0]
            
            # Use weighted average of trends
            trend = (0.7 * recent_trend + 0.3 * long_term_trend)
            
            # Calculate baseline using recent averages
            recent_mean = recent_data['Count'].mean()
            last_value = float(keyword_data['Count'].iloc[-1])
            baseline = (recent_mean + last_value) / 2
            
            # Calculate rolling standard deviation for more stable confidence intervals
            rolling_std = keyword_data['Count'].rolling(window=5, min_periods=1).std().mean()
            
            forecast_results = []
            confidence_intervals = []
            
            for i in range(periods):
                year = int(keyword_data['Year'].max()) + i + 1
                
                # Dampen trend over time
                damping_factor = 1 / (1 + 0.2 * i)
                predicted_value = max(
                    baseline * 0.5,  # Don't go below half the baseline
                    baseline + trend * i * damping_factor
                )
                
                forecast_results.append({
                    'Year': year,
                    'Predicted_Count': predicted_value
                })
                
                # More conservative confidence intervals
                ci_width = rolling_std * (1 + 0.1 * i)  # Increase uncertainty over time
                confidence_intervals.append({
                    'Year': year,
                    'Lower': max(0, predicted_value - ci_width),
                    'Upper': predicted_value + ci_width
                })
            
            # Add trend metrics
            metrics.update({
                'long_term_trend': long_term_trend,
                'recent_trend': recent_trend,
                'weighted_trend': trend,
                'trend_interpretation': self._interpret_trend(trend, recent_trend)
            })
            
            return ForecastResult(
                forecast=forecast_results,
                historical_data=historical_data,
                confidence_intervals=confidence_intervals,
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error in forecast_keyword_trends: {str(e)}")
            raise

    def _interpret_trend(self, weighted_trend: float, recent_trend: float) -> str:
        """Provide interpretation of the trend."""
        if abs(weighted_trend) < 0.5:
            return "stable"
        elif weighted_trend > 0:
            return "increasing" if recent_trend > 0 else "mixed but generally increasing"
        else:
            return "decreasing" if recent_trend < 0 else "mixed but generally decreasing"

    def print_forecast_analysis(self, keyword: str):
        """Print a detailed forecast analysis for a keyword."""
        result = self.forecast_keyword_trends(keyword)
        
        print(f"\nAnalysis for: {keyword}")
        print("=" * 50)
        
        if not result.historical_data:
            print("No data available for this keyword")
            return
        
        print("\nRecent Historical Data (last 5 years):")
        for point in result.historical_data[-5:]:
            print(f"Year {point['Year']}: {point['Count']:.1f}")
        
        print("\nForecast:")
        for point, ci in zip(result.forecast, result.confidence_intervals):
            print(f"Year {point['Year']}: {point['Predicted_Count']:.1f} " +
                f"(95% CI: {ci['Lower']:.1f} - {ci['Upper']:.1f})")
        
        print("\nMetrics:")
        for key, value in result.metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")




# Example usage
if __name__ == "__main__":
    try:
        data_provider = TechLawData('../../datasets/test_all_tech_cases_fixed.csv')
        
        test_keywords = ['technology', 'patent', 'software']
        for keyword in test_keywords:
            data_provider.print_forecast_analysis(keyword)
    except Exception as e:
        print(f"Error in main execution: {str(e)}")