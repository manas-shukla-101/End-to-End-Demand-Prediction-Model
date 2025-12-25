"""
End-to-End Demand Forecasting System
Predicts future demand to help businesses plan inventory and revenue strategies.
Techstacks: Python, Pandas, Scikit-learn, Statsmodels, XGBoost, Prophet, Matplotlib, Seaborn, Streamlit.

Components:
1. Historical Data Analysis
2. Time Series Forecasting using ARIMA and Prophet
3. Seasonality and Trend Decomposition
4. Forecast vs Actual Comparison
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
import xgboost as xgb

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DemandData:
    """Generate realistic demand data for analysis"""
    
    @staticmethod
    def generate_sample_data(days=365*2, trend=True, seasonality=True, noise=True):
        """
        Generate synthetic demand data with trend, seasonality, and noise
        
        Parameters:
        - days: Number of days to generate (default 2 years)
        - trend: Include trend component
        - seasonality: Include seasonality
        - noise: Add random noise
        """
        dates = pd.date_range(start='2022-01-01', periods=days, freq='D')
        np.random.seed(42)
        
        # Base demand
        demand = np.ones(days) * 100
        
        # Add trend (increasing demand over time)
        if trend:
            demand += np.linspace(0, 50, days)
        
        # Add seasonality (weekly and yearly patterns)
        if seasonality:
            weekly = 20 * np.sin(2 * np.pi * np.arange(days) / 7)
            yearly = 30 * np.sin(2 * np.pi * np.arange(days) / 365)
            demand += weekly + yearly
        
        # Add random noise
        if noise:
            demand += np.random.normal(0, 5, days)
        
        # Ensure demand is positive
        demand = np.maximum(demand, 10)
        
        df = pd.DataFrame({
            'date': dates,
            'demand': demand
        })
        
        return df


class HistoricalAnalysis:
    """Analyze historical demand data"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
    
    def get_statistics(self):
        """Calculate basic statistics"""
        stats = {
            'Mean Demand': self.df['demand'].mean(),
            'Std Dev': self.df['demand'].std(),
            'Min Demand': self.df['demand'].min(),
            'Max Demand': self.df['demand'].max(),
            'Total Days': len(self.df),
            'Date Range': f"{self.df['date'].min().date()} to {self.df['date'].max().date()}"
        }
        return stats
    
    def get_monthly_stats(self):
        """Get monthly aggregated statistics"""
        self.df['year_month'] = self.df['date'].dt.to_period('M')
        monthly = self.df.groupby('year_month')['demand'].agg(['mean', 'sum', 'std']).reset_index()
        monthly.columns = ['Year-Month', 'Avg Demand', 'Total Demand', 'Std Dev']
        return monthly
    
    def print_analysis(self):
        """Print analysis summary"""
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("HISTORICAL DATA ANALYSIS")
        print("="*60)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key:.<40} {value:>15.2f}")
            else:
                print(f"{key:.<40} {value:>15}")


class ARIMAForecaster:
    """ARIMA-based time series forecasting"""
    
    def __init__(self, df, p=1, d=1, q=1):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.p, self.d, self.q = p, d, q
        self.model = None
        self.forecast = None
    
    def fit(self):
        """Fit ARIMA model"""
        self.model = ARIMA(self.df['demand'], order=(self.p, self.d, self.q))
        self.model = self.model.fit()
        return self.model.summary()
    
    def predict(self, periods=30):
        """Forecast future demand"""
        if self.model is None:
            self.fit()
        
        forecast_result = self.model.get_forecast(steps=periods)
        forecast_df = forecast_result.conf_int(alpha=0.05).reset_index()
        forecast_df.columns = ['index', 'lower_ci', 'upper_ci']
        forecast_df['forecast'] = forecast_result.predicted_mean.values
        
        # Create dates for forecast
        last_date = self.df['date'].max()
        forecast_df['date'] = [last_date + timedelta(days=x+1) for x in range(periods)]
        
        self.forecast = forecast_df[['date', 'forecast', 'lower_ci', 'upper_ci']].copy()
        return self.forecast


class ProphetForecaster:
    """Prophet-based time series forecasting"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        # Prophet expects 'ds' and 'y' columns
        self.prophet_df = self.df[['date', 'demand']].rename(columns={'date': 'ds', 'demand': 'y'})
        self.model = None
        self.forecast = None
    
    def fit(self):
        """Fit Prophet model"""
        self.model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        self.model.fit(self.prophet_df)
    
    def predict(self, periods=30):
        """Forecast future demand"""
        if self.model is None:
            self.fit()
        
        future = self.model.make_future_dataframe(periods=periods)
        forecast_result = self.model.predict(future)
        
        # Extract forecast for future periods
        self.forecast = forecast_result[forecast_result['ds'] > self.df['date'].max()][
            ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        ].rename(columns={'ds': 'date', 'yhat': 'forecast', 'yhat_lower': 'lower_ci', 'yhat_upper': 'upper_ci'})
        
        self.forecast['forecast'] = self.forecast['forecast'].clip(lower=0)
        
        return self.forecast


class SeasonalityAnalysis:
    """Decompose time series into trend, seasonality, and residuals"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').set_index('date')
        self.decomposition = None
        self.period = None
    
    def get_optimal_period(self):
        """Determine optimal decomposition period based on data size"""
        n_obs = len(self.df)
        
        # Need at least 2 complete cycles for decomposition
        if n_obs >= 730:
            return 365  # Yearly seasonality (1 year)
        elif n_obs >= 60:
            return 30   # Monthly seasonality (1 month)
        elif n_obs >= 14:
            return 7    # Weekly seasonality (1 week)
        else:
            return None  # Not enough data for decomposition
    
    def decompose(self, period=None):
        """Perform seasonal decomposition with adaptive period"""
        if period is None:
            period = self.get_optimal_period()
        
        if period is None:
            return None
        
        self.period = period
        try:
            self.decomposition = seasonal_decompose(self.df['demand'], model='additive', period=period)
            return self.decomposition
        except Exception as e:
            print(f"Warning: Decomposition failed with period {period}. Error: {str(e)}")
            return None
    
    def get_components(self):
        """Get trend, seasonal, and residual components"""
        if self.decomposition is None:
            self.decompose()
        
        if self.decomposition is None:
            return None
        
        return {
            'trend': self.decomposition.trend,
            'seasonal': self.decomposition.seasonal,
            'residual': self.decomposition.resid
        }


class ForecastComparison:
    """Compare forecasts with actual values and calculate metrics"""
    
    @staticmethod
    def calculate_metrics(actual, predicted):
        """Calculate error metrics"""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = mean_absolute_percentage_error(actual, predicted)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    @staticmethod
    def generate_recommendations(df, arima_forecast, prophet_forecast, arima_metrics, prophet_metrics):
        """Generate actionable recommendations based on analysis"""
        recommendations = {
            'demand_trend': '',
            'model_recommendation': '',
            'inventory_strategy': '',
            'revenue_optimization': '',
            'risk_assessment': '',
            'forecast_confidence': ''
        }
        
        # 1. Demand Trend Analysis
        historical_mean = df['demand'].mean()
        forecast_mean = arima_forecast['forecast'].mean()
        trend_change = ((forecast_mean - historical_mean) / historical_mean) * 100
        
        if trend_change > 10:
            recommendations['demand_trend'] = f"📈 **Increasing Demand**: Forecast shows {trend_change:.1f}% increase. Prepare for higher demand volume."
        elif trend_change < -10:
            recommendations['demand_trend'] = f"📉 **Decreasing Demand**: Forecast shows {trend_change:.1f}% decline. Optimize operations for lower volumes."
        else:
            recommendations['demand_trend'] = f"➡️ **Stable Demand**: Forecast shows {abs(trend_change):.1f}% variation. Maintain current operations."
        
        # 2. Model Selection Recommendation
        better_model = 'ARIMA' if arima_metrics['MAPE'] < prophet_metrics['MAPE'] else 'Prophet'
        mape_diff = abs(arima_metrics['MAPE'] - prophet_metrics['MAPE'])
        
        if mape_diff < 2:
            recommendations['model_recommendation'] = f"⚖️ **Both Models Comparable**: {better_model} is slightly better ({mape_diff:.2f}% difference). Use ensemble approach for robustness."
        else:
            recommendations['model_recommendation'] = f"✓ **Recommended Model**: {better_model} (MAPE: {min(arima_metrics['MAPE'], prophet_metrics['MAPE']):.2f}%). Use for production forecasts."
        
        # 3. Inventory Strategy
        forecast_std = arima_forecast['forecast'].std()
        forecast_cv = (forecast_std / forecast_mean) * 100 if forecast_mean > 0 else 0
        
        if forecast_cv > 25:
            recommendations['inventory_strategy'] = f"🔄 **High Variability** (CV: {forecast_cv:.1f}%): Implement just-in-time inventory. Higher safety stock needed."
        elif forecast_cv > 15:
            recommendations['inventory_strategy'] = f"📊 **Moderate Variability** (CV: {forecast_cv:.1f}%): Balance between stock levels and holding costs."
        else:
            recommendations['inventory_strategy'] = f"✓ **Low Variability** (CV: {forecast_cv:.1f}%): Stable demand. Optimize for cost efficiency."
        
        # 4. Revenue Optimization
        min_forecast = arima_forecast['forecast'].min()
        max_forecast = arima_forecast['forecast'].max()
        revenue_range_pct = ((max_forecast - min_forecast) / min_forecast) * 100 if min_forecast > 0 else 0
        
        if revenue_range_pct > 40:
            recommendations['revenue_optimization'] = f"💰 **Dynamic Pricing**: High forecast variance ({revenue_range_pct:.1f}%). Implement surge pricing during peak demand."
        elif revenue_range_pct > 20:
            recommendations['revenue_optimization'] = f"💰 **Seasonal Pricing**: Moderate variance ({revenue_range_pct:.1f}%). Apply seasonal discounts during low demand."
        else:
            recommendations['revenue_optimization'] = f"💰 **Stable Pricing**: Low variance ({revenue_range_pct:.1f}%). Focus on cost reduction and margins."
        
        # 5. Risk Assessment
        mape_threshold = 15
        best_mape = min(arima_metrics['MAPE'], prophet_metrics['MAPE'])
        
        if best_mape < 5:
            recommendations['risk_assessment'] = f"✓ **Low Risk**: MAPE {best_mape:.2f}% indicates reliable forecasts. Confidence: Very High"
        elif best_mape < 10:
            recommendations['risk_assessment'] = f"⚠️ **Moderate Risk**: MAPE {best_mape:.2f}% indicates acceptable forecasts. Maintain contingency plans."
        else:
            recommendations['risk_assessment'] = f"🚨 **High Risk**: MAPE {best_mape:.2f}% indicates variable forecasts. Use conservative safety margins."
        
        # 6. Forecast Confidence
        ci_width = (arima_forecast['upper_ci'] - arima_forecast['lower_ci']).mean()
        ci_percentage = (ci_width / forecast_mean) * 100 if forecast_mean > 0 else 0
        
        if ci_percentage < 20:
            recommendations['forecast_confidence'] = f"🎯 **High Confidence**: 95% CI width {ci_percentage:.1f}%. Narrow range for strategic planning."
        elif ci_percentage < 35:
            recommendations['forecast_confidence'] = f"📌 **Moderate Confidence**: 95% CI width {ci_percentage:.1f}%. Suitable for tactical planning."
        else:
            recommendations['forecast_confidence'] = f"⚡ **Wide Range**: 95% CI width {ci_percentage:.1f}%. Plan with flexibility."
        
        return recommendations
    
    @staticmethod
    def plot_comparison(actual_df, arima_forecast, prophet_forecast):
        """Plot actual vs forecasted demand"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot 1: Historical demand
        axes[0, 0].plot(actual_df['date'], actual_df['demand'], label='Actual Demand', linewidth=2)
        axes[0, 0].set_title('Historical Demand Data', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Demand')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: ARIMA Forecast
        axes[0, 1].plot(actual_df['date'], actual_df['demand'], label='Historical Demand', linewidth=2)
        axes[0, 1].plot(arima_forecast['date'], arima_forecast['forecast'], 
                       label='ARIMA Forecast', linewidth=2, linestyle='--')
        axes[0, 1].fill_between(arima_forecast['date'], arima_forecast['lower_ci'], 
                               arima_forecast['upper_ci'], alpha=0.2)
        axes[0, 1].set_title('ARIMA Forecast with 95% CI', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Demand')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Prophet Forecast
        axes[1, 0].plot(actual_df['date'], actual_df['demand'], label='Historical Demand', linewidth=2)
        axes[1, 0].plot(prophet_forecast['date'], prophet_forecast['forecast'], 
                       label='Prophet Forecast', linewidth=2, linestyle='--', color='orange')
        axes[1, 0].fill_between(prophet_forecast['date'], prophet_forecast['lower_ci'], 
                               prophet_forecast['upper_ci'], alpha=0.2)
        axes[1, 0].set_title('Prophet Forecast with 95% CI', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Demand')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Model Comparison
        axes[1, 1].plot(actual_df['date'], actual_df['demand'], label='Actual Demand', linewidth=2)
        axes[1, 1].plot(arima_forecast['date'], arima_forecast['forecast'], 
                       label='ARIMA', linewidth=2, linestyle='--')
        axes[1, 1].plot(prophet_forecast['date'], prophet_forecast['forecast'], 
                       label='Prophet', linewidth=2, linestyle='--')
        axes[1, 1].set_title('Model Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Demand')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('demand_forecast_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n✓ Forecast comparison chart saved as 'demand_forecast_comparison.png'")


class SeasonalityPlotter:
    """Visualize seasonality and trend components"""
    
    @staticmethod
    def plot_decomposition(decomposition):
        """Plot decomposed time series components"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        # Original
        decomposition.observed.plot(ax=axes[0], title='Observed', color='blue')
        axes[0].set_ylabel('Demand')
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        decomposition.trend.plot(ax=axes[1], title='Trend', color='green')
        axes[1].set_ylabel('Trend')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonality
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal', color='orange')
        axes[2].set_ylabel('Seasonal')
        axes[2].grid(True, alpha=0.3)
        
        # Residual
        decomposition.resid.plot(ax=axes[3], title='Residual', color='red')
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('seasonality_decomposition.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Seasonality decomposition chart saved as 'seasonality_decomposition.png'")


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("END-TO-END DEMAND FORECASTING SYSTEM")
    print("="*70)
    
    # Step 1: Generate sample data
    print("\n[Step 1] Generating sample demand data...")
    data_generator = DemandData()
    df = data_generator.generate_sample_data(days=730, trend=True, seasonality=True, noise=True)
    print(f"✓ Generated {len(df)} days of demand data")
    
    # Step 2: Historical Data Analysis
    print("\n[Step 2] Performing historical data analysis...")
    analyzer = HistoricalAnalysis(df)
    analyzer.print_analysis()
    
    monthly_stats = analyzer.get_monthly_stats()
    print("\nMonthly Statistics (First 5 months):")
    print(monthly_stats.head())
    
    # Step 3: Seasonality and Trend Decomposition
    print("\n[Step 3] Decomposing time series into components...")
    seasonality = SeasonalityAnalysis(df)
    decomposition = seasonality.decompose(period=365)
    SeasonalityPlotter.plot_decomposition(decomposition)
    
    # Step 4: ARIMA Forecasting
    print("\n[Step 4] Training ARIMA model...")
    arima = ARIMAForecaster(df, p=1, d=1, q=1)
    arima.fit()
    arima_forecast = arima.predict(periods=90)
    print("✓ ARIMA model trained and forecast generated")
    print(f"\nARIMA Forecast (First 5 days):")
    print(arima_forecast.head())
    
    # Step 5: Prophet Forecasting
    print("\n[Step 5] Training Prophet model...")
    prophet = ProphetForecaster(df)
    prophet.fit()
    prophet_forecast = prophet.predict(periods=90)
    print("✓ Prophet model trained and forecast generated")
    print(f"\nProphet Forecast (First 5 days):")
    print(prophet_forecast.head())
    
    # Step 6: Forecast Evaluation & Comparison
    print("\n[Step 6] Comparing forecasts and calculating metrics...")
    
    # For comparison, we'll use last 30 days as test data
    test_days = 30
    train_df = df[:-test_days].copy()
    test_df = df[-test_days:].copy()
    
    # Re-train models on training data
    arima_train = ARIMAForecaster(train_df, p=1, d=1, q=1)
    arima_train.fit()
    arima_test_forecast = arima_train.predict(periods=test_days)
    
    prophet_train = ProphetForecaster(train_df)
    prophet_train.fit()
    prophet_test_forecast = prophet_train.predict(periods=test_days)
    
    # Calculate metrics
    arima_metrics = ForecastComparison.calculate_metrics(
        test_df['demand'].values,
        arima_test_forecast['forecast'].values
    )
    
    prophet_metrics = ForecastComparison.calculate_metrics(
        test_df['demand'].values,
        prophet_test_forecast['forecast'].values
    )
    
    print("\n" + "="*60)
    print("FORECAST PERFORMANCE METRICS (on test set)")
    print("="*60)
    print("\nARIMA Model Metrics:")
    for metric, value in arima_metrics.items():
        print(f"  {metric:.<35} {value:.2f}")
    
    print("\nProphet Model Metrics:")
    for metric, value in prophet_metrics.items():
        print(f"  {metric:.<35} {value:.2f}")
    
    # Step 7: Visualizations
    print("\n[Step 7] Generating comparison visualizations...")
    ForecastComparison.plot_comparison(df, arima_forecast, prophet_forecast)
    
    # Summary
    print("\n" + "="*70)
    print("SYSTEM SUMMARY")
    print("="*70)
    print(f"✓ Historical Data: {len(df)} days analyzed")
    print(f"✓ Forecast Horizon: 90 days")
    print(f"✓ Models Trained: ARIMA and Prophet")
    print(f"✓ Best Model: {'ARIMA' if arima_metrics['MAPE'] < prophet_metrics['MAPE'] else 'Prophet'}")
    print(f"✓ Visualizations: 2 charts generated")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

