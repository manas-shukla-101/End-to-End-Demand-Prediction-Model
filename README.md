# 📈 End-to-End Demand Forecasting System

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub last commit](https://img.shields.io/github/last-commit/manas-shukla-101/End-to-End-Demand-Prediction-Model)](https://github.com/manas-shukla-101/End-to-End-Demand-Prediction-Model)
[![Status: Active](https://img.shields.io/badge/status-active-success.svg)](https://github.com/manas-shukla-101/End-to-End-Demand-Prediction-Model)

*An advanced, production-ready demand forecasting system with AI-powered recommendations, interactive dashboard, and intelligent inventory optimization.*

[Features](#-features) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

---

## 🌟 Overview

A **comprehensive end-to-end demand forecasting system** that helps businesses:
- **Predict future demand** with high accuracy using ARIMA and Prophet models
- **Optimize inventory** levels based on demand patterns and forecasts
- **Maximize revenue** through intelligent pricing and planning strategies
- **Manage risks** with confidence intervals and performance metrics
- **Analyze patterns** with advanced seasonality decomposition

Perfect for supply chain professionals, data scientists, and business analysts who need accurate demand predictions and actionable insights.

---

## ✨ Features

### 📈 **Advanced Forecasting**
- ✅ ARIMA (AutoRegressive Integrated Moving Average) with configurable p,d,q parameters
- ✅ Prophet (Facebook's time series forecasting) with automatic seasonality detection
- ✅ Dual-model comparison with performance metrics (MAE, RMSE, MAPE)
- ✅ 95% confidence intervals for all forecasts
- ✅ Supports forecasting up to 1 year into the future

### 🔄 **Time Series Analysis**
- ✅ Adaptive seasonality decomposition (weekly, monthly, yearly)
- ✅ Trend extraction and visualization
- ✅ Seasonal pattern identification
- ✅ Residual analysis
- ✅ Handles datasets with 14+ observations

### 💡 **AI-Powered Recommendations**
- ✅ **Demand Trend Analysis** - Identify growth, decline, or stability
- ✅ **Model Recommendation** - Automatic best model selection
- ✅ **Inventory Strategy** - Optimize stock levels based on variability
- ✅ **Revenue Optimization** - Dynamic pricing suggestions
- ✅ **Risk Assessment** - Forecast reliability evaluation
- ✅ **Confidence Metrics** - Prediction interval analysis

### 📊 **Interactive Dashboard**
- ✅ Streamlit-based web interface
- ✅ Real-time visualizations and charts
- ✅ Custom CSV dataset upload with auto-detection
- ✅ Synthetic data generation for testing
- ✅ Configurable forecasting parameters
- ✅ Responsive design (desktop & mobile)

### 📁 **Data Management**
- ✅ Support for custom CSV datasets
- ✅ Automatic date and demand column mapping
- ✅ Realistic synthetic data generation
- ✅ Data validation and quality checks
- ✅ Monthly statistics and aggregations

### 📉 **Comprehensive Analytics**
- ✅ Historical demand statistics (mean, std, min, max)
- ✅ Coefficient of variation (volatility measure)
- ✅ Monthly and yearly trends
- ✅ Model performance metrics
- ✅ Forecast accuracy validation

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (tested on 3.13.5)
- **pip** or **conda**
- 4GB RAM minimum (for large datasets)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/manas-shukla-101/End-to-End-Demand-Prediction-Model.git
   cd End-to-End-Demand-Prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the System

#### Option 1: Interactive Dashboard (Recommended)
```bash
streamlit run streamlit_app.py
```
Then open `http://localhost:8501` in your browser.

#### Option 2: Command Line
```bash
python main.py
```

---

## 📱 Demo

### Dashboard Preview
- **Step 1**: Load historical data (synthetic or custom CSV)
- **Step 2**: View historical statistics and trends
- **Step 3**: Analyze seasonality decomposition
- **Step 4**: Generate ARIMA forecasts
- **Step 5**: Generate Prophet forecasts
- **Step 6**: Compare model performance
- **Step 7**: View complete forecast comparison
- **Step 8**: Get AI-powered recommendations

### Features in Action
- 🎯 **Demand Trend**: Shows if demand is increasing (+X%), decreasing (-X%), or stable
- 💾 **Model Comparison**: Visual side-by-side comparison with confidence intervals
- 📊 **Metrics Dashboard**: MAE, RMSE, MAPE displayed for both models
- 💡 **Recommendations**: 6 intelligent recommendations across demand, inventory, revenue, and risk
- 📈 **Decomposition**: 4-panel view of observed, trend, seasonal, and residual components

---

## 📋 Project Structure

```
demand-forecasting/
│
├── main.py                      # Core forecasting engine (470+ lines)
├── streamlit_app.py             # Interactive dashboard (350+ lines)
├── icon.png                     # Dashboard branding icon
├── requirements_updated.txt      # Python dependencies
├── sample_demand_data.csv        # Sample dataset (90 rows)
└── README.md  
 
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.8+ | Core development |
| **Data Processing** | Pandas, NumPy | Data manipulation & analysis |
| **Time Series** | Statsmodels | ARIMA modeling & decomposition |
| **Advanced Forecasting** | Prophet | Automated seasonality detection |
| **Machine Learning** | Scikit-learn, XGBoost | Metrics & potential enhancements |
| **Visualization** | Matplotlib, Seaborn, Plotly | Charts and graphs |
| **Web Interface** | Streamlit | Interactive dashboard |

---


## 💻 Usage Examples

### Example 1: Generate Synthetic Data & Forecast
```python
from main import DemandData, ARIMAForecaster, ProphetForecaster, ForecastComparison

# Generate sample data
data_gen = DemandData()
df = data_gen.generate_sample_data(days=730, trend=True, seasonality=True, noise=True)

# ARIMA Forecast
arima = ARIMAForecaster(df, p=1, d=1, q=1)
arima.fit()
arima_forecast = arima.predict(periods=90)

# Prophet Forecast
prophet = ProphetForecaster(df)
prophet.fit()
prophet_forecast = prophet.predict(periods=90)

# Compare models
metrics = ForecastComparison.calculate_metrics(
    actual=df['demand'].tail(30),
    predicted=arima_forecast['forecast'].head(30)
)
print(f"MAPE: {metrics['MAPE']:.2f}%")
```

### Example 2: Load Custom CSV Data
```python
import pandas as pd
from main import HistoricalAnalysis

# Load your data
df = pd.read_csv('your_demand_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Analyze
analyzer = HistoricalAnalysis(df)
stats = analyzer.get_statistics()
monthly_stats = analyzer.get_monthly_stats()
```

### Example 3: Get AI Recommendations
```python
from main import ForecastComparison

recommendations = ForecastComparison.generate_recommendations(
    df=df,
    arima_forecast=arima_forecast,
    prophet_forecast=prophet_forecast,
    arima_metrics=arima_metrics,
    prophet_metrics=prophet_metrics
)

print(recommendations['demand_trend'])
print(recommendations['inventory_strategy'])
print(recommendations['revenue_optimization'])
```

---

## 🎯 Model Selection Guide

### ARIMA (AutoRegressive Integrated Moving Average)
| Aspect | Details |
|--------|---------|
| **Best For** | Stationary or differenced time series |
| **Strengths** | Interpretable, fast, proven |
| **Limitations** | Requires parameter tuning |
| **Parameters** | p (AR order), d (differencing), q (MA order) |
| **Use When** | Data shows clear trend/seasonality patterns |

### Prophet
| Aspect | Details |
|--------|---------|
| **Best For** | Multiple seasonalities & holidays |
| **Strengths** | Automatic detection, robust |
| **Limitations** | Less interpretable |
| **Parameters** | Auto-configured |
| **Use When** | Complex seasonal patterns (daily, weekly, yearly) |

---

## 📊 Performance Metrics

### Explained Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **MAE** | Average absolute error | Average prediction error (same units as demand) |
| **RMSE** | √(mean squared error) | Penalizes large errors more heavily |
| **MAPE** | Average percentage error | Error as % of actual (easy comparison across scales) |

**Rule of Thumb**: MAPE < 10% = Excellent, 10-20% = Good, 20-50% = Fair, > 50% = Poor

---

## ⚙️ Configuration Guide

### ARIMA Parameters
- **p** (0-5): Number of AR (autoregressive) terms
- **d** (0-3): Number of differencing operations
- **q** (0-5): Number of MA (moving average) terms

**Recommended Starting Points**:
- p=1, d=1, q=1 (default, general purpose)
- p=2, d=1, q=1 (stronger trends)
- p=1, d=1, q=2 (more smoothing)

### Forecast Horizon
- **7-30 days**: Tactical planning (highly reliable)
- **30-90 days**: Operational planning (reliable)
- **90-365 days**: Strategic planning (less reliable)

---

## 🐛 Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'prophet'"**
```bash
pip install --upgrade pystan==2.19.1.1
pip install prophet
```

**2. "Decomposition requires at least 730 observations"**
- Use datasets with 2+ years of historical data
- Or adjust decomposition period (7, 30, or 365 days)

**3. "Streamlit deprecated parameter warning"**
- Update Streamlit: `pip install --upgrade streamlit`
- Already fixed in latest version


---

## 🔮 Advanced Features

### Seasonality Decomposition
Adaptive period selection:
- **≥730 observations**: Yearly (365-day) seasonality
- **≥60 observations**: Monthly (30-day) seasonality
- **≥14 observations**: Weekly (7-day) seasonality
- **<14 observations**: Warning (insufficient data)

### Confidence Intervals
All forecasts include 95% confidence intervals:
- **Narrow CI** = High confidence predictions
- **Wide CI** = Uncertain predictions (use conservative planning)

---

## 📈 Future Roadmap

### Planned Features
- [ ] XGBoost integration for non-linear patterns
- [ ] Ensemble methods (weighted averaging)
- [ ] Real-time API data integration
- [ ] Export to Excel/CSV with formatting
- [ ] Hyperparameter optimization (auto p,d,q tuning)
- [ ] Multivariate forecasting (multiple demand factors)
- [ ] Anomaly detection & outlier handling
- [ ] Monthly scheduled forecasting
- [ ] Slack/Email notifications for forecasts

---

## 🤝 Contributing

I welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Guidelines
- Follow PEP 8 style guide
- Add docstrings to functions
- Update README for new features
- Test with various datasets

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 🔗 References & Resources

### Documentation
- [ARIMA Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Learning Resources
- [Time Series Forecasting Guide](https://otexts.com/fpp2/)
- [Demand Planning Best Practices](https://www.apics.org/)
- [Supply Chain Optimization](https://www.gartner.com/en/supply-chain)

### Related Projects
- [Statsmodels](https://www.statsmodels.org/)
- [Facebook Prophet](https://facebook.github.io/prophet/)
- [Scikit-learn](https://scikit-learn.org/)

---

## 📧 Support & Contact

### Getting Help
- 🐛 [Report Issues](https://github.com/manas-shukla-101/End-to-End-Demand-Prediction-Model/issues)
- 💬 [Start a Discussion](https://github.com/manas-shukla-101/End-to-End-Demand-Prediction-Model/discussions)
- 📧 Contact: [G-mail](shuklamanas8928@gmail.com) 

---

## ⭐ Show Your Support

If this project helped you, please:
- ⭐ Star the repository
- 🍴 Fork the project
- 📢 Share with others
- 💬 Leave feedback

---
---
<div align="center">

   **Made with ❤️ by Manas Shukla**

---

## 🌐 Socials:
[![Portfolio](https://img.shields.io/badge/Portfolio-Website-blue)](https://manas-shukla-portfolio.framer.website) [![Instagram](https://img.shields.io/badge/Instagram-%23E4405F.svg?logo=Instagram&logoColor=white)](https://instagram.com/manas_shukla_101) [![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://linkedin.com/in/manas-shukla-006774370) [![email](https://img.shields.io/badge/Email-D14836?logo=gmail&logoColor=white)](mailto:shuklamanas8928@gmail.com) 

---

[Back to Top](#-end-to-end-demand-forecasting-system)

</div>


