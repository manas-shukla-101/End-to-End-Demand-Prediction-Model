"""
Interactive Streamlit Dashboard for End-to-End Demand Forecasting System
Allows users to configure forecasting parameters and visualize results in real-time
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

# Import forecasting modules
from main import (
    DemandData, HistoricalAnalysis, ARIMAForecaster, ProphetForecaster,
    SeasonalityAnalysis, ForecastComparison, SeasonalityPlotter
)

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Demand Forecasting System",
    page_icon="icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.logo("icon.png")
st.markdown(
    """
    ---
    <style>
        [alt=Logo] {
            height: 4rem; /* Adjust this value */
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def main():
    # Display Icon Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image('icon.png', width=300, caption='End-to-End Demand Forecasting Model by MANAS SHUKLA')
        except FileNotFoundError:
            st.warning("⚠️ Icon file not found. Please add 'icon.png' to the project directory.")
    
    st.divider()
    
    # Header
    st.title("📈 End-to-End Demand Forecasting Model")
    st.markdown("""
    Advanced system with intelligent recommendations for demand prediction, inventory planning, and revenue optimization.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Data Source Selection
    st.sidebar.subheader("📁 Data Source")
    data_source = st.sidebar.radio("Select Data Source:", ["Generate Synthetic Data", "Upload Custom Dataset"])
    
    if data_source == "Upload Custom Dataset":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_input = pd.read_csv(uploaded_file)
                
                # Validate dataset
                st.sidebar.info(f"✓ File loaded: {len(df_input)} rows")
                
                # Show preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(df_input.head(10), width='stretch')
                
                # Column mapping
                st.sidebar.subheader("Column Mapping")
                date_col = st.sidebar.selectbox("Date Column", df_input.columns)
                demand_col = st.sidebar.selectbox("Demand Column", [col for col in df_input.columns if col != date_col])
                
                # Prepare dataframe
                df = df_input[[date_col, demand_col]].copy()
                df.columns = ['date', 'demand']
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                
                use_custom_data = True
                
            except Exception as e:
                st.sidebar.error(f"❌ Error loading file: {str(e)}")
                use_custom_data = False
        else:
            st.sidebar.warning("Please upload a CSV file")
            use_custom_data = False
    else:
        use_custom_data = False
        # Data Generation Parameters
        st.sidebar.subheader("Data Generation")
        days = st.sidebar.slider("Historical Days to Generate", min_value=90, max_value=1095, value=730, step=30)
        has_trend = st.sidebar.checkbox("Include Trend", value=True)
        has_seasonality = st.sidebar.checkbox("Include Seasonality", value=True)
        has_noise = st.sidebar.checkbox("Include Noise", value=True)
    
    if data_source == "Generate Synthetic Data" or use_custom_data:
        # Forecasting Parameters
        st.sidebar.subheader("Forecasting Parameters")
        forecast_periods = st.sidebar.slider("Forecast Horizon (days)", min_value=7, max_value=365, value=90, step=7)
        
        # ARIMA Parameters
        st.sidebar.subheader("ARIMA Configuration")
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            p = st.number_input("p", min_value=0, max_value=5, value=1)
        with col2:
            d = st.number_input("d", min_value=0, max_value=3, value=1)
        with col3:
            q = st.number_input("q", min_value=0, max_value=5, value=1)
        
        test_size = st.sidebar.slider("Test Set Size (days)", min_value=7, max_value=90, value=30, step=7)
        
        # Run button
        if st.sidebar.button("🚀 Run Full Analysis", use_container_width=True):
            st.session_state.run_analysis = True
        
        # Initialize session state
        if 'run_analysis' not in st.session_state:
            st.session_state.run_analysis = False
        
        if not st.session_state.run_analysis:
            st.info("""
            👋 **Welcome to the Advanced Demand Forecasting Dashboard!**
            
            This intelligent system helps you:
            
            1. **📊 Analyze** historical demand patterns with your own data
            2. **🔄 Decompose** trends and seasonality patterns
            3. **🔮 Forecast** future demand using ARIMA and Prophet
            4. **⚖️ Compare** model performance with accuracy metrics
            5. **💡 Get AI-Powered Recommendations** for:
               - Inventory optimization
               - Revenue strategy
               - Risk management
               - Demand trend insights
            
            **Getting Started:**
            - Select "Upload Custom Dataset" in the sidebar to use your own data
            - Or generate synthetic data for demo purposes
            - Click "🚀 Run Full Analysis" to see results and recommendations
            
            Let's optimize your demand planning! 📊
            """)
        
        if st.session_state.run_analysis:
            with st.spinner("⏳ Running analysis... This may take a few moments..."):
                # Step 1: Generate or Load Data
                st.header("📈 Step 1: Data Loading & Validation")
                
                if not use_custom_data:
                    data_generator = DemandData()
                    df = data_generator.generate_sample_data(
                        days=days,
                        trend=has_trend,
                        seasonality=has_seasonality,
                        noise=has_noise
                    )
                    st.success(f"✓ Generated {len(df)} days of synthetic demand data")
                else:
                    st.success(f"✓ Loaded {len(df)} records from custom dataset")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Days", len(df))
                with col2:
                    st.metric("Date Range", f"{(df['date'].max() - df['date'].min()).days} days")
                with col3:
                    st.metric("Data Points", f"{len(df):,}")
                with col4:
                    st.metric("Data Quality", "100%")
                
                # Step 2: Historical Analysis
                st.header("📊 Step 2: Historical Data Analysis")
                analyzer = HistoricalAnalysis(df)
                stats = analyzer.get_statistics()
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.metric("Mean Demand", f"{stats['Mean Demand']:.2f}")
                with col2:
                    st.metric("Std Dev", f"{stats['Std Dev']:.2f}")
                with col3:
                    st.metric("Min Demand", f"{stats['Min Demand']:.2f}")
                with col4:
                    st.metric("Max Demand", f"{stats['Max Demand']:.2f}")
                with col5:
                    cv = (stats['Std Dev']/stats['Mean Demand']*100) if stats['Mean Demand'] > 0 else 0
                    st.metric("CV", f"{cv:.1f}%")
                with col6:
                    st.metric("Total Demand", f"{stats['Mean Demand']*len(df):.0f}")
                
                # Plot historical demand
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(df['date'], df['demand'], linewidth=2, color='#1f77b4')
                ax.fill_between(df['date'], df['demand'], alpha=0.3, color='#1f77b4')
                ax.set_title('Historical Demand Data', fontsize=14, fontweight='bold')
                ax.set_xlabel('Date')
                ax.set_ylabel('Demand')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
                # Monthly statistics
                st.subheader("Monthly Statistics")
                monthly_stats = analyzer.get_monthly_stats()
                st.dataframe(monthly_stats, width='stretch', hide_index=True)
                
                # Step 3: Seasonality and Trend Decomposition
                st.header("🔄 Step 3: Seasonality & Trend Decomposition")
                seasonality = SeasonalityAnalysis(df)
                decomposition = seasonality.decompose()
                
                if decomposition is not None:
                    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
                    
                    # Original
                    decomposition.observed.plot(ax=axes[0], color='blue', linewidth=1.5)
                    axes[0].set_title('Observed Demand', fontsize=12, fontweight='bold')
                    axes[0].set_ylabel('Demand')
                    axes[0].grid(True, alpha=0.3)
                    
                    # Trend
                    decomposition.trend.plot(ax=axes[1], color='green', linewidth=1.5)
                    axes[1].set_title('Trend Component', fontsize=12, fontweight='bold')
                    axes[1].set_ylabel('Trend')
                    axes[1].grid(True, alpha=0.3)
                    
                    # Seasonality
                    decomposition.seasonal.plot(ax=axes[2], color='orange', linewidth=1.5)
                    axes[2].set_title('Seasonal Component', fontsize=12, fontweight='bold')
                    axes[2].set_ylabel('Seasonal')
                    axes[2].grid(True, alpha=0.3)
                    
                    # Residual
                    decomposition.resid.plot(ax=axes[3], color='red', linewidth=1.5)
                    axes[3].set_title('Residual Component', fontsize=12, fontweight='bold')
                    axes[3].set_ylabel('Residual')
                    axes[3].set_xlabel('Date')
                    axes[3].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    st.info(f"✓ Decomposition Period: {seasonality.period} days")
                else:
                    st.warning(f"⚠️ Insufficient Data for Decomposition")
                    st.info(f"""
                    Your dataset has only {len(df)} observations.
                    
                    Decomposition requires:
                    - **Yearly**: ≥730 observations (2 complete years)
                    - **Monthly**: ≥60 observations (2 complete months)
                    - **Weekly**: ≥14 observations (2 complete weeks)
                    
                    **Recommendation**: Upload more historical data (at least 2+ years recommended)
                    
                    The system will continue with forecasting, but trend/seasonal insights will be limited.
                    """)
                
                # Step 4: ARIMA Forecasting
                st.header("🔮 Step 4: ARIMA Forecasting")
                arima = ARIMAForecaster(df, p=int(p), d=int(d), q=int(q))
                
                with st.spinner("Training ARIMA model..."):
                    arima.fit()
                    arima_forecast = arima.predict(periods=forecast_periods)
                
                st.success(f"✓ ARIMA({p},{d},{q}) model trained successfully")
                
                # Plot ARIMA forecast
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(df['date'], df['demand'], label='Historical Demand', linewidth=2)
                ax.plot(arima_forecast['date'], arima_forecast['forecast'], 
                       label='ARIMA Forecast', linewidth=2, linestyle='--', color='orange')
                ax.fill_between(arima_forecast['date'], arima_forecast['lower_ci'], 
                               arima_forecast['upper_ci'], alpha=0.2, color='orange', label='95% CI')
                ax.set_title('ARIMA Forecast with Confidence Intervals', fontsize=14, fontweight='bold')
                ax.set_xlabel('Date')
                ax.set_ylabel('Demand')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
                # Display forecast table
                st.subheader("Forecast Values")
                forecast_display = arima_forecast.copy()
                forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
                st.dataframe(forecast_display, use_container_width=True, hide_index=True)
                
                # Step 5: Prophet Forecasting
                st.header("🔮 Step 5: Prophet Forecasting")
                prophet = ProphetForecaster(df)
                
                with st.spinner("Training Prophet model..."):
                    prophet.fit()
                    prophet_forecast = prophet.predict(periods=forecast_periods)
                
                st.success("✓ Prophet model trained successfully")
                
                # Plot Prophet forecast
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(df['date'], df['demand'], label='Historical Demand', linewidth=2)
                ax.plot(prophet_forecast['date'], prophet_forecast['forecast'], 
                       label='Prophet Forecast', linewidth=2, linestyle='--', color='purple')
                ax.fill_between(prophet_forecast['date'], prophet_forecast['lower_ci'], 
                               prophet_forecast['upper_ci'], alpha=0.2, color='purple', label='95% CI')
                ax.set_title('Prophet Forecast with Confidence Intervals', fontsize=14, fontweight='bold')
                ax.set_xlabel('Date')
                ax.set_ylabel('Demand')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
                # Step 6: Model Comparison
                st.header("⚖️ Step 6: Model Performance Comparison")
                
                # Split data for testing
                train_df = df[:-test_size].copy()
                test_df = df[-test_size:].copy()
                
                # Re-train models on training data
                arima_train = ARIMAForecaster(train_df, p=int(p), d=int(d), q=int(q))
                arima_train.fit()
                arima_test_forecast = arima_train.predict(periods=test_size)
                
                prophet_train = ProphetForecaster(train_df)
                prophet_train.fit()
                prophet_test_forecast = prophet_train.predict(periods=test_size)
                
                # Calculate metrics
                arima_metrics = ForecastComparison.calculate_metrics(
                    test_df['demand'].values,
                    arima_test_forecast['forecast'].values
                )
                
                prophet_metrics = ForecastComparison.calculate_metrics(
                    test_df['demand'].values,
                    prophet_test_forecast['forecast'].values
                )
                
                # Display metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("ARIMA Model Metrics")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("MAE", f"{arima_metrics['MAE']:.2f}")
                    with metric_col2:
                        st.metric("RMSE", f"{arima_metrics['RMSE']:.2f}")
                    with metric_col3:
                        st.metric("MAPE", f"{arima_metrics['MAPE']:.2f}%")
                
                with col2:
                    st.subheader("Prophet Model Metrics")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("MAE", f"{prophet_metrics['MAE']:.2f}")
                    with metric_col2:
                        st.metric("RMSE", f"{prophet_metrics['RMSE']:.2f}")
                    with metric_col3:
                        st.metric("MAPE", f"{prophet_metrics['MAPE']:.2f}%")
                
                # Determine best model
                best_model = "ARIMA" if arima_metrics['MAPE'] < prophet_metrics['MAPE'] else "Prophet"
                st.info(f"🏆 Best Model: **{best_model}** (Lower MAPE is better)")
                
                # Comparison visualization
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(test_df['date'], test_df['demand'], label='Actual Demand', linewidth=2.5, marker='o', markersize=4)
                ax.plot(arima_test_forecast['date'], arima_test_forecast['forecast'], 
                       label='ARIMA Forecast', linewidth=2, linestyle='--')
                ax.plot(prophet_test_forecast['date'], prophet_test_forecast['forecast'], 
                       label='Prophet Forecast', linewidth=2, linestyle='--')
                ax.set_title('Model Comparison on Test Set', fontsize=14, fontweight='bold')
                ax.set_xlabel('Date')
                ax.set_ylabel('Demand')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
                # Step 7: Combined Comparison
                st.header("📊 Step 7: Full Forecast Comparison")
                
                fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(df['date'], df['demand'], label='Historical Demand', linewidth=2, color='#1f77b4')
                ax.plot(arima_forecast['date'], arima_forecast['forecast'], 
                       label='ARIMA Forecast', linewidth=2, linestyle='--', color='orange')
                ax.plot(prophet_forecast['date'], prophet_forecast['forecast'], 
                       label='Prophet Forecast', linewidth=2, linestyle='--', color='purple')
                ax.axvline(x=df['date'].max(), color='red', linestyle=':', alpha=0.7, label='Forecast Start')
                ax.set_title('Complete Forecast Comparison: Historical + Future Predictions', fontsize=14, fontweight='bold')
                ax.set_xlabel('Date')
                ax.set_ylabel('Demand')
                ax.legend(loc='best', fontsize=11)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
                # Step 8: INTELLIGENT RECOMMENDATIONS
                st.header("💡 Step 8: AI-Powered Recommendations")
                
                recommendations = ForecastComparison.generate_recommendations(
                    df, arima_forecast, prophet_forecast, arima_metrics, prophet_metrics
                )
                
                # Display recommendations in organized tabs
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "Demand Trend", 
                    "Model Selection", 
                    "Inventory",
                    "Revenue", 
                    "Risk", 
                    "Confidence"
                ])
                
                with tab1:
                    st.markdown(recommendations['demand_trend'])
                
                with tab2:
                    st.markdown(recommendations['model_recommendation'])
                
                with tab3:
                    st.markdown(recommendations['inventory_strategy'])
                
                with tab4:
                    st.markdown(recommendations['revenue_optimization'])
                
                with tab5:
                    st.markdown(recommendations['risk_assessment'])
                
                with tab6:
                    st.markdown(recommendations['forecast_confidence'])
                
                # Summary Statistics
                st.header("📋 Summary")
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.metric("Historical Data Points", f"{len(df):,}")
                with summary_col2:
                    st.metric("Forecast Horizon", f"{forecast_periods} days")
                with summary_col3:
                    st.metric("Models Trained", "2 (ARIMA + Prophet)")
                
                st.success("""
                ✓ Complete demand forecasting analysis with intelligent recommendations completed!
                
                **Action Items:**
                - 📋 Review recommendations across all categories
                - 📊 Use forecast range for contingency planning
                - 💾 Export forecasts for integration with ERP/inventory systems
                - 🔄 Monitor actual vs predicted for model refinement
                - 📈 Re-run analysis monthly with updated data
                """)
    else:
        st.info("""
        👋 **Welcome to the Advanced Demand Forecasting Dashboard!**
        
        This intelligent system helps you:
        
        1. **📊 Analyze** historical demand patterns with your own data
        2. **🔄 Decompose** trends and seasonality patterns
        3. **🔮 Forecast** future demand using ARIMA and Prophet
        4. **⚖️ Compare** model performance with accuracy metrics
        5. **💡 Get AI-Powered Recommendations** for:
           - Inventory optimization
           - Revenue strategy
           - Risk management
           - Demand trend insights
        
        **Getting Started:**
        - Select "Upload Custom Dataset" in the sidebar to use your own data
        - Or generate synthetic data for demo purposes
        - Click "🚀 Run Full Analysis" to see results and recommendations
        
        Let's optimize your demand planning! 📊
        """)


if __name__ == "__main__":
    main()
