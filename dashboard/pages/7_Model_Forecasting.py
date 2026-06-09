import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from utils.helper import (load_sameday_model, load_three_days_forecast_model, 
                        load_seven_days_forecast_model, load_one_day_forecast_model,
                        run_monthly_forecast, run_sameday_prediction, 
                        run_one_day_forecast_prediction, run_three_day_forecast_prediction, 
                        run_seven_day_forecast_prediction)
from components.filters import render_date_controls, render_month_year_controls
from utils.data_loader import load_model_data, get_features_for_date


st.set_page_config(
    page_title="CNN Chlorophyll Predictions",
    page_icon="🤖",
    layout="wide"
)

st.title("CNN Chlorophyll Predictions")
st.markdown("Compare actual vs predicted chlorophyll-a concentrations from trained CNN models")

try:
    model_data = load_model_data()
    test_idx_sameday = model_data['sameday_test_idx']
    test_idx_forecast = model_data['forecast_test_idx']
    norm_stats = model_data['norm_stats']
    chl_ds = model_data['model_chl_ds']
    
except Exception as e:
    st.error(f"Error loading model data: {e}")
    st.stop()
    

times = chl_ds.time.values
lats = chl_ds.latitude.values
lons = chl_ds.longitude.values
all_dates = pd.to_datetime(times)

test_dates_sameday = pd.to_datetime(times[test_idx_sameday])
test_dates_forecast = pd.to_datetime(times[test_idx_forecast])

tab1, tab2, tab3, tab4 = st.tabs(["Sameday Predictions", "1 day Forecast Predictions", "3 days Forecast Predictions", "7 days Forecast Predictions"])

with tab1:
    st.markdown("### Settings")

    selected_date = render_date_controls(all_dates, test_dates_sameday, "sameday")
    
    run_button = st.button("Run Prediction", key="sameday_run_button")
    if run_button:
        model_sameday, device = load_sameday_model()
        if model_sameday is None:
            st.error("Error loading same-day model.")
            st.stop()
        run_sameday_prediction(selected_date, all_dates, norm_stats, model_sameday, device, get_features_for_date)
            
    else:
        st.info("Select a date and click 'Run Prediction' to view results.")


with tab2:
    st.markdown("### Settings")
        
    selected_date = render_date_controls(all_dates, test_dates_forecast, "forecast1day")
    
    run_button = st.button("Run Prediction", key="1_day_forecast_run_button")
    
    if run_button:
        model_forecast, device = load_one_day_forecast_model()
        if model_forecast is None:
            st.error("Error loading one-day forecast model.")
            st.stop()
            
        run_one_day_forecast_prediction(selected_date, all_dates, norm_stats, model_forecast, device, get_features_for_date)
            
    else:
        st.info("Select a date and click 'Run Prediction' to view results.")
        
    with st.expander("Monthly Chlorophyll Forecast with 1 day prediction windows"):
        st.markdown("### Monthly 1-day Chlorophyll Forecast")
        
        selected_ts = render_month_year_controls(all_dates, "monthly_forecast_1day")
        
        if st.button("Run Monthly Forecast", key="monthly_forecast_run_button_1day"):
            model_forecast, device = load_one_day_forecast_model()
            if model_forecast is None:
                st.error("Error loading one-day forecast model.")
                st.stop()
                
            run_monthly_forecast(
                selected_year=selected_ts.year,
                selected_month=selected_ts.month,
                all_dates=all_dates,
                norm_stats=norm_stats,
                model=model_forecast,
                device=device,
                get_features_for_date=get_features_for_date,
                window_size=1,
                min_history=1,
                past_offsets=[],
                chart_key="monthly_forecast_1day_chart"
            )
        else:
            st.info("Select a year and month, then click 'Run Monthly Forecast' to view results.")




with tab3:
    st.markdown("### Settings")
    
    selected_date = render_date_controls(all_dates, test_dates_forecast, "forecast3day")
    
    if st.button("Run Prediction", key="3_day_forecast_run_button"):
        model_taunet, device = load_three_days_forecast_model()
        if model_taunet is None:
            st.error("Error loading three-day forecast model.")
            st.stop()
            
        run_three_day_forecast_prediction(selected_date, all_dates, norm_stats, model_taunet, device, get_features_for_date)
        
    else:
        st.info("Select a date and click 'Run Prediction' to view results.")
        
    with st.expander("Monthly Chlorophyll Forecast with 3 day prediction windows"):
        st.markdown("### Monthly Chlorophyll Forecast")
        
        selected_ts = render_month_year_controls(all_dates, "monthly_forecast_3day")
        
        if st.button("Run Monthly Forecast", key="monthly_forecast_run_button"):
            model_taunet, device = load_three_days_forecast_model()
            if model_taunet is None:
                st.error("Error loading three-day forecast model.")
                st.stop()
            month_mask = (all_dates.year == selected_ts.year) & (all_dates.month == selected_ts.month)
            month_dates = all_dates[month_mask]
            
            if len(month_dates) == 0:
                st.error("No data available for the selected month and year.")
                st.stop()
                
            run_monthly_forecast(
                selected_year=selected_ts.year,
                selected_month=selected_ts.month,
                all_dates=all_dates,
                norm_stats=norm_stats,
                model=model_taunet,
                device=device,
                get_features_for_date=get_features_for_date,
                window_size=3,
                min_history=3,
                past_offsets=[3, 2, 1],
                chart_key="monthly_forecast_3day_chart"
            )
            
        else:
            st.info("Select a year and month, then click 'Run Monthly Forecast' to view results.")
            
            
with tab4:
    st.markdown("### Settings")
    selected_date = render_date_controls(all_dates, test_dates_forecast, "forecast7day")
    
    if st.button("Run Prediction", key="7_day_forecast_run_button"):
        model_seven_days, device = load_seven_days_forecast_model()
        if model_seven_days is None:
            st.error("Error loading seven-day forecast model.")
            st.stop()
            
        run_seven_day_forecast_prediction(selected_date, all_dates, norm_stats, model_seven_days, device, get_features_for_date)

    else:
        st.info("Select a date and click 'Run Prediction' to view results.")
        
    with st.expander("Monthly Chlorophyll Forecast with 7 day prediction windows"):
        st.markdown("### Monthly Chlorophyll Forecast")
        
        selected_ts = render_month_year_controls(all_dates, "monthly_forecast_7day")
        
        if st.button("Run Monthly Forecast", key="monthly_forecast_run_button_7day"):
            model_seven_days, device = load_seven_days_forecast_model()
            if model_seven_days is None:
                st.error("Error loading seven-day forecast model.")
                st.stop()
            
            run_monthly_forecast(
                selected_year=selected_ts.year,
                selected_month=selected_ts.month,
                all_dates=all_dates,
                norm_stats=norm_stats,
                model=model_seven_days,
                device=device,
                get_features_for_date=get_features_for_date,
                window_size=7,
                min_history=14,
                past_offsets=list(range(14, 0, -1)),
                chart_key="monthly_forecast_7day_chart"
            )
            
        else:
            st.info("Select a year and month, then click 'Run Monthly Forecast' to view results.")
    