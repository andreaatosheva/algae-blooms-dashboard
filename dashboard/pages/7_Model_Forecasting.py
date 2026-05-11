import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.helper import make_map, run_prediction, denormalise_chl, load_models
from utils.data_loader import load_variable_data
from plotly.subplots import make_subplots
import gc
import torch


from config import VARIABLE_INFO, SEASONS, SEASON_COLORS, BALTIC_REGIONS, MAP_BOUNDS
from utils.data_loader import load_dataset, get_variable_data, load_model_data, get_features_for_date
from components.filters import create_smooth_interpolated_map, create_scatter_bubble_map


st.set_page_config(
    page_title="CNN Chlorophyll Predictions",
    page_icon="🤖",
    layout="wide"
)

st.title("CNN Chlorophyll Predictions")
st.markdown("Compare actual vs predicted chlorophyll-a concentrations from trained CNN models")

try:
    model_sameday, model_forecast, model_taunet, device = load_models()
    model_data = load_model_data()
    test_idx_sameday  = model_data['sameday_test_idx']
    test_idx_forecast = model_data['forecast_test_idx']
    norm_stats        = model_data['norm_stats']
    chl_ds            = model_data['model_chl_ds']
    
except Exception as e:
    st.error(f"Error loading model data: {e}")
    st.stop()
    

times = chl_ds.time.values
lats = chl_ds.latitude.values
lons = chl_ds.longitude.values
all_dates = pd.to_datetime(times)

test_dates_sameday = pd.to_datetime(times[test_idx_sameday])
test_dates_forecast = pd.to_datetime(times[test_idx_forecast])

tab1, tab2, tab3, tab4 = st.tabs(["Sameday Predictions", "1 day Forecast Predictions", "3 days Forecast Predictions", "Monthly CHL forecast"])

with tab1:
    st.markdown("### Settings")
        
    date_filter = st.radio(
        "**Date Range**",
        options=["All dates (2014-2024)", "Test set only (unseen by model)"],
        help="Test set dates were never seen during training - metrics on these dates are the most reliable indicator of real model performance",
        key = "sameday_date_filter"
    )
    
    if date_filter == "All dates (2014-2024)":
        selected_dates = all_dates
    else:
        selected_dates = test_dates_sameday
        
    selected_date = st.selectbox(
        "Select a date to view predictions",
        options=selected_dates,
        format_func=lambda x: x.strftime("%Y-%m-%d"),
        index=0,
        key="sameday_date_selectbox"
    )
    
    run_button = st.button("Run Prediction", key="sameday_run_button")
    if run_button:
        with st.spinner(f"Loading data for {selected_date.strftime('%Y-%m-%d')}..."):
            features_year, day_idx_year, global_idx = get_features_for_date(selected_date, all_dates)
            
        if features_year is None:
            st.error(f"Features for {selected_date.strftime('%Y-%m-%d')} could not be loaded.")
            st.stop()
        else:
            actual_norm = features_year[day_idx_year, 0, :, :]
            actual_real = denormalise_chl(actual_norm, norm_stats)
            land_mask = np.isnan(actual_norm)
            actual_real[land_mask] = np.nan
            
            with st.spinner("Running model prediction..."):
                x_input = features_year[day_idx_year, 1:, :, :]
                pred_norm = run_prediction(model_sameday, x_input, device)
            
            pred_real = denormalise_chl(pred_norm, norm_stats)
            pred_real[land_mask] = np.nan
            
            
            col1, col2, col3 = st.columns([2, 2, 2])

            with col1:
                fig = make_map(actual_real, f"Actual CHL — {selected_date.strftime('%d %b %Y')}")
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                fig = make_map(pred_real, f"Predicted CHL — {selected_date.strftime('%d %b %Y')}")
                st.plotly_chart(fig, use_container_width=True)
            with col3:
                diff = pred_real - actual_real
                fig = make_map(diff, "Difference (Pred - Actual)",
                                    colorscale='RdBu_r', zmin=-5, zmax=5, zmid=0)
                st.plotly_chart(fig, use_container_width=True)
            
            
    else:
        st.info("Select a date and click 'Run Prediction' to view results.")
        
with tab2:
    st.markdown("### Settings")
        
    date_filter = st.radio(
        "**Date Range**",
        options=["All dates (2014-2024)", "Test set only (unseen by model)"],
        help="Test set dates were never seen during training - metrics on these dates are the most reliable indicator of real model performance",
        key = "forecast_date_filter"
    )
    
    if date_filter == "All dates (2014-2024)":
        selected_dates = all_dates
    else:
        selected_dates = test_dates_forecast
        
    selected_date = st.selectbox(
        "Select a date to view predictions",
        options=selected_dates,
        format_func=lambda x: x.strftime("%Y-%m-%d"),
        index=0,
        key="forecast_date_selectbox"
    )
    
    run_button = st.button("Run Prediction", key="1_day_forecast_run_button")
    
    if run_button:
        global_idx = np.where(all_dates == pd.to_datetime(selected_date))[0][0]
        if global_idx >= len(all_dates) - 1:
                st.error("No data available for the next day after the selected date.")
                st.stop()
        next_date = all_dates[global_idx + 1]
        with st.spinner(f"Loading data for {selected_date.strftime('%Y-%m-%d')}..."):
            features_next, day_idx_next, _ = get_features_for_date(next_date, all_dates)
            
        if features_next is None:
            st.error(f"Features for {next_date.strftime('%Y-%m-%d')} could not be loaded.")
            st.stop()
        else:
            
            actual_next_norm = features_next[day_idx_next, 0, :, :]
            actual_next_real = denormalise_chl(actual_next_norm, norm_stats)
            land_mask_next = np.isnan(actual_next_norm)
            actual_next_real[land_mask_next] = np.nan
            
            with st.spinner("Running model prediction..."):
                x_input = features_next[day_idx_next, :, :, :]
                pred_norm = run_prediction(model_forecast, x_input, device)
            
            pred_real = denormalise_chl(pred_norm, norm_stats)
            pred_real[land_mask_next] = np.nan
            
            
            col1, col2, col3 = st.columns([2, 2, 2])

            with col1:
                fig = make_map(actual_next_real, f"Actual CHL — {selected_date.strftime('%d %b %Y')}")
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                fig = make_map(pred_real, f"Predicted CHL — {selected_date.strftime('%d %b %Y')}")
                st.plotly_chart(fig, use_container_width=True)
            with col3:
                diff = pred_real - actual_next_real
                fig = make_map(diff, "Difference (Pred - Actual)",
                                    colorscale='RdBu_r', zmin=-5, zmax=5, zmid=0)
                st.plotly_chart(fig, use_container_width=True)
            
            
    else:
        st.info("Select a date and click 'Run Prediction' to view results.")


with tab3:
    st.markdown("### Settings")
    
    date_filter = st.radio(
        "**Date Range**",
        options=["All dates (2014-2024)", "Test set only (unseen by model)"],
        help="Test set dates were never seen during training - metrics on these dates are the most reliable indicator of real model performance",
        key = "forecast_3day_date_filter"
    )
    if date_filter == "All dates (2014-2024)":
        selected_dates = all_dates
    else:
        selected_dates = test_dates_forecast
        
    selected_date = st.selectbox(
        "Select a date to view predictions",
        options=selected_dates,
        format_func=lambda x: x.strftime("%Y-%m-%d"),
        index=0,
        key="forecast_3day_date_selectbox"
    )
    
    if st.button("Run Prediction", key="3_day_forecast_run_button"):
        global_idx = np.where(all_dates == pd.to_datetime(selected_date))[0][0]
        
        if global_idx < 3:
            st.error("Not enough previous data available to make a 3-day forecast for the selected date.")
            st.stop()
        if global_idx > len(all_dates) - 4:
            st.error("No data available for the 3 days following the selected date.")
            st.stop()
            
        with st.spinner(f"Loading data for {selected_date.strftime('%Y-%m-%d')}..."):
            past_features = []
            for offset in [3, 2, 1]:
                past_date = all_dates[global_idx - offset]
                features_past, day_idx_past, _ = get_features_for_date(past_date, all_dates)
                if features_past is None:
                    st.error(f"Features for {past_date.strftime('%Y-%m-%d')} could not be loaded.")
                    st.stop()
                past_features.append(features_past[day_idx_past])
            x_input = np.concatenate(past_features, axis=0) 
            
        
        actual_days = []
        land_masks = []
        
        for offset in [1, 2, 3]:
            future_date = all_dates[global_idx + offset]
            features_future, day_idx_future, _ = get_features_for_date(future_date, all_dates)
            if features_future is None:
                st.error(f"Features for {future_date.strftime('%Y-%m-%d')} could not be loaded.")
                st.stop()
            
            actual_norm = features_future[day_idx_future, 0, :, :]
            actual_real = denormalise_chl(actual_norm, norm_stats)
            land_mask = np.isnan(actual_norm)
            actual_real[land_mask] = np.nan
            actual_days.append((actual_real))
            land_masks.append(land_mask)
            
        with st.spinner("Running model prediction..."):
            x = np.nan_to_num(x_input, nan=0.0)
            x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_norm_all = model_taunet(x_tensor).cpu().numpy()[0]
                
        pred_days = []
        for i in range(3):
            pred_real = denormalise_chl(pred_norm_all[i], norm_stats)
            pred_real[land_masks[i]] = np.nan
            pred_days.append(pred_real)
            
        
        for day in range(3):
            future_label = all_dates[global_idx + day + 1].strftime('%d %b %Y')
            st.markdown(f"#### Day t+{day+1} — {future_label}")
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                st.plotly_chart(make_map(actual_days[day], f"Actual CHL (t+{day+1})"),
                                use_container_width=True)
            with col2:
                st.plotly_chart(make_map(pred_days[day], f"Predicted CHL (t+{day+1})"),
                                use_container_width=True)
            with col3:
                st.plotly_chart(make_map(pred_days[day] - actual_days[day],
                                         f"Difference (t+{day+1})",
                                         colorscale='RdBu_r', zmin=-5, zmax=5, zmid=0),
                                use_container_width=True)
    else:
        st.info("Select a date and click 'Run Prediction' to view results.")
        
with tab4:
    st.markdown("### Monthly Chlorophyll Forecast")
    
    col_y, col_m = st.columns(2)
    with col_y:
        selected_year = st.selectbox(
            "Select Year",
            options=sorted(all_dates.year.unique()),
            index=0,
            key="monthly_forecast_year"
        )
    with col_m:
        selected_month = st.selectbox(
            "Select Month",
            options=list(range(1, 13)),
            format_func=lambda x: pd.Timestamp(2000, x, 1).strftime("%B"),
            index=0,
            key="monthly_forecast_month"
        )
    
    if st.button("Run Monthly Forecast", key="monthly_forecast_run_button"):
        month_mask = (all_dates.year == selected_year) & (all_dates.month == selected_month)
        month_dates = all_dates[month_mask]
        
        if len(month_dates) == 0:
            st.error("No data available for the selected month and year.")
            st.stop()
            
        windows = []
        i = 0
        while i + 2 < len(month_dates):
            window_dates = month_dates[i:i+3]
            windows.append(window_dates)
            i += 3
        
        if len(windows) == 0:
            st.error("Not enough data to create 3-day windows for the selected month.")
            st.stop()
            
        st.markdown(f"**{len(windows)} prediction windows** for "
                    f"{pd.Timestamp(selected_year, selected_month, 1).strftime('%B %Y')}")
        
        progress = st.progress(0, text="Running predictions...")
        
        all_window_preds = []
        all_window_actuals = []
        all_window_labels = []
        failed_window = []
        
        for w_idx, window in enumerate(windows):
            try:
                first_day_idx = np.where(all_dates == window[0])[0][0]
                
                if first_day_idx < 3:
                    failed_window.append((w_idx, "Not enough previous data for 3-day input"))
                    continue
                    
                past_features = []
                for offset in [3, 2, 1]:
                    past_date = all_dates[first_day_idx - offset]
                    feats, day_idx, _ = get_features_for_date(past_date, all_dates)
                    if feats is None:
                        raise ValueError(f"Features for {past_date} could not be loaded.")
                    past_features.append(feats[day_idx])
                x_input = np.concatenate(past_features, axis=0)
                
                x = np.nan_to_num(x_input, nan=0.0)
                x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_norm_all = model_taunet(x_tensor).cpu().numpy()[0]
                    
                window_preds = []
                window_actuals = []
                window_labels = []
                
                for d, future_date in enumerate(window):
                    features_future, day_idx_future, _ = get_features_for_date(future_date, all_dates)
                    if features_future is None:
                        raise ValueError(f"Features for {future_date} could not be loaded.")
                    
                    actual_norm = features_future[day_idx_future, 0, :, :]
                    actual_land_mask = np.isnan(actual_norm)
                    
                    actual_real = denormalise_chl(actual_norm, norm_stats)
                    actual_real[actual_land_mask] = np.nan
                    
                    pred_real = denormalise_chl(pred_norm_all[d], norm_stats)
                    pred_real[actual_land_mask] = np.nan
                    
                    window_preds.append(pred_real)
                    window_actuals.append(actual_real)
                    window_labels.append(f"{future_date.strftime('%d %b %Y')}")
                    
                all_window_preds.append(window_preds)
                all_window_actuals.append(window_actuals)
                all_window_labels.append(window_labels)
            except Exception as e:
                failed_window.append((w_idx, str(e)))
                st.warning(f"Window {w_idx} failed: {e}")
            
            progress.progress((w_idx + 1) / len(windows), text=f"Running predictions... (Window {w_idx+1}/{len(windows)})")
            
        progress.empty()
        
        if not all_window_preds:
            st.error("All prediction windows failed. Please check the error messages above.")
            st.stop()
            
        st.markdown("---")
        
        for w_idx, (preds, actuals, labels) in enumerate(
            zip(all_window_preds, all_window_actuals, all_window_labels)
        ):
            with st.expander(f"**Window {w_idx+1}** - {labels[0]} -> {labels[-1]}", expanded=(w_idx==0)):
                
                for d in range(len(labels)):
                    st.markdown(f"##### {labels[d]}")
                    col1, col2, col3 = st.columns([2, 2, 2])
                    with col1:
                        st.plotly_chart(make_map(np.where(np.isnan(actuals[d]), None, actuals[d]), f"Actual CHL ({labels[d]})"),
                                        use_container_width=True, key=f"actual_{w_idx}_{d}")
                        
                    with col2:
                        st.plotly_chart(make_map(np.where(np.isnan(preds[d]), None, preds[d]), f"Predicted CHL ({labels[d]})"),
                                        use_container_width=True, key=f"pred_{w_idx}_{d}")
                        
                    with col3:
                        diff = preds[d] - actuals[d]
                        st.plotly_chart(make_map(np.where(np.isnan(diff), None, diff),
                                                 f"Difference (Pred - Actual) ({labels[d]})",
                                                 colorscale='RdBu_r', zmin=-5, zmax=5, zmid=0),
                                        use_container_width=True, key=f"diff_{w_idx}_{d}")
            
            if failed_window:
                st.warning(f"Could not load windows: {[w+1 for w in failed_window]}")