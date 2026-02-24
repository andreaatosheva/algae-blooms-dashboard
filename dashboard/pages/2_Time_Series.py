import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from utils.helper import show_memory_usage
import gc

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VARIABLE_INFO, SEASONS, SEASON_COLORS
from utils.data_loader import load_dataset, get_variable_data

st.set_page_config(
    page_title="Time Series Analysis",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Time Series Analysis")
st.markdown("Analyze temporal trends, seasonality, and patterns over time")

show_memory_usage()

@st.cache_data
def load_variable_data(variable_key):
    if variable_key in ["nitrate", "phosphate", "ammonia"]:
        dataset = load_dataset("nutrients")
    else:
        dataset = load_dataset(variable_key)
    if dataset is not None:
        return get_variable_data(dataset, variable_key)
    return None


col1, col2 = st.columns(2, gap="xxsmall", border=True)

with col1:
    available_vars = list(VARIABLE_INFO.keys())
    selected_var = st.selectbox(
        "Select Variable",
        options=available_vars,
        format_func=lambda x: VARIABLE_INFO[x]["name"],
        index=0
    )
    var_info = VARIABLE_INFO[selected_var]
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        options=["Trend Analysis", "Seasonal Analysis", "Anomaly Detection", "Year over Year Comparison"],
        index = 0
    )

    spatial_agg = st.selectbox(
        "Spatial Aggregation",
        options=["Mean", "Median", "Max", "Min"],
        index=0
    )
    
data = load_variable_data(selected_var)

if data is None:
    st.error("Failed to load data for the selected variable.")
    st.stop()
    
if spatial_agg == "Mean":
    ts_data = data.mean(dim=['latitude', 'longitude'])
elif spatial_agg == "Median":
    ts_data = data.median(dim=['latitude', 'longitude'])
elif spatial_agg == "Max":
    ts_data = data.max(dim=['latitude', 'longitude'])
else:
    ts_data = data.min(dim=['latitude', 'longitude'])


with col2:
    min_date = pd.Timestamp(data.time.min().values)
    max_date = pd.Timestamp(data.time.max().values)
    
    col_start, col_end = st.columns(2)
    
    with col_start:
        st.write("**Start Date**")
        start_month = st.selectbox("Month", range(1, 13), 
                                   index=min_date.month-1, 
                                   format_func=lambda x: pd.Timestamp(2000, x, 1).strftime('%B'),
                                   key='start_month')
        start_year = st.selectbox("Year", 
                                  range(min_date.year, max_date.year+1),
                                  key='start_year')
    
    with col_end:
        st.write("**End Date**")
        end_month = st.selectbox("Month", range(1, 13), 
                                 index=max_date.month-1,
                                 format_func=lambda x: pd.Timestamp(2000, x, 1).strftime('%B'),
                                 key='end_month')
        end_year = st.selectbox("Year", 
                                range(min_date.year, max_date.year+1),
                                index=max_date.year-min_date.year,
                                key='end_year')
    
    start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
    end_date = pd.Timestamp(year=end_year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
    
    apply_filter = st.checkbox("Apply Time Filter", value=False)
    
    if apply_filter:
        ts_data = ts_data.sel(time=slice(str(start_date), str(end_date)))
        
st.markdown("---")

if analysis_type == "Trend Analysis":
    st.markdown("## 📈 Trend Analysis")
    
    time_numeric = np.arange(len(ts_data))
    valid_mask = ~np.isnan(ts_data.values)
    
    if valid_mask.sum() > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            time_numeric[valid_mask],
            ts_data.values[valid_mask]
        )
        
        trend_line = intercept + slope * time_numeric
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            months_per_year = 12
            trend_per_year = slope * months_per_year
            st.metric(
                label="Trend per Year",
                value=f"{trend_per_year:.4f} {var_info['unit']}/year"
            )
        with col2:
            st.metric(
                label="R-squared",
                value=f"{r_value**2:.4f}"
            )
        
        with col3:
            st.metric(
                label="P-value",
                value=f"{p_value:.4e}"
            )
            
        with col4:
            significance = "Significant ✓" if p_value < 0.05 else "Not Significant"
            st.metric(
                label="Statistical Significance",
                value=significance
            )
            
            # Interpretation
            if p_value < 0.05:
                if slope > 0:
                    st.success(f"✅ **Significant increasing trend detected!** {var_info['name']} is increasing by approximately **{abs(trend_per_year):.3f} {var_info['unit']}** per year.")
                else:
                    st.info(f"📉 **Significant decreasing trend detected!** {var_info['name']} is decreasing by approximately **{abs(trend_per_year):.3f} {var_info['unit']}** per year.")

    
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(
        x=ts_data.time.values,
        y=ts_data.values,
        mode='lines+markers',
        name='Observed Data',
        line=dict(color=var_info['color'], width=2),
        marker=dict(size=4),
        opacity=0.7
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=ts_data.time.values,
        y=trend_line,
        mode='lines',
        name='Linear Trend',
        line=dict(color='red', width=3, dash='dash')
    ))
    
    residuals = ts_data.values[valid_mask] - trend_line[valid_mask]
    std_residual = np.std(residuals)
    
    fig_trend.add_trace(go.Scatter(
        x=ts_data.time.values,
        y=trend_line + 2*std_residual,
        mode='lines',
        name='95% Confidence',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=ts_data.time.values,
        y=trend_line - 2*std_residual,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(width=0),
        name='95% Confidence Interval',
        hoverinfo='skip'
    ))
    
    fig_trend.update_layout(
        title=f'{var_info["name"]} Long-term Trend',
        xaxis_title='Date',
        yaxis_title=f'{var_info["name"]} ({var_info["unit"]})',
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_trend, width='stretch')
    
    with st.expander("📊 View Residual Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Scatter(
                x=ts_data.time.values[valid_mask],
                y=residuals,
                mode='markers',
                marker=dict(color='blue', size=5),
                name='Residuals'
            ))
            
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            
            fig_residuals.update_layout(
                title='Residuals (Observed - Trend)',
                xaxis_title='Date',
                yaxis_title='Residual',
                height=300,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_residuals, width='stretch')
        
        with col2:
            fig_res_hist = go.Figure()
            fig_res_hist.add_trace(go.Histogram(
                x=residuals,
                nbinsx=30,
                marker_color='blue',
                opacity=0.7
            ))
            
            fig_res_hist.update_layout(
                title='Residual Distribution',
                xaxis_title='Residual',
                yaxis_title='Frequency',
                height=300,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_res_hist, width='stretch')
            
elif analysis_type == "Seasonal Analysis":
    st.markdown("## 🌦️ Seasonal Analysis")
    st.markdown("Visualize average seasonal patterns over the years.")
    
    monthly_clim = ts_data.groupby('time.month').mean(dim='time')
    
    seasonal = []
    for t in ts_data.time.values:
        month = pd.Timestamp(t).month
        seasonal.append(float(monthly_clim.sel(month=month)))
    seasonal = np.array(seasonal)
    
    ts_df = pd.DataFrame({
        'value': ts_data.values,
        'time': ts_data.time.values
    })
    ts_df = ts_df.set_index('time')
    trend = ts_df['value'].rolling(window=12, center=True).mean().values
    
    residual = ts_data.values - trend - seasonal
    
    fig_decomp = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Original', 'Trend (12-month avg)', 'Seasonal Component', 'Residual'),
        vertical_spacing=0.08
    )
    
    fig_decomp.add_trace(
        go.Scatter(x=ts_data.time.values, y=ts_data.values,
                   mode='lines', name='Original',
                   line=dict(color=var_info['color'], width=2)),
        row=1, col=1
    )
    
    fig_decomp.add_trace(
        go.Scatter(x=ts_data.time.values, y=trend,
                   mode='lines', name='Trend',
                   line=dict(color='red', width=3)),
        row=2, col=1
    )
    
    fig_decomp.add_trace(
        go.Scatter(x=ts_data.time.values, y=seasonal,
                   mode='lines', name='Seasonal',
                   line=dict(color='green', width=2)),
        row=3, col=1
    )
    
    fig_decomp.add_trace(
        go.Scatter(x=ts_data.time.values, y=residual,
                   mode='markers', name='Residual',
                   marker=dict(color='gray', size=3)),
        row=4, col=1
    )
    
    fig_decomp.update_xaxes(title_text="Date", row=4, col=1)
    fig_decomp.update_yaxes(title_text=var_info['unit'], row=1, col=1)
    fig_decomp.update_yaxes(title_text=var_info['unit'], row=2, col=1)
    fig_decomp.update_yaxes(title_text=var_info['unit'], row=3, col=1)
    fig_decomp.update_yaxes(title_text=var_info['unit'], row=4, col=1)
    
    fig_decomp.update_layout(
        height=900,
        template='plotly_white',
        showlegend=False,
        title_text=f'{var_info["name"]} Seasonal Decomposition'
    )
    
    st.plotly_chart(fig_decomp, width='stretch')
    
    st.markdown("### 📊 Seasonal Component Statistics")
    
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig_seasonal = go.Figure()
        
        # Color by season
        colors = []
        for i in range(12):
            if i+1 in [12, 1, 2]:
                colors.append(SEASON_COLORS['Winter (DJF)'])
            elif i+1 in [3, 4, 5]:
                colors.append(SEASON_COLORS['Spring (MAM)'])
            elif i+1 in [6, 7, 8]:
                colors.append(SEASON_COLORS['Summer (JJA)'])
            else:
                colors.append(SEASON_COLORS['Autumn (SON)'])
        
        fig_seasonal.add_trace(go.Bar(
            x=months,
            y=monthly_clim.values,
            marker_color=colors,
            marker_line_color='black',
            marker_line_width=1,
            opacity=0.8
        ))
        
        fig_seasonal.update_layout(
            title='Average Seasonal Pattern',
            xaxis_title='Month',
            yaxis_title=f'{var_info["name"]} ({var_info["unit"]})',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_seasonal, width='stretch')
    
    with col2:
        st.markdown("**Monthly Averages**")
        seasonal_df = pd.DataFrame({
            'Month': months,
            'Average': [f"{val:.2f}" for val in monthly_clim.values]
        })
        
        st.dataframe(seasonal_df, hide_index=True, width='stretch', height=300)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        seasonal_range = monthly_clim.max() - monthly_clim.min()
        st.metric("Peak Month", months[int(monthly_clim.argmax())])
        
    with col2:
        st.metric("Lowest Month", months[int(monthly_clim.argmin())])
        
    with col3:
        st.metric("Seasonal Amplitude", f"{seasonal_range.values:.2f} {var_info['unit']}")
        
elif analysis_type == "Anomaly Detection":
    st.markdown("## ⚠️ Anomaly Detection")
    st.markdown("Identify significant deviations from normal patterns.")
    
    mean_val = float(ts_data.mean())
    std_val = float(ts_data.std())
    
    threshold_std = st.slider(
        "Anomaly Threshold (Standard Deviations)",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.5,
        help="Values beyond this many standard deviations are considered anomalies"
    )
    
    anomaly_threshold_high = mean_val + threshold_std * std_val
    anomaly_threshold_low = mean_val - threshold_std * std_val
    
    high_anomalies = ts_data.values > anomaly_threshold_high
    low_anomalies = ts_data.values < anomaly_threshold_low
    
    n_high = np.sum(high_anomalies)
    n_low = np.sum(low_anomalies)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{mean_val:.2f} {var_info['unit']}")
    
    with col2:
        st.metric("Std Dev", f"{std_val:.2f} {var_info['unit']}")
    
    with col3:
        st.metric("High Anomalies", f"{n_high} months", 
                 delta=f">{anomaly_threshold_high:.2f}")
    
    with col4:
        st.metric("Low Anomalies", f"{n_low} months",
                 delta=f"<{anomaly_threshold_low:.2f}")
    
    fig_anomaly = go.Figure()
    
    normal_mask = ~(high_anomalies | low_anomalies)
    fig_anomaly.add_trace(go.Scatter(
        x=ts_data.time.values[normal_mask],
        y=ts_data.values[normal_mask],
        mode='markers',
        name='Normal',
        marker=dict(color='gray', size=6, opacity=0.5)
    ))
    
    if n_high > 0:
        fig_anomaly.add_trace(go.Scatter(
            x=ts_data.time.values[high_anomalies],
            y=ts_data.values[high_anomalies],
            mode='markers',
            name='High Anomaly',
            marker=dict(color='red', size=12, symbol='triangle-up',
                       line=dict(width=2, color='darkred'))
        ))
    
    if n_low > 0:
        fig_anomaly.add_trace(go.Scatter(
            x=ts_data.time.values[low_anomalies],
            y=ts_data.values[low_anomalies],
            mode='markers',
            name='Low Anomaly',
            marker=dict(color='blue', size=12, symbol='triangle-down',
                       line=dict(width=2, color='darkblue'))
        ))
    
    fig_anomaly.add_hline(y=mean_val, line_dash="solid", line_color="green",
                         annotation_text="Mean", annotation_position="right")
    fig_anomaly.add_hline(y=anomaly_threshold_high, line_dash="dash", line_color="red",
                         annotation_text=f"+{threshold_std}σ", annotation_position="right")
    fig_anomaly.add_hline(y=anomaly_threshold_low, line_dash="dash", line_color="blue",
                         annotation_text=f"-{threshold_std}σ", annotation_position="right")
    
    fig_anomaly.update_layout(
        title=f'{var_info["name"]} Anomaly Detection',
        xaxis_title='Date',
        yaxis_title=f'{var_info["name"]} ({var_info["unit"]})',
        height=500,
        template='plotly_white',
        hovermode='closest'
    )
    
    st.plotly_chart(fig_anomaly, width='stretch')
    
    if n_high > 0 or n_low > 0:
        st.markdown("### 📋 Anomaly Events")
        
        anomaly_data = []
        
        for i, (is_high, is_low) in enumerate(zip(high_anomalies, low_anomalies)):
            if is_high or is_low:
                anomaly_data.append({
                    'Date': str(ts_data.time.values[i])[:10],
                    'Value': f"{ts_data.values[i]:.2f}",
                    'Type': 'High ▲' if is_high else 'Low ▼',
                    'Deviation': f"{abs(ts_data.values[i] - mean_val) / std_val:.2f}σ"
                })
        
        df_anomalies = pd.DataFrame(anomaly_data)
        st.dataframe(df_anomalies, hide_index=True, width='stretch')
        
        
elif analysis_type == "Year over Year Comparison":
    st.markdown("## 🔄 Year over Year Comparison")
    
    years = sorted(np.unique(ts_data.time.dt.year.values))
    
    fig_yoy = go.Figure()
    
    for year in years:
        year_data = ts_data.sel(time=str(year))
        
        if len(year_data) > 0:
            months = year_data.time.dt.month.values
            
            fig_yoy.add_trace(go.Scatter(
                x=months,
                y=year_data.values,
                mode='lines+markers',
                name=str(year),
                line=dict(width=2.5),
                marker=dict(size=6),
                hovertemplate=f'<b>{year}</b><br>' +
                              'Month: %{x}<br>' +
                              f'{var_info["name"]}: %{{y:.2f}} {var_info["unit"]}<br>' +
                              '<extra></extra>'
            ))
    
    fig_yoy.update_layout(
        title=f'{var_info["name"]} Year-over-Year Comparison',
        xaxis=dict(
            title='Month',
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        ),
        yaxis_title=f'{var_info["name"]} ({var_info["unit"]})',
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_yoy, width='stretch')
    
    st.markdown("### 📊 Annual Statistics")
    
    annual_stats = []
    for year in years:
        year_data = ts_data.sel(time=str(year))
        if len(year_data) > 0:
            valid_year = year_data.values[~np.isnan(year_data.values)]
            if len(valid_year) > 0:
                annual_stats.append({
                    'Year': year,
                    'Mean': np.mean(valid_year),
                    'Max': np.max(valid_year),
                    'Min': np.min(valid_year),
                    'Std Dev': np.std(valid_year)
                })
    
    df_annual = pd.DataFrame(annual_stats)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_annual = go.Figure()
        
        fig_annual.add_trace(go.Bar(
            x=df_annual['Year'],
            y=df_annual['Mean'],
            marker_color=var_info['color'],
            marker_line_color='black',
            marker_line_width=1.5,
            opacity=0.8,
            error_y=dict(
                type='data',
                array=df_annual['Std Dev'],
                visible=True
            )
        ))
        
        overall_mean = df_annual['Mean'].mean()
        fig_annual.add_hline(
            y=overall_mean,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Overall Mean: {overall_mean:.2f}",
            annotation_position="right"
        )
        
        fig_annual.update_layout(
            title='Annual Mean Values',
            xaxis_title='Year',
            yaxis_title=f'{var_info["name"]} ({var_info["unit"]})',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_annual, width='stretch')
    
    with col2:
        st.markdown("**Annual Statistics**")
        
        df_display = df_annual.copy()
        df_display['Mean'] = df_display['Mean'].apply(lambda x: f"{x:.2f}")
        df_display['Max'] = df_display['Max'].apply(lambda x: f"{x:.2f}")
        df_display['Min'] = df_display['Min'].apply(lambda x: f"{x:.2f}")
        df_display['Std Dev'] = df_display['Std Dev'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(df_display, hide_index=True, width='stretch', height=400)

else:
    pass

st.markdown("---")
st.markdown("## 💡 Additional Insights")

tab1, tab2 = st.tabs(["📊 Summary Statistics", "📈 Growth Rates"])

with tab1:
    st.markdown("### Overall Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    valid_values = ts_data.values[~np.isnan(ts_data.values)]
    
    with col1:
        st.markdown("**Central Tendency**")
        st.metric("Mean", f"{np.mean(valid_values):.3f} {var_info['unit']}")
        st.metric("Median", f"{np.median(valid_values):.3f} {var_info['unit']}")
        st.metric("Mode Region", f"{stats.mode(np.round(valid_values, 1), keepdims=True)[0][0]:.1f} {var_info['unit']}")
    
    with col2:
        st.markdown("**Spread**")
        st.metric("Range", f"{np.max(valid_values) - np.min(valid_values):.3f} {var_info['unit']}")
        st.metric("Std Dev", f"{np.std(valid_values):.3f} {var_info['unit']}")
        st.metric("IQR", f"{np.percentile(valid_values, 75) - np.percentile(valid_values, 25):.3f} {var_info['unit']}")
    
    with col3:
        st.markdown("**Extremes**")
        st.metric("Maximum", f"{np.max(valid_values):.3f} {var_info['unit']}")
        st.metric("Minimum", f"{np.min(valid_values):.3f} {var_info['unit']}")
        st.metric("Coefficient of Variation", f"{(np.std(valid_values) / np.mean(valid_values) * 100):.1f}%")

with tab2:
    st.markdown("### Month-to-Month Growth Rates")
    
    changes = np.diff(ts_data.values)
    pct_changes = (changes / ts_data.values[:-1]) * 100
    
    valid_changes = changes[~np.isnan(changes)]
    valid_pct_changes = pct_changes[~np.isnan(pct_changes)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_changes = go.Figure()
        
        fig_changes.add_trace(go.Scatter(
            x=ts_data.time.values[1:],
            y=changes,
            mode='lines',
            line=dict(color='purple', width=2),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.2)'
        ))
        
        fig_changes.add_hline(y=0, line_dash="dash", line_color="black")
        
        fig_changes.update_layout(
            title='Month-to-Month Change',
            xaxis_title='Date',
            yaxis_title=f'Change ({var_info["unit"]})',
            height=350,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_changes, width='stretch')
        
        st.metric("Mean Change", f"{np.mean(valid_changes):+.3f} {var_info['unit']}/month")
        st.metric("Largest Increase", f"{np.max(valid_changes):.3f} {var_info['unit']}")
        st.metric("Largest Decrease", f"{np.min(valid_changes):.3f} {var_info['unit']}")
    
    with col2:
        fig_pct = go.Figure()
        
        fig_pct.add_trace(go.Histogram(
            x=valid_pct_changes,
            nbinsx=30,
            marker_color='orange',
            marker_line_color='black',
            marker_line_width=0.5,
            opacity=0.7
        ))
        
        fig_pct.update_layout(
            title='Distribution of % Changes',
            xaxis_title='% Change',
            yaxis_title='Frequency',
            height=350,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_pct, width='stretch')
        
        st.metric("Mean % Change", f"{np.mean(valid_pct_changes):+.2f}%/month")
        st.metric("Volatility (Std)", f"{np.std(valid_pct_changes):.2f}%")

# Clear memory
gc.collect()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
    💡 Tip: Try different analysis types to understand temporal patterns. 
    Trend analysis shows long-term changes, while seasonal decomposition reveals cyclical patterns.
    </small>
</div>
""", unsafe_allow_html=True)