import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from utils.helper import show_memory_usage
import gc


sys.path.insert(0, str(Path(__file__).parent.parent))
from config import VARIABLE_INFO, DATA_PATHS
from utils.data_loader import load_all_datasets, load_dataset, get_variable_data

st.set_page_config(
    page_title="Data Explorer",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stPlotlyChart {
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)



st.title("🔍 Data Explorer")
st.markdown("Explore and visualize the datasets used in the Algae Bloom Dashboard.")
show_memory_usage()




@st.cache_data(ttl=3600)
def load_variable_data(var_name):
    """Load data for selected variable"""
    if var_name in ['nitrate', 'phosphate', 'ammonia']:
        ds = load_dataset('nutrients')
    else:
        ds = load_dataset(var_name)
    
    if ds is None:
        return None
    
    return get_variable_data(ds, var_name)


col1, col2 = st.columns(2, gap="xxsmall", border=True)
with col1:
    st.markdown("### Select Data")
    available_vars = list(VARIABLE_INFO.keys())
    selected_var = st.selectbox(
        "Variable",
        options=available_vars,
        format_func=lambda x: VARIABLE_INFO[x]['name'],
        index = 0
    )
    var_info = VARIABLE_INFO[selected_var]
    

    agg_method = st.selectbox(
        "Aggregation Method",
        options=['Mean', 'Median', 'Max', 'Min'],
        index=0
    )
    

data = load_variable_data(selected_var)
if data is None:
    st.error(f"Data for variable '{var_info['name']}' is not available.")
    st.stop()
    

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

st.markdown("---")
data_filtered = data.sel(time=slice(str(start_date), str(end_date)))

st.markdown("## 📈 Quick Statistics")
col1, col2, col3, col4 = st.columns(4)

spatial_mean = data_filtered.mean(dim=['latitude', 'longitude'])
valid_data = spatial_mean.values[~np.isnan(spatial_mean.values)]

with col1:
    st.metric(
        label=f"Mean {var_info['name']}",
        value=f"{np.mean(valid_data):.2f} {var_info['unit']}"
    )

with col2:
    st.metric(
        label="Maximum",
        value=f"{np.max(valid_data):.2f} {var_info['unit']}"
    )

with col3:
    st.metric(
        label="Minimum",
        value=f"{np.min(valid_data):.2f} {var_info['unit']}"
    )

with col4:
    st.metric(
        label="Std Deviation",
        value=f"{np.std(valid_data):.2f} {var_info['unit']}"
    )

st.markdown("---")

# Time series plot
st.markdown(f"## 📊 {var_info['name']} Time Series")

if agg_method == "Mean":
    ts_data = data_filtered.mean(dim=['latitude', 'longitude'])
elif agg_method == "Median":
    ts_data = data_filtered.median(dim=['latitude', 'longitude'])
elif agg_method == "Max":
    ts_data = data_filtered.max(dim=['latitude', 'longitude'])
else:  
    ts_data = data_filtered.min(dim=['latitude', 'longitude'])

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=ts_data.time.values,
    y=ts_data.values,
    mode='lines+markers',
    name=f'{agg_method} {var_info["name"]}',
    line=dict(color=var_info['color'], width=2.5),
    marker=dict(size=5),
    fill='tozeroy',
    fillcolor=f'rgba({int(var_info["color"][1:3], 16)}, {int(var_info["color"][3:5], 16)}, {int(var_info["color"][5:7], 16)}, 0.1)',
    hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>' +
                  f'<b>{var_info["name"]}</b>: %{{y:.2f}} {var_info["unit"]}<br>' +
                  '<extra></extra>'
))


st.plotly_chart(fig, width='stretch')

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📈 Statistics", "📅 Seasonal Pattern", "🔄 Compare Variables"])

with tab1:
    st.markdown("### Statistical Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stats_df = pd.DataFrame({
            'Metric': ['Count', 'Mean', 'Std Dev', 'Min', '25th Percentile', 'Median (50th)', '75th Percentile', 'Max'],
            'Value': [
                f"{len(valid_data):,}",
                f"{np.mean(valid_data):.3f}",
                f"{np.std(valid_data):.3f}",
                f"{np.min(valid_data):.3f}",
                f"{np.percentile(valid_data, 25):.3f}",
                f"{np.percentile(valid_data, 50):.3f}",
                f"{np.percentile(valid_data, 75):.3f}",
                f"{np.max(valid_data):.3f}"
            ],
            'Unit': ['-'] + [var_info['unit']] * 7
        })
        
        st.dataframe(stats_df, hide_index=True, width='stretch')
    
    with col2:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=valid_data,
            nbinsx=40,
            name='Distribution',
            marker_color=var_info['color'],
            marker_line_color='black',
            marker_line_width=0.5,
            opacity=0.7
        ))
        
        fig_hist.add_vline(
            x=np.mean(valid_data),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {np.mean(valid_data):.2f}",
            annotation_position="top"
        )
        
        fig_hist.update_layout(
            title=f'{var_info["name"]} Distribution',
            xaxis_title=f'{var_info["name"]} ({var_info["unit"]})',
            yaxis_title='Frequency',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig_hist, width='stretch')
        
with tab2:
    st.markdown("### Seasonal Pattern (Monthly Climatology)")
    
    monthly_clim = data_filtered.groupby('time.month').mean(dim=['time', 'latitude', 'longitude'])
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_seasonal = go.Figure()
        
        colors = []
        for i in range(12):
            if i+1 in [12, 1, 2]:  # Winter
                colors.append('#3498DB')
            elif i+1 in [3, 4, 5]:  # Spring
                colors.append('#2ECC71')
            elif i+1 in [6, 7, 8]:  # Summer
                colors.append('#E74C3C')
            else:  # Autumn
                colors.append('#F39C12')
        
        fig_seasonal.add_trace(go.Bar(
            x=months,
            y=monthly_clim.values,
            marker_color=colors,
            marker_line_color='black',
            marker_line_width=1.5,
            opacity=0.8,
            hovertemplate='<b>%{x}</b><br>' +
                          f'{var_info["name"]}: %{{y:.2f}} {var_info["unit"]}<br>' +
                          '<extra></extra>'
        ))
        
        mean_val = float(monthly_clim.mean())
        fig_seasonal.add_hline(
            y=mean_val,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Annual Mean: {mean_val:.2f}",
            annotation_position="right"
        )
        
        fig_seasonal.update_layout(
            title=f'Average {var_info["name"]} by Month (2014-2024)',
            xaxis_title='Month',
            yaxis_title=f'{var_info["name"]} ({var_info["unit"]})',
            height=450,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig_seasonal, width='stretch')
    
    with col2:
        monthly_stats = pd.DataFrame({
            'Month': months,
            'Value': [f"{val:.2f}" for val in monthly_clim.values]
        })
        
        st.markdown("**Monthly Averages**")
        st.dataframe(monthly_stats, hide_index=True, width='stretch', height=420)

with tab3:
    st.markdown("### Compare with Another Variable")
    
    compare_var = st.selectbox(
        "Select variable to compare",
        options=[v for v in available_vars if v != selected_var],
        format_func=lambda x: VARIABLE_INFO[x]['name']
    )
    
    compare_info = VARIABLE_INFO[compare_var]
    
    with st.spinner(f"Loading {compare_info['name']} data..."):
        compare_data = load_variable_data(compare_var)
    
    if compare_data is not None:
        compare_filtered = compare_data.sel(time=slice(str(start_date), str(end_date)))
        
        if agg_method == "Mean":
            compare_ts = compare_filtered.mean(dim=['latitude', 'longitude'])
        elif agg_method == "Median":
            compare_ts = compare_filtered.median(dim=['latitude', 'longitude'])
        elif agg_method == "Max":
            compare_ts = compare_filtered.max(dim=['latitude', 'longitude'])
        else:
            compare_ts = compare_filtered.min(dim=['latitude', 'longitude'])
        
        fig_compare = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_compare.add_trace(
            go.Scatter(
                x=ts_data.time.values,
                y=ts_data.values,
                mode='lines+markers',
                name=var_info['name'],
                line=dict(color=var_info['color'], width=2.5),
                marker=dict(size=4)
            ),
            secondary_y=False
        )
        
        fig_compare.add_trace(
            go.Scatter(
                x=compare_ts.time.values,
                y=compare_ts.values,
                mode='lines+markers',
                name=compare_info['name'],
                line=dict(color=compare_info['color'], width=2.5),
                marker=dict(size=4)
            ),
            secondary_y=True
        )
        
        fig_compare.update_xaxes(title_text="Date")
        fig_compare.update_yaxes(
            title_text=f"{var_info['name']} ({var_info['unit']})",
            secondary_y=False,
            title_font=dict(color=var_info['color'])
        )
        fig_compare.update_yaxes(
            title_text=f"{compare_info['name']} ({compare_info['unit']})",
            secondary_y=True,
            title_font=dict(color=compare_info['color'])
        )
        
        fig_compare.update_layout(
            title=f"{var_info['name']} vs {compare_info['name']}",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_compare, width='stretch')
    
# Clear memory
gc.collect()
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
    💡 Tip: Use the controls above to filter data, then explore different tabs for various analyses.
    </small>
</div>
""", unsafe_allow_html=True)