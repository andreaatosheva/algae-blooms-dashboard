import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VARIABLE_INFO, SEASONS, SEASON_COLORS
from utils.data_loader import load_dataset, get_variable_data, load_all_datasets

# Page config 
st.set_page_config( 
            page_title="Bloom Detection", 
            page_icon="🌿", 
            layout="wide" 
            )

st.title("🌿 Algae Bloom Detection")
st.markdown("Identify, analyze, and predict algae bloom events in the Baltic Sea")

@st.cache_data
def load_variable_data(var_name):
    """Load data for selected variable"""
    if var_name in ['nitrate', 'phosphate', 'ammonia']:
        ds = load_dataset('nutrients')
    else:
        ds = load_dataset(var_name)
    
    if ds is None:
        return None
    
    return get_variable_data(ds, var_name)


with st.spinner("Loading data..."):
    chl_data = load_variable_data('chlorophyll')
    temp_data = load_variable_data('temperature') 
    wind_data = load_variable_data('wind_speed') 
    solar_data = load_variable_data('solar_radiation')
    no3_data = load_variable_data('nitrate')
    po4_data = load_variable_data('phosphate')
    
if chl_data is None:
    st.error("Could not load chlorophyll data, which is essential for bloom detection.")
    st.stop()
    
st.markdown("## Bloom Detection Analysis")

col1, col2 = st.columns(2)

with col1:
    bloom_threshold = st.slider(
        "Bloom Threshold (mg/m³)",
        min_value=0.5,
        max_value=10.0,
        value=5.0,
        step=0.1,
        help="Chlorophyll-a concentration above this value indicates a bloom"
    )
    
    spatial_extent = st.slider(
        "Min. Spatial Extent (%)",
        min_value=1,
        max_value=50,
        value=5,
        help="Minimum percentage of area affected to count as bloom event"
    )
    intensity_levels = st.checkbox("Show Intensity Levels", value=True)

with col2:
    
    if intensity_levels:
        moderate_threshold = st.number_input(
            "Moderate Bloom (mg/m³)",
            min_value=bloom_threshold,
            max_value=10.0,
            value=5.0
        )
        
        high_threshold = st.number_input(
            "High Bloom (mg/m³)",
            min_value=moderate_threshold,
            max_value=10.0,
            value=7.0
        )

st.markdown("---")

chl_mean = chl_data.mean(dim=['latitude', 'longitude'])
bloom_mask = chl_mean > bloom_threshold

if intensity_levels:
    moderate_mask = chl_mean > moderate_threshold
    high_mask = chl_mean > high_threshold

else:
    moderate_mask = None
    high_mask = None
    
n_bloom_months = bloom_mask.sum().values
total_months = len(chl_mean['time'])
bloom_percentage = (n_bloom_months / total_months) * 100

st.markdown("## 📊 Bloom Event Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Bloom Months Detected",
        value=f"{n_bloom_months}",
        delta=f"{bloom_percentage:.1f}% of total"
    )

with col2:
    if n_bloom_months > 0:
        bloom_values = chl_mean.values[bloom_mask]
        avg_bloom_intensity = np.mean(bloom_values)
        st.metric(
            label="Avg Bloom Intensity",
            value=f"{avg_bloom_intensity:.2f} mg/m³"
        )
    else:
        st.metric(label="Avg Bloom Intensity", value="N/A")

with col3:
    if n_bloom_months > 0:
        max_bloom = float(chl_mean.max())
        max_bloom_date = str(chl_mean.idxmax().values)[:10]
        st.metric(
            label="Peak Bloom",
            value=f"{max_bloom:.2f} mg/m³",
            delta=max_bloom_date
        )
    else:
        st.metric(label="Peak Bloom", value="N/A")

with col4:
    if intensity_levels and moderate_mask is not None:
        n_high = high_mask.sum().values
        st.metric(
            label="High Intensity Events",
            value=f"{n_high}",
            delta=f">{high_threshold} mg/m³"
        )
    else:
        st.metric(label="High Intensity Events", value="N/A")

st.markdown("---")

st.markdown("## 📈 Bloom Event Timeline")

fig_timeline = go.Figure()

fig_timeline.add_trace(go.Scatter(
    x=chl_mean.time.values,
    y=chl_mean.values,
    mode='lines+markers',
    name='Chlorophyll-a',
    line=dict(color='#2ca02c', width=2.5),
    marker=dict(size=5),
    fill='tozeroy',
    fillcolor='rgba(44, 160, 44, 0.1)',
    hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>' +
                  '<b>Chlorophyll</b>: %{y:.2f} mg/m³<br>' +
                  '<extra></extra>'
))

fig_timeline.add_hline(
    y=bloom_threshold,
    line_dash="dash",
    line_color="orange",
    line_width=2,
    annotation_text=f"Bloom Threshold ({bloom_threshold} mg/m³)",
    annotation_position="right"
)

if intensity_levels:
    fig_timeline.add_hline(
        y=moderate_threshold,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Moderate ({moderate_threshold} mg/m³)",
        annotation_position="right"
    )
    
    fig_timeline.add_hline(
        y=high_threshold,
        line_dash="dash",
        line_color="darkred",
        line_width=2,
        annotation_text=f"High ({high_threshold} mg/m³)",
        annotation_position="right"
    )
    
    fig_timeline.add_hrect(
        y0=bloom_threshold, y1=moderate_threshold,
        fillcolor="yellow", opacity=0.1,
        annotation_text="Low Bloom", annotation_position="top left"
    )
    fig_timeline.add_hrect(
        y0=moderate_threshold, y1=high_threshold,
        fillcolor="orange", opacity=0.1,
        annotation_text="Moderate Bloom", annotation_position="top left"
    )
    fig_timeline.add_hrect(
        y0=high_threshold, y1=chl_mean.max().values,
        fillcolor="red", opacity=0.1,
        annotation_text="High Bloom", annotation_position="top right"
    )

bloom_periods_idx = np.where(bloom_mask)[0]
for idx in bloom_periods_idx:
    time_val = chl_mean.time.values[idx]
    fig_timeline.add_vline(
        x=time_val,
        line_color="red",
        line_width=0.5,
        opacity=0.3
    )

fig_timeline.update_layout(
    title='Chlorophyll-a Timeline with Bloom Events',
    xaxis_title='Date',
    yaxis_title='Chlorophyll-a (mg/m³)',
    height=500,
    template='plotly_white',
    hovermode='x unified'
)

st.plotly_chart(fig_timeline, width='stretch')

if n_bloom_months > 0:
    with st.expander(f"📋 View All {n_bloom_months} Bloom Events"):
        bloom_events = []
        
        for i in np.where(bloom_mask)[0]:
            date = str(chl_mean.time.values[i])[:10]
            value = float(chl_mean.values[i])
            
            if intensity_levels:
                if value > high_threshold:
                    intensity = "🔴 High"
                elif value > moderate_threshold:
                    intensity = "🟠 Moderate"
                else:
                    intensity = "🟡 Low"
            else:
                intensity = "🌿 Bloom"
            
            bloom_events.append({
                'Date': date,
                'Chlorophyll (mg/m³)': f"{value:.2f}",
                'Intensity': intensity
            })
        
        df_blooms = pd.DataFrame(bloom_events)
        st.dataframe(df_blooms, hide_index=True, width='stretch', height=400)

st.markdown("---")

# ==================== BLOOM-FAVORABLE CONDITIONS ====================
st.markdown("## 🌡️ Bloom-Favorable Environmental Conditions")

st.info("""
Algae blooms typically occur when multiple favorable conditions coincide:
- **Chlorophyll**: > threshold (bloom indicator)
- **Temperature**: > 15°C (warmer water promotes growth)
- **Solar Radiation**: High (more light for photosynthesis)
- **Nutrients**: Sufficient nitrate and phosphate (growth nutrients)
- **Wind**: Low to moderate (< 7 m/s, allows water stratification)
""")

env_available = {
    'Temperature': temp_data is not None,
    'Wind': wind_data is not None,
    'Solar': solar_data is not None,
    'Nitrate': no3_data is not None,
    'Phosphate': po4_data is not None
}

missing_vars = [k for k, v in env_available.items() if not v]
if missing_vars:
    st.warning(f"⚠️ Missing environmental data: {', '.join(missing_vars)}")
    
st.markdown("### 📊 Environmental Conditions Timeline")

n_subplots = 1 + sum(env_available.values())  

subplot_titles = ['Chlorophyll-a (Bloom Indicator)']
if temp_data is not None:
    subplot_titles.append('Temperature')
if wind_data is not None:
    subplot_titles.append('Wind Speed')
if solar_data is not None:
    subplot_titles.append('Solar Radiation')
if no3_data is not None:
    subplot_titles.append('Nitrate')
if po4_data is not None:
    subplot_titles.append('Phosphate')

fig_conditions = make_subplots(
    rows=n_subplots, cols=1,
    subplot_titles=subplot_titles,
    vertical_spacing=0.05,
    shared_xaxes=True
)

row = 1

fig_conditions.add_trace(
    go.Scatter(
        x=chl_mean.time.values,
        y=chl_mean.values,
        mode='lines',
        name='Chlorophyll',
        line=dict(color='#2ca02c', width=2),
        fill='tozeroy',
        fillcolor='rgba(44, 160, 44, 0.2)',
        showlegend=False
    ),
    row=row, col=1
)

fig_conditions.add_hline(
    y=bloom_threshold,
    line_dash="dash",
    line_color="red",
    line_width=2,
    row=row, col=1
)

for idx in bloom_periods_idx:
    fig_conditions.add_vrect(
        x0=chl_mean.time.values[max(0, idx-1)],
        x1=chl_mean.time.values[min(len(chl_mean)-1, idx+1)],
        fillcolor="red",
        opacity=0.1,
        layer="below",
        line_width=0,
        row=row, col=1
    )

fig_conditions.update_yaxes(title_text="mg/m³", row=row, col=1)
row += 1

if temp_data is not None:
    temp_mean = temp_data.mean(dim=['latitude', 'longitude'])
    
    fig_conditions.add_trace(
        go.Scatter(
            x=temp_mean.time.values,
            y=temp_mean.values,
            mode='lines',
            name='Temperature',
            line=dict(color='#d62728', width=2),
            showlegend=False
        ),
        row=row, col=1
    )
    
    fig_conditions.add_hline(
        y=15,
        line_dash="dash",
        line_color="orange",
        line_width=1.5,
        row=row, col=1
    )
    
    fig_conditions.add_hrect(
        y0=15, y1=temp_mean.max().values,
        fillcolor="orange",
        opacity=0.1,
        layer="below",
        line_width=0,
        row=row, col=1
    )
    
    fig_conditions.update_yaxes(title_text="°C", row=row, col=1)
    row += 1

if wind_data is not None:
    wind_mean = wind_data.mean(dim=['latitude', 'longitude'])
    
    fig_conditions.add_trace(
        go.Scatter(
            x=wind_mean.time.values,
            y=wind_mean.values,
            mode='lines',
            name='Wind Speed',
            line=dict(color='#1f77b4', width=2),
            showlegend=False
        ),
        row=row, col=1
    )
    
    fig_conditions.add_hline(
        y=7,
        line_dash="dash",
        line_color="green",
        line_width=1.5,
        row=row, col=1
    )
    
    fig_conditions.add_hrect(
        y0=0, y1=7,
        fillcolor="green",
        opacity=0.1,
        layer="below",
        line_width=0,
        row=row, col=1
    )
    
    fig_conditions.update_yaxes(title_text="m/s", row=row, col=1)
    row += 1

if solar_data is not None:
    solar_mean = solar_data.mean(dim=['latitude', 'longitude']) / 1e6  
    
    fig_conditions.add_trace(
        go.Scatter(
            x=solar_mean.time.values,
            y=solar_mean.values,
            mode='lines',
            name='Solar Radiation',
            line=dict(color='#ff8c00', width=2),
            showlegend=False
        ),
        row=row, col=1
    )
    
    fig_conditions.update_yaxes(title_text="MJ/m²", row=row, col=1)
    row += 1

if no3_data is not None:
    no3_mean = no3_data.mean(dim=['latitude', 'longitude'])
    
    fig_conditions.add_trace(
        go.Scatter(
            x=no3_mean.time.values,
            y=no3_mean.values,
            mode='lines',
            name='Nitrate',
            line=dict(color='#ff7f0e', width=2),
            showlegend=False
        ),
        row=row, col=1
    )
    
    fig_conditions.update_yaxes(title_text="mmol/m³", row=row, col=1)
    row += 1

if po4_data is not None:
    po4_mean = po4_data.mean(dim=['latitude', 'longitude'])
    
    fig_conditions.add_trace(
        go.Scatter(
            x=po4_mean.time.values,
            y=po4_mean.values,
            mode='lines',
            name='Phosphate',
            line=dict(color='#9467bd', width=2),
            showlegend=False
        ),
        row=row, col=1
    )
    
    fig_conditions.update_yaxes(title_text="mmol/m³", row=row, col=1)
    row += 1

fig_conditions.update_xaxes(title_text="Date", row=n_subplots, col=1)

fig_conditions.update_layout(
    height=150 * n_subplots, 
    template='plotly_white',
    showlegend=False,
    title_text='Environmental Conditions and Bloom Events Timeline'
)

st.plotly_chart(fig_conditions, width='stretch')


# ==================== FAVORABLE CONDITIONS ANALYSIS ====================
favorable_conditions = {}

if temp_data is not None:
    temp_mean = temp_data.mean(dim=['latitude', 'longitude'])
    temp_favorable = temp_mean > 15
    favorable_conditions['Temperature (>15°C)'] = temp_favorable
else:
    temp_favorable = None

if wind_data is not None:
    wind_mean = wind_data.mean(dim=['latitude', 'longitude'])
    wind_favorable = wind_mean < 7
    favorable_conditions['Wind (<7 m/s)'] = wind_favorable
else:
    wind_favorable = None

if solar_data is not None:
    solar_mean = solar_data.mean(dim=['latitude', 'longitude'])
    solar_median = float(solar_mean.median())
    solar_favorable = solar_mean > solar_median
    favorable_conditions['Solar (>median)'] = solar_favorable
else:
    solar_favorable = None

if len(favorable_conditions) > 0:
    all_times = [chl_mean.time.values]
    for condition in favorable_conditions.values():
        all_times.append(condition.time.values)
    
    common_times = all_times[0]
    for times in all_times[1:]:
        common_times = np.intersect1d(common_times, times)
    
    if len(common_times) > 0:
        bloom_aligned = chl_mean.sel(time=common_times) > bloom_threshold
        
        conditions_aligned = {}
        for name, condition in favorable_conditions.items():
            conditions_aligned[name] = condition.sel(time=common_times)
        
        n_favorable = sum(conditions_aligned.values())
        
        all_favorable = bloom_aligned.copy()
        for condition in conditions_aligned.values():
            all_favorable = all_favorable & condition
        
        n_all_favorable = all_favorable.sum().values
        
        st.markdown("### 📊 Favorable Condition Frequency")
        
        cols = st.columns(len(favorable_conditions) + 1)
        
        for idx, (name, condition) in enumerate(favorable_conditions.items()):
            with cols[idx]:
                n_times = conditions_aligned[name].sum().values
                pct = (n_times / len(common_times)) * 100
                st.metric(
                    label=name,
                    value=f"{n_times} months",
                    delta=f"{pct:.1f}%"
                )
        
        with cols[-1]:
            pct_all = (n_all_favorable / len(common_times)) * 100
            st.metric(
                label="All Conditions Met",
                value=f"{n_all_favorable} months",
                delta=f"{pct_all:.1f}%"
            )
        
        bloom_with_favorable = bloom_aligned & all_favorable
        n_bloom_with_favorable = bloom_with_favorable.sum().values
        
        if n_bloom_months > 0:
            coincidence_rate = (n_bloom_with_favorable / n_bloom_months) * 100
            
            st.markdown("### 🎯 Bloom-Condition Coincidence")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Blooms with Favorable Conditions",
                    value=f"{n_bloom_with_favorable} / {n_bloom_months}",
                    delta=f"{coincidence_rate:.1f}%"
                )
            
            with col2:
                blooms_without = n_bloom_months - n_bloom_with_favorable
                st.metric(
                    label="Blooms w/o Favorable Conditions",
                    value=f"{blooms_without}",
                    delta="Unexpected blooms"
                )
            
            with col3:
                favorable_no_bloom = all_favorable & ~bloom_aligned
                n_favorable_no_bloom = favorable_no_bloom.sum().values
                st.metric(
                    label="Favorable w/o Blooms",
                    value=f"{n_favorable_no_bloom}",
                    delta="Missed opportunities"
                )
            
            if coincidence_rate > 70:
                st.success(f"✅ High coincidence rate ({coincidence_rate:.1f}%)! Environmental conditions are strong predictors of blooms.")
            elif coincidence_rate > 40:
                st.info(f"ℹ️ Moderate coincidence rate ({coincidence_rate:.1f}%). Environmental conditions partially explain blooms.")
            else:
                st.warning(f"⚠️ Low coincidence rate ({coincidence_rate:.1f}%). Other factors may be important.")

st.markdown("---")

st.markdown("## 📅 Bloom Seasonality")

tab1, tab2, tab3 = st.tabs(["Monthly Pattern", "Seasonal Distribution", "Spatial Bloom Extent"])

with tab1:
    st.markdown("### Monthly Bloom Frequency")
    
    # Count blooms by month
    bloom_by_month = {}
    for month in range(1, 13):
        month_mask = chl_mean.time.dt.month == month
        month_blooms = (chl_mean.where(month_mask, drop=True) > bloom_threshold).sum().values
        total_months = month_mask.sum().values
        bloom_by_month[month] = {
            'count': int(month_blooms),
            'total': int(total_months),
            'percentage': (month_blooms / total_months * 100) if total_months > 0 else 0
        }
    
    months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig_monthly = go.Figure()
    
    bloom_counts = [bloom_by_month[m]['count'] for m in range(1, 13)]
    total_counts = [bloom_by_month[m]['total'] for m in range(1, 13)]
    non_bloom_counts = [total_counts[i] - bloom_counts[i] for i in range(12)]
    
    fig_monthly.add_trace(go.Bar(
        x=months_names,
        y=bloom_counts,
        name='Bloom Months',
        marker_color='#E74C3C',
        text=bloom_counts,
        textposition='inside'
    ))
    
    fig_monthly.add_trace(go.Bar(
        x=months_names,
        y=non_bloom_counts,
        name='Non-Bloom Months',
        marker_color='#95A5A6',
        text=non_bloom_counts,
        textposition='inside'
    ))
    
    fig_monthly.update_layout(
        barmode='stack',
        title='Bloom Frequency by Month (2014-2024)',
        xaxis_title='Month',
        yaxis_title='Number of Years',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_monthly, width='stretch')
    
    monthly_stats = pd.DataFrame({
        'Month': months_names,
        'Blooms': [bloom_by_month[m]['count'] for m in range(1, 13)],
        'Total': [bloom_by_month[m]['total'] for m in range(1, 13)],
        'Probability': [f"{bloom_by_month[m]['percentage']:.1f}%" for m in range(1, 13)]
    })
    
    st.dataframe(monthly_stats, hide_index=True, width='stretch')

with tab2:
    st.markdown("### Seasonal Bloom Distribution")
    
    seasonal_blooms = {}
    
    for season_name, months in SEASONS.items():
        season_mask = chl_mean.time.dt.month.isin(months)
        season_blooms = (chl_mean.where(season_mask, drop=True) > bloom_threshold).sum().values
        total_season = season_mask.sum().values
        
        seasonal_blooms[season_name] = {
            'count': int(season_blooms),
            'total': int(total_season),
            'percentage': (season_blooms / total_season * 100) if total_season > 0 else 0
        }
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = go.Figure()
        
        fig_pie.add_trace(go.Pie(
            labels=list(seasonal_blooms.keys()),
            values=[seasonal_blooms[s]['count'] for s in seasonal_blooms.keys()],
            marker=dict(colors=[SEASON_COLORS[s] for s in seasonal_blooms.keys()]),
            hovertemplate='<b>%{label}</b><br>' +
                          'Blooms: %{value}<br>' +
                          'Percentage: %{percent}<br>' +
                          '<extra></extra>'
        ))
        
        fig_pie.update_layout(
            title='Bloom Distribution by Season',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_pie, width='stretch')
    
    with col2:
        # Seasonal statistics
        seasonal_stats = pd.DataFrame({
            'Season': list(seasonal_blooms.keys()),
            'Bloom Months': [seasonal_blooms[s]['count'] for s in seasonal_blooms.keys()],
            'Total Months': [seasonal_blooms[s]['total'] for s in seasonal_blooms.keys()],
            'Bloom Probability': [f"{seasonal_blooms[s]['percentage']:.1f}%" for s in seasonal_blooms.keys()]
        })
        
        st.markdown("**Seasonal Bloom Statistics**")
        st.dataframe(seasonal_stats, hide_index=True, width='stretch', height=300)
        
        # Peak season
        peak_season = max(seasonal_blooms.items(), key=lambda x: x[1]['count'])[0]
        st.success(f"🌟 **Peak Bloom Season**: {peak_season}")

with tab3:
    st.markdown("### Spatial Bloom Extent Over Time")
    
    st.info("Analyze how much of the coastal area is affected by blooms")
    
    bloom_extent = []
    
    for t in range(len(chl_data.time)):
        chl_slice = chl_data.isel(time=t)
        total_pixels = (~np.isnan(chl_slice.values)).sum()
        bloom_pixels = (chl_slice.values > bloom_threshold).sum()
        
        if total_pixels > 0:
            extent_pct = (bloom_pixels / total_pixels) * 100
        else:
            extent_pct = 0
        
        bloom_extent.append({
            'time': chl_data.time.values[t],
            'extent_pct': extent_pct,
            'bloom_pixels': int(bloom_pixels),
            'total_pixels': int(total_pixels)
        })
    
    df_extent = pd.DataFrame(bloom_extent)
    
    fig_extent = go.Figure()
    
    fig_extent.add_trace(go.Scatter(
        x=df_extent['time'],
        y=df_extent['extent_pct'],
        mode='lines+markers',
        name='Bloom Extent',
        line=dict(color='#2ca02c', width=2.5),
        marker=dict(size=5),
        fill='tozeroy',
        fillcolor='rgba(44, 160, 44, 0.2)',
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>' +
                      '<b>Extent</b>: %{y:.1f}%<br>' +
                      '<extra></extra>'
    ))
    
    fig_extent.add_hline(
        y=spatial_extent,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Significant extent ({spatial_extent}%)",
        annotation_position="right"
    )
    
    fig_extent.update_layout(
        title='Percentage of Coastal Area Affected by Blooms',
        xaxis_title='Date',
        yaxis_title='Bloom Extent (%)',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_extent, width='stretch')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_extent = df_extent['extent_pct'].mean()
        st.metric("Average Bloom Extent", f"{avg_extent:.1f}%")
    
    with col2:
        max_extent = df_extent['extent_pct'].max()
        max_extent_date = df_extent.loc[df_extent['extent_pct'].idxmax(), 'time']
        st.metric(
            "Maximum Extent",
            f"{max_extent:.1f}%",
            delta=str(max_extent_date)[:10]
        )
    
    with col3:
        significant_events = (df_extent['extent_pct'] > spatial_extent).sum()
        st.metric(
            "Significant Events",
            f"{significant_events}",
            delta=f">{spatial_extent}% area"
        )

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
    💡 Tip: Adjust the bloom threshold to detect different intensity levels. 
    Monitor environmental conditions to understand bloom triggers.
    </small>
</div>
""", unsafe_allow_html=True)