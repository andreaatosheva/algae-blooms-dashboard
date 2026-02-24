import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from utils.helper import show_memory_usage

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VARIABLE_INFO, SEASONS, SEASON_COLORS
from utils.data_loader import load_dataset, get_variable_data, load_all_datasets

# Page config
st.set_page_config(
    page_title="Environmental Factors",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Environmental Factors")
st.markdown("Explore the different environmental factors.")
show_memory_usage()

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

st.markdown("### Analysis Mode")

analysis_mode = st.selectbox(
    "Select Analysis Type",
    options=[
        "Multi-Variable Time Series",
        "Environmental Conditions Profile",
        "Nutrient Ratios"
    ],
    index=0
)

st.markdown("---")
                
# ==================== MULTI-VARIABLE TIME SERIES ====================

if analysis_mode == "Multi-Variable Time Series":
    st.markdown("## 📊 Multi-Variable Time Series")
    
    variables_to_plot = st.multiselect(
        "Select Variables to Display",
        options=list(VARIABLE_INFO.keys()),
        default=['chlorophyll', 'temperature', 'nitrate'],
        format_func=lambda x: VARIABLE_INFO[x]['name']
    )
    
    if len(variables_to_plot) == 0:
        st.warning("Please select at least one variable.")
    else:
        normalize = st.checkbox(
            "Normalize values (0-1 scale)",
            value=False,
            help="Normalize all variables to 0-1 scale for easier comparison"
        )
        
        with st.spinner("Loading data..."):
            fig_multi = go.Figure()
            
            for var in variables_to_plot:
                data = load_variable_data(var)
                if data is not None:
                    ts = data.mean(dim=['latitude', 'longitude'])
                    
                    if normalize:
                        values = ts.values
                        values_norm = (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values))
                        y_values = values_norm
                        y_label = "Normalized Value (0-1)"
                    else:
                        y_values = ts.values
                        y_label = "Value"
                    
                    var_info_plot = VARIABLE_INFO[var]
                    
                    fig_multi.add_trace(go.Scatter(
                        x=ts.time.values,
                        y=y_values,
                        mode='lines',
                        name=var_info_plot['name'],
                        line=dict(color=var_info_plot['color'], width=2.5),
                        hovertemplate=f'<b>{var_info_plot["name"]}</b><br>' +
                                      'Date: %{x|%Y-%m-%d}<br>' +
                                      'Value: %{y:.2f}<br>' +
                                      '<extra></extra>'
                    ))
            
            fig_multi.update_layout(
                title='Multi-Variable Comparison',
                xaxis_title='Date',
                yaxis_title=y_label if normalize else 'Mixed Units',
                height=600,
                template='plotly_white',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_multi, width='stretch')
# ==================== ENVIRONMENTAL CONDITIONS PROFILE ====================
elif analysis_mode == "Environmental Conditions Profile":
    st.markdown("## 🌊 Environmental Conditions Profile")
    
    st.info("View a comprehensive snapshot of environmental conditions for a specific time period")
    
    time_agg = st.radio(
        "Select Time Period for Profile",
        options=["Monthly", "Yearly", "Seasonal"],
        horizontal=True,
        index=0
        )
    months = ['January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December']
    
    st.markdown("---")
    
    profile_year = None
    profile_month = None
    season = None
    season_year = None
    
    if time_agg == "Monthly":
        col1, col2 = st.columns(2)
        
        with col1:
            profile_year = st.selectbox(
                "Year",
                options=sorted(range(2014, 2025), reverse=True),
                index=0)
            
        with col2:
            profile_month = st.selectbox(
                "Month",
                options=range(1, 13),
                format_func=lambda x: months[x-1],
                index=6  # Default to July
            )
    
        date_str = f"{profile_year}-{profile_month:02d}"
        time_label = f"{months[profile_month-1]} {profile_year}"
    
    elif time_agg == "Yearly":
        profile_year = st.selectbox(
            "Select Year",
            options=sorted(range(2014, 2025), reverse=True),
            index=0
        )
        
        date_str = str(profile_year)
        time_label = f"Annual Average {profile_year}"
    
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            season = st.selectbox(
                "Select Season",
                options=["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Autumn (SON)"],
                index = 2)
                
        with col2:
            
            season_year = st.selectbox(
                "Select Year for Season",
                options=sorted(range(2014, 2025), reverse=True),
                index=0
                
            )
            time_label = f"{season} {season_year}"
            
    with st.spinner("Loading environmental data..."):
        env_vars = ['temperature', 'wind_speed', 'solar_radiation', 'rainfall', 
                   'nitrate', 'phosphate', 'ammonia']
        
        profile_data = {}
        
        for var in env_vars:
            data = load_variable_data(var)
            if data is not None:
                try:
                    if time_agg == "Monthly":
                        var_slice = data.sel(time=date_str, method='nearest')
                        mean_val = float(var_slice.mean(dim=['latitude', 'longitude']))
                    
                    elif time_agg == "Yearly":
                        year_data = data.sel(time=str(profile_year))
                        mean_val = float(year_data.mean(dim=['time', 'latitude', 'longitude']))
                        
                    else:
                        season_months = SEASONS[season]
                        season_data = data.where(
                            (data.time.dt.year == season_year) &
                            (data.time.dt.month.isin(season_months)),
                            drop=True
                        )
                        mean_val = float(season_data.mean(dim=['time', 'latitude', 'longitude']))
                        
                    if var == 'solar_radiation':
                        mean_val = mean_val / 1e6  
                    elif var == 'rainfall':
                        mean_val = mean_val * 1000 
                    
                    var_info_temp = VARIABLE_INFO[var]
                    profile_data[var_info_temp['name']] = {
                        'value': mean_val,
                        'unit': var_info_temp['unit'],
                        'color': var_info_temp['color']
                    }
                except:
                    pass
    
    if not profile_data:
        st.error("Could not load environmental data for the selected period.")
    else:
        st.markdown(f"### Environmental Snapshot: {time_label}")
        
        cols = st.columns(len(profile_data))
        
        for col, (var_name, info) in zip(cols, profile_data.items()):
            with col:
                st.metric(
                    label=var_name,
                    value=f"{info['value']:.2f}",
                    delta=info['unit']
                )
        st.markdown("---")
        
        st.markdown("### 📊 Environmental Conditions Radar Chart")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            var_names = list(profile_data.keys())
            values = [profile_data[v]['value'] for v in var_names]
            
            ref_values_min = []
            ref_values_max = []
            
            for var in env_vars:
                data = load_variable_data(var)
                if data is not None and VARIABLE_INFO[var]['name'] in var_names:
                    all_means = data.mean(dim=['latitude', 'longitude']).values
                    all_means = all_means[~np.isnan(all_means)]
                    
                    if var == 'solar_radiation':
                        all_means = all_means / 1e6
                    elif var == 'rainfall':
                        all_means = all_means * 1000
                    
                    ref_values_min.append(np.min(all_means))
                    ref_values_max.append(np.max(all_means))
            
            values_norm = []
            for i, val in enumerate(values):
                if len(ref_values_max) > i and ref_values_max[i] != ref_values_min[i]:
                    norm = (val - ref_values_min[i]) / (ref_values_max[i] - ref_values_min[i])
                    values_norm.append(max(0, min(1, norm)))  
                else:
                    values_norm.append(0.5)
            
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values_norm + [values_norm[0]], 
                theta=var_names + [var_names[0]],
                fill='toself',
                fillcolor='rgba(31, 119, 180, 0.3)',
                line=dict(color='#1f77b4', width=3),
                name=time_label,
                hovertemplate='<b>%{theta}</b><br>Normalized: %{r:.2f}<extra></extra>'
            ))
            
            fig_radar.add_trace(go.Scatterpolar(
                r=[0.5] * (len(var_names) + 1),
                theta=var_names + [var_names[0]],
                line=dict(color='red', width=2, dash='dash'),
                name='Average Level',
                hovertemplate='Average (0.5)<extra></extra>'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickmode='linear',
                        tick0=0,
                        dtick=0.2
                    )
                ),
                showlegend=True,
                title=f'Environmental Conditions Profile',
                height=600,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_radar, width='stretch')
        
        with col2:
            st.markdown("**📊 Normalized Values**")
            st.markdown("*(0 = Minimum, 1 = Maximum)*")
            
            radar_df = pd.DataFrame({
                'Variable': var_names,
                'Actual': [f"{profile_data[v]['value']:.2f} {profile_data[v]['unit']}" for v in var_names],
                'Normalized': [f"{v:.2f}" for v in values_norm],
                'Status': [
                    '🔴 Very High' if v > 0.8 
                    else '🟠 High' if v > 0.6 
                    else '🟡 Average' if v > 0.4 
                    else '🟢 Low' if v > 0.2 
                    else '🔵 Very Low' 
                    for v in values_norm
                ]
            })
            
            st.dataframe(radar_df, hide_index=True, width='stretch', height=300)
            
            st.markdown("---")
            st.markdown("**💡 Interpretation:**")
            st.markdown("""
            - **0.0-0.2**: Very low conditions
            - **0.2-0.4**: Below average
            - **0.4-0.6**: Average conditions
            - **0.6-0.8**: Above average
            - **0.8-1.0**: Very high conditions
            """)
        
        st.markdown("---")
        st.markdown("### 📊 Normalized Values Comparison")
        
        fig_bar = go.Figure()
        
        fig_bar.add_trace(go.Bar(
            x=var_names,
            y=values_norm,
            marker=dict(
                color=values_norm,
                colorscale='RdYlGn',
                cmin=0,
                cmax=1,
                line=dict(color='black', width=1.5)
            ),
            text=[f"{v:.2f}" for v in values_norm],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Normalized: %{y:.2f}<extra></extra>'
        ))
        
        fig_bar.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="gray",
            annotation_text="Average (0.5)",
            annotation_position="right"
        )
        
        fig_bar.update_layout(
            title=f'Environmental Factors - {time_label}',
            xaxis_title='Environmental Variable',
            yaxis_title='Normalized Value (0-1)',
            yaxis_range=[0, 1.1],
            height=450,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig_bar, width='stretch')
        
        st.markdown("---")
        st.markdown("### 🔄 Compare with Another Period")
        
        compare_enabled = st.checkbox("Enable period comparison")
        
        if compare_enabled:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                compare_agg = st.selectbox(
                    "Comparison Period Type",
                    options=["Monthly", "Yearly", "Seasonal"],
                    index=1
                )
            
            months_list = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            
            with col2:
                if compare_agg == "Yearly":
                    compare_year = st.selectbox("Year", options=sorted(range(2014, 2025), reverse=True), index=1, key='comp_year2')
                    compare_date = str(compare_year)
                    compare_label = f"Annual Average {compare_year}"
                    
                elif compare_agg == "Seasonal":
                    compare_year = st.selectbox("Year", options=sorted(range(2014, 2025), reverse=True), index=1, key='comp_year3')
                    
                else:
                    compare_year = st.selectbox("Year", options=sorted(range(2014, 2025), reverse=True), index=1, key='comp_year')

            with col3:
                if compare_agg == "Monthly":
                    compare_month = st.selectbox("Month", options=range(1, 13), 
                                                format_func=lambda x: months_list[x-1], index=6, key='comp_month')
                    compare_date = f"{compare_year}-{compare_month:02d}"
                    compare_label = f"{months_list[compare_month-1]} {compare_year}"

                elif compare_agg == "Seasonal":
                    compare_season = st.selectbox("Season", 
                                                 options=["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Autumn (SON)"],
                                                 index=0, key='comp_season')
                    compare_label = f"{compare_season} {compare_year}"
                    
                    
            with st.spinner("Loading comparison data..."):
                compare_profile = {}
                
                for var in env_vars:
                    data = load_variable_data(var)
                    if data is not None:
                        try:
                            if compare_agg == "Monthly":
                                var_slice = data.sel(time=compare_date, method='nearest')
                                mean_val = float(var_slice.mean(dim=['latitude', 'longitude']))
                            elif compare_agg == "Yearly":
                                year_data = data.sel(time=compare_date)
                                mean_val = float(year_data.mean(dim=['time', 'latitude', 'longitude']))
                            else:  
                                season_months = SEASONS[compare_season]
                                season_data = data.where(
                                    (data.time.dt.year == compare_year) & 
                                    (data.time.dt.month.isin(season_months)),
                                    drop=True
                                )
                                mean_val = float(season_data.mean(dim=['time', 'latitude', 'longitude']))
                            
                            if var == 'solar_radiation':
                                mean_val = mean_val / 1e6
                            elif var == 'rainfall':
                                mean_val = mean_val * 1000
                            
                            var_info_temp = VARIABLE_INFO[var]
                            compare_profile[var_info_temp['name']] = {
                                'value': mean_val,
                                'unit': var_info_temp['unit']
                            }
                        except:
                            pass
            
            if compare_profile:
                compare_var_names = [v for v in var_names if v in compare_profile]
                compare_values = [compare_profile[v]['value'] for v in compare_var_names]
                compare_values_norm = []
                
                for i, (var_name, val) in enumerate(zip(compare_var_names, compare_values)):
                    orig_idx = var_names.index(var_name)
                    if orig_idx < len(ref_values_max) and ref_values_max[orig_idx] != ref_values_min[orig_idx]:
                        norm = (val - ref_values_min[orig_idx]) / (ref_values_max[orig_idx] - ref_values_min[orig_idx])
                        compare_values_norm.append(max(0, min(1, norm)))
                    else:
                        compare_values_norm.append(0.5)
                
                fig_radar_comp = go.Figure()
                
                fig_radar_comp.add_trace(go.Scatterpolar(
                    r=values_norm + [values_norm[0]],
                    theta=var_names + [var_names[0]],
                    fill='toself',
                    fillcolor='rgba(31, 119, 180, 0.3)',
                    line=dict(color='#1f77b4', width=3),
                    name=time_label
                ))
                
                if len(compare_var_names) == len(var_names):
                    fig_radar_comp.add_trace(go.Scatterpolar(
                        r=compare_values_norm + [compare_values_norm[0]],
                        theta=var_names + [var_names[0]],
                        fill='toself',
                        fillcolor='rgba(255, 127, 14, 0.3)',
                        line=dict(color='#ff7f0e', width=3),
                        name=compare_label
                    ))
                
                fig_radar_comp.add_trace(go.Scatterpolar(
                    r=[0.5] * (len(var_names) + 1),
                    theta=var_names + [var_names[0]],
                    line=dict(color='red', width=2, dash='dash'),
                    name='Average (0.5)'
                ))
                
                fig_radar_comp.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title='Period Comparison',
                    height=600,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_radar_comp, width='stretch')
                
                st.markdown("### 📊 Period-to-Period Changes")
                
                diff_data = []
                for var_name in var_names:
                    if var_name in profile_data and var_name in compare_profile:
                        val1 = profile_data[var_name]['value']
                        val2 = compare_profile[var_name]['value']
                        diff = val1 - val2
                        pct_diff = (diff / val2 * 100) if val2 != 0 else 0
                        
                        diff_data.append({
                            'Variable': var_name,
                            time_label: f"{val1:.2f}",
                            compare_label: f"{val2:.2f}",
                            'Difference': f"{diff:+.2f}",
                            '% Change': f"{pct_diff:+.1f}%"
                        })
                
                if diff_data:
                    df_diff = pd.DataFrame(diff_data)
                    st.dataframe(df_diff, hide_index=True, width='stretch')
            else:
                st.warning("Could not load comparison data.")
        
        
# ==================== NUTRIENT RATIOS ====================
# ==================== NUTRIENT RATIOS ====================
elif analysis_mode == "Nutrient Ratios":
    st.markdown("## ⚗️ Nutrient Ratios")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ratio_time_agg = st.selectbox(
            "Select Time Period",
            options=["All Time", "Monthly", "Yearly", "Seasonal"],
            index=0,
            key='ratio_time_agg'
        )
    
    months_list = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    ratio_year = None
    ratio_month = None
    ratio_season = None
    
    with col2:
        if ratio_time_agg != "All Time":
            ratio_year = st.selectbox(
                "Year",
                options=sorted(range(2014, 2025), reverse=True),
                index=0,
                key='ratio_year'
            )
    
    with col3:
        if ratio_time_agg == "Monthly":
            ratio_month = st.selectbox(
                "Month",
                options=range(1, 13),
                format_func=lambda x: months_list[x-1],
                index=6,
                key='ratio_month'
            )
        elif ratio_time_agg == "Seasonal":
            ratio_season = st.selectbox(
                "Season",
                options=["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Autumn (SON)"],
                index=2,
                key='ratio_season'
            )
    
    st.markdown("---")
    
    with st.spinner("Loading nutrient data..."):
        nitrate_data = load_variable_data('nitrate')
        phosphate_data = load_variable_data('phosphate')
        ammonia_data = load_variable_data('ammonia')
    
    if nitrate_data is None or phosphate_data is None:
        st.error("Could not load nutrient data for ratio analysis.")
        
    else:
        if ratio_time_agg == "Monthly":
            time_label = f"{months_list[ratio_month-1]} {ratio_year}"
            date_str = f"{ratio_year}-{ratio_month:02d}"
            nitrate_filtered = nitrate_data.sel(time=date_str, method='nearest')
            phosphate_filtered = phosphate_data.sel(time=date_str, method='nearest')
            if ammonia_data is not None:
                ammonia_filtered = ammonia_data.sel(time=date_str, method='nearest')
            else:
                ammonia_filtered = None
                
        elif ratio_time_agg == "Yearly":
            time_label = f"{ratio_year}"
            nitrate_filtered = nitrate_data.sel(time=str(ratio_year))
            phosphate_filtered = phosphate_data.sel(time=str(ratio_year))
            if ammonia_data is not None:
                ammonia_filtered = ammonia_data.sel(time=str(ratio_year))
            else:
                ammonia_filtered = None
                
        elif ratio_time_agg == "Seasonal":
            time_label = f"{ratio_season} {ratio_year}"
            season_months = SEASONS[ratio_season]
            nitrate_filtered = nitrate_data.where(
                (nitrate_data.time.dt.year == ratio_year) &
                (nitrate_data.time.dt.month.isin(season_months)),
                drop=True
            )
            phosphate_filtered = phosphate_data.where(
                (phosphate_data.time.dt.year == ratio_year) &
                (phosphate_data.time.dt.month.isin(season_months)),
                drop=True
            )
            if ammonia_data is not None:
                ammonia_filtered = ammonia_data.where(
                    (ammonia_data.time.dt.year == ratio_year) &
                    (ammonia_data.time.dt.month.isin(season_months)),
                    drop=True
                )
            else:
                ammonia_filtered = None
        else: 
            time_label = "All Time"
            nitrate_filtered = nitrate_data
            phosphate_filtered = phosphate_data
            ammonia_filtered = ammonia_data
        
        no3_mean = nitrate_filtered.mean(dim=['latitude', 'longitude'])
        po4_mean = phosphate_filtered.mean(dim=['latitude', 'longitude'])
        
        if ammonia_filtered is not None:
            nh4_mean = ammonia_filtered.mean(dim=['latitude', 'longitude'])
            total_nitrogen = no3_mean + nh4_mean
        else:
            total_nitrogen = no3_mean
        
        np_ratio = total_nitrogen / po4_mean
        
        st.markdown(f"### 📊 N:P Ratio Analysis - {time_label}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if ratio_time_agg in ["Monthly", "All Time"]:
                current_np = float(np_ratio[-1] if ratio_time_agg == "All Time" else np_ratio)
            else:
                current_np = float(np_ratio.mean())
            st.metric(
                label="Current N:P Ratio" if ratio_time_agg == "All Time" else "N:P Ratio",
                value=f"{current_np:.1f}:1"
            )
        
        with col2:
            if ratio_time_agg == "All Time":
                mean_np = float(np_ratio.mean())
                st.metric(
                    label="Average N:P Ratio",
                    value=f"{mean_np:.1f}:1"
                )
            else:
                if len(np_ratio.shape) > 0:
                    min_np = float(np_ratio.min())
                    max_np = float(np_ratio.max())
                    st.metric(
                        label="Range",
                        value=f"{min_np:.1f} - {max_np:.1f}"
                    )
                else:
                    st.metric(label="Range", value="Single Value")
        
        with col3:
            if current_np > 20:
                limitation = "P-Limited"
                color = "red"
            elif current_np < 10:
                limitation = "N-Limited"
                color = "blue"
            else:
                limitation = "Balanced"
                color = "green"
            
            st.metric(
                label="Nutrient Status",
                value=limitation
            )
        
        if ratio_time_agg == "All Time" or (len(np_ratio.shape) > 0 and len(np_ratio) > 1):
            fig_np = go.Figure()
            
            fig_np.add_trace(go.Scatter(
                x=np_ratio.time.values,
                y=np_ratio.values,
                mode='lines+markers',
                name='N:P Ratio',
                line=dict(color='#9467bd', width=2.5),
                marker=dict(size=5)
            ))
            
            fig_np.add_hline(
                y=16,
                line_dash="dash",
                line_color="green",
                line_width=2,
                annotation_text="Redfield Ratio (16:1)",
                annotation_position="right"
            )
            
            fig_np.add_hrect(
                y0=0, y1=10,
                fillcolor="blue",
                opacity=0.1,
                annotation_text="N-Limited",
                annotation_position="top left"
            )
            
            fig_np.add_hrect(
                y0=20, y1=np_ratio.max().values if ratio_time_agg == "All Time" else max(np_ratio.max().values, 25),
                fillcolor="red",
                opacity=0.1,
                annotation_text="P-Limited",
                annotation_position="bottom left"
            )
            
            fig_np.update_layout(
                title=f'N:P Ratio Over Time - {time_label}',
                xaxis_title='Date',
                yaxis_title='N:P Ratio',
                height=500,
                template='plotly_white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_np, width='stretch')
        
        st.markdown("### 📈 Individual Nutrient Trends")
        
        if ratio_time_agg == "All Time" or (len(no3_mean.shape) > 0 and len(no3_mean) > 1):
            fig_nutrients = go.Figure()
            
            fig_nutrients.add_trace(go.Scatter(
                x=no3_mean.time.values,
                y=no3_mean.values,
                mode='lines',
                name='Nitrate (NO₃)',
                line=dict(color='#ff7f0e', width=2.5)
            ))
            
            fig_nutrients.add_trace(go.Scatter(
                x=po4_mean.time.values,
                y=po4_mean.values,
                mode='lines',
                name='Phosphate (PO₄)',
                line=dict(color='#9467bd', width=2.5),
                yaxis='y2'
            ))
            
            if ammonia_filtered is not None:
                fig_nutrients.add_trace(go.Scatter(
                    x=nh4_mean.time.values,
                    y=nh4_mean.values,
                    mode='lines',
                    name='Ammonia (NH₄)',
                    line=dict(color='#8c564b', width=2.5)
                ))
            
            fig_nutrients.update_layout(
                title=f'Nutrient Concentrations Over Time - {time_label}',
                xaxis_title='Date',
                yaxis=dict(
                    title='NO₃ & NH₄ (mmol/m³)',
                    side='left'
                ),
                yaxis2=dict(
                    title='PO₄ (mmol/m³)',
                    side='right',
                    overlaying='y'
                ),
                height=500,
                template='plotly_white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_nutrients, width='stretch')
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Nitrate (NO₃)",
                    value=f"{float(no3_mean):.2f}",
                    delta="mmol/m³"
                )
            
            with col2:
                st.metric(
                    label="Phosphate (PO₄)",
                    value=f"{float(po4_mean):.2f}",
                    delta="mmol/m³"
                )
            
            with col3:
                if ammonia_filtered is not None:
                    st.metric(
                        label="Ammonia (NH₄)",
                        value=f"{float(nh4_mean):.2f}",
                        delta="mmol/m³"
                    )
        
        with st.expander("📚 Understanding N:P Ratios"):
            st.markdown("""
            **Redfield Ratio (16:1):** The optimal N:P ratio for phytoplankton growth.
            
            - **N:P > 20:** Phosphorus-limited conditions. Algae growth is constrained by phosphate availability.
            - **N:P < 10:** Nitrogen-limited conditions. Algae growth is constrained by nitrogen availability.
            - **10 < N:P < 20:** Balanced nutrient conditions. Neither nutrient is strongly limiting.
            
            **Why it matters:** Understanding nutrient limitation helps predict which nutrient reductions would be most effective for controlling algal blooms.
            """)
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
    💡 Tip: Environmental factors interact to create bloom conditions. 
    Explore correlations and ratios to understand these relationships.
    </small>
</div>
""", unsafe_allow_html=True)

        
        