import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.helper import show_memory_usage
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VARIABLE_INFO, MAP_BOUNDS
from utils.data_loader import load_dataset, get_variable_data
from components.filters import create_smooth_interpolated_map, create_scatter_bubble_map

st.set_page_config(
    page_title="Spatial Analysis",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Spatial Analysis")
st.markdown("Explore spatial patterns and distributions across the Baltic Sea coastal region")
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


col1, col2 = st.columns(2, gap="xxsmall", border=True)
with col1:
    st.markdown("### Select Data")
    available_vars = list(VARIABLE_INFO.keys())
    selected_var = st.selectbox(
        "Select Variable",
        options=available_vars,
        format_func=lambda x: VARIABLE_INFO[x]['name'],
        index=0
    )
    var_info = VARIABLE_INFO[selected_var]
    
    st.markdown("### Select Map Type")
    map_style = st.selectbox(
        "Map Quality",
        options=["Standard (Original Resolution)", "Smooth (Interpolated)", "Scatter Points"],
        index=0,
        help="Smooth uses interpolation for better appearance with low-resolution data"
    )
    
    map_style_key = "smooth" if map_style == "Smooth (Interpolated)" else "heatmap" if map_style == "Standard (Original Resolution)" else "scatter"
data = load_variable_data(selected_var)

if data is None:
    st.error(f"Failed to load {var_info['name']} data.")
    st.stop()
    
with col2:
    st.markdown("### Select Time Period")
    time_agg = st.selectbox(
        "Time Period",
        options=["Single Month", "Seasonal Average", "Annual Average", "All Time Average"],
        index=0
    )
    if time_agg == "Single Month":
        available_years = sorted(np.unique(data.time.dt.year.values))
        selected_year = st.selectbox("Year", options=available_years, index=len(available_years)-1)
        
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
        selected_month = st.selectbox("Month", options=range(1, 13), 
                                      format_func=lambda x: months[x-1],
                                      index=6)
        date_str = f"{selected_year}-{selected_month:02d}"
        
        try:
            data_plot = data.sel(time=date_str, method='nearest')
            time_label = f"{months[selected_month-1]} {selected_year}"
        except:
            st.error(f"No data available for {months[selected_month-1]} {selected_year}")
            st.stop()
            
    elif time_agg == "Seasonal Average":
        season = st.selectbox(
            "Season",
            options=["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Autumn (SON)"],
            index=2 
        )
        year_option = st.radio("Years", options=["All Years (2014-2024)", "Specific Year"])
        season_months = {
        'Winter (DJF)': [12, 1, 2],
        'Spring (MAM)': [3, 4, 5],
        'Summer (JJA)': [6, 7, 8],
        'Autumn (SON)': [9, 10, 11]
        }[season]
        
        if year_option == "All Years (2014-2024)":
            season_data = data.where(data.time.dt.month.isin(season_months), drop=True)
            data_plot = season_data.mean('time')
            time_label = f"{season} Average (2014-2024)"
        else:
            available_vars = sorted(np.unique(data.time.dt.year.values))
            specific_year = st.selectbox("Select Year", options=available_vars, index=len(available_vars)-1)
            
            season_data = data.where(
                (data.time.dt.year == specific_year) & (data.time.dt.month.isin(season_months)),
                drop=True
            )
            data_plot = season_data.mean('time')
            time_label = f"{season} {specific_year}"
            
    elif time_agg == "Annual Average":
        available_vars = sorted(np.unique(data.time.dt.year.values))
        selected_year = st.selectbox("Select Year", options=available_vars, index=len(available_vars)-1)
        
        year_data = data.sel(time=str(selected_year))
        data_plot = year_data.mean('time')
        time_label = f"Annual Average {selected_year}"
        
    else:
        data_plot = data.mean('time')
        time_label = "All Time Average (2014-2024)"
        
st.markdown("---")
st.markdown(f"## 🗺️ {var_info['name']} - {time_label}")
if map_style_key == "smooth":
    fig_map = create_smooth_interpolated_map(data_plot, var_info, time_label, map_style='smooth')
elif map_style_key == "heatmap":
    fig_map = create_smooth_interpolated_map(data_plot, var_info, time_label, map_style='heatmap')
else:
    fig_map = create_scatter_bubble_map(data_plot, var_info, time_label)

if fig_map is not None:
    st.plotly_chart(fig_map, width='stretch')
else:
    st.error("Could not create map. Please try a different time period or variable.")
    
st.markdown("---")

st.markdown("## 📊 Spatial Statistics")

col1, col2, col3, col4, col5 = st.columns(5)

valid_spatial = data_plot.values[~np.isnan(data_plot.values)]

with col1:
    st.metric("Mean", f"{np.mean(valid_spatial):.2f} {var_info['unit']}")

with col2:
    st.metric("Max", f"{np.max(valid_spatial):.2f} {var_info['unit']}")

with col3:
    st.metric("Min", f"{np.min(valid_spatial):.2f} {var_info['unit']}")

with col4:
    st.metric("Std Dev", f"{np.std(valid_spatial):.2f} {var_info['unit']}")

with col5:
    st.metric("Valid Pixels", f"{len(valid_spatial):,}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📍 Hotspot Analysis", "📏 Spatial Distribution", "🗺️ Regional Comparison"])

with tab1:
    st.markdown("## 📍 Hotspot Identification")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        threshold_pct = st.slider(
            "Hotspot Threshold (Percentile)",
            min_value=50,
            max_value=99,
            value=85,
            step=5,
            help="Values above this percentile will be highlighted as hotspots. Higher percentiles = more exclusive hotspots."
        )
        
        threshold_value = np.nanpercentile(data_plot.values, threshold_pct)
        
        st.info(f"**Hotspot threshold: {threshold_value:.2f} {var_info['unit']}** (≥{threshold_pct}th percentile)")
        
        hotspot_mask = data_plot.values >= threshold_value
        hotspot_count = np.sum(hotspot_mask & ~np.isnan(data_plot.values))
        total_valid_pixels = np.sum(~np.isnan(data_plot.values))
        hotspot_percentage = (hotspot_count / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0
        
        st.metric(
            label="Hotspot Locations", 
            value=f"{hotspot_count:,} pixels",
            delta=f"{hotspot_percentage:.1f}% of area"
        )
        
        if hotspot_count > 0:
            hotspot_values = data_plot.values[hotspot_mask & ~np.isnan(data_plot.values)]
            
            st.markdown("#### Hotspot Statistics")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Mean", f"{np.mean(hotspot_values):.2f}")
                st.metric("Min", f"{np.min(hotspot_values):.2f}")
            with col_b:
                st.metric("Max", f"{np.max(hotspot_values):.2f}")
                st.metric("Std Dev", f"{np.std(hotspot_values):.2f}")
        
        if hotspot_count > 0:
            hotspot_lats, hotspot_lons = np.where(hotspot_mask & ~np.isnan(data_plot.values))
            hotspot_df = pd.DataFrame({
                'Latitude': data_plot.latitude.values[hotspot_lats],
                'Longitude': data_plot.longitude.values[hotspot_lons],
                var_info['name']: data_plot.values[hotspot_mask & ~np.isnan(data_plot.values)]
            })
            
            csv = hotspot_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Hotspot Coordinates",
                data=csv,
                file_name=f"hotspots_{selected_var}_{threshold_pct}pct.csv",
                mime="text/csv"
            )
    
    with col2:
        fig_hotspot = go.Figure()
        
        fig_hotspot.add_trace(go.Heatmap(
            x=data_plot.longitude.values,
            y=data_plot.latitude.values,
            z=data_plot.values,
            colorscale='Greys',
            opacity=0.3,
            showscale=False,
            hoverinfo='skip',
            name='Background'
        ))
        
        hotspot_data = np.where(
            (data_plot.values >= threshold_value) & ~np.isnan(data_plot.values),
            data_plot.values,
            np.nan
        )
        
        fig_hotspot.add_trace(go.Heatmap(
            x=data_plot.longitude.values,
            y=data_plot.latitude.values,
            z=hotspot_data,
            colorscale=[
                [0, '#fee5d9'],
                [0.33, '#fcae91'],
                [0.66, '#fb6a4a'],
                [1, '#cb181d']
            ],
            colorbar=dict(
            title=dict(
                text=f"{var_info['unit']}",
                side='right' 
            )
        ),
            hovertemplate='<b>Lat</b>: %{y:.2f}°N<br>' +
                          '<b>Lon</b>: %{x:.2f}°E<br>' +
                          f'<b>{var_info["name"]}</b>: %{{z:.2f}} {var_info["unit"]}<br>' +
                          '<extra></extra>',
            name='Hotspots'
        ))
        
        fig_hotspot.update_layout(
            title=f'Hotspot Locations (≥{threshold_pct}th percentile)',
            xaxis_title='Longitude (°E)',
            yaxis_title='Latitude (°N)',
            height=500,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig_hotspot, width='stretch')
        if hotspot_count > 0:
            hotspot_indices = np.where(hotspot_mask & ~np.isnan(data_plot.values))
            hotspot_lats = hotspot_indices[0]
            hotspot_lons = hotspot_indices[1]
            
            hotspot_lat_values = data_plot.latitude.values[hotspot_lats]
            hotspot_lon_values = data_plot.longitude.values[hotspot_lons]
            
            center_lat = hotspot_lat_values.mean()
            center_lon = hotspot_lon_values.mean()
            lat_range = np.ptp(hotspot_lat_values)  
            lon_range = np.ptp(hotspot_lon_values)  
            
            if lat_range < 2 and lon_range < 2:
                st.success("🎯 Hotspots are **highly concentrated** in one region")
            elif lat_range < 5 and lon_range < 5:
                st.info("📍 Hotspots are **moderately dispersed**")
            else:
                st.warning("🌍 Hotspots are **widely scattered** across the region")

with tab2:
    st.markdown("### Spatial Value Distribution")
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=valid_spatial,
        nbinsx=50,
        marker_color=var_info['color'],
        marker_line_color='black',
        marker_line_width=0.5,
        opacity=0.7,
        name='Distribution'
    ))
    
    percentiles = [
        (25, 'blue', 'top left'),
        (50, 'green', 'top'),
        (75, 'orange', 'top right'),
        (90, 'red', 'bottom right')
    ]
    
    for pct, color, position in percentiles:
        val = np.nanpercentile(valid_spatial, pct)
        fig_hist.add_vline(
            x=val,
            line_dash="dash",
            line_color=color,
            line_width=2,
            annotation=dict(
                text=f"P{pct}<br>{val:.2f}",
                font=dict(size=11, color=color, family='Arial Black'),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor=color,
                borderwidth=1,
                borderpad=3
            ),
            annotation_position=position
        )
    
    fig_hist.update_layout(
        title='Distribution of Spatial Values',
        xaxis_title=f'{var_info["name"]} ({var_info["unit"]})',
        yaxis_title='Frequency',
        height=500, 
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig_hist, width='stretch')
        
with tab3:
    st.markdown("### Regional Comparison")
    
    regions = {
        'Southern Baltic': {'lat': (53.5, 55.5), 'lon': (12, 20)},
        'Central Baltic': {'lat': (55.5, 58.5), 'lon': (17, 22)},
        'Gulf of Finland': {'lat': (59.0, 60.5), 'lon': (22, 28)},
        'Gulf of Bothnia': {'lat': (60.5, 65.5), 'lon': (18, 24)},
    }
    
    regional_means = []
    
    for region_name, bounds in regions.items():
        try:
            region_data = data_plot.sel(
                latitude=slice(bounds['lat'][0], bounds['lat'][1]),
                longitude=slice(bounds['lon'][0], bounds['lon'][1])
            )
            mean_val = float(region_data.mean())
            max_val = float(region_data.max())
            min_val = float(region_data.min())
            
            regional_means.append({
                'Region': region_name,
                'Mean': mean_val,
                'Max': max_val,
                'Min': min_val,
                'Range': max_val - min_val
            })
        except:
            continue
    
    if regional_means:
        df_regions = pd.DataFrame(regional_means)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_regions = go.Figure()
            
            fig_regions.add_trace(go.Bar(
                x=df_regions['Region'],
                y=df_regions['Mean'],
                marker_color=var_info['color'],
                marker_line_color='black',
                marker_line_width=1.5,
                error_y=dict(
                    type='data',
                    array=df_regions['Range'] / 2,
                    visible=True
                ),
                hovertemplate='<b>%{x}</b><br>' +
                              f'Mean: %{{y:.2f}} {var_info["unit"]}<br>' +
                              '<extra></extra>'
            ))
            
            fig_regions.update_layout(
                title=f'Regional {var_info["name"]} Comparison',
                xaxis_title='Region',
                yaxis_title=f'{var_info["name"]} ({var_info["unit"]})',
                height=400,
                template='plotly_white',
                showlegend=False
            )
            
            st.plotly_chart(fig_regions, width='stretch')
        
        with col2:
            st.markdown("**Regional Statistics**")
            
            df_display = df_regions.copy()
            df_display['Mean'] = df_display['Mean'].apply(lambda x: f"{x:.2f}")
            df_display['Max'] = df_display['Max'].apply(lambda x: f"{x:.2f}")
            df_display['Min'] = df_display['Min'].apply(lambda x: f"{x:.2f}")
            df_display['Range'] = df_display['Range'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(df_display, hide_index=True, width='stretch', height=400)

st.markdown("---")
st.markdown("## 🧭 Spatial Gradients")

col1, col2 = st.columns(2)

with col1:
    lat_profile = data_plot.mean(dim='longitude')
    
    fig_lat = go.Figure()
    fig_lat.add_trace(go.Scatter(
        x=lat_profile.latitude.values,  
        y=lat_profile.values,           
        mode='lines+markers',
        line=dict(color=var_info['color'], width=3),
        marker=dict(size=6),
        fill='tozeroy', 
        fillcolor=f'rgba({int(var_info["color"][1:3], 16)}, {int(var_info["color"][3:5], 16)}, {int(var_info["color"][5:7], 16)}, 0.2)',
        hovertemplate='<b>Latitude</b>: %{x:.2f}°N<br>' +
                      f'<b>{var_info["name"]}</b>: %{{y:.2f}} {var_info["unit"]}<br>' +
                      '<extra></extra>'
    ))
    
    fig_lat.update_layout(
        title='Latitudinal Profile (South to North)',
        xaxis_title='Latitude (°N)', 
        yaxis_title=f'{var_info["name"]} ({var_info["unit"]})',  
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig_lat, width='stretch')

with col2:
    # Longitudinal profile (average across latitudes)
    lon_profile = data_plot.mean(dim='latitude')
    
    fig_lon = go.Figure()
    fig_lon.add_trace(go.Scatter(
        x=lon_profile.longitude.values,
        y=lon_profile.values,
        mode='lines+markers',
        line=dict(color=var_info['color'], width=3),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor=f'rgba({int(var_info["color"][1:3], 16)}, {int(var_info["color"][3:5], 16)}, {int(var_info["color"][5:7], 16)}, 0.2)',
        hovertemplate='<b>Longitude</b>: %{x:.2f}°E<br>' +
                      f'<b>{var_info["name"]}</b>: %{{y:.2f}} {var_info["unit"]}<br>' +
                      '<extra></extra>'
    ))
    
    fig_lon.update_layout(
        title='Longitudinal Profile (West to East)',
        xaxis_title='Longitude (°E)',
        yaxis_title=f'{var_info["name"]} ({var_info["unit"]})',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig_lon, width='stretch')
    
    
max_lat = lat_profile.latitude.values[lat_profile.argmax()]
max_val = float(lat_profile.max())

st.metric(
    "Highest Concentration",
    f"{max_val:.2f} {var_info['unit']}",
    f"at {max_lat:.2f}°N"
)

fig_lat.add_vrect(
    x0=53.5, x1=55.5,
    annotation_text="Southern Baltic",
    fillcolor="blue", opacity=0.1
)

# Clear memory
gc.collect()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
    💡 Tip: Use heatmap for general patterns, contour for detailed gradients. 
    Hover over the map for specific values.
    </small>
</div>
""", unsafe_allow_html=True)
