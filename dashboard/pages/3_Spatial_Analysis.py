import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.helper import show_memory_usage, make_bbox_trace
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VARIABLE_INFO, MAP_BOUNDS, BALTIC_REGIONS
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

@st.cache_data
def get_point_timeseries(var_name, lat, lon):
    """Extract full time series for a single lat/lon point for a given variable."""
    data = load_variable_data(var_name)
    if data is None:
        return None
    try:
        ts = data.sel(latitude=lat, longitude=lon, method='nearest')
        return pd.DataFrame({
            'Date': ts.time.values,
            var_name: ts.values
        })
    except Exception as e:
        return None
    

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
col1, col2 = st.columns([3, 1], gap="xxsmall", border=True)
with col1: 
    st.markdown(f"## 🗺️ {var_info['name']} - {time_label}")
    if map_style_key == "smooth":
        fig_map = create_smooth_interpolated_map(data_plot, var_info, time_label, map_style='smooth')
    elif map_style_key == "heatmap":
        fig_map = create_smooth_interpolated_map(data_plot, var_info, time_label, map_style='heatmap')
    else:
        fig_map = create_scatter_bubble_map(data_plot, var_info, time_label)
        
    if fig_map is not None:
        st.plotly_chart(fig_map, width='stretch', key="main_map")
    else:
        st.error("Could not create map. Please try a different time period or variable.")
    
with col2:
    valid_spatial = data_plot.values[~np.isnan(data_plot.values)]
    st.markdown("### 📊 Spatial Statistics")
    st.metric("Mean", f"{np.mean(valid_spatial):.2f} {var_info['unit']}")
    st.metric("Max", f"{np.max(valid_spatial):.2f} {var_info['unit']}")
    st.metric("Min", f"{np.min(valid_spatial):.2f} {var_info['unit']}")
    st.metric("Std Dev", f"{np.std(valid_spatial):.2f} {var_info['unit']}")
    st.metric("Valid Pixels", f"{len(valid_spatial):,}")

st.markdown("## Point Inspector")
st.caption("Choose any point on the main map and enter it's coordinates to load historical data for that location.")
lats = data_plot.latitude.values
lons = data_plot.longitude.values

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    input_lat = st.number_input(
        "Latitude (°N)",
        min_value=float(lats.min()),
        max_value=float(lats.max()),
        value = float(lats.mean()),
        step = 0.01,
        format="%.2f"
    )

with col2:
    input_lon = st.number_input(
        "Longitude (°E)",
        min_value=float(lons.min()),
        max_value=float(lons.max()),
        value = float(lons.mean()),
        step = 0.01,
        format="%.2f"
    )
    
with col3:
    st.markdown("<br>", unsafe_allow_html=True)  # vertical alignment
    inspect = st.button("🔍 Inspect", width="stretch")

if inspect:
    st.session_state["clicked_lat"] = input_lat
    st.session_state["clicked_lon"] = input_lon
    
clicked_lat = st.session_state.get("clicked_lat", None)
clicked_lon = st.session_state.get("clicked_lon", None)

if clicked_lat is not None and clicked_lon is not None:
    st.markdown(f"### Time Series at ({clicked_lat:.2f}°N, {clicked_lon:.2f}°E)")
    st.markdown("### 📊 All-Variable Summary")
    st.caption("Values at the nearest grid point across all variables, for all available time steps.")
    
    summary_rows = []
    for var_name, info in VARIABLE_INFO.items():
        ts_df = get_point_timeseries(var_name, clicked_lat, clicked_lon)
        if ts_df is not None and not ts_df.empty:
            vals = ts_df[var_name].dropna()
            if len(vals) > 0:
                summary_rows.append({
                    'Variable': info['name'],
                    'Unit': info['unit'],
                    'Mean': f"{vals.mean():.2f} {info['unit']}",
                    'Max': f"{vals.max():.2f} {info['unit']}",
                    'Min': f"{vals.min():.2f} {info['unit']}",
                    'Std Dev': f"{vals.std():.2f} {info['unit']}",
                    'N Months': f"{len(vals)}"
                })
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, hide_index=True, width='stretch')
        
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Point Summary",
            data=csv,
            file_name=f"point_summary_{clicked_lat:.2f}N_{clicked_lon:.2f}E.csv",
            mime="text/csv",
            key="download_summary"
        )
    else:
        st.info("No valid data available at this location for any variable.")
else:
    st.info("No data point selected yet")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📍 Hotspot Analysis", "📏 Distribution of Parameter Values", "🗺️ Regional Comparison"])

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
                mime="text/csv",
                key="download_hotspots"
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
    st.markdown("### Distribution of Parameter Values")
    text = ("This histogram shows how values are spread across all spatial pixels in the selected area at a given point in time. The x-axis represents the variable's value and the y-axis shows how many pixels have that value - tall bars mean many pixels share a similar reading, while a wide spread indicates high spatial variability across the region.")
    st.info(text)
    
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
        title='Distribution of Parameter Values',
        xaxis_title=f'{var_info["name"]} ({var_info["unit"]})',
        yaxis_title='Frequency',
        height=500, 
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig_hist, width='stretch')
        
with tab3:
    st.markdown("### Regional Comparison")

    regional_means = []

    for region_name, bounds in BALTIC_REGIONS.items():
        try:
            region_data = data_plot.sel(
                latitude=slice(bounds['min_lat'], bounds['max_lat']),
                longitude=slice(bounds['min_lon'], bounds['max_lon'])
            )
            mean_val = float(region_data.mean())
            max_val  = float(region_data.max())
            min_val  = float(region_data.min())

            regional_means.append({
                'Region': region_name,
                'Mean':   mean_val,
                'Max':    max_val,
                'Min':    min_val,
                'Range':  max_val - min_val
            })
        except:
            continue

    if regional_means:
        df_regions = pd.DataFrame(regional_means)

        col1, col2 = st.columns(2)

        with col1:
            fig_regions = go.Figure()

            for _, row in df_regions.iterrows():
                region_color = BALTIC_REGIONS[row['Region']]['color']
                fig_regions.add_trace(go.Bar(
                    x=[row['Region']],
                    y=[row['Mean']],
                    marker_color=region_color,
                    marker_line_color='black',
                    marker_line_width=1.5,
                    error_y=dict(
                        type='data',
                        array=[row['Range'] / 2],
                        visible=True
                    ),
                    hovertemplate='<b>%{x}</b><br>' +
                                  f'Mean: %{{y:.2f}} {var_info["unit"]}<br>' +
                                  f'Max: {row["Max"]:.2f} {var_info["unit"]}<br>' +
                                  f'Min: {row["Min"]:.2f} {var_info["unit"]}<br>' +
                                  '<extra></extra>'
                ))

            fig_regions.update_layout(
                title=f'Regional {var_info["name"]} Comparison',
                xaxis_title='Region',
                yaxis_title=f'{var_info["name"]} ({var_info["unit"]})',
                height=400,
                template='plotly_white',
                showlegend=False,
                xaxis_tickangle=-25
            )

            st.plotly_chart(fig_regions, width='stretch')

        with col2:
            fig = go.Figure()

            for name, region in BALTIC_REGIONS.items():
                fig.add_trace(make_bbox_trace(name, region))

            fig.update_layout(
                geo=dict(
                    projection_type='mercator',
                    showland=True,
                    landcolor='rgb(230, 230, 220)',
                    showocean=True,
                    oceancolor='rgb(210, 230, 245)',
                    showcoastlines=True,
                    coastlinecolor='rgb(100, 100, 100)',
                    coastlinewidth=1,
                    showlakes=True,
                    lakecolor='rgb(210, 230, 245)',
                    lonaxis=dict(range=[9, 32]),
                    lataxis=dict(range=[53, 67]),
                ),
                height=400,
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(
                    title="Regions",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="lightgrey",
                    borderwidth=1
                )
            )

            st.plotly_chart(fig, width='stretch')
        st.markdown("**Regional Statistics**")

        df_display = df_regions.copy()
        for col in ['Mean', 'Max', 'Min', 'Range']:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}")

        st.dataframe(df_display, hide_index=True, width='stretch', height=280)
            


st.markdown("---")
st.markdown("## 🧭 Spatial Gradients")

st.markdown("### Select Data")
available_vars = ["Mean", "Max"]
selected_var = st.selectbox(
    "Select Variable",
    options=available_vars,
    index=0
)

if selected_var == "Mean":
    lat_profile = data_plot.mean(dim='longitude')
    lon_profile = data_plot.mean(dim='latitude')
else:
    lat_profile = data_plot.max(dim='longitude')
    lon_profile = data_plot.max(dim='latitude')
    
col1, col2 = st.columns(2)

with col1:
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

    # Add a vrect for each region's latitude range
    for region_name, bounds in BALTIC_REGIONS.items():
        fig_lat.add_vrect(
            x0=bounds['min_lat'],
            x1=bounds['max_lat'],
            annotation_text=region_name,
            annotation_position="top left",
            annotation=dict(
                textangle=-90,
                font=dict(size=9),
            ),
            fillcolor=bounds['color'],
            opacity=0.1,
            line_width=1,
            line_color=bounds['color'],
        )
     
    fig_lat.update_layout(
        title=f'Latitudinal Profile (South to North) — {selected_var}',
        xaxis_title='Latitude (°N)', 
        yaxis_title=f'{var_info["name"]} ({var_info["unit"]})',  
        height=400,
        template='plotly_white',
        showlegend=False
    )
        
    st.plotly_chart(fig_lat, width='stretch')

with col2:
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

    # Add a vrect for each region's longitude range
    for region_name, bounds in BALTIC_REGIONS.items():
        fig_lon.add_vrect(
            x0=bounds['min_lon'],
            x1=bounds['max_lon'],
            annotation_text=region_name,
            annotation_position="top left",
            annotation=dict(
                textangle=-90,
                font=dict(size=9),
            ),
            fillcolor=bounds['color'],
            opacity=0.1,
            line_width=1,
            line_color=bounds['color'],
        )
    
    fig_lon.update_layout(
        title=f'Longitudinal Profile (West to East) — {selected_var}',
        xaxis_title='Longitude (°E)',
        yaxis_title=f'{var_info["name"]} ({var_info["unit"]})',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig_lon, width='stretch')
    
max_lat_idx = lat_profile.argmax()
peak_lat = float(lat_profile.latitude.values[max_lat_idx])
peak_max_lat = float(lat_profile.max())

max_lon_idx = lon_profile.argmax()
peak_lon = float(lon_profile.longitude.values[max_lon_idx])
peak_max_lon = float(lon_profile.max())

overall_val = float(data_plot.mean()) if selected_var == "Mean" else float(data_plot.max())

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label=f"Overall {selected_var}",
        value=f"{overall_val:.2f} {var_info['unit']}",
    )

with col2:
    st.metric(
        label=f"Peak Latitude ({selected_var})",
        value=f"{peak_max_lat:.2f} {var_info['unit']}",
        delta=f"at {peak_lat:.2f}°N"
    )

with col3:
    st.metric(
        label=f"Peak Longitude ({selected_var})",
        value=f"{peak_max_lon:.2f} {var_info['unit']}",
        delta=f"at {peak_lon:.2f}°E"
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
