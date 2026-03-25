import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.helper import show_memory_usage, make_bbox_trace, get_point_timeseries
from utils.data_loader import load_variable_data
from plotly.subplots import make_subplots
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VARIABLE_INFO, SEASONS, SEASON_COLORS, BALTIC_REGIONS, MAP_BOUNDS
from utils.data_loader import load_dataset, get_variable_data
from components.filters import create_smooth_interpolated_map, create_scatter_bubble_map

st.set_page_config(
    page_title="Spatial Analysis of Algae Bloom Factors",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Spatial Analysis of Algae Bloom Factors")
st.markdown("Explore spatial patterns and distributions across the Baltic Sea coastal region")


chl_data = load_variable_data('chlorophyll')
temp_data = load_variable_data('temperature') 
wind_data = load_variable_data('wind_speed') 
solar_data = load_variable_data('solar_radiation')
no3_data = load_variable_data('nitrate')
po4_data = load_variable_data('phosphate')
rain_data = load_variable_data('rainfall')


col1, col2 = st.columns([1.5, 2.5], gap="small", border=False)

with col1:
    map_area_key = st.selectbox(
        "**Map Area**",
        options=["Full Baltic Region"] + list(BALTIC_REGIONS.keys()),
        index=0
    )
    
    map_style_key = st.selectbox(
            "**Map Style**",
            options=["Regular Heatmap", "Hotspot Map"],
            index=0
        )
    
    available_vars = list(VARIABLE_INFO.keys())
    
    selected_var = st.selectbox(
        "**Select Variable**",
        options=available_vars,
        format_func=lambda x: VARIABLE_INFO[x]['name'],
        index=0
    )
    var_info = VARIABLE_INFO[selected_var]
    
    data = load_variable_data(selected_var)

    if data is None:
        st.error(f"Failed to load {var_info['name']} data.")
        st.stop()
    
    time_agg = st.selectbox(
        "**Select Time Period**",
        options=["Single Month", "Seasonal Average", "Annual Average", "All Time Average"],
        index=0
    )

    if time_agg == "Single Month":
        available_years = sorted(np.unique(data.time.dt.year.values))
        selected_year = st.selectbox("**Select Year**", options=available_years, index=len(available_years)-1)
        
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']
        selected_month = st.selectbox("**Select Month**", options=range(1, 13), 
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
            "**Select Season**",
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
            specific_year = st.selectbox("**Select Year**", options=available_vars, index=len(available_vars)-1)
            
            season_data = data.where(
                (data.time.dt.year == specific_year) & (data.time.dt.month.isin(season_months)),
                drop=True
            )
            data_plot = season_data.mean('time')
            time_label = f"{season} {specific_year}"
            
    elif time_agg == "Annual Average":
        available_vars = sorted(np.unique(data.time.dt.year.values))
        selected_year = st.selectbox("**Select Year**", options=available_vars, index=len(available_vars)-1)
        
        year_data = data.sel(time=str(selected_year))
        data_plot = year_data.mean('time')
        time_label = f"Annual Average {selected_year}"
        
    else:
        data_plot = data.mean('time')
        time_label = "All Time Average (2014-2024)"
    
    if map_area_key != "Full Baltic Region":
        bounds = BALTIC_REGIONS[map_area_key]
        def regional_slice(data, bounds):
            if data is None:
                return None
            return data.sel(
                latitude=slice(bounds['min_lat'], bounds['max_lat']),
                longitude=slice(bounds['min_lon'], bounds['max_lon'])
            )
        chl_data_r = regional_slice(chl_data, bounds)
        temp_data_r = regional_slice(temp_data, bounds)
        wind_data_r = regional_slice(wind_data, bounds)
        solar_data_r = regional_slice(solar_data, bounds)
        no3_data_r = regional_slice(no3_data, bounds)
        po4_data_r = regional_slice(po4_data, bounds)
        rain_data_r = regional_slice(rain_data, bounds)
    
    else:
        chl_data_r = chl_data
        temp_data_r = temp_data
        wind_data_r = wind_data
        solar_data_r = solar_data
        no3_data_r = no3_data
        po4_data_r = po4_data
        rain_data_r = rain_data

        
    if time_agg == "Single Month":
        time_filter = {
            'mode': 'Single Month',
            'year': selected_year,
            'month': selected_month
        }

    elif time_agg == "Seasonal Average":
        time_filter = {
            'mode': 'Seasonal Average',
            'months': season_months,
            'year': specific_year if year_option == "Specific Year" else None
        }
        print(season_months)

    elif time_agg == "Annual Average":
        time_filter = {
            'mode': 'Annual Average',
            'year': selected_year
        }

    else:
            time_filter = {
                'mode': 'All Time',
            }
    data_plot = regional_slice(data_plot, BALTIC_REGIONS[map_area_key]) if map_area_key != "Full Baltic Region" else data_plot
    valid_spatial = data_plot.values[~np.isnan(data_plot.values)]
    
    
    if map_style_key == "Hotspot Map":
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
        
        if hotspot_count > 0:
            hotspot_values = data_plot.values[hotspot_mask & ~np.isnan(data_plot.values)]
    
        

with col2:
    if map_style_key == "Regular Heatmap":
        fig_map = create_smooth_interpolated_map(data_plot, var_info, time_label, map_style='heatmap')
        if fig_map is not None:
            st.plotly_chart(fig_map, width="content", key="main_map")
    
    else:
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
            height=550,
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

flat_max_idx = np.nanargmax(data_plot.values)
flat_min_idx = np.nanargmin(data_plot.values)

max_lat_idx, max_lon_idx = np.unravel_index(flat_max_idx, data_plot.values.shape)
min_lat_idx, min_lon_idx = np.unravel_index(flat_min_idx, data_plot.values.shape)

max_lat = float(data_plot.latitude.values[max_lat_idx])
max_lon = float(data_plot.longitude.values[max_lon_idx])
min_lat = float(data_plot.latitude.values[min_lat_idx])
min_lon = float(data_plot.longitude.values[min_lon_idx])

colAAA, colBBB, colCCC, colDDD = st.columns(4)
with colAAA:
    st.metric("Mean", f"{np.mean(valid_spatial):.2f} {var_info['unit']}")
with colBBB:
    st.metric("Max", f"{np.max(valid_spatial):.2f} {var_info['unit']}",
          delta=f"at {max_lat:.2f}°N, {max_lon:.2f}°E",
          delta_color="off")
with colCCC:
    st.metric("Min", f"{np.min(valid_spatial):.2f} {var_info['unit']}",
          delta=f"at {min_lat:.2f}°N, {min_lon:.2f}°E",
          delta_color="off")
with colDDD:
    st.metric("Std Dev", f"{np.std(valid_spatial):.2f} {var_info['unit']}")
                
tabA, tabB, tabC, tabD, tabE = st.tabs(["Point Inspector", "Regional Comparison", "Distribution of Parameter Values", " Spatial Gradients", "Bloom Detection Analysis"])
with tabA:
    st.markdown("#### Point Inspector")
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
        st.markdown(f"#### Time Series at ({clicked_lat:.2f}°N, {clicked_lon:.2f}°E)")
        st.caption("Values at the nearest grid point across all variables, for all available time steps.")
        
        all_time_rows = []
        filetered_rows = []
        
        for var_name, info in VARIABLE_INFO.items():
            df_all, df_filtered = get_point_timeseries(var_name, clicked_lat, clicked_lon, time_filter)
            
            for df, rows in [(df_all, all_time_rows), (df_filtered, filetered_rows)]:
                if df is not None and not df.empty:
                    vals = df[var_name].dropna()
                    if len(vals) > 0:
                        rows.append({
                            'Variable': info['name'],
                            'Unit': info['unit'],
                            'Mean': f"{vals.mean():.2f} {info['unit']}",
                            'Max': f"{vals.max():.2f} {info['unit']}",
                            'Min': f"{vals.min():.2f} {info['unit']}",
                            'Std Dev': f"{vals.std():.2f} {info['unit']}",
                            'N Months': f"{len(vals)}"
                        })
        
        tab_filtered, tab_all = st.tabs([f"📅 {time_label}", "📅 All Time"])
        
        with tab_filtered:
            st.caption(f"Stats for the selected period: **{time_label}**")
            if filetered_rows:
                df_filtered_summary = pd.DataFrame(filetered_rows)
                st.dataframe(df_filtered_summary, hide_index=True, width='stretch')
                st.download_button(
                    label="📥 Download Filtered Summary",
                    data=df_filtered_summary.to_csv(index=False),
                    file_name = f"point_summary_{clicked_lat:.2f}N_{clicked_lon:.2f}E_{time_label.replace(' ', '_')}.csv",
                    mime="text/csv",
                    key="download_filtered_summary")
            else:
                st.info(f"No valid data available at this location for the selected time period ({time_label}).")
                
        with tab_all:
            st.caption("Stats for all available time steps (2014-2024)")
                    
            if all_time_rows:
                summary_df = pd.DataFrame(all_time_rows)
                st.dataframe(summary_df, hide_index=True, width='stretch')
                
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download All-Time Point Summary",
                    data=csv,
                    file_name=f"point_summary_{clicked_lat:.2f}N_{clicked_lon:.2f}E_All_Time.csv",
                    mime="text/csv",
                    key="download_summary"
                )
            else:
                st.info("No valid data available at this location for any variable.")
    else:
        st.info("No data point selected yet")

with tabB:

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
    
with tabC:
    text = ("This histogram shows how values are spread across all spatial pixels in the selected area at a given point in time. The x-axis represents the variable's value and the y-axis shows how many pixels have that value - tall bars mean many pixels share a similar reading, while a wide spread indicates high spatial variability across the region.")
    st.markdown("#### Distribution of Parameter Values", help=text)

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

with tabD:
    available_vars = ["Mean", "Max"]
    selected_var = st.selectbox(
        "Select method",
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

with tabE:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Bloom Detection Analysis")
        bloom_threshold = st.slider(
            "Bloom Threshold (mg/m³)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Chlorophyll-a concentration above this value indicates a bloom"
        )
        
        spatial_extent = st.slider(
            "Min. Spatial Extent (%)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.1,
            help="Minimum percentage of area affected to count as bloom event"
        )
        intensity_levels = st.checkbox("Show Intensity Levels", value=True)
        
        if 'moderate_threshold' not in st.session_state:
            st.session_state.moderate_threshold = 3.0
        if 'high_threshold' not in st.session_state: 
            st.session_state.high_threshold = 5.0

        if st.session_state.moderate_threshold < bloom_threshold:
            st.session_state.moderate_threshold = bloom_threshold
        if st.session_state.high_threshold < st.session_state.moderate_threshold:
            st.session_state.high_threshold = st.session_state.moderate_threshold
        
        if intensity_levels:
            col11, col22 = st.columns(2)
            with col11:
                moderate_threshold = st.number_input(
                    "Moderate Bloom (mg/m³)",
                    min_value=bloom_threshold,
                    max_value=10.0,
                    key = "moderate_threshold"
                )
            with col22: 
            
                high_threshold = st.number_input(
                    "High Bloom (mg/m³)",
                    min_value=moderate_threshold,
                    max_value=10.0,
                    key = "high_threshold"
                )
            
        with col2:

            chl_mean = chl_data_r.mean(dim=['latitude', 'longitude'])
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

            st.markdown("##### Bloom Event Summary")
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Bloom Months Detected",
                    value=f"{n_bloom_months}",
                    delta=f"{bloom_percentage:.1f}% of total"
                )
                if n_bloom_months > 0:
                    bloom_values = chl_mean.values[bloom_mask]
                    avg_bloom_intensity = np.mean(bloom_values)
                    st.metric(
                        label="Avg Bloom Intensity",
                        value=f"{avg_bloom_intensity:.2f} mg/m³"
                    )
                else:
                    st.metric(label="Avg Bloom Intensity", value="N/A")

            with col2:

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
    
    tabAA, tabBB, tabCC, tabDD = st.tabs(["Bloom Timeline", "Environmental Conditions Timeline", "Monthly Bloom Frequency", "Seasonal Bloom Distribution"])
    with tabAA:
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
                
    with tabBB:
        env_available = {
        'Temperature': temp_data_r is not None,
        'Wind': wind_data_r is not None,
        'Solar': solar_data_r is not None,
        'Nitrate': no3_data_r is not None,
        'Phosphate': po4_data_r is not None,
        'Rainfall': rain_data_r is not None
    }

        missing_vars = [k for k, v in env_available.items() if not v]
        if missing_vars:
            st.warning(f"⚠️ Missing environmental data: {', '.join(missing_vars)}")
            

        n_subplots = 1 + sum(env_available.values())  

        subplot_titles = ['Chlorophyll-a (Bloom Indicator)']
        if temp_data_r is not None:
            subplot_titles.append('Temperature')
        if wind_data_r is not None:
            subplot_titles.append('Wind Speed')
        if solar_data_r is not None:
            subplot_titles.append('Solar Radiation')
        if no3_data_r is not None:
            subplot_titles.append('Nitrate')
        if po4_data_r is not None:
            subplot_titles.append('Phosphate')
        if rain_data_r is not None:
            subplot_titles.append('Precipitation')

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

        if temp_data_r is not None:
            temp_mean = temp_data_r.mean(dim=['latitude', 'longitude'])
            
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

        if wind_data_r is not None:
            wind_mean = wind_data_r.mean(dim=['latitude', 'longitude'])
            
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

        if solar_data_r is not None:
            solar_mean = solar_data_r.mean(dim=['latitude', 'longitude']) / 1e6  
            
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

        if no3_data_r is not None:
            no3_mean = no3_data_r.mean(dim=['latitude', 'longitude'])
            
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

        if po4_data_r is not None:
            po4_mean = po4_data_r.mean(dim=['latitude', 'longitude'])
            
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
            
        if rain_data_r is not None:
            rain_mean = rain_data_r.mean(dim=['latitude', 'longitude'], skipna=True)
            
            fig_conditions.add_trace(
                go.Scatter(
                    x=rain_mean.time.values,
                    y=rain_mean.values,
                    mode='lines',
                    name='Precipitation',
                    line=dict(color='#1e90ff', width=2),
                    showlegend=False
                ),
                row=row, col=1
            )
            
            fig_conditions.update_yaxes(title_text="mm", row=row, col=1)
            row += 1

        fig_conditions.update_xaxes(title_text="Date", row=n_subplots, col=1)

        fig_conditions.update_layout(
            height=150 * n_subplots, 
            template='plotly_white',
            showlegend=False,
            title_text='Environmental Conditions and Bloom Events Timeline'
        )

        st.plotly_chart(fig_conditions, width='stretch')
    
    with tabCC:
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
                'Total Months': [bloom_by_month[m]['total'] for m in range(1, 13)],
                'Probability': [f"{bloom_by_month[m]['percentage']:.1f}%" for m in range(1, 13)]
            })
            
            st.dataframe(monthly_stats, hide_index=True, width='stretch')
    
    with tabDD:
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
                'Bloom Probability': [f"{seasonal_blooms[s]['percentage']:.1f}%" for s in seasonal_blooms.keys()]
            })
            
            st.markdown("**Seasonal Bloom Statistics**")
            st.dataframe(seasonal_stats, hide_index=True, width='stretch', height=200)
            
            # Peak season
            peak_season = max(seasonal_blooms.items(), key=lambda x: x[1]['count'])[0]
            st.success(f"🌟 **Peak Bloom Season**: {peak_season}")
            
#Clear memory
gc.collect()

#Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
    💡 Tip: Use heatmap for general patterns, contour for detailed gradients. 
    Hover over the map for specific values.
    </small>
</div>
""", unsafe_allow_html=True)

    # with tabEE:

    #     bloom_extent = []
        
    #     for t in range(len(chl_data.time)):
    #         chl_slice = chl_data.isel(time=t)
    #         total_pixels = (~np.isnan(chl_slice.values)).sum()
    #         bloom_pixels = (chl_slice.values > bloom_threshold).sum()
            
    #         if total_pixels > 0:
    #             extent_pct = (bloom_pixels / total_pixels) * 100
    #         else:
    #             extent_pct = 0
            
    #         bloom_extent.append({
    #             'time': chl_data.time.values[t],
    #             'extent_pct': extent_pct,
    #             'bloom_pixels': int(bloom_pixels),
    #             'total_pixels': int(total_pixels)
    #         })
        
    #     df_extent = pd.DataFrame(bloom_extent)
        
    #     fig_extent = go.Figure()
        
    #     fig_extent.add_trace(go.Scatter(
    #         x=df_extent['time'],
    #         y=df_extent['extent_pct'],
    #         mode='lines+markers',
    #         name='Bloom Extent',
    #         line=dict(color='#2ca02c', width=2.5),
    #         marker=dict(size=5),
    #         fill='tozeroy',
    #         fillcolor='rgba(44, 160, 44, 0.2)',
    #         hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>' +
    #                     '<b>Extent</b>: %{y:.1f}%<br>' +
    #                     '<extra></extra>'
    #     ))
        
    #     fig_extent.add_hline(
    #         y=spatial_extent,
    #         line_dash="dash",
    #         line_color="red",
    #         annotation_text=f"Significant extent ({spatial_extent}%)",
    #         annotation_position="right"
    #     )
        
    #     fig_extent.update_layout(
    #         title='Percentage of Coastal Area Affected by Blooms',
    #         xaxis_title='Date',
    #         yaxis_title='Bloom Extent (%)',
    #         height=400,
    #         template='plotly_white'
    #     )
        
    #     st.plotly_chart(fig_extent, width='stretch')
        
    #     col1, col2, col3 = st.columns(3)
        
    #     with col1:
    #         avg_extent = df_extent['extent_pct'].mean()
    #         st.metric("Average Bloom Extent", f"{avg_extent:.1f}%")
        
    #     with col2:
    #         max_extent = df_extent['extent_pct'].max()
    #         max_extent_date = df_extent.loc[df_extent['extent_pct'].idxmax(), 'time']
    #         st.metric(
    #             "Maximum Extent",
    #             f"{max_extent:.1f}%",
    #             delta=str(max_extent_date)[:10]
    #         )
        
    #     with col3:
    #         significant_events = (df_extent['extent_pct'] > spatial_extent).sum()
    #         st.metric(
    #             "Significant Events",
    #             f"{significant_events}",
    #             delta=f">{spatial_extent}% area"
    #         )

        
        
        
# with col2:
#     valid_spatial = data_plot.values[~np.isnan(data_plot.values)]
#     st.markdown("### 📊 Spatial Statistics")
#     st.metric("Mean", f"{np.mean(valid_spatial):.2f} {var_info['unit']}")
#     st.metric("Max", f"{np.max(valid_spatial):.2f} {var_info['unit']}")
#     st.metric("Min", f"{np.min(valid_spatial):.2f} {var_info['unit']}")
#     st.metric("Std Dev", f"{np.std(valid_spatial):.2f} {var_info['unit']}")
#     st.metric("Valid Pixels", f"{len(valid_spatial):,}")


# tab1, tab2, tab3, tab4 = st.tabs(["📍 Hotspot Analysis", "📏 Distribution of Parameter Values", "🗺️ Regional Comparison", "🧭Spatial Gradients"])

# with tab1:
#     st.markdown("## 📍 Hotspot Identification")
    
#     col1, col2 = st.columns([1, 2])
    
#     with col1:
#         threshold_pct = st.slider(
#             "Hotspot Threshold (Percentile)",
#             min_value=50,
#             max_value=99,
#             value=85,
#             step=5,
#             help="Values above this percentile will be highlighted as hotspots. Higher percentiles = more exclusive hotspots."
#         )
        
#         threshold_value = np.nanpercentile(data_plot.values, threshold_pct)
        
#         st.info(f"**Hotspot threshold: {threshold_value:.2f} {var_info['unit']}** (≥{threshold_pct}th percentile)")
        
#         hotspot_mask = data_plot.values >= threshold_value
#         hotspot_count = np.sum(hotspot_mask & ~np.isnan(data_plot.values))
#         total_valid_pixels = np.sum(~np.isnan(data_plot.values))
#         hotspot_percentage = (hotspot_count / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0
        
#         st.metric(
#             label="Hotspot Locations", 
#             value=f"{hotspot_count:,} pixels",
#             delta=f"{hotspot_percentage:.1f}% of area"
#         )
        
#         if hotspot_count > 0:
#             hotspot_values = data_plot.values[hotspot_mask & ~np.isnan(data_plot.values)]
            
#             st.markdown("#### Hotspot Statistics")
#             col_a, col_b = st.columns(2)
#             with col_a:
#                 st.metric("Mean", f"{np.mean(hotspot_values):.2f}")
#                 st.metric("Min", f"{np.min(hotspot_values):.2f}")
#             with col_b:
#                 st.metric("Max", f"{np.max(hotspot_values):.2f}")
#                 st.metric("Std Dev", f"{np.std(hotspot_values):.2f}")
        
#         if hotspot_count > 0:
#             hotspot_lats, hotspot_lons = np.where(hotspot_mask & ~np.isnan(data_plot.values))
#             hotspot_df = pd.DataFrame({
#                 'Latitude': data_plot.latitude.values[hotspot_lats],
#                 'Longitude': data_plot.longitude.values[hotspot_lons],
#                 var_info['name']: data_plot.values[hotspot_mask & ~np.isnan(data_plot.values)]
#             })
            
#             csv = hotspot_df.to_csv(index=False)
#             st.download_button(
#                 label="📥 Download Hotspot Coordinates",
#                 data=csv,
#                 file_name=f"hotspots_{selected_var}_{threshold_pct}pct.csv",
#                 mime="text/csv",
#                 key="download_hotspots"
#             )
    
#     with col2:
#         fig_hotspot = go.Figure()
        
#         fig_hotspot.add_trace(go.Heatmap(
#             x=data_plot.longitude.values,
#             y=data_plot.latitude.values,
#             z=data_plot.values,
#             colorscale='Greys',
#             opacity=0.3,
#             showscale=False,
#             hoverinfo='skip',
#             name='Background'
#         ))
        
#         hotspot_data = np.where(
#             (data_plot.values >= threshold_value) & ~np.isnan(data_plot.values),
#             data_plot.values,
#             np.nan
#         )
        
#         fig_hotspot.add_trace(go.Heatmap(
#             x=data_plot.longitude.values,
#             y=data_plot.latitude.values,
#             z=hotspot_data,
#             colorscale=[
#                 [0, '#fee5d9'],
#                 [0.33, '#fcae91'],
#                 [0.66, '#fb6a4a'],
#                 [1, '#cb181d']
#             ],
#             colorbar=dict(
#             title=dict(
#                 text=f"{var_info['unit']}",
#                 side='right' 
#             )
#         ),
#             hovertemplate='<b>Lat</b>: %{y:.2f}°N<br>' +
#                           '<b>Lon</b>: %{x:.2f}°E<br>' +
#                           f'<b>{var_info["name"]}</b>: %{{z:.2f}} {var_info["unit"]}<br>' +
#                           '<extra></extra>',
#             name='Hotspots'
#         ))
        
#         fig_hotspot.update_layout(
#             title=f'Hotspot Locations (≥{threshold_pct}th percentile)',
#             xaxis_title='Longitude (°E)',
#             yaxis_title='Latitude (°N)',
#             height=500,
#             template='plotly_white',
#             showlegend=False
#         )
        
#         st.plotly_chart(fig_hotspot, width='stretch')
#         if hotspot_count > 0:
#             hotspot_indices = np.where(hotspot_mask & ~np.isnan(data_plot.values))
#             hotspot_lats = hotspot_indices[0]
#             hotspot_lons = hotspot_indices[1]
            
#             hotspot_lat_values = data_plot.latitude.values[hotspot_lats]
#             hotspot_lon_values = data_plot.longitude.values[hotspot_lons]
            
#             center_lat = hotspot_lat_values.mean()
#             center_lon = hotspot_lon_values.mean()
#             lat_range = np.ptp(hotspot_lat_values)  
#             lon_range = np.ptp(hotspot_lon_values)  
            
#             if lat_range < 2 and lon_range < 2:
#                 st.success("🎯 Hotspots are **highly concentrated** in one region")
#             elif lat_range < 5 and lon_range < 5:
#                 st.info("📍 Hotspots are **moderately dispersed**")
#             else:
#                 st.warning("🌍 Hotspots are **widely scattered** across the region")

# with tab4:

#     st.markdown("## 🧭 Spatial Gradients")

#     st.markdown("### Select Data")
#     available_vars = ["Mean", "Max"]
#     selected_var = st.selectbox(
#         "Select Variable",
#         options=available_vars,
#         index=0
#     )

#     if selected_var == "Mean":
#         lat_profile = data_plot.mean(dim='longitude')
#         lon_profile = data_plot.mean(dim='latitude')
#     else:
#         lat_profile = data_plot.max(dim='longitude')
#         lon_profile = data_plot.max(dim='latitude')
        
#     col1, col2 = st.columns(2)

#     with col1:
#         fig_lat = go.Figure()
#         fig_lat.add_trace(go.Scatter(
#             x=lat_profile.latitude.values,  
#             y=lat_profile.values,           
#             mode='lines+markers',
#             line=dict(color=var_info['color'], width=3),
#             marker=dict(size=6),
#             fill='tozeroy', 
#             fillcolor=f'rgba({int(var_info["color"][1:3], 16)}, {int(var_info["color"][3:5], 16)}, {int(var_info["color"][5:7], 16)}, 0.2)',
#             hovertemplate='<b>Latitude</b>: %{x:.2f}°N<br>' +
#                         f'<b>{var_info["name"]}</b>: %{{y:.2f}} {var_info["unit"]}<br>' +
#                         '<extra></extra>'
#         ))

#         # Add a vrect for each region's latitude range
#         for region_name, bounds in BALTIC_REGIONS.items():
#             fig_lat.add_vrect(
#                 x0=bounds['min_lat'],
#                 x1=bounds['max_lat'],
#                 annotation_text=region_name,
#                 annotation_position="top left",
#                 annotation=dict(
#                     textangle=-90,
#                     font=dict(size=9),
#                 ),
#                 fillcolor=bounds['color'],
#                 opacity=0.1,
#                 line_width=1,
#                 line_color=bounds['color'],
#             )
        
#         fig_lat.update_layout(
#             title=f'Latitudinal Profile (South to North) — {selected_var}',
#             xaxis_title='Latitude (°N)', 
#             yaxis_title=f'{var_info["name"]} ({var_info["unit"]})',  
#             height=400,
#             template='plotly_white',
#             showlegend=False
#         )
            
#         st.plotly_chart(fig_lat, width='stretch')

#     with col2:
#         fig_lon = go.Figure()
#         fig_lon.add_trace(go.Scatter(
#             x=lon_profile.longitude.values,
#             y=lon_profile.values,
#             mode='lines+markers',
#             line=dict(color=var_info['color'], width=3),
#             marker=dict(size=6),
#             fill='tozeroy',
#             fillcolor=f'rgba({int(var_info["color"][1:3], 16)}, {int(var_info["color"][3:5], 16)}, {int(var_info["color"][5:7], 16)}, 0.2)',
#             hovertemplate='<b>Longitude</b>: %{x:.2f}°E<br>' +
#                         f'<b>{var_info["name"]}</b>: %{{y:.2f}} {var_info["unit"]}<br>' +
#                         '<extra></extra>'
#         ))

#         # Add a vrect for each region's longitude range
#         for region_name, bounds in BALTIC_REGIONS.items():
#             fig_lon.add_vrect(
#                 x0=bounds['min_lon'],
#                 x1=bounds['max_lon'],
#                 annotation_text=region_name,
#                 annotation_position="top left",
#                 annotation=dict(
#                     textangle=-90,
#                     font=dict(size=9),
#                 ),
#                 fillcolor=bounds['color'],
#                 opacity=0.1,
#                 line_width=1,
#                 line_color=bounds['color'],
#             )
        
#         fig_lon.update_layout(
#             title=f'Longitudinal Profile (West to East) — {selected_var}',
#             xaxis_title='Longitude (°E)',
#             yaxis_title=f'{var_info["name"]} ({var_info["unit"]})',
#             height=400,
#             template='plotly_white',
#             showlegend=False
#         )
        
#         st.plotly_chart(fig_lon, width='stretch')
        
#     max_lat_idx = lat_profile.argmax()
#     peak_lat = float(lat_profile.latitude.values[max_lat_idx])
#     peak_max_lat = float(lat_profile.max())

#     max_lon_idx = lon_profile.argmax()
#     peak_lon = float(lon_profile.longitude.values[max_lon_idx])
#     peak_max_lon = float(lon_profile.max())

#     overall_val = float(data_plot.mean()) if selected_var == "Mean" else float(data_plot.max())

#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric(
#             label=f"Overall {selected_var}",
#             value=f"{overall_val:.2f} {var_info['unit']}",
#         )

#     with col2:
#         st.metric(
#             label=f"Peak Latitude ({selected_var})",
#             value=f"{peak_max_lat:.2f} {var_info['unit']}",
#             delta=f"at {peak_lat:.2f}°N"
#         )

#     with col3:
#         st.metric(
#             label=f"Peak Longitude ({selected_var})",
#             value=f"{peak_max_lon:.2f} {var_info['unit']}",
#             delta=f"at {peak_lon:.2f}°E"
#         )
        


# with tab2:
#     st.markdown("## 🌡️ Bloom-Favorable Environmental Conditions")
#     with st.expander("🔍 What conditions promote algae blooms?"):
#         st.info("""
#         Algae blooms typically occur when multiple favorable conditions coincide:
#         - **Chlorophyll**: > threshold (bloom indicator)
#         - **Temperature**: > 15°C (warmer water promotes growth)
#         - **Solar Radiation**: High (more light for photosynthesis)
#         - **Nutrients**: Sufficient nitrate and phosphate (growth nutrients)
#         - **Wind**: Low to moderate (< 7 m/s, allows water stratification)
#         """)


    
#     favorable_conditions = {}

#     if temp_data is not None:
#         temp_mean = temp_data.mean(dim=['latitude', 'longitude'])
#         temp_favorable = temp_mean > 15
#         favorable_conditions['Temperature (>15°C)'] = temp_favorable
#     else:
#         temp_favorable = None

#     if wind_data is not None:
#         wind_mean = wind_data.mean(dim=['latitude', 'longitude'])
#         wind_favorable = wind_mean < 7
#         favorable_conditions['Wind (<7 m/s)'] = wind_favorable
#     else:
#         wind_favorable = None

#     if solar_data is not None:
#         solar_mean = solar_data.mean(dim=['latitude', 'longitude'])
#         solar_median = float(solar_mean.median())
#         solar_favorable = solar_mean > solar_median
#         favorable_conditions['Solar (>median)'] = solar_favorable
#     else:
#         solar_favorable = None

#     if len(favorable_conditions) > 0:
#         all_times = [chl_mean.time.values]
#         for condition in favorable_conditions.values():
#             all_times.append(condition.time.values)
        
#         common_times = all_times[0]
#         for times in all_times[1:]:
#             common_times = np.intersect1d(common_times, times)
        
#         if len(common_times) > 0:
#             bloom_aligned = chl_mean.sel(time=common_times) > bloom_threshold
            
#             conditions_aligned = {}
#             for name, condition in favorable_conditions.items():
#                 conditions_aligned[name] = condition.sel(time=common_times)
            
#             n_favorable = sum(conditions_aligned.values())
            
#             all_favorable = bloom_aligned.copy()
#             for condition in conditions_aligned.values():
#                 all_favorable = all_favorable & condition
            
#             n_all_favorable = all_favorable.sum().values
            
#             st.markdown("### 📊 Favorable Condition Frequency")
            
#             cols = st.columns(len(favorable_conditions) + 1)
            
#             for idx, (name, condition) in enumerate(favorable_conditions.items()):
#                 with cols[idx]:
#                     n_times = conditions_aligned[name].sum().values
#                     pct = (n_times / len(common_times)) * 100
#                     st.metric(
#                         label=name,
#                         value=f"{n_times} months",
#                         delta=f"{pct:.1f}%"
#                     )
            
#             with cols[-1]:
#                 pct_all = (n_all_favorable / len(common_times)) * 100
#                 st.metric(
#                     label="All Conditions Met",
#                     value=f"{n_all_favorable} months",
#                     delta=f"{pct_all:.1f}%"
#                 )
            
#             bloom_with_favorable = bloom_aligned & all_favorable
#             n_bloom_with_favorable = bloom_with_favorable.sum().values
            
#             if n_bloom_months > 0:
#                 coincidence_rate = (n_bloom_with_favorable / n_bloom_months) * 100
                
#                 st.markdown("### 🎯 Bloom-Condition Coincidence")
                
#                 col1, col2, col3 = st.columns(3)
                
#                 with col1:
#                     st.metric(
#                         label="Blooms with Favorable Conditions",
#                         value=f"{n_bloom_with_favorable} / {n_bloom_months}",
#                         delta=f"{coincidence_rate:.1f}%"
#                     )
                
#                 with col2:
#                     blooms_without = n_bloom_months - n_bloom_with_favorable
#                     st.metric(
#                         label="Blooms w/o Favorable Conditions",
#                         value=f"{blooms_without}",
#                         delta="Unexpected blooms"
#                     )
                
#                 with col3:
#                     favorable_no_bloom = all_favorable & ~bloom_aligned
#                     n_favorable_no_bloom = favorable_no_bloom.sum().values
#                     st.metric(
#                         label="Favorable w/o Blooms",
#                         value=f"{n_favorable_no_bloom}",
#                         delta="Missed opportunities"
#                     )
                
#                 if coincidence_rate > 70:
#                     st.success(f"✅ High coincidence rate ({coincidence_rate:.1f}%)! Environmental conditions are strong predictors of blooms.")
#                 elif coincidence_rate > 40:
#                     st.info(f"ℹ️ Moderate coincidence rate ({coincidence_rate:.1f}%). Environmental conditions partially explain blooms.")
#                 else:
#                     st.warning(f"⚠️ Low coincidence rate ({coincidence_rate:.1f}%). Other factors may be important.")

# st.markdown("---")

# with tab2:
#     st.markdown("### Seasonal Bloom Distribution")
    

# with tab3:

