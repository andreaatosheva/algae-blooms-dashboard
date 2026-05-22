import numpy as np
from scipy.interpolate import griddata
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def create_smooth_interpolated_map(data_plot, var_info, time_label, map_style='smooth', log_scale=False):
    """
    Create smooth map with interpolation for low-resolution data
    
    Parameters:
    -----------
    data_plot : xr.DataArray
        Data to plot
    var_info : dict
        Variable information from VARIABLE_INFO
    time_label : str
        Time period label
    map_style : str
        'smooth' for interpolated scatter, 'heatmap' for regular heatmap
    """
    # Get data
    lons = data_plot.longitude.values
    lats = data_plot.latitude.values
    values_original = data_plot.values.copy()
    
    if log_scale:
        values_display = np.log1p(values_original)  # log(1 + x) to handle zeros and small values
        colorbar_title = f"{var_info['name']}<br>(log scale, {var_info['unit']})"
    
    else:
        values_display = values_original
        colorbar_title = f"{var_info['name']}<br>({var_info['unit']})"
    
    
    # Create mesh
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    
    # Flatten
    lon_flat = lon_mesh.flatten()
    lat_flat = lat_mesh.flatten()
    val_flat = values_display.flatten()
    val_original_flat = values_original.flatten()
    
    # Remove NaN
    valid_mask = ~np.isnan(val_flat)
    lon_valid = lon_flat[valid_mask]
    lat_valid = lat_flat[valid_mask]
    val_valid = val_flat[valid_mask]
    
    if len(val_valid) < 10:
        st.error("Not enough valid data points to create map.")
        return None
    
    if map_style == 'smooth':
        # Create fine grid for interpolation
        resolution_factor = 5  # 5x finer resolution
        lon_fine = np.linspace(lons.min(), lons.max(), len(lons) * resolution_factor)
        lat_fine = np.linspace(lats.min(), lats.max(), len(lats) * resolution_factor)
        lon_fine_mesh, lat_fine_mesh = np.meshgrid(lon_fine, lat_fine)
        
        # Interpolate
        val_interp = griddata(
            (lon_valid, lat_valid),
            val_valid,
            (lon_fine_mesh, lat_fine_mesh),
            method='cubic',  # Cubic for smoothest result
            fill_value=np.nan
        )
        
        val_original_interp = griddata(
            (lon_flat[valid_mask], lat_flat[valid_mask]),
            val_original_flat[valid_mask],
            (lon_fine_mesh, lat_fine_mesh),
            method='cubic',
            fill_value=np.nan
        )
        
        # Create smooth heatmap
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            x=lon_fine,
            y=lat_fine,
            z=val_interp,
            customdata=val_original_interp,  # Use original values for hover
            colorscale=var_info['cmap'],
            colorbar=dict(
                title=dict(text=f"{var_info['name']}<br>({var_info['unit']})", side='right'),
                thickness=20,
                len=0.7
            ),
            hovertemplate='<b>Lat</b>: %{y:.2f}°N<br>' +
                          '<b>Lon</b>: %{x:.2f}°E<br>' +
                          f'<b>{var_info["name"]}</b>: %{{customdata:.2f}} {var_info["unit"]}<br>' +
                          '<extra></extra>',
            zsmooth='best'
        ))
        
    else:  # Regular heatmap
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            x=lons,
            y=lats,
            z=values_display,
            customdata=values_original,  # Use original values for hover
            colorscale=var_info['cmap'],
            colorbar=dict(
                title=dict(text=f"{var_info['name']}<br>({var_info['unit']})", side='right'),
                thickness=20,
                len=0.7
            ),
            hovertemplate='<b>Lat</b>: %{y:.2f}°N<br>' +
                          '<b>Lon</b>: %{x:.2f}°E<br>' +
                          f'<b>{var_info["name"]}</b>: %{{customdata:.2f}} {var_info["unit"]}<br>' +
                          '<extra></extra>',
            zsmooth='best'
        ))
    
    fig.update_layout(
        title=f'{var_info["name"]} - {time_label}',
        xaxis_title='Longitude (°E)',
        yaxis_title='Latitude (°N)',
        height=550,
        template='plotly_white',
        xaxis=dict(range=[lons.min(), lons.max()]),
        yaxis=dict(range=[lats.min(), lats.max()])
    )
    
    return fig

def create_scatter_bubble_map(data_plot, var_info, time_label):
    """
    Create bubble/scatter map - good for sparse data
    """
    # Get valid data points
    lons = data_plot.longitude.values
    lats = data_plot.latitude.values
    values = data_plot.values
    
    # Create mesh and flatten
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    lon_flat = lon_mesh.flatten()
    lat_flat = lat_mesh.flatten()
    val_flat = values.flatten()
    
    # Remove NaN
    valid_mask = ~np.isnan(val_flat)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scattergeo(
        lon=lon_flat[valid_mask],
        lat=lat_flat[valid_mask],
        mode='markers',
        marker=dict(
            size=15,  # Larger markers
            color=val_flat[valid_mask],
            colorscale=var_info['cmap'],
            cmin=np.nanpercentile(values, 2),
            cmax=np.nanpercentile(values, 98),
            colorbar=dict(
                title=f"{var_info['name']}<br>({var_info['unit']})",
                thickness=20,
                len=0.7
            ),
            line=dict(width=0.5, color='white'),
            opacity=0.8
        ),
        hovertemplate='<b>Lat</b>: %{lat:.2f}°N<br>' +
                      '<b>Lon</b>: %{lon:.2f}°E<br>' +
                      f'<b>{var_info["name"]}</b>: %{{marker.color:.2f}} {var_info["unit"]}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_geos(
        scope='europe',
        projection_type='mercator',
        showcountries=True,
        countrycolor='lightgray',
        showcoastlines=True,
        coastlinecolor='black',
        showland=True,
        landcolor='#f0f0f0',
        center=dict(lat=60, lon=20),
        lataxis_range=[53, 67],
        lonaxis_range=[9, 31]
    )
    
    fig.update_layout(
        title=f'{var_info["name"]} - {time_label}',
        height=600,
        template='plotly_white'
    )
    
    return fig



def render_date_controls(
    all_dates: pd.DatetimeIndex,
    fallback_dates: pd.DatetimeIndex,
    key_prefix: str,
) -> pd.Timestamp:
    """
    Render the date-filter radio + date selectbox that appear in every tab.

    Parameters
    ----------
    all_dates       : Full date range (2014-2024).
    fallback_dates  : Test-set dates shown when the user picks "Test set only".
    key_prefix      : Unique string prefix for Streamlit widget keys.

    Returns
    -------
    The selected pd.Timestamp.
    """
    date_filter = st.radio(
        "**Date Range**",
        options=["All dates (2014-2024)", "Test set only (unseen by model)"],
        help=(
            "Test set dates were never seen during training — metrics on these "
            "dates are the most reliable indicator of real model performance"
        ),
        key=f"{key_prefix}_date_filter",
    )
    selected_dates = all_dates if date_filter == "All dates (2014-2024)" else fallback_dates
    
    return st.selectbox(
        "Select a date to view predictions",
        options=selected_dates,
        format_func=lambda x: x.strftime("%Y-%m-%d"),
        index=0,
        key=f"{key_prefix}_date_selectbox",
    )
    

def render_month_year_controls(
    all_dates: pd.DatetimeIndex,
    key_prefix: str,
) -> tuple[int, int]:
    """
    Render the year + month selector columns used in monthly forecast expanders.

    Returns
    -------
    (selected_year, selected_month) as ints.
    """
    col_y, col_m = st.columns(2)
    with col_y:
        selected_year = st.selectbox(
            "Select Year",
            options=sorted(all_dates.year.unique()),
            index=0,
            key=f"{key_prefix}_year",
        )
    with col_m:
        selected_month = st.selectbox(
            "Select Month",
            options=list(range(1, 13)),
            format_func=lambda x: pd.Timestamp(2000, x, 1).strftime("%B"),
            index=0,
            key=f"{key_prefix}_month",
        )
    return selected_year, selected_month

