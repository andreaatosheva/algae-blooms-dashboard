import numpy as np
from scipy.interpolate import griddata
import streamlit as st
import plotly.graph_objects as go


def create_smooth_interpolated_map(data_plot, var_info, time_label, map_style='smooth'):
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
    values = data_plot.values
    
    # Create mesh
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    
    # Flatten
    lon_flat = lon_mesh.flatten()
    lat_flat = lat_mesh.flatten()
    val_flat = values.flatten()
    
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
        
        # Create smooth heatmap
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            x=lon_fine,
            y=lat_fine,
            z=val_interp,
            colorscale=var_info['cmap'],
            colorbar=dict(
                title=dict(text=f"{var_info['name']}<br>({var_info['unit']})", side='right'),
                thickness=20,
                len=0.7
            ),
            hovertemplate='<b>Lat</b>: %{y:.2f}°N<br>' +
                          '<b>Lon</b>: %{x:.2f}°E<br>' +
                          f'<b>{var_info["name"]}</b>: %{{z:.2f}} {var_info["unit"]}<br>' +
                          '<extra></extra>',
            zsmooth='best'
        ))
        
    else:  # Regular heatmap
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            x=lons,
            y=lats,
            z=values,
            colorscale=var_info['cmap'],
            colorbar=dict(
                title=dict(text=f"{var_info['name']}<br>({var_info['unit']})", side='right'),
                thickness=20,
                len=0.7
            ),
            zsmooth='best'
        ))
    
    fig.update_layout(
        title=f'{var_info["name"]} - {time_label}',
        xaxis_title='Longitude (°E)',
        yaxis_title='Latitude (°N)',
        height=600,
        template='plotly_white',
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(constrain='domain')
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