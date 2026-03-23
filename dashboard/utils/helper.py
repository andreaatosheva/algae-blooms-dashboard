import streamlit as st
import psutil
import plotly.graph_objects as go
import pandas as pd
from utils.data_loader import load_variable_data
from config import SEASONS, VARIABLE_INFO

def show_memory_usage():
    mem = psutil.virtual_memory()
    with st.sidebar:
        st.caption(f"🧠 Memory: {mem.percent}% ({mem.used / 1024**2:.0f} MB)")
        

        
        
def make_bbox_trace(name, region):
    """Create a filled rectangle trace for a region bbox."""
    min_lat = region["min_lat"]
    max_lat = region["max_lat"]
    min_lon = region["min_lon"]
    max_lon = region["max_lon"]
    color   = region["color"]

    # Convert hex to rgba for fill
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    # Close the rectangle: SW -> NW -> NE -> SE -> SW
    lons = [min_lon, min_lon, max_lon, max_lon, min_lon]
    lats = [min_lat, max_lat, max_lat, min_lat, min_lat]

    return go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='lines',
        fill='toself',
        fillcolor=f'rgba({r},{g},{b},0.25)',
        line=dict(color=color, width=2),
        name=name,
        hovertemplate=(
            f"<b>{name}</b><br>"
            f"Lat: {min_lat}°N – {max_lat}°N<br>"
            f"Lon: {min_lon}°E – {max_lon}°E<br>"
            "<extra></extra>"
        )
    )


@st.cache_data
def get_point_timeseries(var_name, lat, lon, time_filter):
    """
    Extract time series for a single lat/lon point, optionally filtered to a time period.
    """
    
    data = load_variable_data(var_name)
    if data is None:
        return None, None

    
    try:
        ts = data.sel(latitude=lat, longitude=lon, method='nearest')
        df_all = pd.DataFrame({
            'Date': ts.time.values,
            var_name: ts.values
        })
        
        mode = time_filter.get('mode') if time_filter else 'Annual Average'
        
        if mode == 'Single Month':
            year = time_filter.get('year')
            month = time_filter.get('month')
            df_filtered = df_all[
                (pd.to_datetime(df_all['Date']).dt.year == year) &
                (pd.to_datetime(df_all['Date']).dt.month == month)
            ]
            
        elif mode == 'Seasonal Average':
            months = time_filter.get('months')
            year = time_filter.get('year')
            mask = pd.to_datetime(df_all['Date']).dt.month.isin(months)
            if year is not None:
                mask &= (pd.to_datetime(df_all['Date']).dt.year == year)
            df_filtered = df_all[mask]
            
        elif mode == 'Annual Average':
            year = time_filter.get('year')
            df_filtered = df_all[pd.to_datetime(df_all['Date']).dt.year == year]
            
        elif mode == 'All Time':
            df_filtered = df_all.copy()
            
        return df_all, df_filtered
    
    
    except Exception as e:
        st.error(f"Error extracting time series for {var_name} at ({lat}, {lon}): {e}")
        return None, None
            
            
        
    
    