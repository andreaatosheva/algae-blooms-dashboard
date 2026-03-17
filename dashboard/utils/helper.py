import streamlit as st
import psutil
import plotly.graph_objects as go

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