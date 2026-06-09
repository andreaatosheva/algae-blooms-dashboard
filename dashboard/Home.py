import streamlit as st
import sys
from pathlib import Path
import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_TITLE, APP_SUBTITLE, PAGE_ICON


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout="wide"
)

def main():
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        
        .sub-header {
            font-size: 1.5rem;
            color: #555;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .metric-card{
            background-color: #f0f2f6;
            border-radius: 0.5rem;
            padding: 1rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
""", unsafe_allow_html=True)
    
    
    st.markdown(f'<div class="main-header">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{APP_SUBTITLE}</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('---')
        mem = psutil.virtual_memory()
        st.caption(f"🧠 Memory: {mem.percent}% ({mem.used / 1024**2:.0f} MB)")
        st.markdown('### 📚 Navigation')
        st.markdown("")
        st.markdown("""
        Use the pages in the sidebar to explore:
        - **Home**: Overview and quick stats
        - **Data Explorer**: Interactive data exploration
        - **Spatial Analysis**: Maps and spatial patterns
        - **Time Series**: Temporal trends
        - **Environmental Factors**: Multi-variable analysis
        - **Bloom Detection**: Algae bloom events
        - **Statistical Analysis**: Correlations and statistics
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        This dashboard analyzes algae bloom patterns in the Baltic Sea 
        coastal waters from 2014-2024 using satellite data.
        """)
        
    # Main content
    st.markdown("## 🎯 Welcome to the Baltic Sea Algae Bloom Dashboard")
    
    st.markdown("""
    This interactive dashboard provides comprehensive analysis of algae bloom patterns 
    in the Baltic Sea coastal regions. Explore environmental factors, identify bloom events, 
    and analyze temporal and spatial patterns.
    
    ### 🔬 Available Data:
    - **Chlorophyll-a**: Surface concentration (algae proxy)
    - **Temperature**: Sea surface temperature
    - **Nutrients**: Nitrate, Phosphate, Ammonia
    - **Wind**: Wind speed at 10m
    - **Solar Radiation**: Surface downwelling radiation
    - **Precipitation**: Rainfall data
    
    ### 📊 Analysis Features:
    - Interactive time series visualization
    - Spatial mapping with zoom and pan
    - Bloom event detection and tracking
    - Multi-variable correlation analysis
    - Seasonal and monthly climatologies
    - Statistical summaries and trends
    """)
    
    # Quick stats
    st.markdown("---")
    st.markdown("## 📈 Quick Statistics")
    
    try:
        from utils.data_loader import get_data_summary
        
        with st.spinner("Loading data summaries..."):
            summary_stats = get_data_summary()
            
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label = "Datasets",
                value = summary_stats['total_datasets']
            )
            
        with col2:
            if summary_stats['date_range']:
                st.metric(
                    label = "Date Range",
                    value = f"{summary_stats['date_range'][0][:4]}  to  {summary_stats['date_range'][1][:4]}"
                )
        
        with col3:
            st.metric(
                label = "Variables",
                value = "8"
            )
    except Exception as e:
        st.error(f"Error loading data summaries: {e}")
        
        
        
    st.markdown("---")
    st.markdown("## 🚀 Getting Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 1️⃣ Explore Data
        Navigate to **Data Explorer** to:
        - View time series for different variables
        - Select custom date ranges
        - Compare multiple variables
        """)
    
    with col2:
        st.markdown("""
        ### 2️⃣ Analyze Patterns
        Use **Spatial Analysis** and **Time Series** to:
        - Visualize spatial distributions
        - Identify bloom hotspots
        - Analyze seasonal patterns
        - Detect trends over time
        """)
    
    
    # ── Data Sources ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🛰️ Data Sources")
    st.markdown(
        "The datasets used in this dashboard are sourced from the "
        "**Copernicus** Earth observation programme and **NASA GES DISC**, "
        "and are freely accessible via the portals below."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="source-card">
            <strong>🌊 Copernicus Marine Service</strong><br>
            Provides ocean variables including chlorophyll-a, sea surface
            temperature, and nutrient concentrations derived from satellite
            observations.<br><br>
            <a href="https://data.marine.copernicus.eu" target="_blank">
                🔗 data.marine.copernicus.eu
            </a>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="source-card">
            <strong>🌤️ Copernicus Climate Data Store (CDS)</strong><br>
            Provides atmospheric and climate variables including wind speed and
            solar radiation from ERA5 and related
            reanalysis products.<br><br>
            <a href="https://cds.climate.copernicus.eu" target="_blank">
                🔗 cds.climate.copernicus.eu
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="source-card">
            <strong>🛰️ NASA GES DISC — IMERG</strong><br>
            Provides high-resolution precipitation data (0.1°) from the 
            GPM IMERG Final Run product, combining multi-satellite 
            microwave estimates with gauge calibration.<br><br>
            <a href="https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGM_07/summary?keywords=%22IMERG%20final%22" target="_blank">
                🔗 disc.gsfc.nasa.gov
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>
        Baltic Sea Algae Bloom Dashboard | Data: 2014-2024 | 
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()