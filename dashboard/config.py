from pathlib import Path
from typing import Dict, Optional
# Paths
BASE_DIR = Path(__file__).parent.parent.resolve()
print(f"Base directory: {BASE_DIR}")
DATA_DIR = BASE_DIR / 'data'
PLOTS_DIR = BASE_DIR / 'exploration_plots'

# Data file paths
DATA_PATHS = {
    'chl': DATA_DIR / 'chl' / 'processed_data' / 'chl_full_combined.nc',
    'solar': DATA_DIR / 'solar' / 'processed_data' / 'solar_full_combined.nc',
    'temperature': DATA_DIR / 'temperature' / 'processed_data' / 'temperature_full_combined.nc',
    'wind': DATA_DIR / 'wind' / 'processed_data' / 'wind_full_combined.nc',
    'rain': DATA_DIR / 'rain' / 'processed_data' / 'rain_full_combined.nc',
    'nutrients': DATA_DIR / 'nutrients' / 'processed_data' / 'nutrients_full_combined.nc'
}

# Print for debugging (remove later)
print("\nDEBUG - Checking file existence:")
for var, path in DATA_PATHS.items():
    print(f"  {var}: {path.exists()} - {path}")

# Variable metadata
VARIABLE_INFO = {
    'chlorophyll': {
        'name': 'Chlorophyll-a',
        'unit': 'mg/m³',
        'color': '#2ca02c',
        'cmap': 'YlGnBu',
        'description': 'Surface chlorophyll-a concentration (0-10m average)'
    },
    'temperature': {
        'name': 'Sea Surface Temperature',
        'unit': '°C',
        'color': '#d62728',
        'cmap': 'RdYlBu_r',
        'description': 'Sea surface temperature'
    },
    'nitrate': {
        'name': 'Nitrate',
        'unit': 'mmol/m³',
        'color': '#ff7f0e',
        'cmap': 'Reds',
        'description': 'Nitrate concentration (0-10m average)'
    },
    'phosphate': {
        'name': 'Phosphate',
        'unit': 'mmol/m³',
        'color': '#9467bd',
        'cmap': 'Purples',
        'description': 'Phosphate concentration (0-10m average)'
    },
    'ammonia': {
        'name': 'Ammonia',
        'unit': 'mmol/m³',
        'color': '#8c564b',
        'cmap': 'Oranges',
        'description': 'Ammonia concentration (0-10m average)'
    },
    'wind_speed': {
        'name': 'Wind Speed',
        'unit': 'm/s',
        'color': '#1f77b4',
        'cmap': 'viridis',
        'description': 'Wind speed at 10m'
    },
    'solar_radiation': {
        'name': 'Solar Radiation',
        'unit': 'MJ/m²',
        'color': '#ff8c00',
        'cmap': 'YlOrRd',
        'description': 'Surface solar radiation downwards'
    },
    'rainfall': {
        'name': 'Precipitation',
        'unit': 'mm',
        'color': '#1e90ff',
        'cmap': 'Blues',
        'description': 'Total precipitation'
    }
}

# Map settings
MAP_BOUNDS = {
    'lat_min': 53.0,
    'lat_max': 67.0,
    'lon_min': 9.0,
    'lon_max': 31.0
}

# Bloom thresholds
BLOOM_THRESHOLDS = {
    'low': 5.0,      # mg/m³
    'moderate': 10.0,
    'high': 20.0
}

# Seasons
SEASONS = {
    'Winter (DJF)': [12, 1, 2],
    'Spring (MAM)': [3, 4, 5],
    'Summer (JJA)': [6, 7, 8],
    'Autumn (SON)': [9, 10, 11]
}

# Color schemes
SEASON_COLORS = {
    'Winter (DJF)': '#3498DB',
    'Spring (MAM)': '#2ECC71',
    'Summer (JJA)': '#E74C3C',
    'Autumn (SON)': '#F39C12'
}

# App settings
APP_TITLE = "Baltic Sea Algae Bloom Dashboard"
APP_SUBTITLE = "Coastal Water Analysis (2014-2024)"
PAGE_ICON = "🌊"