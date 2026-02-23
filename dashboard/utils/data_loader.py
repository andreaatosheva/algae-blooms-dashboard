import streamlit as st
import xarray as xr
from pathlib import Path
from typing import Dict, Optional
import logging
from config import DATA_PATHS
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


logger = logging.getLogger(__name__)

def check_data_files():
    """
    Check which data files exist
    Returns dict of {variable: exists (bool)}
    """
    status = {}
    for var_name, file_path in DATA_PATHS.items():
        exists = file_path.exists() if file_path else False
        status[var_name] = exists
        if not exists:
            logger.warning(f"Data file missing: {var_name} at {file_path}")
    return status

@st.cache_data(ttl=3600)
def load_dataset(variable: str) -> Optional[xr.Dataset]:
    var_mapping_paths = {
        'chlorophyll': 'chl',
        'temperature': 'temperature',
        'nitrate': 'nutrients',
        'phosphate': 'nutrients',
        'ammonia': 'nutrients',
        'wind_speed': 'wind',
        'solar_radiation': 'solar',
        'rainfall': 'rain'
    }
    try:
        path_var = var_mapping_paths.get(variable, variable)
        file_path = DATA_PATHS.get(path_var)
        if file_path is None or not file_path.exists():
            logger.error(f"Data file for variable '{variable}' not found at path: {file_path}")
            return None
        
        ds = xr.open_dataset(file_path)
        logger.info(f"Successfully loaded dataset for variable '{variable}' from {file_path}")
        return ds
    
    except Exception as e:
        logger.error(f"Error loading dataset for variable '{variable}': {e}")
        st.error(f"Failed to load data for {variable}. Please check the logs for details.")
        return None
    

@st.cache_data(ttl=3600)
def load_all_datasets() -> Dict[str, Optional[xr.Dataset]]:
    datasets = {}
    
    with st.spinner("Loading datasets..."):
        for var_name in DATA_PATHS.keys():
            ds = load_dataset(var_name)
            if ds is not None:
                datasets[var_name] = ds
    return datasets

@st.cache_data(ttl=3600)
def get_data_summary() -> Dict:
    datasets = load_all_datasets()
    
    summary = {
        'total_datasets': len(datasets),
        'variables': list(datasets.keys()),
        'date_range': None,
        'spatial_coverage': None,
        'data_size_mb': 0
    }
    
    if 'chl' in datasets:
        chl = datasets['chl']
        summary['date_range'] = (str(chl.time.min().values)[:10], str(chl.time.max().values)[:10])
        summary['spatial_coverage'] = {
            'lat': (float(chl.latitude.min()), float(chl.latitude.max())),
            'lon': (float(chl.longitude.min()), float(chl.longitude.max()))
        }
        
    for ds in datasets.values():
        summary['data_size_mb'] += ds.nbytes / (1024 * 1024)
        
    return summary

def get_variable_data(dataset: xr.Dataset, variable_name: str) -> Optional[xr.DataArray]:
    var_mapping = {
        'chlorophyll': 'chlorophyll',      
        'temperature': 'temperature',      
        'nitrate': 'nitrate',              
        'phosphate': 'phosphate',          
        'ammonia': 'ammonium',             
        'wind_speed': 'wind_speed',           
        'solar_radiation': 'solar_radiation', 
        'rainfall': 'rainfall'                
    }
    
    actual_var = var_mapping.get(variable_name, variable_name)
    if actual_var in dataset:
        return dataset[actual_var]
    else:
        logger.warning(f"Variable '{actual_var}' not found in dataset.")
        return None
    
    
    