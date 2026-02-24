import streamlit as st
import xarray as xr
from pathlib import Path
from typing import Dict, Optional
import logging
from config import DATA_PATHS
import sys
from huggingface_hub import hf_hub_download, login
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


logger = logging.getLogger(__name__)

HF_REPO_ID = "andreaatosheva/data"
HF_REPO_TYPE = "dataset"

HF_FILE_MAPPING = {
    'chl':         'chl/chl_full.nc',
    'temperature': 'temperature/temperature_full.nc',
    'nutrients':   'nutrients/nutrients_full.nc',
    'wind':        'wind/wind_full.nc',
    'solar':       'solar/solar_full.nc',
    'rain':        'rain/rain_full.nc'
}

VAR_TO_PATH = {
    'chlorophyll':     'chl',
    'temperature':     'temperature',
    'nitrate':         'nutrients',
    'phosphate':       'nutrients',
    'ammonia':         'nutrients',
    'wind_speed':      'wind',
    'solar_radiation': 'solar',
    'rainfall':        'rain'
}

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
def download_file(path_key: str) -> Optional[Path]:
    try:
        repo_file = HF_FILE_MAPPING.get(path_key)
        if repo_file is None:
            logger.error(f"No file mapped for key '{path_key}'")
            return None
        
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=repo_file,
            repo_type=HF_REPO_TYPE
            )
        return Path(local_path)
    
    except Exception as e:
        logger.error(f"Failed to download '{path_key}': {e}")
        return None

@st.cache_data(ttl=3600)
def load_dataset(variable: str) -> Optional[xr.Dataset]:
    try:
        path_var = VAR_TO_PATH.get(variable, variable)
        file_path = download_file(path_var)
        if file_path is None or not file_path.exists():
            logger.error(f"Data file for variable '{variable}' not found at path: {file_path}")
            return None
        
        ds = xr.open_dataset(file_path, chunks={'time': 12})
        
        vars_to_keep = [VAR_TO_PATH.get(variable, variable)]
        ds = ds[vars_to_keep]
        ds = ds.load()
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
    
    
    