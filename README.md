# Algae Blooms Dashboard

A Streamlit dashboard for monitoring and analyzing algae bloom events in the Baltic Sea using data in the time period 2014–2024.  
## Prerequisites
Make sure you have the following installed:
- VS Code or another suitable IDE
- Python 3.13+
- [Poetry](https://python-poetry.org/docs/#installation)

## Getting Started
### 1. Clone the Repository
You can either use the command:
```bash
git clone https://github.com/andreaatosheva/algae-blooms-dashboard.git
cd algae-blooms-dashboard
```
Or you can use the **GitHub Desktop** app.

### 2. Install Dependencies
```bash
poetry install
```

### 3. Run the Dashboard
```bash
poetry run streamlit run dashboard/Home.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## Data

The oceanographic datasets are hosted on Hugging Face and will be downloaded automatically when you first run the dashboard. The following variables are available:

- **Chlorophyll-a** — Chlorophyll-a concentration (mg/m³)
- **Sea Surface Temperature** — Water temperature (°C)
- **Nutrients** — Nitrate, Phosphate, Ammonia (mmol/m³)
- **Wind Speed** — Surface wind speed (m/s)
- **Solar Radiation** — Incoming solar radiation (W/m²)
- **Rainfall** — Precipitation (mm/day)

##  Project Structure
```
algae-blooms-dashboard/
├── dashboard/
│   ├── Home.py               # Main entry point
│   ├── config.py             # File with configuration settings
│   ├── pages/                # Dashboard pages
│   ├── components/           # Shared UI components
│   └── utils/                # Data loading and processing
├── pyproject.toml            # Poetry dependencies
├── poetry.lock         
└── README.md
```

## Tech Stack

- [Streamlit](https://streamlit.io/) — Dashboard framework
- [xArray](https://docs.xarray.dev/) — NetCDF data handling
- [Plotly](https://plotly.com/python/) — Interactive visualizations
- [SciPy](https://scipy.org/) — Statistical analysis
- [Hugging Face Hub](https://huggingface.co/) — Data hosting
