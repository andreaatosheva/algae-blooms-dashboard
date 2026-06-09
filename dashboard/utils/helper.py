import streamlit as st
import psutil
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from utils.data_loader import load_variable_data
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import torch.nn.functional as F
from plotly.subplots import make_subplots


HF_REPO_ID_MODEL = "andreaatosheva/model"
HF_REPO_TYPE = "dataset"
MODEL_FILES = {
    "sameday_model" : "model/best_unet_chl.pth",
    "forecast_model": "model/best_unet_forecast.pth",
    "3_days_forecast_model": "model/best_taunet_20260506_1533.pth",
    "7_days_forecast_model": "model/best_taunet_7day_20260519_1034.pth"
}


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

    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

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


def compute_threshold_for_slice(data_slice, pct):
        vals = data_slice.values
        return np.nanpercentile(vals, pct)

def compute_centroid(data_slice, threshold):
    """Weighted centroid of pixels above threshold."""
    mask = data_slice.values > threshold
    if not mask.any():
        return None, None
    lats = data_slice.latitude.values
    lons = data_slice.longitude.values
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    weights = np.where(mask & ~np.isnan(data_slice.values), data_slice.values, 0.0)
    total = weights.sum()
    if total == 0:
        return None, None
    return float((lat_grid * weights).sum() / total), float((lon_grid * weights).sum() / total)

def get_time_groups(data, granularity):
    """
    Returns a list of (label, data_slice) tuples based on granularity.
    Each slice is a single time step or a mean over grouped steps.
    """
    groups = []
    times = data.time

    if granularity == "Month-by-month":
        for t in times:
            label = str(t.values)[:7]  # "YYYY-MM"
            groups.append((label, data.sel(time=t)))

    elif granularity == "Season-by-season":
        season_map = {
            'DJF': [12, 1, 2],
            'MAM': [3, 4, 5],
            'JJA': [6, 7, 8],
            'SON': [9, 10, 11]
        }
        years = sorted(np.unique(data.time.dt.year.values))
        for year in years:
            for season_abbr, months in season_map.items():
                subset = data.where(
                    (data.time.dt.year == year) & (data.time.dt.month.isin(months)),
                    drop=True
                )
                if len(subset.time) == 0:
                    continue
                label = f"{season_abbr} {year}"
                groups.append((label, subset.mean('time')))

    elif granularity == "Year-by-year":
        years = sorted(np.unique(data.time.dt.year.values))
        for year in years:
            subset = data.sel(time=str(year))
            if len(subset.time) == 0:
                continue
            label = str(year)
            groups.append((label, subset.mean('time')))

    return groups


class ResDoubleConv(nn.Module):
    """DoubleConv with residual connection for better gradient flow."""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if dropout > 0:
            self.conv.add_module('dropout', nn.Dropout2d(dropout))
        # 1x1 conv to match channels for residual if needed
        self.residual = nn.Conv2d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.residual(x)

class SqueezeExcitation(nn.Module):
    """

    Channel attention block (SE block).

    Learns to re-weight feature channels — helps the bottleneck focus

    on the most informative variable combinations.

    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


class TemporalAttention(nn.Module):
    """

    Spatial-temporal attention across T past timesteps.

    For each spatial location (h, w), learns a weighted combination
    of the T timestep features. This is more flexible than ConvLSTM
    because it can directly attend to any past day regardless of order.

    Input:  (B, T, C, H, W)

    Output: (B, C_out, H, W)  — attended temporal summary

    """
    def __init__(self, in_channels_per_step, out_channels, n_steps=3):
        super().__init__()
        self.n_steps = n_steps

        # Project each timestep independently
        self.step_proj = nn.Sequential(
            nn.Conv2d(in_channels_per_step, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Attention scoring: for each position, score each timestep
        self.attn_score = nn.Conv2d(out_channels * n_steps, n_steps, 1)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # Project each timestep
        step_features = []
        for t in range(T):
            feat = self.step_proj(x[:, t])  
            step_features.append(feat)

        # Concatenate all timesteps for attention scoring
        stacked = torch.cat(step_features, dim=1)  

        # Compute attention weights per spatial location
        attn_logits = self.attn_score(stacked)          
        attn_weights = F.softmax(attn_logits, dim=1)    

        # Weighted sum of timestep features
        out = torch.zeros_like(step_features[0])
        for t in range(T):
            out = out + attn_weights[:, t:t+1] * step_features[t]

        return out  


def pad_to_match(x, target):
    dh = target.shape[2] - x.shape[2]
    dw = target.shape[3] - x.shape[3]
    return F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])


class TAUNet(nn.Module):
    """

    Temporal Attention UNet for multi-step CHL forecasting.
    
    Input:  (B, 27, H, W)  — reshaped to (B, 3, 9, H, W) internally
    Output: (B, 3,  H, W)  — CHL at t+1, t+2, t+3

    """
    def __init__(self,
                vars_per_step=9,
                n_steps=3,
                out_channels=3,
                temporal_hidden=64,
                features=[32, 64, 128, 256]):
        super().__init__()
        self.vars_per_step = vars_per_step
        self.n_steps = n_steps

        # Temporal attention encoder
        self.temporal_attn = TemporalAttention(vars_per_step, temporal_hidden, n_steps)

        # 4-level UNet encoder
        f = features
        self.enc1 = ResDoubleConv(temporal_hidden, f[0])
        self.enc2 = ResDoubleConv(f[0], f[1])
        self.enc3 = ResDoubleConv(f[1], f[2])
        self.enc4 = ResDoubleConv(f[2], f[3])
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck with SE channel attention
        self.bottleneck = nn.Sequential(
            ResDoubleConv(f[3], f[3] * 2, dropout=0.3),
            SqueezeExcitation(f[3] * 2),
        )

        # 4-level decoder
        self.up4  = nn.ConvTranspose2d(f[3] * 2, f[3], 2, stride=2)
        self.dec4 = ResDoubleConv(f[3] * 2, f[3])

        self.up3  = nn.ConvTranspose2d(f[3], f[2], 2, stride=2)
        self.dec3 = ResDoubleConv(f[2] * 2, f[2])

        self.up2  = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec2 = ResDoubleConv(f[1] * 2, f[1])

        self.up1  = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec1 = ResDoubleConv(f[0] * 2, f[0])

        self.output = nn.Conv2d(f[0], out_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = x.view(B, self.n_steps, self.vars_per_step, H, W)

        # Temporal attention — (B, temporal_hidden, H, W)
        t_feat = self.temporal_attn(x_seq)

        # Encoder
        e1 = self.enc1(t_feat)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b);  d4 = pad_to_match(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4); d3 = pad_to_match(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3); d2 = pad_to_match(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2); d1 = pad_to_match(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.output(d1)
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class UNetChl(nn.Module):
    def __init__(self, in_channels=8, features=[32, 64, 128, 256]):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, features[0])
        self.enc2 = DoubleConv(features[0], features[1])
        self.enc3 = DoubleConv(features[1], features[2])
        self.enc4 = DoubleConv(features[2], features[3])
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[3], features[3] * 2)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(features[3] * 2, features[3])
        
        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features[2] * 2, features[2])
        
        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features[1] * 2, features[1])
        
        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(features[0] * 2, features[0])
        
        # Output
        self.output = nn.Conv2d(features[0], 1, kernel_size=1)
    
    def pad_to_match(self, x, target):
        """Pad x to match target spatial dimensions"""
        diff_h = target.shape[2] - x.shape[2]
        diff_w = target.shape[3] - x.shape[3]
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        return x
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections and padding
        d4 = self.up4(b)
        d4 = self.pad_to_match(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        d3 = self.pad_to_match(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.pad_to_match(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.pad_to_match(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.output(d1).squeeze(1)
    
    
@st.cache_resource
def load_sameday_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sameday_path = hf_hub_download(
            repo_id = HF_REPO_ID_MODEL,
            filename = MODEL_FILES["sameday_model"],
            repo_type = HF_REPO_TYPE
        )
        model_sameday = UNetChl(in_channels=8).to(device)
        model_sameday.load_state_dict(torch.load(sameday_path, map_location=device))
        model_sameday.eval()
        return model_sameday, device
    except Exception as e:
        print(f"Error loading same-day model: {e}")
        return None, None
    
@st.cache_resource
def load_three_days_forecast_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        three_days_forecast_path = hf_hub_download(
            repo_id = HF_REPO_ID_MODEL,
            filename = MODEL_FILES["3_days_forecast_model"],
            repo_type = HF_REPO_TYPE
        )
        three_days_forecast_model = TAUNet(vars_per_step=9, n_steps=3, out_channels=3,
                                            temporal_hidden=64, features=[32, 64, 128, 256]).to(device)
        three_days_forecast_model.load_state_dict(torch.load(three_days_forecast_path, map_location=device))
        three_days_forecast_model.eval()
        return three_days_forecast_model, device
    except Exception as e:
        print(f"Error loading three-day forecast model: {e}")
        return None, None
    
@st.cache_resource
def load_seven_days_forecast_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seven_days_forecast_path = hf_hub_download(
            repo_id = HF_REPO_ID_MODEL,
            filename = MODEL_FILES["7_days_forecast_model"],
            repo_type = HF_REPO_TYPE
        )
        seven_days_forecast_model = TAUNet(vars_per_step=9, n_steps=14, out_channels=7,
                                            temporal_hidden=128, features=[64, 128, 256, 512]).to(device)
        seven_days_forecast_model.load_state_dict(torch.load(seven_days_forecast_path, map_location=device))
        seven_days_forecast_model.eval()
        return seven_days_forecast_model, device
    except Exception as e:
        print(f"Error loading seven-day forecast model: {e}")
        return None, None

@st.cache_resource
def load_one_day_forecast_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        forecast_path = hf_hub_download(
            repo_id = HF_REPO_ID_MODEL,
            filename = MODEL_FILES["forecast_model"],
            repo_type = HF_REPO_TYPE
        )
        model_forecast = UNetChl(in_channels=9).to(device)
        model_forecast.load_state_dict(torch.load(forecast_path, map_location=device))
        model_forecast.eval()
        return model_forecast, device
    except Exception as e:
        print(f"Error loading one-day forecast model: {e}")
        return None, None


def denormalise_chl(x_norm, norm_stats):
    chl_min = norm_stats['chl']['min']
    chl_max = norm_stats['chl']['max']
    x_log = x_norm * (chl_max - chl_min) + chl_min
    return np.expm1(x_log)


def run_prediction(model, x_input, device):
    """x_input is already sliced — shape (channels, lat, lon)"""
    x = np.nan_to_num(x_input, nan=0.0)
    x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(x_tensor).cpu().numpy()[0]
    
    return pred

def make_map(data, title, colorscale='Viridis', zmin=0, zmax=None, zmid=None, log_scale=True):
    if log_scale:
        plot_data = np.log1p(np.clip(data, 0, None))
        tick_vals_real = [0, 1, 2.5, 5, 10, 20]
        tick_vals_log  = [np.log1p(v) for v in tick_vals_real]
        tick_labels    = [str(v) for v in tick_vals_real]
        colorbar = dict(
            title='mg/m³',
            tickvals=tick_vals_log,
            ticktext=tick_labels,
        )
        plot_zmin = np.log1p(zmin) if zmin is not None else None
        plot_zmax = np.log1p(zmax) if zmax is not None else None
        plot_zmid = None
        
    else:
        plot_data = data
        colorbar = dict(title='mg/m³')
        plot_zmin = zmin
        plot_zmax = zmax
        plot_zmid = zmid
        
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=plot_data,
        customdata = data,
        colorscale=colorscale,
        zmin=plot_zmin,
        zmax=plot_zmax,
        zmid=plot_zmid,
        colorbar=colorbar,
        hovertemplate='<b>Value</b>: %{customdata:.2f} mg/m³<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        height=400,
        template='plotly_white',
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def render_prediction_columns(
    actual: np.ndarray,
    predicted: np.ndarray,
    date_label: str,
) -> None:
    """
    Render the standard three-column layout: Actual | Predicted | Difference.

    Parameters
    ----------
    actual      : Denormalised actual CHL array (NaN on land).
    predicted   : Denormalised predicted CHL array (NaN on land).
    date_label  : Human-readable date string used in plot titles.
    """

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        st.plotly_chart(
            make_map(actual, f"Actual CHL — {date_label}"),
            width="stretch",
        )
    with col2:
        st.plotly_chart(
            make_map(predicted, f"Predicted CHL — {date_label}"),
            width="stretch",
        )
    with col3:
        st.plotly_chart(
            make_map(
                predicted - actual,
                "Difference (Pred - Actual)",
                colorscale="RdBu_r",
                zmin=-5,
                zmax=5,
                zmid=0,
                log_scale=False
            ),
            width="stretch",
        )

def render_multiday_prediction_columns(
    actual_days: list[np.ndarray],
    pred_days: list[np.ndarray],
    all_dates: pd.DatetimeIndex,
    global_idx: int,
) -> None:
    """
    Render one three-column row per forecast day (used by 3-day and 7-day tabs).

    Parameters
    ----------
    actual_days : List of actual CHL arrays, one per forecast day.
    pred_days   : List of predicted CHL arrays, one per forecast day.
    all_dates   : Full date index (needed to look up future date labels).
    global_idx  : Index of the *selected* date in all_dates.
    """
    for day_offset, (actual, pred) in enumerate(zip(actual_days, pred_days)):
        future_label = all_dates[global_idx + day_offset + 1].strftime("%d %b %Y")
        st.markdown(f"#### Day t+{day_offset + 1} — {future_label}")
        render_prediction_columns(actual, pred, f"t+{day_offset + 1}")


def _nan_to_none(arr: np.ndarray) -> list[list]:
    """Convert a 2-D numpy array to nested lists, replacing NaN with None."""
    return [[None if np.isnan(x) else float(x) for x in row] for row in arr]


def build_animated_forecast_figure(
    daily_results: list[dict],
    title: str,
    zmin: float = 0,
    zmax: float = 10,
) -> go.Figure:
    """
    Build an animated side-by-side Actual vs Predicted heatmap figure.

    Parameters
    ----------
    daily_results : List of dicts with keys 'date', 'actual', 'pred'.
    title         : Figure title string.
    zmin / zmax   : Colour-scale limits.

    Returns
    -------
    A Plotly Figure with frames and a play/pause slider.
    """
    first = daily_results[0]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Actual CHL", "Predicted CHL"],
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Heatmap(
            z=_nan_to_none(first["actual"]),
            zmin=zmin,
            zmax=zmax,
            colorscale="Viridis",
            colorbar=dict(title="mg/m³", x=0.45),
            hovertemplate="<b>Actual</b>: %{z:.2f} mg/m³<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=_nan_to_none(first["pred"]),
            zmin=zmin,
            zmax=zmax,
            colorscale="Viridis",
            colorbar=dict(title="mg/m³", x=1.0),
            hovertemplate="<b>Predicted</b>: %{z:.2f} mg/m³<extra></extra>",
        ),
        row=1,
        col=2,
    )

    frames = [
        go.Frame(
            data=[
                go.Heatmap(z=_nan_to_none(r["actual"])),
                go.Heatmap(z=_nan_to_none(r["pred"])),
            ],
            name=pd.Timestamp(r["date"]).strftime("%d %b %Y"),
            layout=go.Layout(
                title_text=f"CHL — {pd.Timestamp(r['date']).strftime('%d %b %Y')}"
            ),
        )
        for r in daily_results
    ]
    fig.frames = frames

    slider_steps = [
        dict(
            args=[
                [f["name"]],
                dict(
                    frame=dict(duration=500, redraw=True),
                    mode="immediate",
                    transition=dict(duration=300),
                ),
            ],
            label=f["name"],
            method="animate",
        )
        for f in frames
    ]

    fig.update_layout(
        title=title,
        height=550,
        template="plotly_white",
        margin=dict(l=0, r=0, t=60, b=120),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=-0.12,
                x=0.00,
                xanchor="left",
                yanchor="top",
                direction="right",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=600, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=300),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=slider_steps,
                x=0.0,
                y=-0.08,
                len=1.0,
                currentvalue=dict(
                    prefix="Date: ",
                    visible=True,
                    xanchor="center",
                    font=dict(size=13),
                ),
                transition=dict(duration=300),
            )
        ],
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

def run_sameday_prediction(
    selected_date: pd.Timestamp,
    all_dates: pd.DatetimeIndex,
    norm_stats: dict,
    model,
    device,
    get_features_for_date,
) -> None:
    """Load features and run the same-day CNN, then display results."""
    with st.spinner(f"Loading data for {selected_date.strftime('%Y-%m-%d')}..."):
        features_year, day_idx_year, _ = get_features_for_date(selected_date, all_dates)

    if features_year is None:
        st.error(f"Features for {selected_date.strftime('%Y-%m-%d')} could not be loaded.")
        return

    actual_norm = features_year[day_idx_year, 0, :, :]
    actual_real = denormalise_chl(actual_norm, norm_stats)
    land_mask = np.isnan(actual_norm)
    actual_real[land_mask] = np.nan

    with st.spinner("Running model prediction..."):
        x_input = features_year[day_idx_year, 1:, :, :]
        pred_norm = run_prediction(model, x_input, device)

    pred_real = denormalise_chl(pred_norm, norm_stats)
    pred_real[land_mask] = np.nan

    render_prediction_columns(actual_real, pred_real, selected_date.strftime("%d %b %Y"))


def run_one_day_forecast_prediction(
    selected_date: pd.Timestamp,
    all_dates: pd.DatetimeIndex,
    norm_stats: dict,
    model,
    device,
    get_features_for_date,
) -> None:
    """Load features and run the 1-day forecast CNN, then display results."""
    global_idx = np.where(all_dates == pd.to_datetime(selected_date))[0][0]
    if global_idx >= len(all_dates) - 1:
        st.error("No data available for the next day after the selected date.")
        return

    next_date = all_dates[global_idx + 1]
    with st.spinner(f"Loading data for {selected_date.strftime('%Y-%m-%d')}..."):
        features_next, day_idx_next, _ = get_features_for_date(next_date, all_dates)

    if features_next is None:
        st.error(f"Features for {next_date.strftime('%Y-%m-%d')} could not be loaded.")
        return

    actual_norm = features_next[day_idx_next, 0, :, :]
    actual_real = denormalise_chl(actual_norm, norm_stats)
    land_mask = np.isnan(actual_norm)
    actual_real[land_mask] = np.nan

    with st.spinner("Running model prediction..."):
        x_input = features_next[day_idx_next, :, :, :]
        pred_norm = run_prediction(model, x_input, device)

    pred_real = denormalise_chl(pred_norm, norm_stats)
    pred_real[land_mask] = np.nan

    render_prediction_columns(actual_real, pred_real, selected_date.strftime("%d %b %Y"))


def run_three_day_forecast_prediction(
    selected_date: pd.Timestamp,
    all_dates: pd.DatetimeIndex,
    norm_stats: dict,
    model,
    device,
    get_features_for_date,
) -> None:
    """Load features and run the 3-day forecast CNN, then display results."""
    global_idx = np.where(all_dates == pd.to_datetime(selected_date))[0][0]

    if global_idx < 3:
        st.error("Not enough previous data to make a 3-day forecast.")
        return
    if global_idx > len(all_dates) - 4:
        st.error("No data available for the 3 days following the selected date.")
        return

    with st.spinner(f"Loading data for {selected_date.strftime('%Y-%m-%d')}..."):
        past_features = []
        for offset in [3, 2, 1]:
            past_date = all_dates[global_idx - offset]
            features_past, day_idx_past, _ = get_features_for_date(past_date, all_dates)
            if features_past is None:
                st.error(f"Features for {past_date.strftime('%Y-%m-%d')} could not be loaded.")
                return
            past_features.append(features_past[day_idx_past])
        x_input = np.concatenate(past_features, axis=0)

    actual_days, land_masks = [], []
    for offset in [1, 2, 3]:
        future_date = all_dates[global_idx + offset]
        features_future, day_idx_future, _ = get_features_for_date(future_date, all_dates)
        if features_future is None:
            st.error(f"Features for {future_date.strftime('%Y-%m-%d')} could not be loaded.")
            return
        actual_norm = features_future[day_idx_future, 0, :, :]
        actual_real = denormalise_chl(actual_norm, norm_stats)
        land_mask = np.isnan(actual_norm)
        actual_real[land_mask] = np.nan
        actual_days.append(actual_real)
        land_masks.append(land_mask)

    with st.spinner("Running model prediction..."):
        x = np.nan_to_num(x_input, nan=0.0)
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm_all = model(x_tensor).cpu().numpy()[0]

    pred_days = []
    for i in range(3):
        pred_real = denormalise_chl(pred_norm_all[i], norm_stats)
        pred_real[land_masks[i]] = np.nan
        pred_days.append(pred_real)

    render_multiday_prediction_columns(actual_days, pred_days, all_dates, global_idx)


def run_seven_day_forecast_prediction(
    selected_date: pd.Timestamp,
    all_dates: pd.DatetimeIndex,
    norm_stats: dict,
    model,
    device,
    get_features_for_date,
) -> None:
    """Load features and run the 7-day forecast CNN, then display results."""
    global_idx = np.where(all_dates == pd.to_datetime(selected_date))[0][0]

    if global_idx < 14:
        st.error("Not enough previous data to make a 7-day forecast.")
        return
    if global_idx > len(all_dates) - 8:
        st.error("No data available for the 7 days following the selected date.")
        return

    with st.spinner(f"Loading data for {selected_date.strftime('%Y-%m-%d')}..."):
        past_features = []
        for offset in range(14, 0, -1):
            past_date = all_dates[global_idx - offset]
            features_past, day_idx_past, _ = get_features_for_date(past_date, all_dates)
            if features_past is None:
                st.error(f"Features for {past_date.strftime('%Y-%m-%d')} could not be loaded.")
                return
            past_features.append(features_past[day_idx_past])
        x_input = np.concatenate(past_features, axis=0)

    actual_days, land_masks = [], []
    for offset in range(1, 8):
        future_date = all_dates[global_idx + offset]
        features_future, day_idx_future, _ = get_features_for_date(future_date, all_dates)
        if features_future is None:
            st.error(f"Features for {future_date.strftime('%Y-%m-%d')} could not be loaded.")
            return
        actual_norm = features_future[day_idx_future, 0, :, :]
        actual_real = denormalise_chl(actual_norm, norm_stats)
        land_mask = np.isnan(actual_norm)
        actual_real[land_mask] = np.nan
        actual_days.append(actual_real)
        land_masks.append(land_mask)

    with st.spinner("Running model prediction..."):
        x = np.nan_to_num(x_input, nan=0.0)
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm_all = model(x_tensor).cpu().numpy()[0]

    pred_days = []
    for i in range(7):
        pred_real = denormalise_chl(pred_norm_all[i], norm_stats)
        pred_real[land_masks[i]] = np.nan
        pred_days.append(pred_real)

    render_multiday_prediction_columns(actual_days, pred_days, all_dates, global_idx)


def run_monthly_forecast(
    selected_year: int,
    selected_month: int,
    all_dates: pd.DatetimeIndex,
    norm_stats: dict,
    model,
    device,
    get_features_for_date,
    window_size: int,          # 1, 3, or 7
    min_history: int,          # minimum past days needed (1 for 1-day, 3 for 3-day, 14 for 7-day)
    past_offsets: list[int],   # e.g. [1] / [3,2,1] / range(14,0,-1)
    chart_key: str,
) -> None:
    """
    Generic monthly forecast runner shared by 1-day, 3-day, and 7-day expanders.

    Loops over non-overlapping `window_size`-day windows within the selected
    month, runs the model for each window, collects results, and renders an
    animated Plotly figure.
    """
    month_mask = (all_dates.year == selected_year) & (all_dates.month == selected_month)
    month_dates = all_dates[month_mask]

    if len(month_dates) == 0:
        st.error("No data available for the selected month and year.")
        return

    if window_size == 1:
        windows = [[d] for d in month_dates[:-1]]
    else:
        windows = []
        i = 0
        while i + window_size - 1 < len(month_dates):
            windows.append(month_dates[i : i + window_size])
            i += window_size

    if not windows:
        st.error(f"Not enough data to create {window_size}-day windows for the selected month.")
        return

    st.markdown(
        f"**{len(windows)} prediction windows** for "
        f"{pd.Timestamp(selected_year, selected_month, 1).strftime('%B %Y')}"
    )

    progress = st.progress(0, text="Running predictions...")
    daily_results = []

    for w_idx, window in enumerate(windows):
        try:
            if window_size == 1:
                # 1-day: predict the day after `window[0]`
                target_date = window[0] + pd.Timedelta(days=1)
                features_next, day_idx_next, _ = get_features_for_date(target_date, all_dates)
                if features_next is None:
                    raise ValueError(f"Features for {target_date} could not be loaded.")

                actual_norm = features_next[day_idx_next, 0, :, :]
                land_mask = np.isnan(actual_norm)
                actual_real = denormalise_chl(actual_norm, norm_stats)
                actual_real[land_mask] = np.nan

                x_input = features_next[day_idx_next, :, :, :]
                pred_norm = run_prediction(model, x_input, device)
                pred_real = denormalise_chl(pred_norm, norm_stats)
                pred_real[land_mask] = np.nan

                daily_results.append({"date": target_date, "pred": pred_real, "actual": actual_real})

            else:
                first_day_idx = np.where(all_dates == window[0])[0][0]
                if first_day_idx < min_history:
                    continue

                past_features = []
                for offset in past_offsets:
                    past_date = all_dates[first_day_idx - offset]
                    feats, day_idx, _ = get_features_for_date(past_date, all_dates)
                    if feats is None:
                        raise ValueError(f"Features for {past_date} could not be loaded.")
                    past_features.append(feats[day_idx])
                x_input = np.concatenate(past_features, axis=0)

                x = np.nan_to_num(x_input, nan=0.0)
                x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_norm_all = model(x_tensor).cpu().numpy()[0]

                for d, future_date in enumerate(window):
                    features_future, day_idx_future, _ = get_features_for_date(future_date, all_dates)
                    if features_future is None:
                        raise ValueError(f"Features for {future_date} could not be loaded.")

                    actual_norm = features_future[day_idx_future, 0, :, :]
                    land_mask = np.isnan(actual_norm)
                    actual_real = denormalise_chl(actual_norm, norm_stats)
                    actual_real[land_mask] = np.nan
                    pred_real = denormalise_chl(pred_norm_all[d], norm_stats)
                    pred_real[land_mask] = np.nan

                    daily_results.append({"date": future_date, "pred": pred_real, "actual": actual_real})

        except Exception as e:
            st.warning(f"Window {w_idx} failed: {e}")

        progress.progress(
            (w_idx + 1) / len(windows),
            text=f"Running predictions... (Window {w_idx + 1}/{len(windows)})",
        )

    progress.empty()

    if not daily_results:
        st.error("All prediction windows failed. Please check the error messages above.")
        return

    st.markdown("---")
    month_label = pd.Timestamp(selected_year, selected_month, 1).strftime("%B %Y")
    fig = build_animated_forecast_figure(
        daily_results,
        title=f"{window_size}-Day CHL Forecast — {month_label}",
    )
    st.plotly_chart(fig, width="stretch", key=chart_key)