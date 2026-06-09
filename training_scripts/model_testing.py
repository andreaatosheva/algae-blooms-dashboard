import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

def pad_to_match(x, target):
    dh = target.shape[2] - x.shape[2]
    dw = target.shape[3] - x.shape[3]
    return F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])


class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell operating on spatial feature maps.
    Gates are computed with convolutions so spatial structure is preserved.
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels
        # All four gates in one conv for efficiency
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size, padding=pad
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch, H, W, device):
        return (
            torch.zeros(batch, self.hidden_channels, H, W, device=device),
            torch.zeros(batch, self.hidden_channels, H, W, device=device),
        )

class ConvLSTMEncoder(nn.Module):
    """
    Processes the 3 past timesteps sequentially.

    Input:  (batch, T, C_per_step, H, W)  e.g. (B, 3, 9, 155, 130)
    Output: final hidden state (batch, hidden_channels, H, W)

    This replaces naive channel-stacking and lets the model learn
    *how CHL and forcing variables evolved* over the 3 days,
    not just their snapshot values.
    """
    def __init__(self, in_channels_per_step=9, hidden_channels=64):
        super().__init__()
        # Project each timestep to a richer representation first
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels_per_step, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.cell = ConvLSTMCell(hidden_channels, hidden_channels)

    def forward(self, x):
        # x: (batch, T, C, H, W)
        B, T, C, H, W = x.shape
        h, c = self.cell.init_hidden(B, H, W, x.device)
        for t in range(T):
            xt = self.input_proj(x[:, t])  # (B, hidden, H, W)
            h, c = self.cell(xt, h, c)
        return h  # final hidden state encodes temporal dynamics

class ConvLSTMUNet(nn.Module):
    """
    ConvLSTM temporal encoder → UNet spatial decoder.

    Input:  (batch, 27, H, W)  — reshaped internally to (batch, 3, 9, H, W)
    Output: (batch, 3,  H, W)  — CHL at t+1, t+2, t+3

    Why this helps with blurriness vs. plain UNet:
      - The ConvLSTM sees each day as a coherent state update, learning
        physical dynamics (upwelling, bloom growth) rather than static patterns.
      - This gives sharper predictions at bloom boundaries and coastal zones
        where CHL changes between days.
      - The UNet decoder still gets full-resolution skip connections for detail.
    """
    def __init__(self,
                 vars_per_step=9,
                 n_steps=3,
                 out_channels=3,
                 lstm_hidden=64,
                 features=[64, 128, 256]):
        super().__init__()
        self.vars_per_step = vars_per_step
        self.n_steps = n_steps

        # Temporal encoder
        self.temporal_enc = ConvLSTMEncoder(vars_per_step, lstm_hidden)

        # UNet encoder from temporal features
        f = features
        self.enc1 = DoubleConv(lstm_hidden, f[0])
        self.enc2 = DoubleConv(f[0], f[1])
        self.enc3 = DoubleConv(f[1], f[2])
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = DoubleConv(f[2], f[2] * 2, dropout=0.3)

        # Decoder
        self.up3  = nn.ConvTranspose2d(f[2] * 2, f[2], 2, stride=2)
        self.dec3 = DoubleConv(f[2] * 2, f[2])
 
        self.up2  = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec2 = DoubleConv(f[1] * 2, f[1])
 
        self.up1  = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec1 = DoubleConv(f[0] * 2, f[0])
 
        self.output = nn.Conv2d(f[0], out_channels, 1)
 
    def forward(self, x):
        # Reshape: (B, 27, H, W) → (B, 3, 9, H, W)
        B, C, H, W = x.shape
        x_seq = x.view(B, self.n_steps, self.vars_per_step, H, W)
 
        # Temporal encoding — produces (B, lstm_hidden, H, W)
        temporal_feat = self.temporal_enc(x_seq)
 
        # Spatial encoder
        e1 = self.enc1(temporal_feat)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
 
        # Decoder with skip connections
        d3 = self.up3(b);  d3 = pad_to_match(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
 
        d2 = self.up2(d3); d2 = pad_to_match(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
 
        d1 = self.up1(d2); d1 = pad_to_match(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
 
        return self.output(d1)
 

def ssim_loss(pred, target, window_size=7):
    """
    Structural Similarity loss — penalises blurry predictions directly.
    Operates per output day and averages.
    pred, target: (B, C, H, W)
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    losses = []
    for c in range(pred.shape[1]):
        p = pred[:, c:c+1]
        t = target[:, c:c+1]
        pad = window_size // 2
        mu_p  = F.avg_pool2d(p, window_size, stride=1, padding=pad)
        mu_t  = F.avg_pool2d(t, window_size, stride=1, padding=pad)
        mu_pp = F.avg_pool2d(p * p, window_size, stride=1, padding=pad)
        mu_tt = F.avg_pool2d(t * t, window_size, stride=1, padding=pad)
        mu_pt = F.avg_pool2d(p * t, window_size, stride=1, padding=pad)
        sig_p  = mu_pp - mu_p ** 2
        sig_t  = mu_tt - mu_t ** 2
        sig_pt = mu_pt - mu_p * mu_t
        ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)) / \
                   ((mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2))
        losses.append(1 - ssim_map.mean())
    return torch.stack(losses).mean()
 
 
def spectral_loss(pred, target):
    """
    FFT-based loss in frequency domain.
    Penalises missing high-frequency content (edges, fine spatial structure).
    pred, target: (B, C, H, W)
    """
    pred_fft   = torch.fft.rfft2(pred,   norm='ortho')
    target_fft = torch.fft.rfft2(target, norm='ortho')
    pred_mag   = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    return F.l1_loss(pred_mag, target_mag)
 
 
def gradient_loss(pred, target):
    if pred.dim() == 3:
        pred   = pred.unsqueeze(1)
        target = target.unsqueeze(1)
    pred_dx   = pred[:, :, :, 1:]  - pred[:, :, :, :-1]
    pred_dy   = pred[:, :, 1:, :]  - pred[:, :, :-1, :]
    tgt_dx    = target[:, :, :, 1:]  - target[:, :, :, :-1]
    tgt_dy    = target[:, :, 1:, :]  - target[:, :, :-1, :]
    return (F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy))
 
def sharpness_loss(pred, target,
                   mse_w=1.0,
                   grad_w=0.2,
                   ssim_w=0.3,
                   day_weights=(0.5, 0.3, 0.2)):
    w = torch.tensor(day_weights, device=pred.device)

    mse_days = torch.stack([
        F.mse_loss(pred[:, i], target[:, i]) for i in range(3)
    ])
    mse = (w * mse_days).sum()

    grad = gradient_loss(pred, target)
    ssim = ssim_loss(pred, target)

    # Guard each component individually
    if torch.isnan(ssim): ssim = torch.tensor(0.0, device=pred.device)
    if torch.isnan(grad): grad = torch.tensor(0.0, device=pred.device)
    if torch.isnan(mse):  mse  = torch.tensor(0.0, device=pred.device)

    return mse_w * mse + grad_w * grad + ssim_w * ssim
 

class ChlMultiOutDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(np.nan_to_num(X, nan=0.0))
        self.y = torch.FloatTensor(np.nan_to_num(y, nan=0.0))
        self.augment = augment
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.augment:
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[-1]); y = torch.flip(y, dims=[-1])
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[-2]); y = torch.flip(y, dims=[-2])
        return x, y


# Load data
features_norm = np.load('/home/admin/Desktop/Andrea_model/features_normalised.npy')
n_days = len(features_norm)
X_list, y_list = [], []

for i in range(3, n_days - 3):
    window = np.concatenate([
        features_norm[i-3].astype(np.float32),
        features_norm[i-2].astype(np.float32),
        features_norm[i-1].astype(np.float32),
    ], axis=0)
    target = np.stack([
        features_norm[i+1, 0].astype(np.float32),
        features_norm[i+2, 0].astype(np.float32),
        features_norm[i+3, 0].astype(np.float32),
    ], axis=0)
    X_list.append(window)
    y_list.append(target)

X = np.stack(X_list, axis=0)
y = np.stack(y_list, axis=0)

indices = np.arange(len(X))
_, temp_idx = train_test_split(indices, test_size=0.30, random_state=42)
_, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42)

X_test = torch.FloatTensor(np.nan_to_num(X[test_idx], nan=0.0))
y_test = torch.FloatTensor(np.nan_to_num(y[test_idx], nan=0.0))

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvLSTMUNet(vars_per_step=9, n_steps=3, out_channels=3,
                     lstm_hidden=64, features=[64, 128, 256]).to(device)
model.load_state_dict(torch.load('/home/admin/Desktop/Andrea_model/models/best_convlstm_unet.pth',
                                  map_location=device))
model.eval()

# Run predictions in batches
all_preds, all_targets = [], []
batch_size = 32

with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        xb = X_test[i:i+batch_size].to(device)
        pred = model(xb).cpu().numpy()
        all_preds.append(pred)
        all_targets.append(y_test[i:i+batch_size].numpy())

preds   = np.concatenate(all_preds,   axis=0)
targets = np.concatenate(all_targets, axis=0)
print(f"Predictions shape: {preds.shape}")

# Metrics per day
for day in range(3):
    p = preds[:, day].flatten()
    t = targets[:, day].flatten()
    mae  = np.mean(np.abs(p - t))
    rmse = np.sqrt(np.mean((p - t) ** 2))
    corr = np.corrcoef(p, t)[0, 1]
    bias = np.mean(p - t)
    print(f"Day t+{day+1}: MAE={mae:.3f}, RMSE={rmse:.3f}, Corr={corr:.3f}, Bias={bias:.3f}")

# Plot example
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
sample = 0
for day in range(3):
    vmin = min(targets[sample, day].min(), preds[sample, day].min())
    vmax = max(targets[sample, day].max(), preds[sample, day].max())
    
    axes[day, 0].imshow(targets[sample, day], vmin=vmin, vmax=vmax, cmap='viridis')
    axes[day, 0].set_title(f'Actual CHL (day t+{day+1})')
    
    axes[day, 1].imshow(preds[sample, day], vmin=vmin, vmax=vmax, cmap='viridis')
    axes[day, 1].set_title(f'Predicted CHL (day t+{day+1})')
    
    diff = preds[sample, day] - targets[sample, day]
    axes[day, 2].imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[day, 2].set_title(f'Difference (day t+{day+1})')

plt.tight_layout()
plt.savefig('/home/admin/Desktop/Andrea_model/evaluation_plot.png', dpi=150)
print("Plot saved to evaluation_plot.png")