import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import joblib
import random
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
from torch.optim.lr_scheduler import StepLR
import warnings


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler_path = np.load(os.path.join('model', 'norm_stats.npz'))
pos_max = scaler_path['pos_max']
vel_max = scaler_path['vel_max']

model_path = os.path.join('model', 'best_0p3_0p1_11.pth')

CONFIG = {
    'seq_len': 50,
    'pred_len': 20,
    'pos_max': pos_max,
    'vel_max': vel_max,
    'model_path': model_path,

    'input_dim': 6,
    'hidden_dim': 64,
    'dt': 0.1,

    'test_list': 'test_list.txt',
    'data_dir': 'test_trajs',
}




def compute_velocity(pos_seq, dt=0.1, window_length=11, polyorder=3):
    velocity = savgol_filter(pos_seq, window_length=window_length, polyorder=polyorder, deriv=1, delta=dt, axis=0)
    return velocity

def max_normalize_sequence(sequence, max):
    return sequence / max

def max_denormalize_sequence(sequence, max):
    return sequence * max

def predict_trajectory(model, file_path, window_length=11, polyorder=3):
    if not os.path.exists(file_path):
        return None, None, None

    df = pd.read_csv(file_path)
    raw_coords = df[['tx', 'ty', 'tz']].values.astype(np.float32)
    
    if len(raw_coords) < CONFIG['seq_len'] + CONFIG['pred_len']:
        return None, None, None

    # Input history (first 50 steps) for visualization
    input_history = raw_coords[:CONFIG['seq_len']]

    vel = compute_velocity(input_history, CONFIG['dt'], window_length, polyorder)
    n_pos = input_history / CONFIG['pos_max']
    n_vel = vel / CONFIG['vel_max']
    state_scaled = np.concatenate([n_pos, n_vel], axis=1)
    
    # first 50 steps = input
    current_input_tensor = torch.FloatTensor(state_scaled).unsqueeze(0).to(device)
    
    limit_len = len(raw_coords) - CONFIG['seq_len']
    truth_future = raw_coords[CONFIG['seq_len'] : CONFIG['seq_len'] + limit_len]
    
    predictions_scaled = []
    num_blocks = int(np.ceil(limit_len / CONFIG['pred_len']))
    
    with torch.no_grad():
        for _ in range(num_blocks):
            pred_block = model(current_input_tensor)
            block_np = pred_block.detach().cpu().numpy()[0]
            predictions_scaled.extend(block_np.tolist())
            remaining = current_input_tensor[:, CONFIG['pred_len']:, :]
            # sliding window (remove first 20, add newly predicted 20)
            current_input_tensor = torch.cat((remaining, pred_block), dim=1)

    predictions_scaled = predictions_scaled[:limit_len]
    pred_coords = max_denormalize_sequence(np.array(predictions_scaled)[:, :3], CONFIG['pos_max'])
    
    return input_history, truth_future, pred_coords
    
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5):
        super(GRUModel, self).__init__()

        self.gru1 = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_pos = nn.Linear(hidden_dim, input_dim//2)
        self.fc_vel = nn.Linear(hidden_dim, input_dim//2)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        out, h_n = self.gru1(x)
        dec_input = torch.zeros(x.size(0), CONFIG['pred_len'], self.hidden_dim).to(x.device)
        out, _ = self.gru2(dec_input, h_n)
        out_pos = self.fc_pos(out)
        out_vel = self.fc_vel(out)
        return torch.cat([out_pos, out_vel], dim=2)
    

def plot_3d_trajectory(truth):
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], 'b-', linewidth=2, label='Ground Truth')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    return fig