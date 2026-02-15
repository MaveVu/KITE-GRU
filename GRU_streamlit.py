import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import os


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import warnings


from utils import *

warnings.filterwarnings('ignore')


st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
    model = GRUModel(
        input_dim=CONFIG['input_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=2
    ).to(device)

    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
    model.eval()

    return model


model = load_model()

pos_max = CONFIG['pos_max']
vel_max = CONFIG['vel_max']

@st.cache_data
def load_test_files(path):
    with open(path, 'r') as f:
        files = [line.strip() for line in f.readlines() if line.strip()]
    return files

test_files = load_test_files(CONFIG['test_list'])

# Pick a file
st.title("KITE-GRU DEMO")

file_idx = st.number_input(
    f"Select a test trajectory (from 0 to {len(test_files)-1}):",
    min_value=0,
    max_value=len(test_files)-1,
    value=20, 
    step=1
)

file = test_files[file_idx]
file_path = os.path.join(CONFIG['data_dir'], file)


# advanced setting
with st.expander("Advanced setting"):
    window_length = st.slider(
        "SG window length",
        min_value=3,
        max_value=49,
        value=11,
        step=2,  
        help="Must be an odd integer and smaller than input size (50)."
    )

    polyorder = st.slider(
        "SG polynomial degree",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Must be less than the window length."
    )

    threshold = st.slider(
        "Safety threshold",
        min_value=0.1,
        max_value=10.0,
        value=2.0,
        step=0.1
    )

if polyorder >= window_length:
    st.error("Polynomial degree must be smaller than window length.")
    st.stop()

# predict
_, truth, pred = predict_trajectory(model, file_path, window_length, polyorder)

if truth is None:
    st.error("Could not process this file")
    st.stop()

# Evaluation metrics
E = np.linalg.norm(truth - pred, axis=1)

idx = np.argmax(E >= threshold)
vt = len(truth) if E[idx] < threshold else idx

st.write(f"Valid Time: **{vt}** steps - Safety threshold: **{threshold}** meters")


# full trajectory
full_truth = truth
full_pred = pred
total_frames = len(full_truth)


# axis limits
all_data = np.vstack([full_truth, full_pred])
margin = 0.0

xr = all_data[:, 0].max() - all_data[:, 0].min()
yr = all_data[:, 1].max() - all_data[:, 1].min()
zr = all_data[:, 2].max() - all_data[:, 2].min()

xmin, xmax = all_data[:, 0].min() - margin * xr, all_data[:, 0].max() + margin * xr
ymin, ymax = all_data[:, 1].min() - margin * yr, all_data[:, 1].max() + margin * yr
zmin, zmax = all_data[:, 2].min() - margin * zr, all_data[:, 2].max() + margin * zr


# layout
col1, col2 = st.columns([2, 1]) 
traj_slot = col1.empty()
loss_slot = col2.empty()

# static 3D trajectory visualization (matplotlib)
fig_static = plot_3d_trajectory(full_truth)
traj_slot.pyplot(fig_static, use_container_width=False)
plt.close(fig_static)

loss_slot.plotly_chart(go.Figure(), width='stretch')

# animation
if st.button("Start"):
    for frame in range(total_frames):

        # trajectory
        fig_traj = go.Figure()

        # faint paths
        fig_traj.add_trace(go.Scatter3d(
            x=full_truth[:, 0], y=full_truth[:, 1], z=full_truth[:, 2],
            mode="lines", line=dict(color="blue", width=4, dash="solid"),
            opacity=0.3, name="Ground Truth"
        ))

        fig_traj.add_trace(go.Scatter3d(
            x=full_pred[:, 0], y=full_pred[:, 1], z=full_pred[:, 2],
            mode="lines", line=dict(color="red", width=4, dash="dot"),
            opacity=0.3, name="Prediction"
        ))

        # moving points
        fig_traj.add_trace(go.Scatter3d(
            x=[full_truth[frame, 0]],
            y=[full_truth[frame, 1]],
            z=[full_truth[frame, 2]],
            mode="markers",
            marker=dict(size=5, color='blue'),
            name="Ground Truth"
        ))

        fig_traj.add_trace(go.Scatter3d(
            x=[full_pred[frame, 0]],
            y=[full_pred[frame, 1]],
            z=[full_pred[frame, 2]],
            mode="markers",
            marker=dict(size=5, color='red'),
            name="Prediction"
        ))

        # threshold location
        valid_frame = vt
        if valid_frame < total_frames:
            fig_traj.add_trace(go.Scatter3d(
                x=[full_truth[valid_frame, 0]],
                y=[full_truth[valid_frame, 1]],
                z=[full_truth[valid_frame, 2]],
                mode="markers",
                marker=dict(size=3, symbol="x"),
                name="Threshold Hit"
            ))


        fig_traj.update_layout(
            scene=dict(
                xaxis=dict(range=[xmin, xmax]),
                yaxis=dict(range=[ymin, ymax]),
                zaxis=dict(range=[zmin, zmax]),
                aspectmode="cube",
            )
        )

        traj_slot.plotly_chart(fig_traj, width='stretch')

        # loss
        t = frame

        fig_loss = go.Figure()

        fig_loss.add_trace(go.Scatter(
            x=np.arange(t + 1),
            y=E[:t + 1],
            mode="lines",
            name="Error"
        ))

        fig_loss.add_hline(y=threshold)

        if vt <= t:
            fig_loss.add_trace(go.Scatter(
                x=[vt],
                y=[E[vt]],
                mode="markers",
                marker=dict(size=12, symbol="diamond"),
                name="Valid Time"
            ))

        fig_loss.update_layout(
            height=450, 
            xaxis_title="Step",
            yaxis_title="Error"
        )

        loss_slot.plotly_chart(fig_loss, width='stretch')

        time.sleep(1 / 25)

# per-axis plot
with st.expander("Show per-axis predictions"):
    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    labels = ['tx', 'ty', 'tz']
    t_axis = np.arange(len(truth))
    for i, label in enumerate(labels):
        axes[i].plot(t_axis, truth[:, i], 'b-', linewidth=1.5, label='Ground Truth')
        axes[i].plot(t_axis, pred[:, i], 'r--', linewidth=1.5, label='Prediction')
        axes[i].axvline(x=vt, color='k', linewidth=2, linestyle='--',
                        label='Valid Time' if i == 0 else '')
        axes[i].set_ylabel(label)
        axes[i].set_xlim(0, len(t_axis))
        axes[i].grid(True, alpha=0.3)

    axes[0].legend(loc='upper right')
    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()
    st.pyplot(fig)


# Statistics
with st.expander("Additional Statistics"):
    rmse = np.sqrt(np.mean(E**2))
    ade = np.mean(E)
    fde = E[-1]

    st.write(f"**ADE:** {ade:.4f}")
    st.write(f"**FDE:** {fde:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")