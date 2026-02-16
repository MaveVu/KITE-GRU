import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import warnings
from utils import *

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")

# Load model
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

# Load test files
@st.cache_data
def load_test_files(path):
    with open(path, 'r') as f:
        files = [line.strip() for line in f.readlines() if line.strip()]
    return files

test_files = load_test_files(CONFIG['test_list'])

# Header layout
header_col1, header_col2 = st.columns([1, 1])
with header_col1:
    st.title("KITE-GRU")
with header_col2:
    file_idx = st.number_input(
        "Select a file",
        min_value=0,
        max_value=len(test_files)-1,
        value=20,
        step=1,
        label_visibility="visible",
        help=f"Select a file from {0} to {len(test_files)-1}"
    )

file = test_files[file_idx]
file_path = os.path.join(CONFIG['data_dir'], file)

# Advanced settings
with st.expander("Advanced Settings"):
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
    threshold = st.number_input(
        "Safety threshold (m)",
        min_value=0.1,
        max_value=10.0,
        value=2.0,
        step=0.1,
    )
    FPS = st.number_input("FPS", min_value=5, max_value=60, value=10, step=1, help="Frames per second for the animation.")

if polyorder >= window_length:
    st.error("Polynomial degree must be smaller than window length.")
    st.stop()

# Prediction
_, truth, pred = predict_trajectory(model, file_path, window_length, polyorder)

if truth is None:
    st.error("Could not process this file")
    st.stop()

# Metrics
E = np.linalg.norm(truth - pred, axis=1)
idx = np.argmax(E >= threshold)
vt = len(truth) if E[idx] < threshold else idx

full_truth = truth
full_pred = pred
total_frames = len(full_truth)

# Axis limits for 3D plot
all_data = np.vstack([full_truth, full_pred])
margin = 0.0
xr = all_data[:, 0].max() - all_data[:, 0].min()
yr = all_data[:, 1].max() - all_data[:, 1].min()
zr = all_data[:, 2].max() - all_data[:, 2].min()
xmin, xmax = all_data[:, 0].min() - margin * xr, all_data[:, 0].max() + margin * xr
ymin, ymax = all_data[:, 1].min() - margin * yr, all_data[:, 1].max() + margin * yr
zmin, zmax = all_data[:, 2].min() - margin * zr, all_data[:, 2].max() + margin * zr

# Axis limits for per-axis plots
t_axis = np.arange(len(truth))
tx_range = [min(truth[:, 0].min(), pred[:, 0].min()), max(truth[:, 0].max(), pred[:, 0].max())]
ty_range = [min(truth[:, 1].min(), pred[:, 1].min()), max(truth[:, 1].max(), pred[:, 1].max())]
tz_range = [min(truth[:, 2].min(), pred[:, 2].min()), max(truth[:, 2].max(), pred[:, 2].max())]

# Main layout
row1_col1, row1_col2 = st.columns([1, 1])
row2_col1, row2_col2 = st.columns([2, 1])

with row1_col1:
    st.markdown("### 3D Trajectory")
    traj_3d_slot = st.empty()
with row1_col2:
    st.markdown("### Prediction Error")
    error_slot = st.empty()

with row2_col1:
    st.markdown("### Per-Axis Trajectory")
    per_axis_slot = st.empty()
with row2_col2:
    st.markdown("### Statistics")
    stats_slot = st.empty()

# Static 3D ground truth (replaced by animation when Start is pressed)
fig_static_3d = go.Figure()
fig_static_3d.add_trace(go.Scatter3d(
    x=full_truth[:, 0], y=full_truth[:, 1], z=full_truth[:, 2],
    mode="lines", line=dict(color="blue", width=4),
    name="Ground Truth"
))
fig_static_3d.update_layout(
    scene=dict(
        xaxis=dict(range=[xmin, xmax], showticklabels=False),
        yaxis=dict(range=[ymin, ymax], showticklabels=False),
        zaxis=dict(range=[zmin, zmax], showticklabels=False),
        aspectmode="cube",
        domain=dict(x=[0, 1], y=[0, 1])
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=300,
    showlegend=True,
    legend=dict(x=0, y=1, font=dict(size=10))
)

# Initialize plots
traj_3d_slot.plotly_chart(fig_static_3d, width='stretch', key="3d-traj-init")
error_slot.plotly_chart(go.Figure().update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0)), width='stretch', key="error-init")
per_axis_slot.plotly_chart(go.Figure().update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0)), width='stretch', key="per-axis-init")

# Start button
start_col1, start_col2, start_col3 = st.columns([1, 1, 1])
with start_col2:
    start_button = st.button("Start", width='stretch')

# Animation
if start_button:
    for frame in range(total_frames):
        
        # 3D trajectory animation
        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter3d(
            x=full_truth[:, 0], y=full_truth[:, 1], z=full_truth[:, 2],
            mode="lines", line=dict(color="blue", width=4),
            opacity=0.3, name="Ground Truth", showlegend=False
        ))
        fig_traj.add_trace(go.Scatter3d(
            x=full_pred[:, 0], y=full_pred[:, 1], z=full_pred[:, 2],
            mode="lines", line=dict(color="red", width=4, dash="dot"),
            opacity=0.3, name="Prediction", showlegend=False
        ))
        fig_traj.add_trace(go.Scatter3d(
            x=[full_truth[frame, 0]], y=[full_truth[frame, 1]], z=[full_truth[frame, 2]],
            mode="markers", marker=dict(size=6, color='blue'),
            name="Ground Truth"
        ))
        fig_traj.add_trace(go.Scatter3d(
            x=[full_pred[frame, 0]], y=[full_pred[frame, 1]], z=[full_pred[frame, 2]],
            mode="markers", marker=dict(size=6, color='red'),
            name="Prediction"
        ))
        if vt < total_frames:
            fig_traj.add_trace(go.Scatter3d(
                x=[full_truth[vt, 0]], y=[full_truth[vt, 1]], z=[full_truth[vt, 2]],
                mode="markers", marker=dict(size=4, symbol="diamond", color="black"),
                name="Threshold Hit"
            ))
        fig_traj.update_layout(
            scene=dict(
                xaxis=dict(range=[xmin, xmax], showticklabels=False),
                yaxis=dict(range=[ymin, ymax], showticklabels=False),
                zaxis=dict(range=[zmin, zmax], showticklabels=False),
                aspectmode="cube",
                domain=dict(x=[0, 1], y=[0, 1])
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=300,
            showlegend=True,
            legend=dict(x=0, y=1, font=dict(size=10))
        )
        traj_3d_slot.plotly_chart(fig_traj, width='stretch', key=f"3d-traj-{frame}")

        # Error animation
        fig_error = go.Figure()
        fig_error.add_trace(go.Scatter(
            x=np.arange(frame + 1), y=E[:frame + 1],
            mode="lines", line=dict(color="orange", width=3),
            name="Error", showlegend=False
        ))
        fig_error.add_hline(y=threshold, line_color="red", annotation_text="Threshold")
        if vt <= frame:
            fig_error.add_trace(go.Scatter(
                x=[vt], y=[E[vt]],
                mode="markers", marker=dict(size=12, symbol="diamond", color="red"),
                name="Valid Time"
            ))
        fig_error.update_layout(
            xaxis_title="Step", yaxis_title="Error (m)",
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=True,
            legend=dict(x=0.7, y=1.08, yanchor="bottom")
        )
        error_slot.plotly_chart(fig_error, width='stretch', key=f"error-{frame}")

        # Per-axis: ground truth static, prediction animated
        fig_per_axis = go.Figure()
        
        # tx subplot
        fig_per_axis.add_trace(go.Scatter(
            x=t_axis, y=truth[:, 0],
            mode="lines", line=dict(color="blue", width=3),
            name="Ground Truth", legendgroup="truth", showlegend=True,
            xaxis="x1", yaxis="y1"
        ))
        fig_per_axis.add_trace(go.Scatter(
            x=t_axis[:frame+1], y=pred[:frame+1, 0],
            mode="lines", line=dict(color="red", width=3, dash="dash"),
            name="Prediction", legendgroup="pred", showlegend=True,
            xaxis="x1", yaxis="y1"
        ))
        
        # ty subplot
        fig_per_axis.add_trace(go.Scatter(
            x=t_axis, y=truth[:, 1],
            mode="lines", line=dict(color="blue", width=3),
            name="Ground Truth", legendgroup="truth", showlegend=False,
            xaxis="x2", yaxis="y2"
        ))
        fig_per_axis.add_trace(go.Scatter(
            x=t_axis[:frame+1], y=pred[:frame+1, 1],
            mode="lines", line=dict(color="red", width=3, dash="dash"),
            name="Prediction", legendgroup="pred", showlegend=False,
            xaxis="x2", yaxis="y2"
        ))
        
        # tz subplot
        fig_per_axis.add_trace(go.Scatter(
            x=t_axis, y=truth[:, 2],
            mode="lines", line=dict(color="blue", width=3),
            name="Ground Truth", legendgroup="truth", showlegend=False,
            xaxis="x3", yaxis="y3"
        ))
        fig_per_axis.add_trace(go.Scatter(
            x=t_axis[:frame+1], y=pred[:frame+1, 2],
            mode="lines", line=dict(color="red", width=3, dash="dash"),
            name="Prediction", legendgroup="pred", showlegend=False,
            xaxis="x3", yaxis="y3"
        ))
        
        # Valid time vertical lines
        if vt <= frame:
            for i in range(3):
                fig_per_axis.add_vline(
                    x=vt, line_dash="dash", line_color="black", line_width=3,
                    xref=f"x{i+1}", yref=f"y{i+1}"
                )
        
        fig_per_axis.update_layout(
            xaxis=dict(domain=[0, 1], anchor='y1', range=[0, len(t_axis)], showticklabels=False),
            yaxis=dict(domain=[0.72, 1], anchor='x1', title="tx", range=tx_range),
            xaxis2=dict(domain=[0, 1], anchor='y2', range=[0, len(t_axis)], showticklabels=False),
            yaxis2=dict(domain=[0.38, 0.66], anchor='x2', title="ty", range=ty_range),
            xaxis3=dict(domain=[0, 1], anchor='y3', range=[0, len(t_axis)], title="Step"),
            yaxis3=dict(domain=[0.0, 0.28], anchor='x3', title="tz", range=tz_range),
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True,
            legend=dict(x=1.02, y=0.5, xanchor="left", yanchor="middle", font=dict(size=10))
        )
        per_axis_slot.plotly_chart(fig_per_axis, width='stretch', key=f"per-axis-{frame}")

        time.sleep(1 / FPS)
    
    # Display statistics after animation
    rmse = np.sqrt(np.mean(E**2))
    ade = np.mean(E)
    fde = E[-1]
    
    stats_html = f"""
    <div style="background-color: #1f77b4; color: white; padding: 20px; border-radius: 10px; height: 100%;">
        <p style="margin: 10px 0; font-size: 16px;"><strong>ADE:</strong> {ade:.4f} m</p>
        <p style="margin: 10px 0; font-size: 16px;"><strong>RMSE:</strong> {rmse:.4f} m</p>
        <p style="margin: 10px 0; font-size: 16px;"><strong>FDE:</strong> {fde:.4f} m</p>
        <p style="margin: 10px 0; font-size: 16px;"><strong>Valid Time:</strong> {vt} steps</p>
    </div>
    """
    stats_slot.markdown(stats_html, unsafe_allow_html=True)