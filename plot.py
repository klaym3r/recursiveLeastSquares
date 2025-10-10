from __future__ import annotations
import numpy as np

def build_plotly_html(raw_arr: np.ndarray, corr_arr: np.ndarray, params: dict, out_html: str):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], subplot_titles=("Raw (before)", "Corrected (after)"))

    if raw_arr.size:
        fig.add_trace(go.Scatter3d(x=raw_arr[:, 0], y=raw_arr[:, 1], z=raw_arr[:, 2], mode='markers', marker=dict(size=2), name='raw'), row=1, col=1)
    est_center = params.get('center') if params.get('center') is not None else np.zeros(3)
    fig.add_trace(go.Scatter3d(x=[est_center[0]], y=[est_center[1]], z=[est_center[2]], mode='markers', marker=dict(size=6, symbol='x'), name='est_center'), row=1, col=1)

    if corr_arr.size:
        fig.add_trace(go.Scatter3d(x=corr_arr[:, 0], y=corr_arr[:, 1], z=corr_arr[:, 2], mode='markers', marker=dict(size=2), name='corrected_after'), row=1, col=2)
    # unit sphere mesh on the after plot
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.2, showscale=False, name='unit_sphere'), row=1, col=2)

    fig.update_layout(height=700, width=1400, showlegend=True, title_text='RLS Magnetometer Calibration — before / after')

    print(f"Writing interactive HTML to {out_html} ...")
    fig.write_html(out_html, auto_open=True)
    print("HTML written and opened in browser (if available).")


def save_results(npz_path: str, raw_arr: np.ndarray, corr_arr: np.ndarray, params: dict):
    np.savez_compressed(npz_path, raw=raw_arr, corrected_batch=corr_arr, center=params.get('center'), r=params.get('r'), theta=params.get('theta'))
    print(f"Saved results to {npz_path}")
