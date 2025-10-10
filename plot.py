from __future__ import annotations
import numpy as np


def build_plotly_html(
    raw_arr: np.ndarray,
    corr_arr: np.ndarray,
    params: dict,
    out_html: str,
    max_points: int = 80000,
    downsample_seed: int = 42,
    auto_open: bool = True,
):
    """
    Построить interactive HTML с тремя 3D графиками:
      1) Raw (before)
      2) Corrected (after)
      3) Overlay (raw + corrected в одном пространстве)

    Делает опциональный downsample, чтобы не перегружать браузер.
    Явно выставляет aspectmode='data' для всех сцен.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Downsample (если много точек) — стратифицированная случайная выборка
    def maybe_downsample(arr: np.ndarray, max_pts: int):
        if arr is None:
            return np.empty((0, 3))
        if arr.size == 0:
            return arr
        n = arr.shape[0]
        if n <= max_pts:
            return arr
        rng = np.random.default_rng(downsample_seed)
        idx = np.sort(rng.choice(n, size=max_pts, replace=False))
        return arr[idx]

    raw_plot = maybe_downsample(raw_arr, max_points)
    corr_plot = maybe_downsample(corr_arr, max_points)

    # 3 колонки: raw | corrected | overlay
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Raw (before)", "Corrected (after)", "Overlay (raw + after)"),
    )

    # --- Raw plot (col 1) ---
    if raw_plot.size:
        fig.add_trace(
            go.Scatter3d(
                x=raw_plot[:, 0],
                y=raw_plot[:, 1],
                z=raw_plot[:, 2],
                mode="markers",
                marker=dict(size=2, color="blue", opacity=0.7),
                name="raw",
            ),
            row=1,
            col=1,
        )

    est_center = params.get("center") if params.get("center") is not None else np.zeros(3)
    fig.add_trace(
        go.Scatter3d(
            x=[est_center[0]],
            y=[est_center[1]],
            z=[est_center[2]],
            mode="markers",
            marker=dict(size=6, symbol="x", color="black"),
            name="est_center",
        ),
        row=1,
        col=1,
    )

    # --- Corrected plot (col 2) ---
    if corr_plot.size:
        fig.add_trace(
            go.Scatter3d(
                x=corr_plot[:, 0],
                y=corr_plot[:, 1],
                z=corr_plot[:, 2],
                mode="markers",
                marker=dict(size=2, color="red", opacity=0.7),
                name="corrected_after",
            ),
            row=1,
            col=2,
        )

    # unit sphere mesh on the after plot (col 2)
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(
        go.Surface(x=sx, y=sy, z=sz, opacity=0.25, showscale=False, name="unit_sphere"),
        row=1,
        col=2,
    )

    # --- Overlay plot (col 3): both raw and corrected in same axes ---
    # Draw raw (blue, slightly transparent)
    if raw_plot.size:
        fig.add_trace(
            go.Scatter3d(
                x=raw_plot[:, 0],
                y=raw_plot[:, 1],
                z=raw_plot[:, 2],
                mode="markers",
                marker=dict(size=2, color="blue", opacity=0.4),
                name="raw_overlay",
            ),
            row=1,
            col=3,
        )
    # Draw corrected (red) over it
    if corr_plot.size:
        fig.add_trace(
            go.Scatter3d(
                x=corr_plot[:, 0],
                y=corr_plot[:, 1],
                z=corr_plot[:, 2],
                mode="markers",
                marker=dict(size=2, color="red", opacity=0.6),
                name="corrected_overlay",
            ),
            row=1,
            col=3,
        )
    # add unit sphere to overlay for reference
    fig.add_trace(
        go.Surface(x=sx, y=sy, z=sz, opacity=0.18, showscale=False, name="unit_sphere_overlay"),
        row=1,
        col=3,
    )

    # Общие настройки layout — выставляем aspectmode='data' для всех сцен
    scene_common = dict(
        xaxis=dict(title="X", showspikes=False),
        yaxis=dict(title="Y", showspikes=False),
        zaxis=dict(title="Z", showspikes=False),
        aspectmode="data",  # сохраняет реальные числовые пропорции осей
    )

    fig.update_layout(
        height=720,
        width=1600,
        showlegend=True,
        title_text="RLS Magnetometer Calibration — before / after / overlay",
        margin=dict(l=10, r=10, t=60, b=10),
    )

    # применяем отдельно для scene, scene2, scene3
    fig.update_layout(scene=scene_common, scene2=scene_common, scene3=scene_common)

    # Настройка камеры (одинаковая для всех сцен)
    cam = dict(eye=dict(x=1.25, y=1.25, z=1.25))
    fig.update_layout(scene_camera=cam, scene2_camera=cam, scene3_camera=cam)

    print(f"Writing interactive HTML to {out_html} ...")
    fig.write_html(out_html, auto_open=auto_open)
    print("HTML written" + (" and opened in browser." if auto_open else "."))


def save_results(
    npz_path: str, raw_arr: np.ndarray, corr_arr: np.ndarray, params: dict
):
    """
    Сохраняет numpy-архив с результатами.
    """
    # Приводим пустые массивы к shape (0,3) чтобы в ходе загрузки не было неожиданностей
    if raw_arr is None or raw_arr.size == 0:
        raw_arr = np.empty((0, 3))
    if corr_arr is None or corr_arr.size == 0:
        corr_arr = np.empty((0, 3))

    np.savez_compressed(
        npz_path,
        raw=raw_arr,
        corrected_batch=corr_arr,
        center=params.get("center"),
        r=params.get("r"),
        theta=params.get("theta"),
    )
    print(f"Saved results to {npz_path}")
