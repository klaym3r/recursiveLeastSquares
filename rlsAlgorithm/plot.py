import numpy as np

# Импортируем настройки
try:
    from config import CFG_PLOT
except ImportError:
    print("ОШИBКА: Не удалось импортировать настройки. Убедитесь, что config.py существует.")
    # Запасные значения
    CFG_PLOT = {
        "plot_output_html": "mag_raw_vs_calibrated.html",
        "plot_max_points": 20000,
        "plot_marker_size": 2.0,
        "plot_auto_open": True,
        "plot_3d_height": 700,
        "plot_3d_width": 1400,
        "histogram_bins": 60,
        "histogram_output_html": "hist_calibrated.html",
        "histogram_show_stats": True,
        "histogram_auto_open": False,
        "histogram_height": 500,
        "histogram_width": 800
    }

# ---- Plotting with plotly ----
def plot_raw_vs_calibrated(
    raw: np.ndarray,
    calibrated: np.ndarray,
    out_html: str = None,
    max_plot_points: int = None,
    marker_size: float = None,
    auto_open: bool = None,
):
    """
    Создаёт HTML с двумя 3D scatter (raw и calibrated).
    Параметры, если None, берутся из config.json.
    """
    # Загружаем значения из конфига, если они не заданы
    if out_html is None: out_html = CFG_PLOT["plot_output_html"]
    if max_plot_points is None: max_plot_points = CFG_PLOT["plot_max_points"]
    if marker_size is None: marker_size = CFG_PLOT["plot_marker_size"]
    if auto_open is None: auto_open = CFG_PLOT["plot_auto_open"]

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.offline as pyo

    # ... (остальная логика функции не меняется) ...
    
    assert raw.shape == calibrated.shape
    N = raw.shape[0]

    if N > max_plot_points:
        idx = np.linspace(0, N - 1, max_plot_points, dtype=int)
        raw_plot = raw[idx]
        cal_plot = calibrated[idx]
        print(
            f"Plotting subsample {len(idx)} / {N} points (max_plot_points={max_plot_points})."
        )
    else:
        raw_plot = raw
        cal_plot = calibrated
        print(f"Plotting all {N} points.")

    raw_norms = np.linalg.norm(raw_plot, axis=1)
    cal_norms = np.linalg.norm(cal_plot, axis=1)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=["Raw magnetometer", "Calibrated (ellipsoid->sphere)"],
    )
    
    # ... (код добавления trace не меняется) ...
    fig.add_trace(
        go.Scatter3d(
            x=raw_plot[:, 0], y=raw_plot[:, 1], z=raw_plot[:, 2], mode="markers",
            marker=dict(size=marker_size, color=raw_norms, colorbar=dict(title="norm"), showscale=True),
            name="raw",
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=cal_plot[:, 0], y=cal_plot[:, 1], z=cal_plot[:, 2], mode="markers",
            marker=dict(size=marker_size, color=cal_norms, colorbar=dict(title="norm"), showscale=True),
            name="calibrated",
        ), row=1, col=2,
    )
    
    fig.update_layout(
        height=CFG_PLOT["plot_3d_height"],
        width=CFG_PLOT["plot_3d_width"],
        title_text="Magnetometer: raw vs calibrated",
        scene=dict(aspectmode="data"),
        scene2=dict(aspectmode="data"),
    )

    pyo.plot(fig, filename=out_html, auto_open=auto_open)
    print(
        f"Saved interactive plot to '{out_html}'. Open it in a browser to inspect the results."
    )


# ---- Histogram function integrated here ----
def plot_distance_histogram(
    *,
    calibrated: np.ndarray = None,
    raw: np.ndarray = None,
    params: dict = None,
    bins: int = None,
    out_html: str = None,
    show_stats: bool = None,
    scale_to: float = None,
    auto_open: bool = None,
):
    """
    Строит интерактивную гистограмму расстояний.
    Параметры, если None, берутся из config.json.
    """
    # Загружаем значения из конфига, если они не заданы
    if bins is None: bins = CFG_PLOT["histogram_bins"]
    if out_html is None: out_html = CFG_PLOT["histogram_output_html"]
    if show_stats is None: show_stats = CFG_PLOT["histogram_show_stats"]
    if auto_open is None: auto_open = CFG_PLOT["histogram_auto_open"]

    import plotly.graph_objects as go
    import plotly.offline as pyo
    
    # ... (остальная логика функции не меняется) ...
    cal = np.asarray(calibrated, dtype=float)
    if cal.ndim != 2 or cal.shape[1] != 3:
        raise ValueError("Координаты должны иметь форму (N,3).")

    distances = np.linalg.norm(cal, axis=1)
    mean = float(np.mean(distances))
    med = float(np.median(distances))
    std = float(np.std(distances))
    mn = float(np.min(distances))
    mx = float(np.max(distances))
    p90 = float(np.percentile(distances, 90))

    if show_stats:
        print(
            f"Points: {len(distances)}  mean={mean:.6f}  median={med:.6f}  std={std:.6f}  min={mn:.6f}  max={mx:.6f}  90%={p90:.6f}"
        )

    hist = go.Histogram(
        x=distances, nbinsx=bins, marker=dict(line=dict(width=0.5, color="white"))
    )

    layout = go.Layout(
        title="Histogram of distances to sphere center",
        xaxis=dict(title="Distance"),
        yaxis=dict(title="Count"),
        # ... (остальная часть layout не меняется) ...
        shapes=[
            dict(type="line", x0=mean, x1=mean, y0=0, y1=1, yref="paper", line=dict(color="red", width=2, dash="dash")),
            dict(type="line", x0=med, x1=med, y0=0, y1=1, yref="paper", line=dict(color="green", width=2, dash="dot")),
        ],
        annotations=[
            dict(x=mean, y=1.02, xref="x", yref="paper", text=f"mean={mean:.4f}", showarrow=False, font=dict(color="red")),
            dict(x=med, y=1.06, xref="x", yref="paper", text=f"median={med:.4f}", showarrow=False, font=dict(color="green")),
        ],
        bargap=0.02,
        height=CFG_PLOT["histogram_height"],
        width=CFG_PLOT["histogram_width"],
    )

    fig = go.Figure(data=[hist], layout=layout)

    try:
        xs = np.linspace(mn, mx, 200)
        from scipy.stats import norm
        pdf = norm.pdf(xs, loc=mean, scale=std)
        max_count = np.histogram(distances, bins=bins)[0].max()
        if pdf.max() > 0:
            pdf_scaled = pdf * (max_count / pdf.max())
            fig.add_trace(go.Scatter(x=xs, y=pdf_scaled, mode="lines", name="normal approx", line=dict(color="black")))
    except Exception:
        pass

    pyo.plot(fig, filename=out_html, auto_open=auto_open)
    print(f"Saved histogram to '{out_html}'.")
    return distances, fig
