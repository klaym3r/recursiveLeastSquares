import numpy as np

# ---- Plotting with plotly ----
def plot_raw_vs_calibrated(
    raw: np.ndarray,
    calibrated: np.ndarray,
    out_html: str = "mag_raw_vs_calibrated.html",
    max_plot_points: int = 20000,
    marker_size: float = 2.0,
    auto_open: bool = True,
):
    """
    Создаёт HTML с двумя 3D scatter (raw и calibrated).
    Если данных > max_plot_points, выполняется детерминированная подвыборка.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.offline as pyo

    assert raw.shape == calibrated.shape
    N = raw.shape[0]

    # Deterministic sampling: take indices linspace to keep distribution
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

    # color by norm to give visual depth
    raw_norms = np.linalg.norm(raw_plot, axis=1)
    cal_norms = np.linalg.norm(cal_plot, axis=1)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=["Raw magnetometer", "Calibrated (ellipsoid->sphere)"],
    )

    fig.add_trace(
        go.Scatter3d(
            x=raw_plot[:, 0],
            y=raw_plot[:, 1],
            z=raw_plot[:, 2],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=raw_norms,
                colorbar=dict(title="norm"),
                showscale=True,
            ),
            name="raw",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter3d(
            x=cal_plot[:, 0],
            y=cal_plot[:, 1],
            z=cal_plot[:, 2],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=cal_norms,
                colorbar=dict(title="norm"),
                showscale=True,
            ),
            name="calibrated",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=700,
        width=1400,
        title_text="Magnetometer: raw vs calibrated",
        scene=dict(aspectmode="data"),
        scene2=dict(aspectmode="data"),
    )

    # Save to HTML
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
    bins: int = 60,
    out_html: str = "hist_calibrated.html",
    show_stats: bool = True,
    scale_to: float = None,
    auto_open: bool = False,
):
    """
    Строит интерактивную гистограмму расстояний точек до центра сферы.
    Входные варианты:
      - calibrated: (N,3) — уже откалиброванные точки (центр = [0,0,0]).
      - raw и params: raw (N,3) + params (результат fit_ellipsoid) -> сначала выполняется apply_calibration(raw, params, scale_to).
    Параметры:
      bins     - число корзин гистограммы
      out_html - имя выходного HTML-файла с интерактивным графиком (plotly)
      show_stats- печатать среднее/медиану/стд
      scale_to  - если задан и мы используем raw+params, передаётся в apply_calibration
      auto_open - открывать ли HTML автоматически (по умолчанию False)
    Возвращает: (distances, fig) — массив расстояний и объект plotly Figure.
    """
    import plotly.graph_objects as go
    import plotly.offline as pyo

    cal = np.asarray(calibrated, dtype=float)
    if cal.ndim != 2 or cal.shape[1] != 3:
        raise ValueError("Координаты должны иметь форму (N,3).")

    # расстояния до центра сферы (0,0,0)
    distances = np.linalg.norm(cal, axis=1)

    # статистика
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
        shapes=[
            dict(
                type="line",
                x0=mean,
                x1=mean,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="red", width=2, dash="dash"),
            ),
            dict(
                type="line",
                x0=med,
                x1=med,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="green", width=2, dash="dot"),
            ),
        ],
        annotations=[
            dict(
                x=mean,
                y=1.02,
                xref="x",
                yref="paper",
                text=f"mean={mean:.4f}",
                showarrow=False,
                font=dict(color="red"),
            ),
            dict(
                x=med,
                y=1.06,
                xref="x",
                yref="paper",
                text=f"median={med:.4f}",
                showarrow=False,
                font=dict(color="green"),
            ),
        ],
        bargap=0.02,
        height=500,
        width=800,
    )

    fig = go.Figure(data=[hist], layout=layout)

    # попытка добавить нормальную аппроксимацию, если scipy есть
    try:
        xs = np.linspace(mn, mx, 200)
        from scipy.stats import norm

        pdf = norm.pdf(xs, loc=mean, scale=std)
        max_count = np.histogram(distances, bins=bins)[0].max()
        if pdf.max() > 0:
            pdf_scaled = pdf * (max_count / pdf.max())
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=pdf_scaled,
                    mode="lines",
                    name="normal approx",
                    line=dict(color="black"),
                )
            )
    except Exception:
        pass

    pyo.plot(fig, filename=out_html, auto_open=auto_open)
    print(f"Saved histogram to '{out_html}'.")
    return distances, fig
