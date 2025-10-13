import numpy as np


def plot_distance_histogram(
    *,
    calibrated: np.ndarray = None,
    raw: np.ndarray = None,
    params: dict = None,
    bins: int = 60,
    out_html: str = "mag_distance_hist.html",
    show_stats: bool = True,
    scale_to: float = None,
):
    """
    Строит гистограмму расстояний точек до центра сферы.
    Входные варианты:
      - calibrated: (N,3) — уже откалиброванные точки (центр = [0,0,0]).
      - raw и params: raw (N,3) + params (результат fit_ellipsoid) -> сначала выполняется apply_calibration(raw, params, scale_to).
    Параметры:
      bins     - число корзин гистограммы
      out_html - имя выходного HTML-файла с интерактивным графиком (plotly)
      show_stats- печатать среднее/медиану/стд
      scale_to  - если задан и мы используем raw+params, передаётся в apply_calibration
    Возвращает: (distances, fig) — массив расстояний и объект plotly Figure.
    """

    # Проверка входа
    if calibrated is None:
        if raw is None or params is None:
            raise ValueError(
                "Нужно либо передать `calibrated`, либо `raw` и `params` вместе."
            )
        # lazy-import apply_calibration from вашего модуля; предполагается, что функция присутствует
        # если apply_calibration определена в том же файле, просто вызовите её напрямую
        try:
            cal = apply_calibration(raw, params, scale_to=scale_to)
        except NameError:
            raise RuntimeError(
                "Функция apply_calibration не найдена. Поместите plot_distance_histogram в тот же модуль, где определена apply_calibration."
            )
    else:
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

    # Plotly histogram
    import plotly.graph_objects as go
    import plotly.offline as pyo

    hist = go.Histogram(
        x=distances, nbinsx=bins, marker=dict(line=dict(width=0.5, color="white"))
    )
    layout = go.Layout(
        title="Histogram of distances to sphere center",
        xaxis=dict(title="Distance"),
        yaxis=dict(title="Count"),
        shapes=[
            # vertical lines for mean and median
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

    # если хотим, можно добавить распределение нормального приближения (необязательно)
    try:
        xs = np.linspace(mn, mx, 200)
        from scipy.stats import norm

        pdf = norm.pdf(xs, loc=mean, scale=std)
        # масштабируем pdf к высоте гистограммы
        max_count = np.histogram(distances, bins=bins)[0].max()
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
        # scipy не обязателен — просто пропускаем, если его нет
        pass

    # Сохраняем интерактивный html
    pyo.plot(fig, filename=out_html, auto_open=False)
    print(f"Saved histogram to '{out_html}'.")

    return distances, fig


# ---------------- Пример использования ----------------
# (предполагается, что в вашем модуле есть load_magnetometer_raw, fit_ellipsoid, apply_calibration)
if __name__ == "__main__":
    # Пример: если у вас уже есть calibrated (из apply_calibration)
    # distances, fig = plot_distance_histogram(calibrated=calibrated, out_html='hist_calibrated.html')

    # Или: если есть только raw и params:
    # distances, fig = plot_distance_histogram(raw=raw, params=params, out_html='hist_raw_center_dist.html', scale_to=1.0)

    pass
