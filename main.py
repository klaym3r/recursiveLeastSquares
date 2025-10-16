"""
mag_calib_plot.py

Читает файл с большим количеством строк, извлекает колонки (по умолчанию 0-based 8,9,10),
подгоняет эллипсоид к сырым значениям магнитометра и строит два 3D scatter:
  - "Raw" (сырые точки)
  - "Calibrated" (после преобразования, эллипсоид -> сфера)

Также строит интерактивную гистограмму расстояний точек до центра сферы.

Графики сохраняются в HTML (Plotly), который можно открыть в браузере.

Настройте filepath прямо в __main__.
"""

from typing import Sequence, Optional, Tuple
import os
import numpy as np


# ---- Ellipsoid fit / calibration (как раньше) ----
def fit_ellipsoid(raw: np.ndarray):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("raw must be (N,3) array")
    N = raw.shape[0]
    if N < 9:
        print("Warning: fitting ellipsoid with <9 points — solution may be unstable.")

    x = raw[:, 0]
    y = raw[:, 1]
    z = raw[:, 2]
    D = np.column_stack([x * x, y * y, z * z, x * y, x * z, y * z, x, y, z, np.ones(N)])
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    v = Vt.T[:, -1]

    def unpack(v):
        A, B, C, D12, D13, D23, G, H, I, J = v
        Q = np.array(
            [
                [A, D12 / 2.0, D13 / 2.0],
                [D12 / 2.0, B, D23 / 2.0],
                [D13 / 2.0, D23 / 2.0, C],
            ]
        )
        p_vec = np.array([G, H, I])
        return Q, p_vec, J

    Q, p_vec, J = unpack(v)

    def center_and_Jc(Q, p_vec, J):
        Qinv = np.linalg.inv(Q)
        center = -0.5 * Qinv.dot(p_vec)
        Jc = center.dot(Q.dot(center)) + p_vec.dot(center) + J
        return center, Jc, Qinv

    center, Jc, Qinv = center_and_Jc(Q, p_vec, J)
    if -Jc <= 0:
        v = -v
        Q, p_vec, J = unpack(v)
        center, Jc, Qinv = center_and_Jc(Q, p_vec, J)
        if -Jc <= 0:
            raise ValueError(
                f"Fitted quadric does not represent an ellipsoid (Jc={Jc})."
            )

    Qs = Q / (-Jc)
    evals, evecs = np.linalg.eigh(Qs)
    evals = np.clip(evals, 1e-12, None)
    S = np.diag(np.sqrt(evals)).dot(evecs.T)
    M = evecs.dot(np.diag(1.0 / np.sqrt(evals)))

    params = {
        "center": center,
        "transform": S,
        "inv_transform": M,
        "eigvals": evals,
        "eigvecs": evecs,
        "Jc": Jc,
    }
    return params


def apply_calibration(
    raw: np.ndarray, params: dict, scale_to: Optional[float] = 1.0
) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    center = params["center"]
    S = params["transform"]
    cal = (S.dot((raw - center).T)).T
    if scale_to is not None:
        norms = np.linalg.norm(cal, axis=1)
        mean_norm = np.mean(norms) if norms.size > 0 else 0.0
        if mean_norm > 0:
            cal = cal * (scale_to / mean_norm)
    return cal


# ---- File loader ----
def load_magnetometer_raw(
    path: str,
    cols: Sequence[int] = (8, 9, 10),
    max_lines: Optional[int] = None,
    progress_every: int = 50000,
) -> np.ndarray:
    """
    Read file line-by-line, extract float values at positions cols (0-based).
    Returns numpy array (M,3).
    """
    data = []
    max_col = max(cols)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) <= max_col:
                continue
            try:
                triple = [float(parts[c]) for c in cols]
            except (ValueError, IndexError):
                continue
            data.append(triple)
            if max_lines is not None and len(data) >= max_lines:
                break
            if (i + 1) % progress_every == 0:
                print(f"Read {i+1} lines, collected {len(data)} vectors...")
    if len(data) == 0:
        return np.empty((0, 3), dtype=float)
    return np.array(data, dtype=float)


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

    # Проверка входа
    if calibrated is None:
        if raw is None or params is None:
            raise ValueError(
                "Нужно либо передать `calibrated`, либо `raw` и `params` вместе."
            )
        cal = apply_calibration(raw, params, scale_to=scale_to)
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


def pca_diagnostics(raw: np.ndarray):
    """Возвращает собственные значения (desc), собственные векторы (cols), и отношения."""
    raw = np.asarray(raw, dtype=float)
    C = np.cov(raw.T)
    evals, evecs = np.linalg.eigh(C)  # ascending
    evals = evals[::-1]
    evecs = evecs[:, ::-1]
    ratios = evals / (evals.sum() + 1e-15)
    return evals, evecs, ratios


def fit_circle_in_plane(raw: np.ndarray):
    """
    1) Находит плоскость через PCA (две главные компоненты),
    2) Проецирует точки на эту плоскость,
    3) Подгоняет окружность алгебраическим методом,
    4) Возвращает center3d, radius, plane basis (u,v), plane_origin (mean), residuals.
    """
    raw = np.asarray(raw, dtype=float)
    mean = raw.mean(axis=0)
    C = np.cov(raw.T)
    evals, evecs = np.linalg.eigh(C)
    # выбираем две главные компоненты (наибольшие)
    u = evecs[:, -1]  # largest
    v = evecs[:, -2]  # second
    # ортонормируем u,v на всякий случай
    u = u / np.linalg.norm(u)
    v = v - u * np.dot(v, u)
    v = v / np.linalg.norm(v)

    rel = raw - mean
    X = np.column_stack([rel.dot(u), rel.dot(v)])  # N x 2

    x = X[:, 0]
    y = X[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    # robust lstsq
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b0, c = sol
    center2 = np.array([a, b0])
    # radius
    rad_sq = c + a * a + b0 * b0
    if rad_sq <= 0:
        raise ValueError("Computed non-positive radius^2 in circle fit.")
    radius = np.sqrt(rad_sq)
    center3d = mean + a * u + b0 * v

    dists = np.sqrt((X[:, 0] - a) ** 2 + (X[:, 1] - b0) ** 2)
    residuals = dists - radius

    return {
        "center3d": center3d,
        "radius": radius,
        "residuals": residuals,
        "plane_u": u,
        "plane_v": v,
        "plane_origin": mean,
        "proj2d": X,
    }


def apply_calibration_with_fallback(
    raw: np.ndarray,
    ellipsoid_params: Optional[dict] = None,
    scale_to: Optional[float] = 1.0,
    planar_thresh: float = 0.05,
):
    """
    Универсальная функция калибровки:
     - если данные 3D (неплоские) -> стандартный ellipsoid transform (apply_calibration).
     - если данные почти плоские (малое отношение EV3/EV1 < planar_thresh) -> circle-in-plane fallback.
    Возвращает calibrated (N,3) — откалиброванные 3D-векторы.
    """
    raw = np.asarray(raw, dtype=float)
    evals, evecs, ratios = pca_diagnostics(raw)
    planar_ratio = evals[-1] / (evals[0] + 1e-15)

    if planar_ratio < planar_thresh:
        # planar: сделать 2D circle-fit и нормировать радиус
        try:
            circ = fit_circle_in_plane(raw)
        except Exception as e:
            # если circle fit упал, fallback на ellipsoid (если есть params) или просто центрирование
            if ellipsoid_params is not None:
                return apply_calibration(raw, ellipsoid_params, scale_to=scale_to)
            else:
                mean = raw.mean(axis=0)
                return raw - mean

        u = circ["plane_u"]
        v = circ["plane_v"]
        n = np.cross(u, v)
        mean = circ["plane_origin"]
        center3d = circ["center3d"]
        radius = circ["radius"]

        # Проекция на плоскость (2D)
        rel = raw - mean
        coords2 = np.column_stack([rel.dot(u), rel.dot(v)])  # N x 2
        # центр в 2D:
        center2 = np.array([np.dot(center3d - mean, u), np.dot(center3d - mean, v)])

        # нормируем радиальную компоненту: (coords2 - center2) * (scale_to / radius)
        if radius <= 0:
            scale = 1.0
        else:
            scale = (scale_to / radius) if scale_to is not None else 1.0

        rel2_norm = (coords2 - center2[None, :]) * scale  # N x 2

        # восстанавливаем 3D для плоскости
        cal_plane3d = (
            mean[None, :] + np.outer(rel2_norm[:, 0], u) + np.outer(rel2_norm[:, 1], v)
        )

        # перпендикулярная компонента (сохраняем вариацию, но центрируем)
        perp = rel.dot(n)  # signed distances along normal
        perp_mean = perp.mean()
        perp_centered = (
            perp - perp_mean
        )  # оставляем как небольшое смещение вдоль нормали
        cal = cal_plane3d + np.outer(perp_centered, n)

        # Дополнительно: если хочется, можно масштабировать так, чтобы средняя норма == scale_to:
        if scale_to is not None:
            norms = np.linalg.norm(cal, axis=1)
            mean_norm = norms.mean() if norms.size > 0 else 0.0
            if mean_norm > 0:
                cal = cal * (scale_to / mean_norm)

        return cal

    else:
        # не плоские -> используем полный ellipsoid (если нет params, попытка вызвать fit)
        if ellipsoid_params is None:
            # попробуем подогнать эллипсоид прямо
            params = fit_ellipsoid(raw)
        else:
            params = ellipsoid_params
        return apply_calibration(raw, params, scale_to=scale_to)


# ---- Main execution (no argparse) ----
if __name__ == "__main__":
    # === Настройки (изменяйте тут) ===
    filepath = "data/data4.txt"  # <- укажите ваш файл здесь
    cols = (8, 9, 10)  # 0-based индексы колонок, которые интересуют
    scale_to = (
        1.0  # масштаб результата (1.0 => unit sphere). Можно поставить ≈50 для µT
    )
    max_plot_points = (
        20000  # максимальное число точек для отрисовки (подвыборка при больших файлах)
    )
    # ==================================

    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Input file not found: '{filepath}'. Убедитесь, что путь указан верно."
        )

    print("Loading raw magnetometer data from file...")
    raw = load_magnetometer_raw(filepath, cols=cols)
    if raw.shape[0] == 0:
        raise ValueError(
            "No valid vectors were extracted from the file. Проверьте формат/индексы колонок."
        )

    print(
        f"Loaded {raw.shape[0]} vectors. Fitting ellipsoid (this may take a moment)..."
    )
    params = None
    try:
        params = fit_ellipsoid(raw)
    except Exception:
        params = None
    calibrated = apply_calibration_with_fallback(raw, params, scale_to=scale_to)

    print("Fit complete.")
    print("Center (hard-iron offset):", params["center"])
    print("Eigenvalues (shape):", params["eigvals"])

    # Plot and save HTML (opens browser by default; change auto_open if нужно)
    plot_raw_vs_calibrated(
        raw,
        calibrated,
        out_html="mag_raw_vs_calibrated.html",
        max_plot_points=max_plot_points,
        auto_open=True,
    )

    # гистограмма расстояний (не открываем автоматически, чтобы не мешать)
    distances, hist_fig = plot_distance_histogram(
        calibrated=calibrated, out_html="hist_calibrated.html", auto_open=False
    )
