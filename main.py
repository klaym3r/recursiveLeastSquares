"""
mag_calib_plot.py

Читает файл с большим количеством строк, извлекает колонки (по умолчанию 0-based 8,9,10),
подгоняет эллипсоид к сырым значениям магнитометра и строит два 3D scatter:
  - "Raw" (сырые точки)
  - "Calibrated" (после преобразования, эллипсоид -> сфера)
Графики сохраняются в HTML (Plotly), который можно открыть в браузере.

Настройте filepath прямо в __main__.
"""
from typing import Sequence, Optional, Tuple
import os
import numpy as np

from histogramByDistance import plot_distance_histogram 

# ---- Ellipsoid fit / calibration (как раньше) ----
def fit_ellipsoid(raw: np.ndarray):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("raw must be (N,3) array")
    N = raw.shape[0]
    if N < 9:
        print("Warning: fitting ellipsoid with <9 points — solution may be unstable.")

    x = raw[:,0]; y = raw[:,1]; z = raw[:,2]
    D = np.column_stack([x*x, y*y, z*z, x*y, x*z, y*z, x, y, z, np.ones(N)])
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    v = Vt.T[:, -1]

    def unpack(v):
        A, B, C, D12, D13, D23, G, H, I, J = v
        Q = np.array([[A, D12/2.0, D13/2.0],
                      [D12/2.0, B, D23/2.0],
                      [D13/2.0, D23/2.0, C]])
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
            raise ValueError(f"Fitted quadric does not represent an ellipsoid (Jc={Jc}).")

    Qs = Q / (-Jc)
    evals, evecs = np.linalg.eigh(Qs)
    evals = np.clip(evals, 1e-12, None)
    S = np.diag(np.sqrt(evals)).dot(evecs.T)
    M = evecs.dot(np.diag(1.0/np.sqrt(evals)))

    params = {
        'center': center,
        'transform': S,
        'inv_transform': M,
        'eigvals': evals,
        'eigvecs': evecs,
        'Jc': Jc
    }
    return params

def apply_calibration(raw: np.ndarray, params: dict, scale_to: Optional[float] = 1.0) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    center = params['center']
    S = params['transform']
    cal = (S.dot((raw - center).T)).T
    if scale_to is not None:
        norms = np.linalg.norm(cal, axis=1)
        mean_norm = np.mean(norms) if norms.size > 0 else 0.0
        if mean_norm > 0:
            cal = cal * (scale_to / mean_norm)
    return cal

# ---- File loader ----
def load_magnetometer_raw(path: str,
                          cols: Sequence[int] = (8,9,10),
                          max_lines: Optional[int] = None,
                          progress_every: int = 50000) -> np.ndarray:
    """
    Read file line-by-line, extract float values at positions cols (0-based).
    Returns numpy array (M,3).
    """
    data = []
    max_col = max(cols)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
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
            if (i+1) % progress_every == 0:
                print(f"Read {i+1} lines, collected {len(data)} vectors...")
    if len(data) == 0:
        return np.empty((0,3), dtype=float)
    return np.array(data, dtype=float)

# ---- Plotting with plotly ----
def plot_raw_vs_calibrated(raw: np.ndarray,
                           calibrated: np.ndarray,
                           out_html: str = 'mag_raw_vs_calibrated.html',
                           max_plot_points: int = 20000,
                           marker_size: float = 2.0):
    """
    Создаёт HTML с двумя 3D scatter (raw и calibrated).
    Если данных > max_plot_points, выполняется детерминированная подвыборка.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    assert raw.shape == calibrated.shape
    N = raw.shape[0]

    # Deterministic sampling: take indices linspace to keep distribution
    if N > max_plot_points:
        idx = np.linspace(0, N-1, max_plot_points, dtype=int)
        raw_plot = raw[idx]
        cal_plot = calibrated[idx]
        print(f"Plotting subsample {len(idx)} / {N} points (max_plot_points={max_plot_points}).")
    else:
        raw_plot = raw
        cal_plot = calibrated
        print(f"Plotting all {N} points.")

    # color by norm to give visual depth
    raw_norms = np.linalg.norm(raw_plot, axis=1)
    cal_norms = np.linalg.norm(cal_plot, axis=1)

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                        subplot_titles=['Raw magnetometer', 'Calibrated (ellipsoid->sphere)'])

    fig.add_trace(
        go.Scatter3d(x=raw_plot[:,0], y=raw_plot[:,1], z=raw_plot[:,2],
                     mode='markers',
                     marker=dict(size=marker_size, color=raw_norms, colorbar=dict(title='norm'), showscale=True),
                     name='raw'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter3d(x=cal_plot[:,0], y=cal_plot[:,1], z=cal_plot[:,2],
                     mode='markers',
                     marker=dict(size=marker_size, color=cal_norms, colorbar=dict(title='norm'), showscale=True),
                     name='calibrated'),
        row=1, col=2
    )

    fig.update_layout(height=700, width=1400,
                      title_text="Magnetometer: raw vs calibrated",
                      scene=dict(aspectmode='data'),
                      scene2=dict(aspectmode='data'))

    # Save to HTML
    import plotly.offline as pyo
    pyo.plot(fig, filename=out_html, auto_open=False)
    print(f"Saved interactive plot to '{out_html}'. Open it in a browser to inspect the results.")

# ---- Main execution (no argparse) ----
if __name__ == '__main__':
    # === Настройки (изменяйте тут) ===
    filepath = 'data2.txt'          # <- укажите ваш файл здесь
    cols = (8, 9, 10)               # 0-based индексы колонок, которые интересуют
    scale_to = 1.0                  # масштаб результата (1.0 => unit sphere). Можно поставить ≈50 для µT
    max_plot_points = 20000         # максимальное число точек для отрисовки (подвыборка при больших файлах)
    # ==================================

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Input file not found: '{filepath}'. Убедитесь, что путь указан верно.")

    print("Loading raw magnetometer data from file...")
    raw = load_magnetometer_raw(filepath, cols=cols)
    if raw.shape[0] == 0:
        raise ValueError("No valid vectors were extracted from the file. Проверьте формат/индексы колонок.")

    print(f"Loaded {raw.shape[0]} vectors. Fitting ellipsoid (this may take a moment)...")
    params = fit_ellipsoid(raw)
    calibrated = apply_calibration(raw, params, scale_to=scale_to)

    print("Fit complete.")
    print("Center (hard-iron offset):", params['center'])
    print("Eigenvalues (shape):", params['eigvals'])

    # Plot and save HTML
    plot_raw_vs_calibrated(raw, calibrated, out_html='mag_raw_vs_calibrated.html', max_plot_points=max_plot_points)

    # гистограмма
    distances, fig = plot_distance_histogram(calibrated=calibrated, out_html='hist_calibrated.html')
