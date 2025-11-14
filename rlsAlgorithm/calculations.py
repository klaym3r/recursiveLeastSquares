from typing import Sequence, Optional, Tuple
import numpy as np
import os

# Импортируем настройки
try:
    from config import CFG_LOAD, CFG_ALG
except ImportError:
    print("ОШИBКА: Не удалось импортировать настройки. Убедитесь, что config.py существует.")
    # Запасные значения
    CFG_LOAD = {
        "data_columns": (8, 9, 10),
        "load_max_lines": None,
        "load_progress_interval": 50000,
        "file_encoding": "utf-8",
        "file_read_errors": "ignore"
    }
    CFG_ALG = {
        "min_points_for_fit": 9,
        "ellipsoid_clip_min_eigenvalue": 1e-12,
        "pca_numerical_epsilon": 1e-15,
        "lstsq_rcond": None,
        "planar_detection_threshold": 0.05,
        "fallback_target_scale": 1.0,
        "use_analytic_transform": True
    }


# ---- Ellipsoid fit / calibration ----
def fit_ellipsoid(raw: np.ndarray):
    """
    Подгоняет эллипсоид к набору 3D-точек.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("raw must be (N,3) array")
    N = raw.shape[0]
    
    min_points = CFG_ALG["min_points_for_fit"]
    if N < min_points:
        print(f"Warning: fitting ellipsoid with {N} points (need >= {min_points}) — solution may be unstable.")

    x = raw[:, 0]
    y = raw[:, 1]
    z = raw[:, 2]
    D = np.column_stack([x * x, y * y, z * z, x * y, x * z, y * z, x, y, z, np.ones(N)])
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    v = Vt.T[:, -1]

    # ... (логика unpack не меняется) ...
    def unpack(v):
        A, B, C, D12, D13, D23, G, H, I, J = v
        Q = np.array([[A, D12/2.0, D13/2.0], [D12/2.0, B, D23/2.0], [D13/2.0, D23/2.0, C]])
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
    
    # Используем значение из config
    evals = np.clip(evals, CFG_ALG["ellipsoid_clip_min_eigenvalue"], None)

    D_sqrt = np.diag(np.sqrt(evals))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(evals))
    S = evecs.dot(D_sqrt).dot(evecs.T)
    M = evecs.dot(D_inv_sqrt).dot(evecs.T)

    params = {
        "center": center, "transform": S, "inv_transform": M,
        "eigvals": evals, "eigvecs": evecs, "Jc": Jc,
    }
    return params


def apply_calibration(raw: np.ndarray, params: dict) -> np.ndarray:
    """
    Применяет параметры калибровки к сырым данным.
    Масштабирует итоговую сферу к средней норме сырых данных.
    """
    raw = np.asarray(raw, dtype=float)
    center = params["center"]
    S = params["transform"]
    cal = (S.dot((raw - center).T)).T
    avg_raw_norm = np.mean(np.linalg.norm(raw, axis=1)) if raw.size > 0 else 0.0
    if avg_raw_norm == 0.0:
        return cal 
    current_norms = np.linalg.norm(cal, axis=1)
    current_mean_norm = np.mean(current_norms) if current_norms.size > 0 else 0.0
    if current_mean_norm > 0:
        scale_factor = avg_raw_norm / current_mean_norm
        cal = cal * scale_factor
    return cal


# ---- File loader ----
def load_magnetometer_raw(
    path: str,
    cols: Sequence[int] = None,
    max_lines: Optional[int] = None,
    progress_every: int = None,
) -> np.ndarray:
    """
    Читает файл, извлекает float-значения из колонок cols.
    Параметры, если None, берутся из config.json.
    """
    # Загружаем значения из конфига, если они не заданы
    if cols is None: cols = CFG_LOAD["data_columns"]
    if max_lines is None: max_lines = CFG_LOAD["load_max_lines"]
    if progress_every is None: progress_every = CFG_LOAD["load_progress_interval"]
    
    file_encoding = CFG_LOAD["file_encoding"]
    file_errors = CFG_LOAD["file_read_errors"]

    data = []
    max_col = max(cols)
    
    with open(path, "r", encoding=file_encoding, errors=file_errors) as f:
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
            if progress_every and (i + 1) % progress_every == 0:
                print(f"Read {i+1} lines, collected {len(data)} vectors...")
                
    if len(data) == 0:
        return np.empty((0, 3), dtype=float)
    return np.array(data, dtype=float)


def pca_diagnostics(raw: np.ndarray):
    """Возвращает собственные значения (desc), собственные векторы (cols), и отношения."""
    raw = np.asarray(raw, dtype=float)
    C = np.cov(raw.T)
    evals, evecs = np.linalg.eigh(C)  # ascending
    evals = evals[::-1]
    evecs = evecs[:, ::-1]
    # Используем значение из config
    ratios = evals / (evals.sum() + CFG_ALG["pca_numerical_epsilon"])
    return evals, evecs, ratios


def fit_circle_in_plane(raw: np.ndarray):
    """
    Подгоняет окружность в 2D (после PCA).
    """
    raw = np.asarray(raw, dtype=float)
    mean = raw.mean(axis=0)
    C = np.cov(raw.T)
    evals, evecs = np.linalg.eigh(C)
    u = evecs[:, -1]
    v = evecs[:, -2]
    u = u / np.linalg.norm(u)
    v = v - u * np.dot(v, u)
    v = v / np.linalg.norm(v)

    rel = raw - mean
    X = np.column_stack([rel.dot(u), rel.dot(v)])
    x = X[:, 0]
    y = X[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    
    # Используем значение из config
    sol, *_ = np.linalg.lstsq(A, b, rcond=CFG_ALG["lstsq_rcond"])
    
    a, b0, c = sol
    center2 = np.array([a, b0])
    rad_sq = c + a * a + b0 * b0
    if rad_sq <= 0:
        raise ValueError("Computed non-positive radius^2 in circle fit.")
    radius = np.sqrt(rad_sq)
    center3d = mean + a * u + b0 * v
    dists = np.sqrt((X[:, 0] - a) ** 2 + (X[:, 1] - b0) ** 2)
    residuals = dists - radius

    return {
        "center3d": center3d, "radius": radius, "residuals": residuals,
        "plane_u": u, "plane_v": v, "plane_origin": mean, "proj2d": X,
    }


def apply_calibration_with_fallback(
    raw: np.ndarray,
    ellipsoid_params: Optional[dict] = None,
    planar_thresh: float = None,
):
    """
    Универсальная функция калибровки (2D или 3D).
    """
    if planar_thresh is None:
        planar_thresh = CFG_ALG["planar_detection_threshold"]
        
    raw = np.asarray(raw, dtype=float)
    evals, evecs, ratios = pca_diagnostics(raw)
    
    planar_ratio = evals[-1] / (evals[0] + CFG_ALG["pca_numerical_epsilon"])

    target_scale = np.mean(np.linalg.norm(raw, axis=1)) if raw.size > 0 else 0.0
    if target_scale == 0:
        target_scale = CFG_ALG["fallback_target_scale"]

    if planar_ratio < planar_thresh:
        # planar: 2D circle-fit
        try:
            circ = fit_circle_in_plane(raw)
        except Exception as e:
            if ellipsoid_params is not None:
                return apply_calibration(raw, ellipsoid_params)
            else:
                mean = raw.mean(axis=0)
                return raw - mean
        
        # ... (логика 2D калибровки не меняется) ...
        u, v = circ["plane_u"], circ["plane_v"]
        n = np.cross(u, v)
        mean, center3d, radius = circ["plane_origin"], circ["center3d"], circ["radius"]
        rel = raw - mean
        coords2 = np.column_stack([rel.dot(u), rel.dot(v)])
        center2 = np.array([np.dot(center3d - mean, u), np.dot(center3d - mean, v)])
        scale = (target_scale / radius) if radius > 0 else 1.0
        rel2_norm = (coords2 - center2[None, :]) * scale
        cal_plane3d = (mean[None, :] + np.outer(rel2_norm[:, 0], u) + np.outer(rel2_norm[:, 1], v))
        perp = rel.dot(n)
        perp_centered = perp - perp.mean()
        cal = cal_plane3d + np.outer(perp_centered, n)
        norms = np.linalg.norm(cal, axis=1)
        mean_norm = norms.mean() if norms.size > 0 else 0.0
        if mean_norm > 0:
            cal = cal * (target_scale / mean_norm)
        return cal
    else:
        # не плоские -> 3D ellipsoid
        if ellipsoid_params is None:
            params = fit_ellipsoid(raw)
        else:
            params = ellipsoid_params
        return apply_calibration(raw, params)


def compute_affine_from_raw_file(
    file_raw: str, cols=None, min_points: int = None, use_analytic_if_ellipsoid=None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    1) загружает сырьё, 2) подгоняет ellipsoid, 3) возвращает A, b, params.
    """
    # Загружаем значения из конфига, если они не заданы
    if cols is None: cols = CFG_LOAD["data_columns"]
    if min_points is None: min_points = CFG_ALG["min_points_for_fit"]
    if use_analytic_if_ellipsoid is None:
        use_analytic_if_ellipsoid = CFG_ALG["use_analytic_transform"]

    if not os.path.isfile(file_raw):
        raise FileNotFoundError(file_raw)

    raw = load_magnetometer_raw(file_raw, cols=cols)
    if raw.shape[0] < min_points:
        raise ValueError(
            f"Not enough points in {file_raw}: got {raw.shape[0]}, need >= {min_points}"
        )

    params = None
    try:
        params = fit_ellipsoid(raw)
    except Exception:
        params = None

    if params is not None and use_analytic_if_ellipsoid:
        S = params["transform"]
        center = params["center"]
        inter = (S.dot((raw - center).T)).T
        mean_inter_norm = np.mean(np.linalg.norm(inter, axis=1))
        avg_raw_norm = np.mean(np.linalg.norm(raw, axis=1))
        alpha = 1.0 if mean_inter_norm == 0 else (avg_raw_norm / mean_inter_norm)
        A = alpha * S
        b = -alpha * S.dot(center)
        return A, b, params

    # Fallback
    calibrated = apply_calibration_with_fallback(raw, ellipsoid_params=params)
    N = raw.shape[0]
    Phi = np.hstack([raw, np.ones((N, 1), dtype=float)])
    X, *_ = np.linalg.lstsq(Phi, calibrated, rcond=CFG_ALG["lstsq_rcond"])
    A = X[:3, :].T
    b = X[3, :].T
    return A, b, params


def apply_affine_to_file(
    input_file: str, A: np.ndarray, b: np.ndarray, cols=None
) -> np.ndarray:
    """
    Считает из input_file колонки cols и применяет calibrated = raw @ A.T + b.
    """
    if cols is None:
        cols = CFG_LOAD["data_columns"]
        
    raw2 = load_magnetometer_raw(input_file, cols=cols)
    if raw2.size == 0:
        return np.empty((0, 3), dtype=float)
    cal2 = (raw2 @ A.T) + b
    return cal2
