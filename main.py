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

import os

from calculations import (
    compute_affine_from_raw_file,
    load_magnetometer_raw,
    apply_affine_to_file,
    fit_ellipsoid,
    apply_calibration_with_fallback
)
from plot import plot_raw_vs_calibrated, plot_distance_histogram

def applyParamsToAnother():
    # === Настройки (изменяйте тут) ===
    file1 = "data/data4.txt"
    file2 = "data/data3.txt"
    cols = (8, 9, 10)  # 0-based индексы колонок, которые интересуют
    scale_to = (
        1.0  # масштаб результата (1.0 => unit sphere). Можно поставить ≈50 для µT
    )
    max_plot_points = (
        20000  # максимальное число точек для отрисовки (подвыборка при больших файлах)
    )
    # ==================================

    if not os.path.isfile(file1):
        raise FileNotFoundError(
            f"Input file not found: '{file1}'. Убедитесь, что путь указан верно."
        )

    A, B, params1 = compute_affine_from_raw_file(file1, cols=cols)
    raw = load_magnetometer_raw(file2, cols=cols)
    calibrated = apply_affine_to_file(file2, A, B, cols=cols)

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

def main():
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

# ---- Main execution (no argparse) ----
if __name__ == "__main__":
    main()
