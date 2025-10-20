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
import numpy as np

from MagnetometerCalibrator import MagnetometerCalibrator
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

def buildPlot():


def realTimeProcessing():

    calibration_data_file = "data/data4.txt"
    calibrator = MagnetometerCalibrator()

    calibrator.calibrate(calibration_data_file)

    if calibrator.is_calibrated:
        calibrator.save_calibration("mag_calib.npz")

    print("\n" + "="*50 + "\n")


    real_time_calibrator = MagnetometerCalibrator()
    try:
        real_time_calibrator.load_calibration("mag_calib.npz")
    except FileNotFoundError as e:
        print(e)
        print("Программа продолжит работу без калибровки.")

    print("\nСимуляция Real-Time коррекции:")

    while True:
        data_input = input("x y z: ")
        raw_xyz = np.array(list(map(float, data_input.split(" "))))

        # real_time_raw_data = [
        #     np.array([10.5, -30.1, 55.2]),
        #     np.array([25.0, -15.8, 40.7]),
        #     np.array([-5.3, -45.6, 61.0]),
        # ]

        calibrated_xyz = real_time_calibrator.correct(raw_xyz)
        print(f"    Сырые данные: {np.round(raw_xyz, 2)}")
        print(f"    Откалиброванные: {np.round(calibrated_xyz, 2)}\n")

# ---- Main execution (no argparse) ----
if __name__ == "__main__":
    realTimeProcessing()
