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
    # === Настройки (изменяйте тут) ===
    filepath = "data/data6.txt"  # <- укажите ваш файл здесь
    cols = (8, 9, 10)  # 0-based индексы колонок, которые интересуют
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
    calibrated = apply_calibration_with_fallback(raw, params)

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

def realTimeProcessing():
    N_POINTS_FOR_RECALIBRATION = 9 

    # --- Начальная калибровка по файлу ---
    calibration_data_file = "data/data4.txt"
    calibrator = MagnetometerCalibrator()

    print(f"Выполняется начальная калибровка по файлу: {calibration_data_file}...")
    calibrator.calibrate(calibration_data_file)

    if calibrator.is_calibrated:
        calibrator.save_calibration("mag_calib.npz")
        print("Начальная калибровка сохранена в 'mag_calib.npz'")
    else:
        print("Ошибка начальной калибровки. Файл не сохранен.")

    print("\n" + "="*50 + "\n")

    # --- Настройка калибратора для реального времени ---
    real_time_calibrator = MagnetometerCalibrator()
    try:
        real_time_calibrator.load_calibration("mag_calib.npz")
        print("Загружена начальная калибровка 'mag_calib.npz'.")
    except FileNotFoundError as e:
        print(e)
        print("Программа продолжит работу без начальной калибровки.")

    print("\nСимуляция Real-Time коррекции и рекалибровки:")

    # --- Настройка для рекалибровки ---
    # Задайте N - количество точек для сбора перед рекалибровкой
    real_time_data_buffer = [] # Буфер для сбора N точек
    temp_calib_file = "temp_recalib_data.txt" # Файл для временных данных

    while True:
        try:
            data_input = input(f"x y z (собрано {len(real_time_data_buffer)}/{N_POINTS_FOR_RECALIBRATION}, 'q' для выхода): ")
            
            if data_input.lower() == 'q':
                print("Выход из программы.")
                break

            raw_xyz = np.array(list(map(float, data_input.split(" "))))
            
            # 1. Добавляем новые данные в буфер
            real_time_data_buffer.append(raw_xyz)

            # 2. Корректируем данные с использованием *текущей* калибровки
            calibrated_xyz = real_time_calibrator.correct(raw_xyz)
            print(f"    Сырые данные: {np.round(raw_xyz, 2)}")
            print(f"    Откалиброванные: {np.round(calibrated_xyz, 2)}\n")

            # 3. Проверяем, не пора ли делать рекалибровку
            if len(real_time_data_buffer) >= N_POINTS_FOR_RECALIBRATION:
                print("\n" + "="*20 + " РЕКАЛИБРОВКА " + "="*20)
                print(f"Накоплено {len(real_time_data_buffer)} точек. Выполняется рекалибровка...")
                
                # Конвертируем список массивов в 2D-массив
                data_to_calibrate = np.array(real_time_data_buffer)
                
                # Сохраняем накопленные данные во временный файл
                # (Предполагаем, что .calibrate() ожидает имя файла)
                np.savetxt(temp_calib_file, data_to_calibrate, fmt='%.8f') 

                # Выполняем калибровку по этому файлу
                real_time_calibrator.calibrate(temp_calib_file)

                if real_time_calibrator.is_calibrated:
                    print("Рекалибровка успешна. Обновленные параметры сохранены.")
                    # Сохраняем обновленную калибровку
                    real_time_calibrator.save_calibration("mag_calib.npz") 
                else:
                    print("Ошибка рекалибровки. Будут использоваться старые параметры.")
                
                # 4. Очищаем буфер для сбора новой партии данных
                real_time_data_buffer.clear()
                print("Буфер очищен. Сбор новых данных...")
                print("="*54 + "\n")

        except ValueError:
            print("Ошибка: Введите 3 числа, разделенных пробелом, или 'q'.")
        except KeyboardInterrupt:
            print("\nВыход из программы (Ctrl+C).")
            break

# ---- Main execution (no argparse) ----
if __name__ == "__main__":
    realTimeProcessing()
