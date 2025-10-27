"""
... (описание файла) ...

Все настройки загружаются из config.json через модуль config.py
"""

import os
import numpy as np

# Импортируем наши модули
from MagnetometerCalibrator import MagnetometerCalibrator
from calculations import (
    compute_affine_from_raw_file,
    load_magnetometer_raw,
    apply_affine_to_file,
    fit_ellipsoid,
    apply_calibration_with_fallback
)
from plot import plot_raw_vs_calibrated, plot_distance_histogram

# Импортируем настройки из config.py
try:
    from config import CFG_FILE, CFG_LOAD, CFG_PLOT, CFG_RT
except ImportError:
    print("ОШИBКА: Не удалось импортировать настройки. Убедитесь, что config.py существует.")
    exit(1)


def applyParamsToAnother():
    # === Настройки (берутся из config.py) ===
    file1 = CFG_FILE["comparison_file_1"]
    file2 = CFG_FILE["comparison_file_2"]
    cols = CFG_LOAD["data_columns"]
    max_plot_points = CFG_PLOT["plot_max_points"]
    # ========================================

    if not os.path.isfile(file1):
        raise FileNotFoundError(
            f"Input file not found: '{file1}'. Убедитесь, что путь указан верно в config.json."
        )

    A, B, params1 = compute_affine_from_raw_file(file1, cols=cols)
    raw = load_magnetometer_raw(file2, cols=cols)
    calibrated = apply_affine_to_file(file2, A, B, cols=cols)

    # Plot and save HTML
    plot_raw_vs_calibrated(
        raw,
        calibrated,
        out_html=CFG_FILE["plot_output_html"],
        max_plot_points=max_plot_points,
        auto_open=CFG_PLOT["plot_auto_open"],
    )

    # гистограмма расстояний
    distances, hist_fig = plot_distance_histogram(
        calibrated=calibrated, 
        out_html=CFG_FILE["histogram_output_html"], 
        auto_open=CFG_PLOT["histogram_auto_open"]
    )

def buildPlot():
    # === Настройки (берутся из config.py) ===
    filepath = CFG_FILE["main_data_file"]
    cols = CFG_LOAD["data_columns"]
    max_plot_points = CFG_PLOT["plot_max_points"]
    # ========================================

    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Input file not found: '{filepath}'. Убедитесь, что путь указан верно в config.json."
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
    if params:
        print("Center (hard-iron offset):", params["center"])
        print("Eigenvalues (shape):", params["eigvals"])

    # Plot and save HTML
    plot_raw_vs_calibrated(
        raw,
        calibrated,
        out_html=CFG_FILE["plot_output_html"],
        max_plot_points=max_plot_points,
        auto_open=CFG_PLOT["plot_auto_open"],
    )

    # гистограмма расстояний
    distances, hist_fig = plot_distance_histogram(
        calibrated=calibrated, 
        out_html=CFG_FILE["histogram_output_html"], 
        auto_open=CFG_PLOT["histogram_auto_open"]
    )

def realTimeProcessing():
    # === Настройки (берутся из config.py) ===
    N_POINTS_FOR_RECALIBRATION = CFG_RT["recalibration_point_threshold"]
    calibration_data_file = CFG_FILE["realtime_initial_calib_file"]
    calib_storage_file = CFG_FILE["realtime_calib_storage_file"]
    temp_calib_file = CFG_FILE["realtime_temp_recalib_file"]
    exit_char = CFG_RT["realtime_exit_char"]
    # ========================================

    # --- Начальная калибровка по файлу ---
    calibrator = MagnetometerCalibrator()

    print(f"Выполняется начальная калибровка по файлу: {calibration_data_file}...")
    try:
        calibrator.calibrate(calibration_data_file)
    except FileNotFoundError:
        print(f"Файл {calibration_data_file} не найден, начальная калибровка пропущена.")

    if calibrator.is_calibrated:
        calibrator.save_calibration(calib_storage_file)
        print(f"Начальная калибровка сохранена в '{calib_storage_file}'")
    else:
        print("Ошибка начальной калибровки. Файл не сохранен.")

    print("\n" + "="*50 + "\n")

    # --- Настройка калибратора для реального времени ---
    real_time_calibrator = MagnetometerCalibrator()
    try:
        real_time_calibrator.load_calibration(calib_storage_file)
        print(f"Загружена начальная калибровка '{calib_storage_file}'.")
    except FileNotFoundError:
        print(f"Файл {calib_storage_file} не найден.")
        print("Программа продолжит работу без начальной калибровки.")

    print("\nСимуляция Real-Time коррекции и рекалибровки:")

    # --- Настройка для рекалибровки ---
    real_time_data_buffer = [] # Буфер для сбора N точек

    while True:
        try:
            data_input = input(f"x y z (собрано {len(real_time_data_buffer)}/{N_POINTS_FOR_RECALIBRATION}, '{exit_char}' для выхода): ")
            
            if data_input.lower() == exit_char:
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
                
                data_to_calibrate = np.array(real_time_data_buffer)
                
                # Сохраняем накопленные данные во временный файл
                np.savetxt(temp_calib_file, data_to_calibrate, fmt='%.8f') 

                # Выполняем калибровку по этому файлу
                # Используем колонки 0, 1, 2, т.к. мы сами создали этот файл
                real_time_calibrator.calibrate(temp_calib_file, cols=(0, 1, 2))

                if real_time_calibrator.is_calibrated:
                    print("Рекалибровка успешна. Обновленные параметры сохранены.")
                    real_time_calibrator.save_calibration(calib_storage_file) 
                else:
                    print("Ошибка рекалибровки. Будут использоваться старые параметры.")
                
                real_time_data_buffer.clear()
                print("Буфер очищен. Сбор новых данных...")
                print("="*54 + "\n")

        except ValueError:
            print(f"Ошибка: Введите 3 числа, разделенных пробелом, или '{exit_char}'.")
        except KeyboardInterrupt:
            print("\nВыход из программы (Ctrl+C).")
            break

# ---- Main execution ----
if __name__ == "__main__":

    choice = input("1 - applyParamsToAnother()\n2 - buildPlot()\n3 - realTimeProcessing()\nВыбор: ")
    match choice:
        case '1':
            applyParamsToAnother()
        case '2':
            buildPlot()
        case '3':
            realTimeProcessing()
        case _:
            pass
