import numpy as np
import os
from calculations import compute_affine_from_raw_file, load_magnetometer_raw

# Импортируем настройки
try:
    from config import CFG_FILE, CFG_LOAD, CFG_ALG
except ImportError:
    print("ОШИBКА: Не удалось импортировать настройки. Убедитесь, что config.py существует.")
    # Устанавливаем запасные значения, чтобы модуль мог хотя бы импортироваться
    CFG_FILE = {"realtime_calib_storage_file": "mag_calib.npz"}
    CFG_LOAD = {"data_columns": (8, 9, 10)}
    CFG_ALG = {"calibrator_initial_scale": 1.0}


class MagnetometerCalibrator:
    """
    Класс для калибровки магнитометра и применения коррекции в реальном времени.
    """

    def __init__(self, initial_scale: float = None):
        if initial_scale is None:
            initial_scale = CFG_ALG["calibrator_initial_scale"]
            
        # По умолчанию, калибровка отсутствует (единичная матрица, нулевой сдвиг)
        self.A = np.eye(3) * initial_scale
        self.b = np.zeros(3)
        self.is_calibrated = False

    def calibrate(self, calibration_file: str, cols: tuple = None):
        """
        Выполняет оффлайн-калибровку на основе файла с данными.
        Этот метод вычисляет матрицу A и вектор b.

        :param calibration_file: Путь к файлу с сырыми данными для калибровки.
        :param cols: Индексы колонок (x, y, z). Если None, берутся из config.
        """
        if cols is None:
            cols = CFG_LOAD["data_columns"]
            
        print(f"🔬 Выполняется калибровка по файлу: {calibration_file} (колонки {cols})...")
        if not os.path.isfile(calibration_file):
            raise FileNotFoundError(
                f"Файл для калибровки не найден: {calibration_file}"
            )

        try:
            # Используем вашу функцию для вычисления аффинного преобразования
            self.A, self.b, params = compute_affine_from_raw_file(
                calibration_file, cols=cols
            )
            self.is_calibrated = True
            print("✅ Калибровка успешно завершена.")
            print("  Матрица A (soft-iron):\n", self.A)
            print("  Вектор b (hard-iron):", self.b)
        except Exception as e:
            print(f"❌ Ошибка во время калибровки: {e}")
            self.is_calibrated = False

    def correct(self, raw_point: np.ndarray) -> np.ndarray:
        """
        Применяет вычисленную калибровку к одному вектору (x, y, z).
        Это функция для использования в реальном времени.

        :param raw_point: Сырые данные [x, y, z] в виде numpy array.
        :return: Откалиброванные данные [x_cal, y_cal, z_cal].
        """
        return raw_point @ self.A.T + self.b

    def save_calibration(self, filepath: str = None):
        """
        Сохраняет параметры калибровки (A и b) в файл.

        :param filepath: Путь к файлу для сохранения. Если None, берется из config.
        """
        if filepath is None:
            filepath = CFG_FILE["realtime_calib_storage_file"]
            
        if not self.is_calibrated:
            print("⚠️ Предупреждение: Калибровка не выполнена. Нечего сохранять.")
            return
        np.savez(filepath, A=self.A, b=self.b)
        print(f"💾 Параметры калибровки сохранены в файл: {filepath}")

    def load_calibration(self, filepath: str = None):
        """
        Загружает параметры калибровки (A и b) из файла.

        :param filepath: Путь к файлу для загрузки. Если None, берется из config.
        """
        if filepath is None:
            filepath = CFG_FILE["realtime_calib_storage_file"]

        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"Файл с параметрами калибровки не найден: {filepath}"
            )

        data = np.load(filepath)
        self.A = data["A"]
        self.b = data["b"]
        self.is_calibrated = True
        print(f"✅ Параметры калибровки успешно загружены из файла: {filepath}")
