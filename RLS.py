from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
import os


class RLSMagCalibrator:
    """
    Метод update(x,y,z) возвращает откалиброванный вектор (x,y,z) как tuple[float,float,float].
    """

    def __init__(self, lam: float = 0.999, delta: float = 1e6):
        self.lam = float(lam)
        self.dim = 9
        self.theta = np.zeros((self.dim, 1), dtype=float)
        self.P = np.eye(self.dim, dtype=float) * float(delta)
        self.y_value = -1.0

        # геометрические параметры (обновляются внутри update)
        self.A: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.c: Optional[float] = None
        self.center: Optional[np.ndarray] = None
        self.sqrtA: Optional[np.ndarray] = None
        self.r: Optional[float] = None

    def _phi_from_sample(self, m: np.ndarray) -> np.ndarray:
        x, y, z = m
        return np.array(
            [
                x * x,
                y * y,
                z * z,
                2 * x * y,
                2 * x * z,
                2 * y * z,
                2 * x,
                2 * y,
                2 * z,
            ],
            dtype=float,
        ).reshape((self.dim, 1))

    def update(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Обновляет RLS по новому измерению и возвращает откалиброванный вектор (x,y,z).
        Возвращаемые значения — простые float (не numpy-массивы).
        """
        m = np.array([x, y, z], dtype=float)
        phi = self._phi_from_sample(m)

        # RLS-обновление (с безопасным извлечением скаляров через .item())
        P_phi = self.P @ phi
        denom = float(self.lam + (phi.T @ P_phi).item())
        K = P_phi / denom
        y_pred = float((phi.T @ self.theta).item())
        err = self.y_value - y_pred
        self.theta = self.theta + K * err
        self.P = (self.P - K @ (phi.T @ self.P)) / self.lam

        # Реконструкция параметров эллипсоида (a0..a8, a9=1)
        a = np.vstack((self.theta.reshape((-1, 1)), np.array([[1.0]])))
        A = np.array(
            [
                [a[0, 0], a[3, 0] / 2.0, a[4, 0] / 2.0],
                [a[3, 0] / 2.0, a[1, 0], a[5, 0] / 2.0],
                [a[4, 0] / 2.0, a[5, 0] / 2.0, a[2, 0]],
            ],
            dtype=float,
        )
        b_vec = np.array([a[6, 0], a[7, 0], a[8, 0]], dtype=float).reshape((3, 1))
        c = float(a[9, 0])

        # вычисление центра и прочего (с регуляризацией)
        reg = 1e-12
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.inv(A + np.eye(3) * reg)
        center = -0.5 * (A_inv @ b_vec).reshape((3,))

        r2 = float((center @ A @ center) - c)
        if r2 <= 0:
            r2 = max(r2, 1e-6)
        r = float(np.sqrt(r2))

        try:
            D, U = np.linalg.eigh(A)
            D = np.clip(D, 1e-12, None)
            sqrtA = (U * np.sqrt(D)) @ U.T
        except Exception:
            sqrtA = np.eye(3)

        # сохранить параметры
        self.A = A
        self.b = b_vec
        self.c = c
        self.center = center
        self.sqrtA = sqrtA
        self.r = r

        # вычислить и вернуть откалиброванный вектор в виде кортежа (x,y,z)
        corrected = sqrtA @ (m - center)
        if r != 0:
            corrected = corrected / r

        # гарантируем, что на выходе простые float
        return float(corrected[0]), float(corrected[1]), float(corrected[2])

    def get_params(self) -> dict:
        """Вернуть текущие параметры калибровки (если понадобятся)."""
        return {
            "theta": self.theta.flatten(),
            "A": self.A,
            "b": self.b.flatten() if self.b is not None else None,
            "c": self.c,
            "center": self.center,
            "sqrtA": self.sqrtA,
            "r": self.r,
        }

def process_file_batch(path: str, cal: RLSMagCalibrator, indices=(8, 9, 10)):
    """Reads file, counts lines, processes all samples sequentially and returns arrays."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    print(f"Counting lines in {path}...")
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        total_lines = sum(1 for _ in fh)
    print(f"Total lines: {total_lines}")

    # cal = RLSMagCalibrator(lam=0.999, delta=1e5)
    raw_list = []
    processed = 0
    skipped = 0

    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        for i, line in enumerate(fh, 1):
            parts = line.strip().split()
            if len(parts) <= max(indices):
                skipped += 1
                continue
            try:
                x = float(parts[indices[0]])
                y = float(parts[indices[1]])
                z = float(parts[indices[2]])
            except Exception:
                skipped += 1
                continue
            # update RLS using this sample
            cal.update(x, y, z)
            raw_list.append((x, y, z))
            processed += 1
            if processed % 20000 == 0:
                print(f"Processed {processed}/{total_lines} lines...")

    raw_arr = np.array(raw_list)
    params = cal.get_params()

    # compute batch-corrected points using final parameters
    if params.get('sqrtA') is not None and params.get('center') is not None and params.get('r') is not None:
        center = params['center']
        sqrtA = params['sqrtA']
        r = params['r']
        # apply: corrected = sqrtA @ (raw - center) / r
        if raw_arr.size:
            corrected_batch = (sqrtA @ (raw_arr - center).T).T
            if r != 0:
                corrected_batch = corrected_batch / r
        else:
            corrected_batch = np.empty_like(raw_arr)
    else:
        corrected_batch = np.empty((0, 3))

    print(f"Done. Processed: {processed}, skipped: {skipped}")
    return raw_arr, corrected_batch, params

# Пример использования:
if __name__ == "__main__":
    cal = RLSMagCalibrator(lam=0.999, delta=1e5)
    # пример одного обновления
    x_raw, y_raw, z_raw = 0.1, -0.02, 0.98
    x_c, y_c, z_c = cal.update(x_raw, y_raw, z_raw)
    print("corrected:", x_c, y_c, z_c)
