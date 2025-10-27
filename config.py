import json
import sys
import os

# --- Имя файла конфигурации ---
CONFIG_FILE_PATH = "config.json"

def _extract_values(config_group):
    """Вспомогательная функция для извлечения 'value' из структуры JSON."""
    return {key: data["value"] for key, data in config_group.items()}

def load_config(path=CONFIG_FILE_PATH):
    """
    Загружает, парсит и извлекает 'value' из файла config.json.
    """
    if not os.path.isfile(path):
        print(f"ОШИБКА: Файл конфигурации не найден: {path}", file=sys.stderr)
        print("Пожалуйста, убедитесь, что config.json находится в том же каталоге.", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        # Извлекаем значения из вложенной структуры
        cfg = {
            "FilePaths": _extract_values(config_data["FilePaths"]),
            "DataLoading": _extract_values(config_data["DataLoading"]),
            "Plotting": _extract_values(config_data["Plotting"]),
            "RealTime": _extract_values(config_data["RealTime"]),
            "Algorithm": _extract_values(config_data["Algorithm"]),
        }
        return cfg

    except (json.JSONDecodeError, KeyError) as e:
        print(f"ОШИБКА: Файл конфигурации {path} поврежден или имеет неверную структуру.", file=sys.stderr)
        print(f"Детали ошибки: {e}", file=sys.stderr)
        sys.exit(1)

# --- Загружаем конфигурацию при первом импорте этого модуля ---
_config = load_config()

# --- Экспортируем группы настроек для импорта в других файлах ---
CFG_FILE = _config["FilePaths"]
CFG_LOAD = _config["DataLoading"]
CFG_PLOT = _config["Plotting"]
CFG_RT = _config["RealTime"]
CFG_ALG = _config["Algorithm"]

# --- Выполняем постобработку для удобства ---
# Преобразуем список колонок из JSON в кортеж (tuple), 
# так как он часто используется для индексации
try:
    CFG_LOAD["data_columns"] = tuple(CFG_LOAD["data_columns"])
except Exception:
    print(f"ОШИБКА: 'data_columns' в config.json (внутри DataLoading) должен быть списком чисел.", file=sys.stderr)
    sys.exit(1)
