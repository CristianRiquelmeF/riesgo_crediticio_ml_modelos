"""
Módulo de utilidades generales para el proyecto de riesgo crediticio.

Incluye funciones para crear directorios, guardar modelos entrenados
y cargar modelos serializados.
"""

from pathlib import Path
from typing import Any

import joblib


MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"


def create_project_directories() -> None:
    """
    Crea los directorios necesarios para almacenar modelos, reportes y figuras.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_model(model: Any, path: Path = BEST_MODEL_PATH) -> None:
    """
    Serializa y guarda un modelo entrenado usando joblib.

    Parameters
    ----------
    model : Any
        Modelo o pipeline entrenado que será guardado.
    path : Path
        Ruta donde se guardará el modelo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path = BEST_MODEL_PATH) -> Any:
    """
    Carga un modelo previamente serializado con joblib.

    Parameters
    ----------
    path : Path
        Ruta del modelo guardado.

    Returns
    -------
    Any
        Modelo cargado desde disco.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en la ruta: {path}")

    return joblib.load(path)


if __name__ == "__main__":
    create_project_directories()

    print("Directorios del proyecto verificados correctamente.")
    print(f"Directorio de modelos: {MODELS_DIR}")
    print(f"Directorio de reportes: {REPORTS_DIR}")
    print(f"Directorio de figuras: {FIGURES_DIR}")