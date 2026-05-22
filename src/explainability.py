"""
Módulo de explicabilidad para el proyecto de riesgo crediticio.

Genera gráficos SHAP para interpretar el comportamiento del modelo ganador.
Actualmente está diseñado para modelos basados en árboles, especialmente XGBoost.
"""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


FIGURES_DIR = Path("reports") / "figures"


def extract_preprocessor_and_model(best_model: Pipeline):
    """
    Extrae el preprocesador y el estimador final desde un pipeline entrenado.

    Parameters
    ----------
    best_model : Pipeline
        Pipeline entrenado que contiene pasos 'preprocessor' y 'model'.

    Returns
    -------
    tuple
        preprocessor, model
    """
    preprocessor = best_model.named_steps["preprocessor"]
    model = best_model.named_steps["model"]

    return preprocessor, model


def prepare_data_for_shap(
    best_model: Pipeline,
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, object]:
    """
    Aplica el preprocesamiento del pipeline al conjunto de datos y retorna
    los datos transformados junto con el modelo final.

    Parameters
    ----------
    best_model : Pipeline
        Pipeline entrenado.
    X : pd.DataFrame
        Variables predictoras originales.

    Returns
    -------
    Tuple[pd.DataFrame, object]
        X transformado como DataFrame y modelo final extraído del pipeline.
    """
    preprocessor, model = extract_preprocessor_and_model(best_model)

    X_processed = preprocessor.transform(X)

    X_processed = pd.DataFrame(
        X_processed,
        columns=X.columns,
        index=X.index,
    )

    return X_processed, model


def generate_shap_plots(
    best_model: Pipeline,
    X_test: pd.DataFrame,
    sample_size: int = 1000,
) -> None:
    """
    Genera gráficos SHAP globales para el modelo ganador.

    Se generan:
    - shap_feature_importance.png
    - shap_summary.png

    Parameters
    ----------
    best_model : Pipeline
        Pipeline entrenado del mejor modelo.
    X_test : pd.DataFrame
        Conjunto de prueba.
    sample_size : int
        Número máximo de observaciones usadas para SHAP.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_sample = X_test.sample(
        n=min(sample_size, len(X_test)),
        random_state=42,
    )

    X_processed, model = prepare_data_for_shap(best_model, X_sample)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)

    # Gráfico 1: importancia global de variables
    plt.figure(figsize=(7, 5), dpi=120)
    shap.summary_plot(
        shap_values,
        X_processed,
        plot_type="bar",
        show=False,
        max_display=10,
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "shap_feature_importance.png",
        bbox_inches="tight",
    )
    plt.close()

    # Gráfico 2: summary plot con dirección del efecto
    plt.figure(figsize=(7, 5), dpi=120)
    shap.summary_plot(
        shap_values,
        X_processed,
        show=False,
        max_display=10,
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "shap_summary.png",
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    import sys

    from pathlib import Path
    from sklearn.model_selection import train_test_split

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SRC_DIR = PROJECT_ROOT / "src"
    sys.path.append(str(SRC_DIR))

    from data_loader import load_credit_data, get_features_target
    from train import RANDOM_STATE
    from utils import load_model

    df = load_credit_data()
    X, y = get_features_target(df)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    best_model = load_model()

    print("Generando explicabilidad SHAP para el modelo guardado en models/best_model.pkl")
    generate_shap_plots(best_model, X_test)

    print("Gráficos SHAP generados correctamente:")
    print("- reports/figures/shap_feature_importance.png")
    print("- reports/figures/shap_summary.png")