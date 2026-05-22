"""
Módulo de evaluación para el proyecto de riesgo crediticio.

Calcula métricas de clasificación para los modelos entrenados, exporta una
tabla comparativa y genera la matriz de confusión del mejor modelo.
"""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def evaluate_models(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evalúa modelos entrenados sobre el conjunto de prueba.

    Parameters
    ----------
    models : Dict[str, Pipeline]
        Diccionario con modelos entrenados.
    X_test : pd.DataFrame
        Variables predictoras del conjunto de prueba.
    y_test : pd.Series
        Variable objetivo del conjunto de prueba.

    Returns
    -------
    pd.DataFrame
        Tabla comparativa de métricas.
    """
    results = []

    for model_name, model in models.items():
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc_roc = roc_auc_score(y_test, y_proba)
        else:
            auc_roc = None

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "auc_roc": auc_roc,
            }
        )

    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df.sort_values(by="auc_roc", ascending=False).reset_index(drop=True)

    return metrics_df


def save_metrics_report(metrics_df: pd.DataFrame) -> None:
    """
    Guarda la tabla comparativa de métricas en reports/metrics.csv.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(REPORTS_DIR / "metrics.csv", index=False)


def get_best_model(
    models: Dict[str, Pipeline],
    metrics_df: pd.DataFrame,
) -> Tuple[str, Pipeline]:
    """
    Identifica el mejor modelo según AUC-ROC.

    Parameters
    ----------
    models : Dict[str, Pipeline]
        Diccionario con modelos entrenados.
    metrics_df : pd.DataFrame
        Tabla comparativa de métricas.

    Returns
    -------
    Tuple[str, Pipeline]
        Nombre y objeto del mejor modelo.
    """
    best_model_name = metrics_df.iloc[0]["model"]
    best_model = models[best_model_name]

    return best_model_name, best_model


def save_confusion_matrix(
    model: Pipeline,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Genera y guarda la matriz de confusión del modelo seleccionado.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No riesgo", "Riesgo"],
    )

    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    display.plot(ax=ax, values_format="d", colorbar=False)

    ax.set_title(f"Matriz de confusión - {model_name}")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_best_model.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from data_loader import load_credit_data, get_features_target
    from train import RANDOM_STATE, train_all_models

    df = load_credit_data()
    X, y = get_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = train_all_models(X_train, y_train)

    metrics_df = evaluate_models(models, X_test, y_test)
    save_metrics_report(metrics_df)

    best_model_name, best_model = get_best_model(models, metrics_df)
    save_confusion_matrix(best_model, best_model_name, X_test, y_test)

    print("\nEvaluación finalizada.")
    print("\nMétricas comparativas:")
    print(metrics_df.round(4))

    print(f"\nMejor modelo según AUC-ROC: {best_model_name}")
    print("Reporte guardado en: reports/metrics.csv")
    print("Matriz de confusión guardada en: reports/figures/confusion_matrix_best_model.png")