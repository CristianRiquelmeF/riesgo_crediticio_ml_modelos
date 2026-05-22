"""
Punto de entrada principal del proyecto de predicción de riesgo crediticio.

Este script orquesta el flujo completo:
1. Carga de datos.
2. Separación train/test.
3. Entrenamiento de modelos.
4. Optimización de Random Forest.
5. Evaluación comparativa.
6. Exportación de métricas.
7. Generación de matriz de confusión.
8. Guardado del mejor modelo.
9. Generación de explicabilidad SHAP.
"""

import sys
from pathlib import Path

from sklearn.model_selection import train_test_split


# Permite importar módulos desde la carpeta src/
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))


from data_loader import load_credit_data, get_features_target
from evaluate import (
    evaluate_models,
    get_best_model,
    save_confusion_matrix,
    save_metrics_report,
)
from explainability import generate_shap_plots
from train import RANDOM_STATE, train_all_models
from utils import create_project_directories, save_model


def main() -> None:
    """
    Ejecuta el pipeline completo de Machine Learning.
    """
    print("Iniciando pipeline de riesgo crediticio...\n")

    # 1. Crear directorios necesarios
    create_project_directories()

    # 2. Cargar datos
    print("Cargando datos...")
    df = load_credit_data()
    X, y = get_features_target(df)

    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Variables predictoras: {X.shape[1]}")
    print(f"Distribución del target:\n{y.value_counts(normalize=True).round(4)}\n")

    # 3. Train/test split
    print("Separando datos en train y test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}\n")

    # 4. Entrenar modelos
    print("Entrenando modelos...")
    models = train_all_models(X_train, y_train)

    # 5. Evaluar modelos
    print("\nEvaluando modelos en conjunto de prueba...")
    metrics_df = evaluate_models(models, X_test, y_test)
    save_metrics_report(metrics_df)

    # 6. Seleccionar mejor modelo
    best_model_name, best_model = get_best_model(models, metrics_df)

    # 7. Guardar matriz de confusión del mejor modelo
    save_confusion_matrix(best_model, best_model_name, X_test, y_test)

    # 8. Guardar mejor modelo
    save_model(best_model)

    # 9. Generar explicabilidad SHAP
    print("\nGenerando gráficos de explicabilidad SHAP...")
    generate_shap_plots(best_model, X_test)

    # 10. Mostrar resumen final
    print("\nPipeline finalizado correctamente.")

    print("\nMétricas comparativas:")
    print(metrics_df.round(4))

    print(f"\nMejor modelo según AUC-ROC: {best_model_name}")

    print("\nArchivos generados:")
    print("- reports/metrics.csv")
    print("- reports/figures/confusion_matrix_best_model.png")
    print("- reports/figures/shap_feature_importance.png")
    print("- reports/figures/shap_summary.png")
    print("- models/best_model.pkl")


if __name__ == "__main__":
    main()