"""
Módulo de entrenamiento para el proyecto de riesgo crediticio.

Define, entrena y optimiza modelos de clasificación binaria utilizando
pipelines de preprocesamiento diferenciados según la familia de algoritmo.
"""

from typing import Dict

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing import build_linear_preprocessor, build_tree_preprocessor


RANDOM_STATE = 42


def build_base_models() -> Dict[str, Pipeline]:
    """
    Construye los pipelines de los modelos base.

    Returns
    -------
    Dict[str, Pipeline]
        Diccionario con nombre del modelo y pipeline correspondiente.
    """
    linear_preprocessor = build_linear_preprocessor()
    tree_preprocessor = build_tree_preprocessor()

    models = {
        "Logistic Regression Elastic Net": Pipeline(
            steps=[
                ("preprocessor", linear_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.5,
                        max_iter=5000,
                        random_state=RANDOM_STATE
                    )
                )
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    )
                )
            ]
        ),
        "AdaBoost": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    AdaBoostClassifier(
                        n_estimators=200,
                        learning_rate=0.5,
                        random_state=RANDOM_STATE
                    )
                )
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    )
                )
            ]
        )
    }

    return models


def train_base_models(X_train, y_train) -> Dict[str, Pipeline]:
    """
    Entrena los modelos base sobre el conjunto de entrenamiento.

    Parameters
    ----------
    X_train : pd.DataFrame
        Variables predictoras de entrenamiento.
    y_train : pd.Series
        Variable objetivo de entrenamiento.

    Returns
    -------
    Dict[str, Pipeline]
        Diccionario con modelos entrenados.
    """
    models = build_base_models()
    trained_models = {}

    for model_name, pipeline in models.items():
        print(f"Entrenando modelo base: {model_name}")
        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline

    return trained_models


def optimize_random_forest(X_train, y_train) -> RandomizedSearchCV:
    """
    Optimiza Random Forest mediante RandomizedSearchCV usando sólo el conjunto
    de entrenamiento y validación cruzada interna.

    Parameters
    ----------
    X_train : pd.DataFrame
        Variables predictoras de entrenamiento.
    y_train : pd.Series
        Variable objetivo de entrenamiento.

    Returns
    -------
    RandomizedSearchCV
        Objeto RandomizedSearchCV ajustado con el mejor estimador encontrado.
    """
    tree_preprocessor = build_tree_preprocessor()

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", tree_preprocessor),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ]
    )

    param_distributions = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 5, 10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
        "model__bootstrap": [True, False]
    }

    random_search = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    print("Optimizando Random Forest con RandomizedSearchCV...")
    random_search.fit(X_train, y_train)

    print("Mejores hiperparámetros encontrados:")
    print(random_search.best_params_)

    print(f"Mejor AUC-ROC en validación cruzada: {random_search.best_score_:.4f}")

    return random_search


def train_all_models(X_train, y_train) -> Dict[str, Pipeline]:
    """
    Entrena modelos base y agrega Random Forest optimizado.

    Parameters
    ----------
    X_train : pd.DataFrame
        Variables predictoras de entrenamiento.
    y_train : pd.Series
        Variable objetivo de entrenamiento.

    Returns
    -------
    Dict[str, Pipeline]
        Diccionario con todos los modelos entrenados.
    """
    trained_models = train_base_models(X_train, y_train)

    optimized_rf = optimize_random_forest(X_train, y_train)
    trained_models["Random Forest Optimized"] = optimized_rf.best_estimator_

    return trained_models


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from data_loader import load_credit_data, get_features_target

    df = load_credit_data()
    X, y = get_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    models = train_all_models(X_train, y_train)

    print("\nEntrenamiento finalizado.")
    print("Modelos entrenados:")
    for model_name in models.keys():
        print(f"- {model_name}")