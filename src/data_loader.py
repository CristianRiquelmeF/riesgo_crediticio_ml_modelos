"""
Módulo de carga de datos para el proyecto de predicción de riesgo crediticio.

"""

from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_openml


TARGET_NAME = "riesgo"


def load_credit_data() -> pd.DataFrame:
    """
    Descarga el dataset de riesgo crediticio desde OpenML y retorna un
    DataFrame consolidado con la variable objetivo renombrada como 'riesgo'.

    Returns
    -------
    pd.DataFrame
        DataFrame con variables predictoras numéricas y columna objetivo.
    """
    credit_data = fetch_openml(
        name="credit",
        version=1,
        as_frame=True,
        parser="auto"
    )

    df = credit_data.data.copy()
    target = credit_data.target.copy()

    df[TARGET_NAME] = target.astype(int)

    df = df.drop_duplicates().reset_index(drop=True)

    return df


def get_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa el DataFrame consolidado en variables predictoras y variable objetivo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame consolidado que contiene la columna objetivo 'riesgo'.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X: variables predictoras.
        y: variable objetivo.
    """
    X = df.drop(columns=[TARGET_NAME])
    y = df[TARGET_NAME]

    return X, y


if __name__ == "__main__":
    df = load_credit_data()
    X, y = get_features_target(df)

    print("Datos cargados exitosamente.")
    print(f"Dimensiones del DataFrame: {df.shape}")
    print(f"Dimensiones de X: {X.shape}")
    print(f"Dimensiones de y: {y.shape}")

    print("\nDistribución absoluta de la variable objetivo:")
    print(y.value_counts().sort_index())

    print("\nDistribución porcentual de la variable objetivo:")
    print(y.value_counts(normalize=True).sort_index().round(4))