"""
Módulo de preprocesamiento para el proyecto de riesgo crediticio.

Incluye un transformer personalizado para aplicar clipping de outliers
y funciones para construir pipelines diferenciados según el tipo de modelo.
"""

from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Transformer personalizado para aplicar clipping de outliers usando percentiles.

    Los límites se calculan durante fit() utilizando sólo el conjunto de entrenamiento.
    Luego, transform() aplica esos límites a nuevos datos, evitando fuga de información.
    """

    def __init__(self, lower_quantile: Optional[float] = None, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Calcula los límites de clipping a partir de X.

        Parameters
        ----------
        X : pd.DataFrame
            Variables predictoras.
        y : None
            No utilizado. Se incluye por compatibilidad con Scikit-learn.

        Returns
        -------
        self
        """
        X = pd.DataFrame(X).copy()

        if self.lower_quantile is not None:
            self.lower_bounds_ = X.quantile(self.lower_quantile)
        else:
            self.lower_bounds_ = None

        self.upper_bounds_ = X.quantile(self.upper_quantile)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica clipping usando los límites aprendidos en fit().

        Parameters
        ----------
        X : pd.DataFrame
            Variables predictoras.

        Returns
        -------
        pd.DataFrame
            DataFrame con valores extremos recortados.
        """
        X_clipped = pd.DataFrame(X).copy()

        if self.lower_bounds_ is not None:
            X_clipped = X_clipped.clip(lower=self.lower_bounds_, axis=1)

        X_clipped = X_clipped.clip(upper=self.upper_bounds_, axis=1)

        return X_clipped


def build_linear_preprocessor() -> Pipeline:
    """
    Construye el pipeline de preprocesamiento para modelos lineales.

    Incluye:
    - clipping de outliers al percentil 99;
    - escalamiento estándar.

    Returns
    -------
    Pipeline
        Pipeline de preprocesamiento para modelos lineales.
    """
    return Pipeline(
        steps=[
            ("outlier_clipper", OutlierClipper(upper_quantile=0.99)),
            ("scaler", StandardScaler())
        ]
    )


def build_tree_preprocessor() -> Pipeline:
    """
    Construye el pipeline de preprocesamiento para modelos basados en árboles.

    Incluye:
    - clipping de outliers al percentil 99.

    No aplica escalamiento, ya que Random Forest, AdaBoost y XGBoost
    no lo requieren para operar correctamente.

    Returns
    -------
    Pipeline
        Pipeline de preprocesamiento para modelos basados en árboles.
    """
    return Pipeline(
        steps=[
            ("outlier_clipper", OutlierClipper(upper_quantile=0.99))
        ]
    )


if __name__ == "__main__":
    from data_loader import load_credit_data, get_features_target

    df = load_credit_data()
    X, y = get_features_target(df)

    linear_preprocessor = build_linear_preprocessor()
    tree_preprocessor = build_tree_preprocessor()

    X_linear = linear_preprocessor.fit_transform(X)
    X_tree = tree_preprocessor.fit_transform(X)

    print("Preprocesamiento ejecutado correctamente.")
    print(f"Shape original: {X.shape}")
    print(f"Shape pipeline lineal: {X_linear.shape}")
    print(f"Shape pipeline árboles: {X_tree.shape}")