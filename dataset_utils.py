# guardar como dataset_utils.py
import os
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

COMMON_TARGET_NAMES = ['target','label','class','y','output','salida','etiqueta']

def detect_io(df: pd.DataFrame, target_cols: Optional[List[str]] = None
             ) -> Tuple[List[str], List[str], int]:
    """
    Detecta automáticamente columnas de entrada (features) y salida(s).
    Devuelve (inputs, outputs, n_patterns).
    - Si target_cols se pasa, se respeta.
    - Heurísticas: busca nombres comunes; si no encuentra, usa la última columna
      si tiene pocas categorías; detecta columnas binarias (0/1) como salidas multi.
    """
    cols = list(df.columns)
    n_patterns = len(df)

    if target_cols is not None:
        outputs = [c for c in cols if c in target_cols]
        if not outputs:
            raise ValueError("target_cols no coinciden con columnas del DataFrame")
    else:
        # 1) buscar nombres comunes
        lower_cols = [c.lower() for c in cols]
        outputs = []
        for name in COMMON_TARGET_NAMES:
            if name in lower_cols:
                idx = lower_cols.index(name)
                outputs.append(cols[idx])
        # 2) si no hay, mirar columnas binarias (0/1)
        if not outputs:
            bin_cols = []
            for c in cols:
                vals = df[c].dropna().unique()
                try:
                    smallset = set(np.unique(vals))
                except Exception:
                    smallset = set(vals)
                # considerar 0/1 o 0.0/1.0
                if smallset.issubset({0,1,0.0,1.0}):
                    bin_cols.append(c)
            if bin_cols:
                # si hay varias binarias, pueden ser salidas multi-output
                outputs = bin_cols
        # 3) heurística: si ninguna, usar la última columna si tiene "pocas" categorías
        if not outputs:
            last = cols[-1]
            nunique = df[last].nunique(dropna=True)
            if nunique <= max(2, min(10, max(2, n_patterns // 10))):
                outputs = [last]
            else:
                outputs = [last]

    inputs = [c for c in cols if c not in outputs]
    return inputs, outputs, n_patterns


def preprocess_for_perceptron(df: pd.DataFrame, inputs: List[str], outputs: List[str],
                             fillna_strategy: str = 'mean',
                             scale: Optional[str] = None,
                             map_output_to: str = 'binary'
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    - Llena NA, convierte categóricas (inputs) con one-hot.
    - NO escala: se usan los valores originales del dataset.
    - Convierte salidas a formato numérico apropiado para un perceptrón.
    Retorna (X, y) como numpy arrays.
    """
    X = df[inputs].copy()
    Y = df[outputs].copy()

    # ---- inputs: fillna ----
    for c in X.columns:
        if X[c].dtype == object or pd.api.types.is_categorical_dtype(X[c]):
            X[c] = X[c].fillna('MISSING')
        else:
            if fillna_strategy == 'mean':
                X[c] = X[c].fillna(X[c].mean())
            else:
                X[c] = X[c].fillna(0)

    # ---- inputs: categorical -> one-hot ----
    X = pd.get_dummies(X, drop_first=False)

    # ---- sin escalado ----
    X_np = X.values.astype(float)

    # ---- outputs: manejar diferentes casos ----
    if Y.shape[1] == 1:
        y_series = Y.iloc[:,0]
        if y_series.dtype == object or pd.api.types.is_categorical_dtype(y_series):
            codes, uniques = pd.factorize(y_series)
            y = codes
        else:
            y = y_series.fillna(y_series.mode()[0]).astype(int).values.ravel()
    else:
        y = Y.fillna(0).values.astype(int)

    # ---- opcional: mapear a bipolar ----
    if map_output_to == 'bipolar':
        if y.ndim == 1 and set(np.unique(y)).issubset({0,1}):
            y = np.where(y==0, -1, 1)
        elif y.ndim == 2 and set(np.unique(y)).issubset({0,1}):
            y = np.where(y==0, -1, 1)

    return X_np, y
