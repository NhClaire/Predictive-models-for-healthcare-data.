from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class Preprocessor:
    imputer: SimpleImputer
    scaler: StandardScaler
    label_encoder: LabelEncoder
    feature_names: list[str]

    def transform_X(self, X: pd.DataFrame) -> pd.DataFrame:
        values = self.imputer.transform(X)
        values = self.scaler.transform(values)
        return pd.DataFrame(values, columns=self.feature_names, index=X.index)

    def transform_y(self, y: pd.Series):
        return self.label_encoder.transform(y.astype(str))

    def inverse_y(self, y):
        return self.label_encoder.inverse_transform(y)


def fit_preprocessor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series, Preprocessor]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    label_encoder = LabelEncoder()

    imputed = imputer.fit_transform(X_train)
    scaled = scaler.fit_transform(imputed)
    y_encoded = label_encoder.fit_transform(y_train.astype(str))
    feature_names = list(X_train.columns)

    X_processed = pd.DataFrame(scaled, columns=feature_names, index=X_train.index)
    y_processed = pd.Series(y_encoded, index=y_train.index, name=y_train.name)
    processor = Preprocessor(imputer, scaler, label_encoder, feature_names)
    return X_processed, y_processed, processor


def preprocess_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    X_train_p, y_train_p, processor = fit_preprocessor(X_train, y_train)
    X_test_p = processor.transform_X(X_test)
    y_test_p = pd.Series(processor.transform_y(y_test), index=y_test.index, name=y_test.name)
    return X_train_p, X_test_p, y_train_p, y_test_p, processor

