"""Preprocessing helpers for Bank Customer Churn notebooks."""

import pandas as pd
import matplotlib.pyplot as plt

# 🤖 scikit-learn імпорти
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import List, Optional, Tuple

def get_input_columns(
    raw_df: pd.DataFrame,
    target_col: str = "Exited",
    drop_cols: Optional[List[str]] = None,
) -> List[str]:
    """Return model input columns after excluding target and optional columns."""
    optional_drop = drop_cols or []
    columns_to_drop = set(optional_drop + [target_col])
    return [col for col in raw_df.columns if col not in columns_to_drop]


def split_inputs_targets(
    raw_df: pd.DataFrame,
    input_cols: List[str],
    target_col: str = "Exited",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split raw data into train/validation parts with stratification by target."""
    train_df, val_df = train_test_split(
        raw_df,
        test_size=test_size,
        random_state=random_state,
        stratify=raw_df[target_col],
    )

    train_inputs = train_df[input_cols].copy()
    val_inputs = val_df[input_cols].copy()
    train_targets = train_df[target_col].copy()
    val_targets = val_df[target_col].copy()

    return train_inputs, train_targets, val_inputs, val_targets


def detect_column_types(input_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Detect numeric and categorical columns from input features."""
    categorical_cols = input_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [col for col in input_df.columns if col not in categorical_cols]
    return numeric_cols, categorical_cols


def scale_numeric_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    numeric_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Fit MinMaxScaler on train numeric columns and transform train/validation."""
    scaler = MinMaxScaler()
    scaler.fit(train_inputs[numeric_cols])

    train_scaled = train_inputs.copy()
    val_scaled = val_inputs.copy()
    train_scaled.loc[:, numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_scaled.loc[:, numeric_cols] = scaler.transform(val_inputs[numeric_cols])

    return train_scaled, val_scaled, scaler


def encode_categorical_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """Fit OneHotEncoder on train categoricals and add encoded columns."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(train_inputs[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()

    train_encoded_array = encoder.transform(train_inputs[categorical_cols])
    val_encoded_array = encoder.transform(val_inputs[categorical_cols])

    train_encoded_df = pd.DataFrame(
        train_encoded_array,
        columns=encoded_cols,
        index=train_inputs.index,
    )
    val_encoded_df = pd.DataFrame(
        val_encoded_array,
        columns=encoded_cols,
        index=val_inputs.index,
    )

    train_result = pd.concat(
        [train_inputs.drop(columns=categorical_cols), train_encoded_df],
        axis=1,
    )
    val_result = pd.concat(
        [val_inputs.drop(columns=categorical_cols), val_encoded_df],
        axis=1,
    )

    return train_result, val_result, encoder


def preprocess_data(
    raw_df: pd.DataFrame,
    target_col: str = "Exited",
    drop_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler_numeric: bool = True,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    List[str],
    Optional[MinMaxScaler],
    Optional[OneHotEncoder],
]:
    """Prepare raw train data for modeling and return train/validation matrices.

    Returns
    -------
    X_train, y_train, X_val, y_val, input_cols, scaler, encoder
    """
    input_cols = get_input_columns(raw_df, target_col=target_col, drop_cols=drop_cols)

    train_inputs, train_targets, val_inputs, val_targets = split_inputs_targets(
        raw_df=raw_df,
        input_cols=input_cols,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    numeric_cols, categorical_cols = detect_column_types(train_inputs)

    scaler: Optional[MinMaxScaler] = None
    if scaler_numeric and numeric_cols:
        train_inputs, val_inputs, scaler = scale_numeric_features(
            train_inputs=train_inputs,
            val_inputs=val_inputs,
            numeric_cols=numeric_cols,
        )

    encoder: Optional[OneHotEncoder] = None
    if categorical_cols:
        train_inputs, val_inputs, encoder = encode_categorical_features(
            train_inputs=train_inputs,
            val_inputs=val_inputs,
            categorical_cols=categorical_cols,
        )

    return train_inputs, train_targets, val_inputs, val_targets, input_cols, scaler, encoder


def preprocess_new_data(
    raw_df: pd.DataFrame,
    input_cols: Optional[List[str]] = None,
    scaler: Optional[MinMaxScaler] = None,
    encoder: Optional[OneHotEncoder] = None,
    scaler_numeric: bool = True,
    target_col: str = "Exited",
    drop_cols: Optional[List[str]] = None,
):
    """Preprocess data without train/validation split.

    Modes
    -----
    1) Fit mode (no input_cols/scaler/encoder passed):
       returns (X, input_cols, scaler, encoder)
    2) Transform mode (input_cols and/or fitted objects passed):
       returns X
    """
    fit_mode = input_cols is None and scaler is None and encoder is None

    if input_cols is None:
        input_cols = get_input_columns(raw_df, target_col=target_col, drop_cols=drop_cols)

    missing_cols = [col for col in input_cols if col not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"New data is missing required columns: {missing_cols}")

    inputs = raw_df[input_cols].copy()

    if encoder is None:
        _, categorical_cols = detect_column_types(inputs)
    else:
        categorical_cols = list(encoder.feature_names_in_)
        missing_cat_cols = [col for col in categorical_cols if col not in inputs.columns]
        if missing_cat_cols:
            raise ValueError(
                f"New data is missing categorical columns required by encoder: {missing_cat_cols}"
            )

    numeric_cols = [col for col in input_cols if col not in categorical_cols]

    if scaler_numeric and numeric_cols:
        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(inputs[numeric_cols])
        inputs.loc[:, numeric_cols] = scaler.transform(inputs[numeric_cols])

    if categorical_cols:
        if encoder is None:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoder.fit(inputs[categorical_cols])

        encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()
        encoded_array = encoder.transform(inputs[categorical_cols])
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=inputs.index)
        inputs = pd.concat([inputs.drop(columns=categorical_cols), encoded_df], axis=1)

    if fit_mode:
        return inputs, input_cols, scaler, encoder
    return inputs


def preprocess__new_data(*args, **kwargs):
    """Backward-compatible alias for preprocess_new_data."""
    return preprocess_new_data(*args, **kwargs)

def compute_auroc_and_build_roc(
    model,
    inputs: pd.DataFrame,
    targets: pd.Series | pd.DataFrame,
    name: str = "Validation",
) -> float:
    """Computes AUROC, plots the ROC curve, and returns the AUROC value."""
    y_true = targets.squeeze()
    y_pred_proba = model.predict_proba(inputs)[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f"AUROC for {name}: {roc_auc:.4f}")

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.show()

    return roc_auc

__all__ = [
    "get_input_columns",
    "split_inputs_targets",
    "detect_column_types",
    "scale_numeric_features",
    "encode_categorical_features",
    "preprocess_data",
    "preprocess_new_data",
    "preprocess__new_data",
]
