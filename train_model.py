from catboost import CatBoostClassifier
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)

import joblib


def train_model(df: pd.DataFrame) -> CatBoostClassifier:
    categorical_features = ["carmodel", "carname", "platform"]
    target_column = "is_done"

    X = df.drop(columns=[target_column], errors="ignore")
    y = df[target_column]

    for col in categorical_features:
        if col in X.columns:
            counts = X[col].value_counts()
            rare_categories = counts[
                counts < 10
            ].index.tolist()  # Categories appearing less than 10 times
            X[col] = X[col].replace(rare_categories, "Other").astype(str)

    # Split data into training+validation and a final test set
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Split the training+validation set into a training and a validation set for early stopping
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=42,
        stratify=y_train_full,
    )

    model = CatBoostClassifier(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=6,
        l2_leaf_reg=5,
        subsample=0.8,
        random_state=42,
        verbose=0,
        cat_features=categorical_features,
        auto_class_weights="Balanced",
        early_stopping_rounds=50,
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    print("\n--- Evaluation on the Test Set ---")
    y_pred_proba = model.predict_proba(X_test)

    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred_thresholded = (y_pred_proba[:, 1] >= threshold).astype(int)
        current_f1 = f1_score(y_test, y_pred_thresholded, pos_label=1)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    print("\n--- Поиск оптимального порога ---")
    print(
        f"Лучший F1-score для класса 1: {best_f1:.3f} при пороге: {best_threshold:.2f}"
    )

    y_pred_final = (y_pred_proba[:, 1] >= best_threshold).astype(int)

    print(f"Оптимальное количество деревьев: {model.get_best_iteration()}")
    print(f"Точность модели (Accuracy): {accuracy_score(y_test, y_pred_final):.3f}")
    print("Матрица ошибок:\n", confusion_matrix(y_test, y_pred_final))
    print("\nОтчет по классификации:\n", classification_report(y_test, y_pred_final))

    # Важность признаков
    feature_importances = pd.DataFrame(
        {"feature": model.feature_names_, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\n--- Топ-15 самых важных признаков ---")
    print(feature_importances.head(15))

    return model


if __name__ == "__main__":
    data_path = "data_processed.csv"

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        print(
            f"Ошибка: файл {data_path} не найден. Сначала запустите data_preprocessing.py"
        )
        raise e

    model = train_model(df)
    model_path = "catboost_predictor.joblib"
    joblib.dump(model, model_path)
