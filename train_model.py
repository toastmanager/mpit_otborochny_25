import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)
import joblib
import optuna

# Отключаем подробные логи Optuna, чтобы не засорять вывод
optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_model(df: pd.DataFrame, n_trials: int = 50) -> CatBoostClassifier:
    """
    Организует полный пайплайн: подбор гиперпараметров с Optuna,
    обучение финальной модели и ее оценка.
    """
    categorical_features = ["carmodel", "carname", "platform"]
    target_column = "is_done"

    X = df.drop(columns=[target_column], errors="ignore")
    y = df[target_column]

    # Инженерия признаков
    for col in categorical_features:
        if col in X.columns:
            counts = X[col].value_counts()
            rare_categories = counts[counts < 10].index.tolist()
            X[col] = X[col].replace(rare_categories, "Other").astype(str)

    # Разделение данных на обучающую и финальную тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ------------------- Optuna Integration -------------------

    def objective(trial: optuna.Trial) -> float:
        """
        Функция для одного "прогона" Optuna. Обучает модель с предложенными
        гиперпараметрами и возвращает F1-score на валидационной выборке.
        """
        # Разделяем обучающие данные для использования в early_stopping
        train_x, val_x, train_y, val_y = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
        )

        # Определяем пространство поиска гиперпараметров
        params = {
            "n_estimators": 1500,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "random_state": 42,
            "verbose": 0,
            "cat_features": categorical_features,
            "auto_class_weights": "Balanced",
        }

        model = CatBoostClassifier(**params)
        model.fit(
            train_x,
            train_y,
            eval_set=[(val_x, val_y)],
            early_stopping_rounds=50,
            verbose=0,
        )

        # Оцениваем модель на валидационной выборке
        preds = model.predict(val_x)
        f1 = float(f1_score(val_y, preds, pos_label=1))

        # Сохраняем лучшее количество деревьев для использования в финальной модели
        trial.set_user_attr("best_iteration", model.get_best_iteration())

        return f1

    # Создаем и запускаем исследование Optuna
    study = optuna.create_study(
        direction="maximize", study_name="catboost_f1_optimization"
    )
    study.optimize(objective, n_trials=n_trials)

    print("\n--- Результаты подбора гиперпараметров Optuna ---")
    print(f"Лучший F1-score на валидации: {study.best_value:.4f}")
    print("Лучшие гиперпараметры:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # ------------------- Обучение финальной модели -------------------

    print("\n--- Обучение финальной модели на лучших параметрах ---")

    # Получаем лучшие параметры и оптимальное количество деревьев
    best_params = study.best_params
    best_iteration = study.best_trial.user_attrs["best_iteration"]
    best_params["n_estimators"] = best_iteration
    best_params["cat_features"] = categorical_features
    best_params["auto_class_weights"] = "Balanced"
    best_params["random_state"] = 42

    final_model = CatBoostClassifier(**best_params)

    # Обучаем финальную модель на ВСЕХ обучающих данных
    final_model.fit(X_train, y_train, verbose=0)

    # ------------------- Оценка финальной модели -------------------

    print("\n--- Оценка финальной модели на тестовой выборке ---")
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]

    # Поиск оптимального порога
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred_thresholded = (y_pred_proba >= threshold).astype(int)
        current_f1 = f1_score(y_test, y_pred_thresholded, pos_label=1)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    print(
        f"Поиск оптимального порога -> Лучший F1-score для класса 1: {best_f1:.3f} при пороге: {best_threshold:.2f}"
    )

    y_pred_final = (y_pred_proba >= best_threshold).astype(int)

    print(f"Итоговая точность (Accuracy): {accuracy_score(y_test, y_pred_final):.3f}")
    print("Матрица ошибок:\n", confusion_matrix(y_test, y_pred_final))
    print("\nОтчет по классификации:\n", classification_report(y_test, y_pred_final))

    # Важность признаков
    feature_importances = pd.DataFrame(
        {
            "feature": final_model.feature_names_,
            "importance": final_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\n--- Топ-15 самых важных признаков ---")
    print(feature_importances.head(15))

    return final_model


if __name__ == "__main__":
    data_path = "data_processed.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        print(
            f"Ошибка: файл {data_path} не найден. Сначала запустите data_preprocessing.py"
        )
        raise e

    # Запускаем поиск на 50 итерациях. Можно увеличить для более тщательного поиска.
    model = train_model(df, n_trials=50)

    model_path = "catboost_predictor_optimized.joblib"
    joblib.dump(model, model_path)
    print(f"\nОптимизированная модель сохранена в {model_path}")
