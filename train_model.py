import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib
import optuna
import time
from typing import Dict, Any

# Устанавливаем уровень логирования Optuna, чтобы избежать лишнего вывода
optuna.logging.set_verbosity(optuna.logging.WARNING)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Применяет инжиниринг признаков к датафрейму.
    - Логарифмирует асимметричные признаки для нормализации распределения.
    - Создает новые признаки для улучшения предсказательной силы модели.
    """
    # Логарифмирование признака distance_in_meters, чтобы сгладить его распределение
    # Добавляем +1 (псевдосчет), чтобы избежать ошибки логарифмирования нуля
    if "distance_in_meters" in df.columns:
        df["distance_in_meters_log"] = np.log1p(df["distance_in_meters"])

    # --- Сюда можно добавлять другие преобразования признаков ---
    # Например, создание полиномиальных признаков или комбинаций
    # df['price_per_km'] = df['price_start_local'] / (df['distance_in_meters'] / 1000)

    return df


def train_pipeline(df: pd.DataFrame, n_trials: int = 50) -> Dict[str, Any]:
    """
    Организует полный пайплайн: инжиниринг признаков, подбор гиперпараметров,
    обучение и оценка модели.
    """
    # 1. Инжиниринг признаков
    df = feature_engineering(df)

    # Исключаем исходный столбец дистанции, так как мы используем его логарифм
    columns_to_drop = ["is_done", "distance_in_meters"]
    categorical_features = ["carmodel", "carname", "platform"]

    X = df.drop(columns=columns_to_drop, errors="ignore")
    y = df["is_done"]

    # 2. Обработка редких категорий
    for col in categorical_features:
        if col in X.columns:
            counts = X[col].value_counts()
            rare_categories = counts[counts < 10].index.tolist()
            X[col] = X[col].replace(rare_categories, "Other").astype(str)

    # 3. Разделение данных (стратифицированное)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4. Подбор гиперпараметров с Optuna
    def objective(trial: optuna.Trial) -> float:
        train_x, val_x, train_y, val_y = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
        )

        # Расширенный и уточненный диапазон гиперпараметров
        params = {
            "n_estimators": 2000,  # Увеличиваем, чтобы early_stopping нашел лучший момент
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "random_state": 42,
            "verbose": 0,
            "cat_features": categorical_features,
            "auto_class_weights": "Balanced",  # Ключевой параметр для борьбы с дисбалансом
        }

        model = CatBoostClassifier(**params)
        model.fit(
            train_x,
            train_y,
            eval_set=[(val_x, val_y)],
            early_stopping_rounds=70,  # Увеличиваем для стабильности
            verbose=0,
        )

        preds = model.predict(val_x)
        f1 = float(f1_score(val_y, preds, pos_label=1))
        trial.set_user_attr("best_iteration", model.get_best_iteration())
        return f1

    study = optuna.create_study(
        direction="maximize", study_name="catboost_f1_optimization"
    )
    study.optimize(objective, n_trials=n_trials)

    # 5. Обучение финальной модели на лучших параметрах
    best_params = study.best_params
    best_iteration = study.best_trial.user_attrs["best_iteration"]

    final_model_params = best_params.copy()
    final_model_params.update(
        {
            "n_estimators": best_iteration,
            "cat_features": categorical_features,
            "auto_class_weights": "Balanced",
            "random_state": 42,
        }
    )

    final_model = CatBoostClassifier(**final_model_params)
    final_model.fit(X_train, y_train, verbose=100)  # Можно включить вывод для контроля

    # 6. Сборка артефактов для возврата
    artifacts = {
        "model": final_model,
        "study": study,
        "X_test": X_test,
        "y_test": y_test,
        "best_params": study.best_params,
        "feature_names": X_train.columns.tolist(),  # Сохраняем порядок признаков
    }
    return artifacts


if __name__ == "__main__":
    data_path = "data_processed.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        print(f"Ошибка: файл {data_path} не найден.")
        raise e

    print(f"Старт обучения модели в {time.ctime()}")
    start_time = time.time()

    artifacts = train_pipeline(
        df, n_trials=50
    )  # Для тестов можно ставить мало, для реального поиска 50-100

    duration = time.time() - start_time
    print(f"Конец обучения модели в {time.ctime()}")
    print(f"Затрачено на обучение: {duration:.2f} сек.")

    # Сохранение всех артефактов
    joblib.dump(artifacts["model"], "final_model.joblib")
    joblib.dump(artifacts["study"], "optuna_study.joblib")
    joblib.dump(
        artifacts["feature_names"], "feature_names.joblib"
    )  # Сохраняем признаки

    # Сохраняем тестовые данные для воспроизводимости оценки
    test_df = artifacts["X_test"].copy()
    test_df["is_done"] = artifacts["y_test"]
    test_df.to_csv("test_data.csv", index=False)

    print(
        "\nАртефакты (модель, исследование Optuna, имена признаков, тестовые данные) успешно сохранены."
    )
