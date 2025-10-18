import os
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import optuna
import time
from typing import Dict, Any

# Устанавливаем уровень логирования Optuna, чтобы избежать лишнего вывода
optuna.logging.set_verbosity(optuna.logging.WARNING)

### ИЗМЕНЕНИЕ 1: Добавляем контрольный флаг и путь к файлу исследования ###
# --- Контрольные параметры ---
# Поставьте True, чтобы загрузить существующее исследование и пропустить оптимизацию.
# Поставьте False, чтобы запустить новый подбор гиперпараметров.
USE_EXISTING_STUDY = False
STUDY_PATH = "models/optuna_study.joblib"
# --------------------------------


def train_pipeline(
    df: pd.DataFrame,
    n_trials: int = 50,
    ### ИЗМЕНЕНИЕ 2: Добавляем флаг и путь в аргументы функции ###
    use_existing_study: bool = False,
    study_path: str = "",
) -> Dict[str, Any]:
    """
    Организует полный пайплайн: инжиниринг признаков, подбор гиперпараметров,
    обучение и оценка модели.
    """
    # ... (код подготовки данных остается без изменений)
    columns_to_drop = ["is_done"]
    categorical_features = ["carmodel", "carname", "platform"]

    X = df.drop(columns=columns_to_drop, errors="ignore")
    y = df["is_done"]

    for col in categorical_features:
        if col in X.columns:
            counts = X[col].value_counts()
            rare_categories = counts[counts < 10].index.tolist()
            X[col] = X[col].replace(rare_categories, "Other").astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    try:
        test_model = CatBoostClassifier(
            task_type="GPU", devices="0", cat_features=categorical_features, verbose=0
        )
        test_model.fit(X_train.head(10), y_train.head(10), verbose=0)
        task_type = "GPU"
        devices = "0"
        print("Используется GPU")
    except Exception as e:
        task_type = "CPU"
        devices = None
        print(e)
        print("Используется СPU")

    ### ИЗМЕНЕНИЕ 3: Основной блок логики для переключения режимов ###
    if use_existing_study and os.path.exists(study_path):
        print(f"Загрузка существующего исследования Optuna из файла: {study_path}")
        study = joblib.load(study_path)
        best_params = study.best_params
        # .get() используется для безопасности, если в старом файле нет этого атрибута
        best_iteration = study.best_trial.user_attrs.get("best_iteration", 2000)
        print("Исследование успешно загружено. Оптимизация будет пропущена.")
    else:
        if use_existing_study:
            print(
                f"Внимание: Файл исследования '{study_path}' не найден. Запускается новая оптимизация."
            )
        else:
            print("Запуск нового исследования Optuna для подбора гиперпараметров...")

        def objective(trial: optuna.Trial) -> float:
            train_x, val_x, train_y, val_y = train_test_split(
                X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
            )
            params = {
                "n_estimators": 2000,
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.1, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 4, 9),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
                "random_state": 42,
                "verbose": 0,
                "task_type": task_type,
                "cat_features": categorical_features,
                "auto_class_weights": "Balanced",
            }
            if devices:
                params["devices"] = devices
            model = CatBoostClassifier(**params)
            model.fit(
                train_x,
                train_y,
                eval_set=[(val_x, val_y)],
                early_stopping_rounds=70,
                verbose=0,
            )

            # Получаем вероятности вместо предсказанных классов
            preds_proba = model.predict_proba(val_x)[:, 1]

            # Вычисляем ROC-AUC вместо F1-score
            roc_auc = float(roc_auc_score(val_y, preds_proba))

            trial.set_user_attr("best_iteration", model.get_best_iteration())
            return roc_auc

        study = optuna.create_study(
            direction="maximize", study_name="catboost_f1_optimization"
        )
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        best_iteration = study.best_trial.user_attrs["best_iteration"]

    # 5. Обучение финальной модели на лучших параметрах (этот блок выполняется в любом случае)
    print("\nОбучение финальной модели на лучших параметрах...")
    final_model_params = best_params.copy()
    final_model_params.update(
        {
            "n_estimators": best_iteration,
            "cat_features": categorical_features,
            "auto_class_weights": "Balanced",
            "random_state": 42,
            "task_type": task_type,
            "verbose": 100,
        }
    )
    if devices:
        final_model_params["devices"] = devices

    final_model = CatBoostClassifier(**final_model_params)
    final_model.fit(X_train, y_train, verbose=100)

    # 6. Сборка артефактов для возврата
    artifacts = {
        "model": final_model,
        "study": study,
        "X_test": X_test,
        "y_test": y_test,
        "best_params": study.best_params,
        "feature_names": X_train.columns.tolist(),
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

    ### ИЗМЕНЕНИЕ 4: Передаем флаг и путь в функцию ###
    artifacts = train_pipeline(
        df, n_trials=40, use_existing_study=USE_EXISTING_STUDY, study_path=STUDY_PATH
    )

    duration = time.time() - start_time
    print(f"Конец обучения модели в {time.ctime()}")
    print(f"Затрачено на обучение: {duration:.2f} сек.")

    # Сохранение всех артефактов (код без изменений)
    os.makedirs("models", exist_ok=True)
    joblib.dump(artifacts["model"], "models/final_model.joblib")
    # Если мы не запускали новую оптимизацию, мы все равно сохраняем загруженное исследование.
    # Это не повредит, но если хотите, можно добавить проверку.
    joblib.dump(artifacts["study"], "models/optuna_study.joblib")
    joblib.dump(artifacts["feature_names"], "models/feature_names.joblib")

    test_df = artifacts["X_test"].copy()
    test_df["is_done"] = artifacts["y_test"]
    test_df.to_csv("models/test_data.csv", index=False)

    print(
        "\nАртефакты (модель, исследование Optuna, имена признаков, тестовые данные) успешно сохранены."
    )
