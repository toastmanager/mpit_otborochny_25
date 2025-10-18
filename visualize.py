import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
)
import os

# --- Константы и настройки ---
# Папка для сохранения визуализаций
ARTIFACTS_DIR = "models/"
PLOTS_DIR = "visualizations"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "final_model.joblib")
STUDY_PATH = os.path.join(ARTIFACTS_DIR, "optuna_study.joblib")
TEST_DATA_PATH = os.path.join(ARTIFACTS_DIR, "test_data.csv")

# Создаем папку для графиков, если ее нет
os.makedirs(PLOTS_DIR, exist_ok=True)

# Настройки для графиков
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

# Устанавливаем бэкенд для matplotlib чтобы не открывать окна
plt.switch_backend("Agg")


def load_artifacts(model_path: str, study_path: str, test_data_path: str) -> dict:
    """Загружает все сохраненные артефакты."""
    try:
        model = joblib.load(model_path)
        study = joblib.load(study_path)
        test_df = pd.read_csv(test_data_path)
        print("Артефакты успешно загружены.")
        return {"model": model, "study": study, "test_df": test_df}
    except FileNotFoundError as e:
        print(f"Ошибка: Не удалось найти файл артефакта. {e}")
        raise


def plot_optuna_study(study):
    """Создает и сохраняет графики исследования Optuna."""
    fig_history = plot_optimization_history(study)
    history_path = os.path.join(PLOTS_DIR, "optuna_optimization_history.png")
    fig_history.write_image(history_path)
    print(f"График истории оптимизации сохранен в {history_path}")

    fig_params = plot_param_importances(study)
    params_path = os.path.join(PLOTS_DIR, "optuna_param_importances.png")
    fig_params.write_image(params_path)
    print(f"График важности параметров сохранен в {params_path}")


def plot_feature_importance(model, feature_names):
    """Строит и сохраняет график важности признаков."""
    importances = pd.DataFrame(
        {"feature": feature_names, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x="importance", y="feature", data=importances.head(20))
    plt.title("Топ-20 Важных признаков модели")
    plt.xlabel("Важность")
    plt.ylabel("Признак")
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.savefig(save_path)
    plt.close()
    print(f"График важности признаков сохранен в {save_path}")


def plot_roc_curve(model, X_test, y_test):
    """Создает и сохраняет ROC-кривую."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC кривая (AUC = {roc_auc:.4f})"
    )
    plt.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=2,
        linestyle="--",
        label="Случайный классификатор (AUC = 0.5)",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (Доля ложно-положительных)")
    plt.ylabel("True Positive Rate (Доля истинно-положительных)")
    plt.title("ROC-кривая (Receiver Operating Characteristic)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    roc_path = os.path.join(PLOTS_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC-кривая сохранена в {roc_path}")
    print(f"ROC AUC на тестовых данных: {roc_auc:.4f}")


def plot_model_performance(model, X_test, y_test):
    """Создает и сохраняет графики для оценки производительности модели."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_class = model.predict(X_test)

    plot_roc_curve(model, X_test, y_test)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba, pos_label=1)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, lw=2, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Полнота (Recall)")
    plt.ylabel("Точность (Precision)")
    plt.title("Кривая Точности-Полноты (Precision-Recall Curve)")
    plt.legend(loc="best")
    pr_curve_path = os.path.join(PLOTS_DIR, "precision_recall_curve.png")
    plt.savefig(pr_curve_path)
    plt.close()
    print(f"График PR-кривой сохранен в {pr_curve_path}")

    cm = confusion_matrix(y_test, y_pred_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues")
    plt.title("Матрица ошибок (Confusion Matrix)")
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Матрица ошибок сохранена в {cm_path}")


def plot_error_analysis(model, X_test, y_test, top_n_features=3):
    """
    Создает и сохраняет графики для анализа ошибок модели.
    """
    print("\n--- Начало анализа ошибок модели ---")

    y_pred_class = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    analysis_df = X_test.copy()
    analysis_df["true_label"] = y_test
    analysis_df["predicted_label"] = y_pred_class
    analysis_df["predicted_proba"] = y_pred_proba

    conditions = [
        (analysis_df["true_label"] == 0) & (analysis_df["predicted_label"] == 1),
        (analysis_df["true_label"] == 1) & (analysis_df["predicted_label"] == 0),
        (analysis_df["true_label"] == 1) & (analysis_df["predicted_label"] == 1),
        (analysis_df["true_label"] == 0) & (analysis_df["predicted_label"] == 0),
    ]
    outcomes = ["False Positive", "False Negative", "True Positive", "True Negative"]
    analysis_df["outcome"] = pd.Series(
        np.select(conditions, outcomes, default="Other"), index=analysis_df.index
    )

    plt.figure(figsize=(12, 7))
    sns.histplot(
        data=analysis_df,
        x="predicted_proba",
        hue="true_label",
        multiple="layer",
        bins=50,
        palette=["#4c72b0", "#dd8452"],
    )
    plt.title("Распределение предсказанных вероятностей для каждого класса")
    plt.xlabel("Предсказанная вероятность положительного класса")
    plt.ylabel("Количество")
    prob_path = os.path.join(PLOTS_DIR, "error_analysis_probabilities.png")
    plt.savefig(prob_path)
    plt.close()
    print(f"График распределения вероятностей сохранен в {prob_path}")

    importances = pd.DataFrame(
        {
            "feature": X_test.columns.tolist(),
            "importance": model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)

    top_features = importances["feature"].head(top_n_features).tolist()

    for feature in top_features:
        plt.figure(figsize=(12, 7))

        is_numeric = pd.api.types.is_numeric_dtype(analysis_df[feature])
        treat_as_categorical = analysis_df[feature].nunique() < 25

        palette = {
            "False Positive": "#d62728",
            "False Negative": "#ff7f0e",
            "True Positive": "#2ca02c",
            "True Negative": "#1f77b4",
        }

        # Если признак числовой и непрерывный -> kdeplot
        if is_numeric and not treat_as_categorical:
            sns.kdeplot(
                data=analysis_df,
                x=feature,
                hue="outcome",
                fill=True,
                common_norm=False,
                palette=palette,
            )
            plt.title(f"Распределение плотности вероятности для признака: {feature}")
            # Строка ниже УДАЛЕНА, так как Seaborn создает легенду автоматически
            # plt.legend(title='Outcome')
        # Иначе (категориальный или дискретный числовой) -> countplot
        else:
            order = sorted(analysis_df[feature].unique()) if is_numeric else None
            sns.countplot(
                data=analysis_df, x=feature, hue="outcome", order=order, palette=palette
            )
            plt.title(f"Распределение ошибок по признаку: {feature}")
            if analysis_df[feature].dtype == "object":
                plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        save_path = os.path.join(PLOTS_DIR, f"error_analysis_{feature}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"График анализа ошибок для '{feature}' сохранен в {save_path}")


if __name__ == "__main__":
    artifacts = load_artifacts(MODEL_PATH, STUDY_PATH, TEST_DATA_PATH)
    model = artifacts["model"]
    study = artifacts["study"]
    test_df = artifacts["test_df"]

    X_test = test_df.drop(columns=["is_done"])
    y_test = test_df["is_done"]

    print("\n--- Создание визуализаций для исследования Optuna ---")
    plot_optuna_study(study)

    print("\n--- Создание визуализации важности признаков ---")
    plot_feature_importance(model, X_test.columns.tolist())

    print("\n--- Создание визуализаций для оценки качества модели ---")
    plot_model_performance(model, X_test, y_test)

    plot_error_analysis(model, X_test, y_test, top_n_features=3)

    print(f"\nВсе визуализации сохранены в папку '{PLOTS_DIR}'.")
