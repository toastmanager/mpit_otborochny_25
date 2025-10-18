import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    roc_curve,  # <-- ДОБАВЛЕНО ИМПОРТ
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
ARTIFACTS_DIR = "."
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
    # 1. График истории оптимизации
    fig_history = plot_optimization_history(study)
    history_path = os.path.join(PLOTS_DIR, "optuna_optimization_history.png")
    fig_history.write_image(history_path)
    print(f"График истории оптимизации сохранен в {history_path}")

    # 2. График важности гиперпараметров
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
    print(f"График важности признаков сохранен в {save_path}")
    plt.show()


def plot_roc_curve(model, X_test, y_test):
    """Создает и сохраняет ROC-кривую."""
    # Получаем предсказания вероятностей для положительного класса
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Вычисляем ROC-кривую и AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Строим график
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

    # Сохраняем график
    roc_path = os.path.join(PLOTS_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    print(f"ROC-кривая сохранена в {roc_path}")
    plt.show()

    # Выводим AUC в консоль для удобства
    print(f"ROC AUC на тестовых данных: {roc_auc:.4f}")

    return roc_auc


def plot_model_performance(model, X_test, y_test):
    """Создает и сохраняет графики для оценки производительности модели."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_class = model.predict(X_test)

    # 1. ROC-кривая (ВЫЗЫВАЕМ ПЕРВОЙ, так как она показывает AUC)
    roc_auc = plot_roc_curve(model, X_test, y_test)

    # 2. Кривая Точности-Полноты (Precision-Recall Curve)
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
    print(f"График PR-кривой сохранен в {pr_curve_path}")
    plt.show()

    # 3. Матрица ошибок (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues")
    plt.title("Матрица ошибок (Confusion Matrix)")
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Матрица ошибок сохранена в {cm_path}")
    plt.show()


if __name__ == "__main__":
    # Загружаем артефакты
    artifacts = load_artifacts(MODEL_PATH, STUDY_PATH, TEST_DATA_PATH)
    model = artifacts["model"]
    study = artifacts["study"]
    test_df = artifacts["test_df"]

    # Готовим тестовые данные
    X_test = test_df.drop(columns=["is_done"])
    y_test = test_df["is_done"]

    print("\n--- Создание визуализаций для исследования Optuna ---")
    plot_optuna_study(study)

    print("\n--- Создание визуализации важности признаков ---")
    plot_feature_importance(model, X_test.columns.tolist())

    print("\n--- Создание визуализаций для оценки качества модели ---")
    plot_model_performance(model, X_test, y_test)

    print(f"\nВсе визуализации сохранены в папку '{PLOTS_DIR}'.")
