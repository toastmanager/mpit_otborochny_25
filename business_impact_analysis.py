import pandas as pd
import joblib
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- Импортируем вашу функцию предобработки ---
try:
    from data_preprocessing import preprocess_data
except ImportError:
    print("Ошибка: не удалось найти файл 'data_preprocessing.py'.")
    exit()

# --- Конфигурация симуляции ---
RAW_DATA_PATH = "train.csv"  # Убедитесь, что ваш файл с данными так называется
MODEL_FILE_PATH = "models/final_model.joblib"
FEATURES_FILE_PATH = "models/feature_names.joblib"

MINIMUM_ABSOLUTE_BID = 120  # Минимальная цена любой поездки (в руб.)
MINIMUM_BID_PER_KM = 20
N_SIMULATION_ROWS = 100  # Сколько поездок из конца файла взять для анализа
SERVICE_COMMISSION_RATE = 0.14
BID_RANGE_SIM = 100  # Уменьшим диапазон для ускорения симуляции
BID_STEP_SIM = 10  # Уменьшим шаг для ускорения


# --- Копируем сюда класс BidOptimizer и немного его "облегчаем" для симуляции ---
# Мы уберем из него все print(), чтобы не засорять лог во время анализа тысяч поездок
class BidOptimizer:
    def __init__(self, model_path: str, features_path: str):
        try:
            self.model = joblib.load(model_path)
            self.feature_names = [
                f for f in joblib.load(features_path) if f != "is_done"
            ]
        except FileNotFoundError as e:
            print(f"Ошибка: Не удалось найти файлы модели: {e}")
            raise

    def calculate_optimal_bid(
        self, initial_features: Dict[str, Any], initial_bid: float
    ):
        bids = np.arange(
            max(initial_bid - BID_RANGE_SIM, BID_STEP_SIM),
            initial_bid + BID_RANGE_SIM + BID_STEP_SIM,
            BID_STEP_SIM,
        )
        results = []
        for bid in bids:
            current_features = initial_features.copy()
            current_features["price_bid_local"] = bid

            df_for_prediction = preprocess_data(
                pd.DataFrame(current_features, index=[0])
            )

            if df_for_prediction.empty:
                probability_of_success = 0.0
            else:
                try:
                    df_ordered = df_for_prediction[self.feature_names]
                    probability_of_success = self.model.predict_proba(df_ordered)[0, 1]
                except Exception:
                    probability_of_success = 0.0

            expected_profit = probability_of_success * (bid * SERVICE_COMMISSION_RATE)
            results.append(
                {
                    "bid": bid,
                    "probability": probability_of_success,
                    "expected_profit": expected_profit,
                }
            )

        if not results:
            return None

        best_result = max(results, key=lambda x: x["expected_profit"])

        # --- ЗАЩИТНЫЙ ФИЛЬТР ---
        # Рассчитываем минимально допустимую цену для этой конкретной поездки
        distance_km = initial_features.get("distance_in_meters", 0) / 1000
        min_price_for_ride = max(MINIMUM_ABSOLUTE_BID, distance_km * MINIMUM_BID_PER_KM)

        # Если модель предложила цену ниже минимальной, принудительно поднимаем ее
        if best_result["bid"] < min_price_for_ride:
            best_result["bid"] = min_price_for_ride
            # Важно: остальные метрики (вероятность, прибыль) станут неточными,
            # но мы предотвратим нерентабельный заказ.
        # ------------------------

        return best_result


def run_simulation(data_path: str, n_rows: int) -> pd.DataFrame:
    """Запускает симуляцию на последних N строках данных."""
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Ошибка: не найден файл {data_path}")
        return pd.DataFrame()

    test_df = df.tail(n_rows).copy()
    optimizer = BidOptimizer(
        model_path=MODEL_FILE_PATH, features_path=FEATURES_FILE_PATH
    )

    simulation_results = []
    print(f"Запуск симуляции на {len(test_df)} поездках...")

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        ride_data = row.to_dict()
        initial_bid_for_search = ride_data.get("price_start_local", 200)

        if not isinstance(initial_bid_for_search, (int, float)):
            initial_bid_for_search = 200

        optimal_bid_info = optimizer.calculate_optimal_bid(
            initial_features=ride_data, initial_bid=initial_bid_for_search
        )

        if optimal_bid_info:
            simulation_results.append(
                {
                    "original_bid": ride_data.get("price_bid_local"),
                    "is_done": 1 if ride_data.get("is_done") == "done" else 0,
                    "model_bid": optimal_bid_info["bid"],
                    "model_probability": optimal_bid_info["probability"],
                }
            )

    return pd.DataFrame(simulation_results)


def analyze_and_visualize(results_df: pd.DataFrame):
    """Считает метрики и строит графики."""
    if results_df.empty:
        print("Нет данных для анализа.")
        return

    # --- 1. Расчет метрик ---
    # Исторические показатели ("Было")
    historical_rides = results_df["is_done"].sum()
    historical_revenue = (results_df["original_bid"] * results_df["is_done"]).sum()
    historical_profit = historical_revenue * SERVICE_COMMISSION_RATE
    historical_ar = results_df["is_done"].mean()  # Acceptance Rate

    # Прогнозные показатели ("Стало бы")
    # Ожидаемое число поездок = сумма вероятностей
    model_expected_rides = results_df["model_probability"].sum()
    # Ожидаемая выручка = сумма (цена * вероятность)
    model_expected_revenue = (
        results_df["model_bid"] * results_df["model_probability"]
    ).sum()
    model_expected_profit = model_expected_revenue * SERVICE_COMMISSION_RATE
    model_expected_ar = results_df["model_probability"].mean()

    print("\n--- СРАВНЕНИЕ БИЗНЕС-ПОКАЗАТЕЛЕЙ ---")
    print(
        f"{'Метрика':<25} | {'Было (история)':<20} | {'Стало бы (модель)':<20} | {'Изменение':<10}"
    )
    print("-" * 85)

    profit_uplift = (model_expected_profit / historical_profit - 1) * 100
    print(
        f"{'Прибыль сервиса':<25} | {historical_profit:17.2f} ₽ | {model_expected_profit:17.2f} ₽ | {profit_uplift:+7.2f}%"
    )

    revenue_uplift = (model_expected_revenue / historical_revenue - 1) * 100
    print(
        f"{'Общая выручка':<25} | {historical_revenue:17.2f} ₽ | {model_expected_revenue:17.2f} ₽ | {revenue_uplift:+7.2f}%"
    )

    rides_uplift = (model_expected_rides / historical_rides - 1) * 100
    print(
        f"{'Кол-во поездок':<25} | {historical_rides:<20.0f} | {model_expected_rides:<20.2f} | {rides_uplift:+7.2f}%"
    )

    ar_uplift = (model_expected_ar / historical_ar - 1) * 100
    print(
        f"{'Acceptance Rate':<25} | {historical_ar:19.2%} | {model_expected_ar:19.2%} | {ar_uplift:+7.2f}%"
    )

    # --- 2. Визуализация ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(18, 6))

    # График 1: Сравнение ключевых метрик
    plt.subplot(1, 2, 1)
    metrics_data = {
        "Показатель": ["Прибыль", "Выручка", "Кол-во поездок"],
        "Было (история)": [historical_profit, historical_revenue, historical_rides],
        "Стало бы (модель)": [
            model_expected_profit,
            model_expected_revenue,
            model_expected_rides,
        ],
    }
    metrics_df = pd.DataFrame(metrics_data).melt(
        id_vars="Показатель", var_name="Тип", value_name="Значение"
    )
    sns.barplot(
        data=metrics_df, x="Показатель", y="Значение", hue="Тип", palette="viridis"
    )
    plt.title("Сравнение ключевых бизнес-метрик", fontsize=16)
    plt.ylabel("Абсолютное значение")
    plt.xlabel("")

    # График 2: Распределение цен
    plt.subplot(1, 2, 2)
    sns.kdeplot(
        results_df["original_bid"],
        fill=True,
        label="Исторические цены",
        color="skyblue",
    )
    sns.kdeplot(
        results_df["model_bid"], fill=True, label="Цены модели", color="lightcoral"
    )
    plt.title("Распределение цен (бидов)", fontsize=16)
    plt.xlabel("Цена поездки, ₽")
    plt.ylabel("Плотность")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Совет: чтобы лог не засорялся сообщениями из preprocess_data,
    # можно временно закомментировать print() в том файле на время анализа.
    results = run_simulation(RAW_DATA_PATH, N_SIMULATION_ROWS)
    analyze_and_visualize(results)
