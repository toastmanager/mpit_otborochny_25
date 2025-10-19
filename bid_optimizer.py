import pandas as pd
import joblib
import numpy as np
from typing import Dict, Any
from data_preprocessing import preprocess_data

SERVICE_COMMISSION_RATE = 0.14
DRIVER_TAX_RATE = 0.05


class BidOptimizer:
    """
    Класс для расчета оптимального бида с использованием обученной ML-модели.
    ОПТИМИЗИРОВАННАЯ ВЕРСИЯ.
    """

    def __init__(self, model_path: str, features_path: str):
        try:
            self.model = joblib.load(model_path)
            self.feature_names = joblib.load(features_path)
            print("Модель и признаки успешно загружены.")
        except FileNotFoundError as e:
            print(
                f"Ошибка: Не удалось найти файлы модели или признаков. Проверьте пути: {e}"
            )
            raise

    def calculate_optimal_bid(
        self,
        initial_features: Dict[str, Any],
        initial_bid: float,
        bid_range: int = 100,
        bid_step: int = 10,
    ) -> Dict[str, Any]:
        """
        Рассчитывает оптимальный бид путем векторной обработки всех возможных цен,
        что значительно быстрее, чем итеративный перебор.
        """
        # 1. Создаем массив всех возможных бидов
        bids = np.arange(
            max(initial_bid - bid_range, bid_step),
            initial_bid + bid_range + bid_step,
            bid_step,
        )
        num_bids = len(bids)
        if num_bids == 0:
            return {}

        # 2. Создаем "батч" DataFrame: дублируем исходные признаки N раз,
        # где N - количество проверяемых бидов.
        features_df = pd.DataFrame([initial_features] * num_bids)
        features_df["price_bid_local"] = bids  # Заменяем цену на массив бидов

        # 3. Вызываем предобработку ОДИН РАЗ для всего батча
        preprocessed_df = preprocess_data(features_df)

        # 4. Принудительно приводим типы и порядок колонок (как в предыдущем исправлении)
        categorical_cols = ["platform", "order_hour", "order_dayofweek"]
        for col in categorical_cols:
            if col in preprocessed_df.columns:
                preprocessed_df[col] = preprocessed_df[col].astype(str)

        try:
            preprocessed_df = preprocessed_df[self.feature_names]
        except KeyError as e:
            print(f"Ошибка согласования колонок: {e}")
            return {}

        # 5. Вызываем предсказание ОДИН РАЗ и сразу получаем все вероятности
        # probabilities[:, 1] - берем вероятности для класса '1' (успех)
        try:
            probabilities = self.model.predict_proba(preprocessed_df)[:, 1]
        except Exception as e:
            print(f"Ошибка при пакетном предсказании: {e}")
            return {}

        # 6. Выполняем финансовые расчеты с помощью быстрых операций numpy
        service_profits = bids * SERVICE_COMMISSION_RATE
        expected_profits = probabilities * service_profits

        # 7. Находим индекс бида с максимальной ожидаемой прибылью
        best_idx = np.argmax(expected_profits)

        # 8. Собираем и возвращаем лучший результат
        best_bid = bids[best_idx]
        best_probability = probabilities[best_idx]
        best_expected_profit = expected_profits[best_idx]

        driver_income = best_bid * (1 - SERVICE_COMMISSION_RATE)
        driver_net_income = driver_income * (1 - DRIVER_TAX_RATE)

        best_result = {
            "bid": best_bid,
            "probability_of_success": best_probability,
            "service_profit": service_profits[best_idx],
            "expected_profit": best_expected_profit,
            "driver_net_income": driver_net_income,
        }

        return best_result


if __name__ == "__main__":
    MODEL_FILE_PATH = "models/final_model.joblib"
    FEATURES_FILE_PATH = "models/feature_names.joblib"

    optimizer = BidOptimizer(
        model_path=MODEL_FILE_PATH, features_path=FEATURES_FILE_PATH
    )

    df = pd.read_csv("train.csv")
    for i in range(48210, 48216):
        row = df.iloc[i].to_dict()
        row["price_bid_local"] = None
        optimal_bid_info = optimizer.calculate_optimal_bid(
            initial_features=row,
            initial_bid=row["price_start_local"],
            bid_range=150,
            bid_step=5,
        )

        if optimal_bid_info:
            print("\n--- Результаты расчета оптимального бида ---")
            print(f"Начальный бид: {row['price_start_local']:.2f} руб.")
            print("-" * 40)
            print(f"Оптимальный бид для пассажира: {optimal_bid_info['bid']:.2f} руб.")
            print(
                f"Вероятность согласия при этом биде: {optimal_bid_info['probability_of_success']:.2%}"
            )
            print(
                f"Ожидаемая прибыль сервиса: {optimal_bid_info['expected_profit']:.2f} руб."
            )
            print(
                f"Чистый доход водителя (после комиссии и налога): {optimal_bid_info['driver_net_income']:.2f} руб."
            )
            print("---" * 20)
