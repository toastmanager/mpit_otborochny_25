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
    """

    def __init__(self, model_path: str, features_path: str):
        """
        Инициализация класса, загрузка модели и списка признаков.

        :param model_path: Путь к файлу с обученной моделью (.joblib).
        :param features_path: Путь к файлу со списком названий признаков (.joblib).
        """
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
        Рассчитывает оптимальный бид путем итеративного перебора цен
        и максимизации ожидаемой прибыли сервиса.

        :param initial_features: Словарь с признаками поездки (без цены).
        :param initial_bid: Начальная цена, предложенная системой (например, на основе расстояния).
        :param bid_range: Диапазон для поиска вокруг начальной цены (в рублях).
        :param bid_step: Шаг изменения цены при поиске.
        :return: Словарь с результатами расчета.
        """
        bids = np.arange(
            max(initial_bid - bid_range, bid_step),  # Не даем цене упасть до нуля
            initial_bid + bid_range + bid_step,
            bid_step,
        )

        results = []

        for bid in bids:
            # 1. Рассчитываем финансовые показатели для текущего бида
            driver_income_before_tax = bid * (1 - SERVICE_COMMISSION_RATE)
            driver_tax = driver_income_before_tax * DRIVER_TAX_RATE
            driver_net_income = driver_income_before_tax - driver_tax
            service_profit = bid * SERVICE_COMMISSION_RATE

            # 2. Формируем вектор признаков для модели
            # Важно: В модели может быть признак, отвечающий за цену.
            # Предположим, он называется 'bid' или 'price'. Здесь мы его устанавливаем.
            current_features = initial_features.copy()
            current_features["price_bid_local"] = (
                bid  # Убедитесь, что 'bid' - это правильное имя признака
            )
            df_for_prediction = preprocess_data(
                pd.DataFrame(current_features, index=[1]),
            )

            # Создаем DataFrame с правильным порядком колонок
            # df_for_prediction = pd.DataFrame(
            #     [current_features], columns=self.feature_names
            # )

            # 3. Получаем вероятность согласия от модели
            try:
                # predict_proba возвращает вероятности для каждого класса, нам нужен класс "1" (согласие)
                probability_of_success = self.model.predict_proba(df_for_prediction)[
                    0, 1
                ]
            except Exception as e:
                print(f"Ошибка при предсказании для бида {bid}: {e}")
                continue

            # 4. Рассчитываем ожидаемую прибыль
            expected_profit = probability_of_success * service_profit

            results.append(
                {
                    "bid": bid,
                    "probability_of_success": probability_of_success,
                    "service_profit": service_profit,
                    "expected_profit": expected_profit,
                    "driver_net_income": driver_net_income,
                }
            )

        if not results:
            print("Не удалось получить ни одного результата. Проверьте входные данные.")
            return {}

        # 5. Находим лучший бид, который максимизирует ожидаемую прибыль
        best_result = max(results, key=lambda x: x["expected_profit"])

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
