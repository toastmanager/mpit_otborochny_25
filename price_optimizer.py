import warnings
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class ServicePriceOptimizer:
    def __init__(
        self,
        model_path: str = "models/final_model.joblib",
        feature_names_path: str = "models/feature_names.joblib",
    ):
        """
        Финальная версия оптимизатора цен для сервиса такси
        """
        try:
            self.model = joblib.load(model_path)
            self.feature_names = joblib.load(feature_names_path)
            print("✅ Модель и признаки успешно загружены")
        except FileNotFoundError:
            print("❌ Ошибка: файлы модели не найдены. Запустите обучение сначала.")
            raise

        # Оптимизированные настройки на основе анализа
        self.PRICE_STEP = 10
        self.SERVICE_COST_PER_KM = 8
        self.SERVICE_MARGIN_MIN = 0.20
        self.MIN_PRICE_PER_KM = 15
        self.MAX_PRICE_PER_KM = 45
        self.BASE_PRICE = 120

    def prepare_features(
        self,
        carmodel: str,
        carname: str,
        platform: str,
        distance: float,
        price: float,
        order_hour: int = 12,
        driver_rating: float = 4.8,
    ) -> pd.DataFrame:
        """
        Подготовка признаков только с теми данными, которые есть в датасете
        """
        if platform not in ["ios", "android"]:
            platform = "ios"

        # Используем только те признаки, которые были при обучении
        features = {
            "carmodel": carmodel,
            "carname": carname,
            "platform": platform,
            "distance": distance,
            "price": price,
            "price_per_km": price / distance if distance > 0 else 0,
            "duration_in_seconds": distance * 180 + 300,  # 3 мин/км + 5 мин подачи
            "driver_rating": driver_rating,
            "pickup_in_meters": 500,
            "pickup_in_seconds": 300,
            "price_start_local": max(price * 0.7, self.BASE_PRICE),
            "price_bid_local": price,
            "order_hour": order_hour,
            "order_dayofweek": 1,
            "is_night": 1 if 22 <= order_hour <= 23 or 0 <= order_hour <= 5 else 0,
            "price_per_meter": price / (distance * 1000) if distance > 0 else 0,
            "price_per_second": price / (distance * 180 + 300) if distance > 0 else 0,
            "price_increase_abs": price * 0.2,
            "price_increase_perc": 20.0,
            "driver_experience_days": 365,
            "distance_in_meters_log": np.log(distance * 1000) if distance > 0 else 0,
        }

        # Создаем DataFrame
        df = pd.DataFrame([features])

        # Обработка категориальных признаков
        categorical_features = ["carmodel", "carname", "platform"]
        for col in categorical_features:
            if col in df.columns:
                if (
                    col in self.feature_names
                    and df[col].iloc[0] not in self.feature_names
                ):
                    df[col] = "Other"
                df[col] = df[col].astype(str)

        # Убеждаемся, что все признаки в правильном порядке
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        return df[self.feature_names]

    def predict_success_probability(self, features: pd.DataFrame) -> float:
        """Предсказание вероятности успеха"""
        try:
            return self.model.predict_proba(features)[0, 1]
        except Exception as e:
            print(f"❌ Ошибка при предсказании: {e}")
            return 0.0

    def calculate_realistic_price_range(self, distance: float) -> Tuple[float, float]:
        """Расчет реалистичного диапазона цен"""
        min_price = max(self.BASE_PRICE, distance * self.MIN_PRICE_PER_KM)
        max_price = distance * self.MAX_PRICE_PER_KM
        return min_price, max_price

    def calculate_service_profit(
        self, price: float, success_prob: float, distance: float
    ) -> Dict:
        """Расчет прибыли сервиса"""
        service_cost = distance * self.SERVICE_COST_PER_KM + 80
        gross_profit = price - service_cost
        expected_profit = gross_profit * success_prob
        margin = gross_profit / price if price > 0 else 0

        return {
            "price": price,
            "service_cost": service_cost,
            "gross_profit": gross_profit,
            "expected_profit": expected_profit,
            "margin": margin,
            "success_probability": success_prob,
            "price_per_km": price / distance if distance > 0 else 0,
        }

    def find_optimal_service_price(
        self,
        carmodel: str,
        carname: str,
        platform: str,
        distance: float,
        base_price: float,
        order_hour: int = 12,
        driver_rating: float = 4.8,
    ) -> Dict:
        """
        Поиск оптимальной цены с использованием только существующих признаков
        """
        print(f"🔍 Анализ заказа: {carmodel} {carname}, {distance} км")
        print(f"💡 Базовая цена: {base_price} руб.")
        print(f"🕒 Время заказа: {order_hour}:00, Рейтинг водителя: {driver_rating}")

        # Рассчитываем реалистичный диапазон
        min_price, max_price = self.calculate_realistic_price_range(distance)

        # Корректируем базовую цену если нужно
        adjusted_base = max(min_price, min(base_price, max_price))
        if adjusted_base != base_price:
            print(f"📝 Скорректированная базовая цена: {adjusted_base} руб.")
            base_price = adjusted_base

        print(f"📊 Диапазон цен: {min_price:.0f} - {max_price:.0f} руб.")
        print(f"   ({self.MIN_PRICE_PER_KM} - {self.MAX_PRICE_PER_KM} руб./км)")

        # Умный подбор диапазона тестирования
        test_min = max(min_price, base_price - 50)
        test_max = min(max_price, base_price + 100)

        test_prices = np.arange(test_min, test_max + self.PRICE_STEP, self.PRICE_STEP)
        if base_price not in test_prices:
            test_prices = np.append(test_prices, base_price)
        test_prices = np.sort(test_prices)

        print(
            f"🔎 Тестируем {len(test_prices)} цен от {test_min:.0f} до {test_max:.0f} руб."
        )

        results = []
        for price in test_prices:
            features = self.prepare_features(
                carmodel, carname, platform, distance, price, order_hour, driver_rating
            )
            success_prob = self.predict_success_probability(features)
            profit_data = self.calculate_service_profit(price, success_prob, distance)
            results.append(profit_data)

        if not results:
            return {}

        # Сортируем по ожидаемой прибыли
        results.sort(key=lambda x: x["expected_profit"], reverse=True)
        base_option = next(r for r in results if r["price"] == base_price)
        best_option = results[0]

        # Находим вариант с минимальной маржой
        min_margin_options = [
            r for r in results if r["margin"] >= self.SERVICE_MARGIN_MIN
        ]
        min_margin_option = min_margin_options[0] if min_margin_options else base_option

        recommendation = {
            "base_price": base_option,
            "optimal_price": best_option,
            "min_margin_price": min_margin_option,
            "all_options": results[:6],
            "service_cost": base_option["service_cost"],
            "min_margin": self.SERVICE_MARGIN_MIN,
        }

        return recommendation

    def print_recommendation(self, recommendation: Dict):
        """Четкий вывод рекомендаций"""
        if not recommendation:
            print("❌ Нет данных для рекомендации")
            return

        base = recommendation["base_price"]
        optimal = recommendation["optimal_price"]
        min_margin = recommendation["min_margin_price"]

        print("\n" + "=" * 60)
        print("💰 ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ ДЛЯ СЕРВИСА")
        print("=" * 60)

        improvement = optimal["expected_profit"] - base["expected_profit"]

        # Основная рекомендация
        if improvement > 5:  # Значительное улучшение
            print(f"🎯 РЕКОМЕНДАЦИЯ: Увеличить цену до {optimal['price']:.0f} руб.")
            print(
                f"   📈 Прибыль вырастет на +{improvement:.0f} руб. ({improvement / base['expected_profit']:+.0%})"
            )
        elif improvement < -5:  # Ухудшение
            print(f"🎯 РЕКОМЕНДАЦИЯ: Снизить цену до {optimal['price']:.0f} руб.")
            print(f"   📉 Потери сократятся на {abs(improvement):.0f} руб.")
        else:
            print(f"🎯 РЕКОМЕНДАЦИЯ: Оставить цену {base['price']:.0f} руб.")
            print("   ✅ Текущая цена близка к оптимальной")

        print("\n📊 ДЕТАЛИ РЕКОМЕНДАЦИИ:")
        print(
            f"   • Цена: {optimal['price']:.0f} руб. ({optimal['price_per_km']:.1f} руб./км)"
        )
        print(f"   • Вероятность успеха: {optimal['success_probability']:.1%}")
        print(f"   • Ожидаемая прибыль: {optimal['expected_profit']:.0f} руб.")
        print(f"   • Маржа: {optimal['margin']:.1%}")
        print(f"   • Себестоимость: {recommendation['service_cost']:.0f} руб.")

        # Показываем сравнение с текущей ценой
        if optimal["price"] != base["price"]:
            print("\n📈 СРАВНЕНИЕ С ТЕКУЩЕЙ ЦЕНОЙ:")
            print(
                f"   Текущая: {base['price']:.0f} руб. → {base['expected_profit']:.0f} руб. прибыли"
            )
            print(
                f"   Рекомендуемая: {optimal['price']:.0f} руб. → {optimal['expected_profit']:.0f} руб. прибыли"
            )

        # Альтернативные варианты
        if len(recommendation["all_options"]) > 1:
            print("\n🏆 ЛУЧШИЕ ВАРИАНТЫ:")
            for option in recommendation["all_options"]:
                markers = []
                if option["price"] == optimal["price"]:
                    markers.append("РЕКОМЕНДУЕМ")
                if (
                    option["price"] == base["price"]
                    and base["price"] != optimal["price"]
                ):
                    markers.append("ТЕКУЩАЯ")
                if (
                    option["price"] == min_margin["price"]
                    and min_margin["price"] != optimal["price"]
                ):
                    markers.append("МИН.МАРЖА")

                marker_str = " 👈 " + ", ".join(markers) if markers else ""
                print(
                    f"   {option['price']:4.0f}р | {option['expected_profit']:4.0f}р прибыли | {option['success_probability']:4.1%} шанс{marker_str}"
                )


def main():
    """Демонстрация работы оптимизатора"""
    try:
        optimizer = ServicePriceOptimizer()

        # Тестовые сценарии с разным временем суток
        test_orders = [
            {
                "carmodel": "Toyota Camry",
                "carname": "Camry 2.5",
                "platform": "ios",
                "distance": 8.5,
                "base_price": 200,
                "order_hour": 18,  # Вечер
                "driver_rating": 4.9,
            },
            {
                "carmodel": "Kia Rio",
                "carname": "Rio Classic",
                "platform": "android",
                "distance": 5.2,
                "base_price": 150,
                "order_hour": 10,  # Утро
                "driver_rating": 4.7,
            },
            {
                "carmodel": "Hyundai Solaris",
                "carname": "Solaris 1.6",
                "platform": "ios",
                "distance": 12.0,
                "base_price": 300,
                "order_hour": 23,  # Ночь
                "driver_rating": 4.8,
            },
        ]

        for i, order in enumerate(test_orders, 1):
            print(f"\n{'#' * 60}")
            print(f"📦 ЗАКАЗ #{i}")
            print(f"{'#' * 60}")

            recommendation = optimizer.find_optimal_service_price(**order)
            optimizer.print_recommendation(recommendation)

        print(f"\n{'=' * 60}")
        print("✅ АНАЛИЗ ЗАВЕРШЕН! Рекомендации готовы к использованию.")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
