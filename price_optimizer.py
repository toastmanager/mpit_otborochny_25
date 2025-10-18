import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")


class PriceOptimizer:
    def __init__(
        self,
        model_path: str = "models/final_model.joblib",
        feature_names_path: str = "models/feature_names.joblib",
    ):
        """
        Инициализация оптимизатора цен

        Args:
            model_path: путь к обученной модели
            feature_names_path: путь к файлу с именами признаков
        """
        try:
            self.model = joblib.load(model_path)
            self.feature_names = joblib.load(feature_names_path)
            print("✅ Модель и признаки успешно загружены")
        except FileNotFoundError:
            print("❌ Ошибка: файлы модели не найдены. Запустите обучение сначала.")
            raise

        # Настройки цен
        self.PRICE_STEP = 10  # Уменьшили шаг до 10 рублей

    def prepare_features(
        self, carmodel: str, carname: str, platform: str, distance: float, price: float
    ) -> pd.DataFrame:
        """
        Подготовка признаков для модели
        """
        # Проверяем корректность платформы
        if platform not in ["ios", "android"]:
            print(
                f"⚠️  Платформа '{platform}' не 'ios' или 'android'. Используем 'ios' по умолчанию."
            )
            platform = "ios"

        # Создаем базовые фичи
        features = {
            "carmodel": carmodel,
            "carname": carname,
            "platform": platform,
            "distance": distance,
            "price": price,
            "price_per_km": price / distance if distance > 0 else 0,
        }

        # Добавляем недостающие фичи с значениями по умолчанию
        additional_features = {
            "duration_in_seconds": 1800,  # 30 минут по умолчанию
            "driver_rating": 4.8,  # высокий рейтинг
            "pickup_in_meters": 500,  # 500 метров до подачи
            "pickup_in_seconds": 300,  # 5 минут до подачи
            "price_start_local": price * 0.8,  # стартовая цена на 20% ниже
            "price_bid_local": price,  # текущая ставка
            "order_hour": 12,  # полдень
            "order_dayofweek": 1,  # понедельник
            "is_night": 0,  # не ночь
            "price_per_meter": price / (distance * 1000) if distance > 0 else 0,
            "price_per_second": price / 1800 if distance > 0 else 0,
            "price_increase_abs": price * 0.2,  # увеличение цены на 20%
            "price_increase_perc": 20.0,  # 20% увеличение
            "driver_experience_days": 365,  # 1 год опыта
            "distance_in_meters_log": np.log(distance * 1000) if distance > 0 else 0,
        }

        # Объединяем все фичи
        all_features = {**features, **additional_features}

        # Создаем DataFrame
        df = pd.DataFrame([all_features])

        # Обработка редких категорий
        categorical_features = ["carmodel", "carname", "platform"]
        for col in categorical_features:
            if col in df.columns:
                if (
                    col in self.feature_names
                    and df[col].iloc[0] not in self.feature_names
                ):
                    df[col] = "Other"
                df[col] = df[col].astype(str)

        # Убеждаемся, что все признаки в правильном порядке и есть все нужные колонки
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0  # добавляем недостающие колонки

        return df[self.feature_names]

    def predict_success_probability(self, features: pd.DataFrame) -> float:
        """
        Предсказание вероятности успешного завершения поездки
        """
        try:
            # Предсказываем вероятность класса 1 (успешная поездка)
            probability = self.model.predict_proba(features)[0, 1]
            return probability
        except Exception as e:
            print(f"❌ Ошибка при предсказании: {e}")
            return 0.0

    def calculate_expected_value(self, price: float, success_prob: float) -> float:
        """
        Расчет ожидаемого дохода: цена × вероятность успеха
        """
        return price * success_prob

    def find_optimal_price(
        self,
        carmodel: str,
        carname: str,
        platform: str,
        distance: float,
        proposed_price: float,
    ) -> Dict:
        """
        Поиск оптимальной цены для максимизации ожидаемого дохода

        Returns:
            Словарь с рекомендациями
        """
        print(
            f"🔍 Анализ заказа: {carmodel} {carname}, {distance} км, платформа: {platform}"
        )
        print(f"💡 Предложенная цена: {proposed_price} руб.")

        results = []

        # Генерируем цены от предложенной до +300 рублей с шагом 10
        min_test_price = proposed_price
        max_test_price = proposed_price + 300

        test_prices = np.arange(
            min_test_price, max_test_price + self.PRICE_STEP, self.PRICE_STEP
        )

        # Добавляем предложенную цену если ее нет в списке (на всякий случай)
        if proposed_price not in test_prices:
            test_prices = np.append(test_prices, proposed_price)

        test_prices = np.sort(test_prices)

        print(
            f"📊 Тестируем цены от {min_test_price} до {max_test_price} руб. (шаг: {self.PRICE_STEP} руб.)"
        )

        for price in test_prices:
            features = self.prepare_features(
                carmodel, carname, platform, distance, price
            )
            success_prob = self.predict_success_probability(features)
            expected_value = self.calculate_expected_value(price, success_prob)

            results.append(
                {
                    "price": price,
                    "success_probability": success_prob,
                    "expected_value": expected_value,
                }
            )

        # Сортируем по ожидаемому доходу
        results.sort(key=lambda x: x["expected_value"], reverse=True)

        best_option = results[0]
        proposed_option = next(r for r in results if r["price"] == proposed_price)

        recommendation = {
            "proposed_price": {
                "price": proposed_price,
                "success_probability": proposed_option["success_probability"],
                "expected_value": proposed_option["expected_value"],
            },
            "recommended_price": {
                "price": best_option["price"],
                "success_probability": best_option["success_probability"],
                "expected_value": best_option["expected_value"],
            },
        }

        return recommendation

    def print_recommendation(self, recommendation: Dict):
        """
        Красивый вывод рекомендации
        """
        prop = recommendation["proposed_price"]
        rec = recommendation["recommended_price"]

        print("\n" + "=" * 60)
        print("💰 РЕКОМЕНДАЦИЯ ПО ЦЕНООБРАЗОВАНИЮ")
        print("=" * 60)

        print(f"📊 Предложенная цена: {prop['price']:.0f} руб.")
        print(f"   Вероятность успеха: {prop['success_probability']:.1%}")
        print(f"   Ожидаемый доход: {prop['expected_value']:.0f} руб.")

        print(f"🎯 Рекомендуемая цена: {rec['price']:.0f} руб.")
        print(f"   Вероятность успеха: {rec['success_probability']:.1%}")
        print(f"   Ожидаемый доход: {rec['expected_value']:.0f} руб.")

        improvement = rec["expected_value"] - prop["expected_value"]
        price_difference = rec["price"] - prop["price"]

        if improvement > 0:
            print(
                f"📈 Увеличение дохода: +{improvement:.0f} руб. ({improvement / prop['expected_value']:+.1%})"
            )
            print(f"💸 Наценка: +{price_difference:.0f} руб.")
        else:
            print(f"📉 Ухудшение дохода: {improvement:.0f} руб.")


def main():
    """
    Пример использования оптимизатора цен
    """
    try:
        optimizer = PriceOptimizer()

        # Пример данных заказа
        test_orders = [
            {
                "carmodel": "Toyota Camry",
                "carname": "Camry 2.5",
                "platform": "ios",  # Только ios или android
                "distance": 15.5,
                "proposed_price": 450,
            },
            {
                "carmodel": "Kia Rio",
                "carname": "Rio Classic",
                "platform": "android",  # Только ios или android
                "distance": 8.2,
                "proposed_price": 300,
            },
        ]

        for order in test_orders:
            recommendation = optimizer.find_optimal_price(**order)
            optimizer.print_recommendation(recommendation)
            print("\n")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
