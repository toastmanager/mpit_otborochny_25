import pandas as pd
from tqdm import tqdm
from bid_optimizer import BidOptimizer

# --- НАСТРОЙКИ ---
# Пути к файлам. Убедитесь, что они верные.
MODEL_FILE_PATH = "models/final_model.joblib"
FEATURES_FILE_PATH = "models/feature_names.joblib"
INPUT_TEST_FILE = "test.csv"
OUTPUT_PREDICTIONS_FILE = "predictions.csv"


def run_predictions(test_df: pd.DataFrame, optimizer: BidOptimizer) -> pd.DataFrame:
    """
    Итерируется по тестовому DataFrame, для каждой строки находит оптимальный бид
    и ОБНОВЛЯЕТ колонку 'price_bid_local' в исходном DataFrame.
    """
    # Создаем копию, чтобы не изменять оригинальный DataFrame, переданный в функцию
    result_df = test_df.copy()

    print("Начинаем расчет и обновление цен для каждой поездки...")
    for index, row in tqdm(result_df.iterrows(), total=result_df.shape[0]):
        # 1. Преобразуем строку в словарь для передачи в оптимизатор
        features_dict = row.to_dict()

        # 2. Добавляем фиктивную колонку 'is_done' для корректной работы препроцессора
        features_dict["is_done"] = 0

        # 3. Получаем начальную цену из данных
        initial_bid = features_dict["price_start_local"]

        # 4. Вызываем оптимизатор для поиска лучшей цены
        try:
            optimal_bid_info = optimizer.calculate_optimal_bid(
                initial_features=features_dict,
                initial_bid=initial_bid,
                bid_range=150,
                bid_step=10,
            )
        except Exception as e:
            print(f"Ошибка при обработке строки {index}: {e}")
            optimal_bid_info = {}

        # <<< ГЛАВНОЕ ИЗМЕНЕНИЕ ЗДЕСЬ >>>
        # 5. Если оптимальный бид найден, обновляем колонку 'price_bid_local'
        if optimal_bid_info and "bid" in optimal_bid_info:
            # Используем .loc[index, column_name] для надежного присвоения значения
            result_df.loc[index, "price_bid_local"] = optimal_bid_info["bid"]  # type: ignore
        # Если оптимальный бид не найден, мы ничего не делаем,
        # оставляя исходное значение в 'price_bid_local'.
        # <<< КОНЕЦ ИЗМЕНЕНИЯ >>>

    return result_df


if __name__ == "__main__":
    try:
        # Загружаем тестовые данные
        print(f"Загрузка данных из {INPUT_TEST_FILE}...")
        test_data = pd.read_csv(INPUT_TEST_FILE)

        # Инициализируем наш оптимизатор
        print("Инициализация BidOptimizer...")
        bid_optimizer = BidOptimizer(
            model_path=MODEL_FILE_PATH, features_path=FEATURES_FILE_PATH
        )

        # Запускаем процесс предсказания и обновления
        predictions_df = run_predictions(test_data, bid_optimizer)

        # Сохраняем результат
        predictions_df.to_csv(OUTPUT_PREDICTIONS_FILE, index=False, encoding="utf-8")

        print("\n" + "=" * 50)
        print(f"Готово! Результаты сохранены в файл: {OUTPUT_PREDICTIONS_FILE}")
        print("Пример результата (колонка 'price_bid_local' обновлена):")
        # Сравниваем исходную и новую цену
        print(
            pd.concat(
                [
                    test_data[["order_id", "price_bid_local"]]
                    .head()
                    .rename(columns={"price_bid_local": "original_bid"}),
                    predictions_df[["price_bid_local"]]
                    .head()
                    .rename(columns={"price_bid_local": "optimal_bid"}),
                ],
                axis=1,
            )
        )
        print("=" * 50)

    except FileNotFoundError as e:
        print(f"\n[ОШИБКА] Необходимый файл не найден: {e}")
    except Exception as e:
        print(f"\n[КРИТИЧЕСКАЯ ОШИБКА] Произошла непредвиденная ошибка: {e}")
