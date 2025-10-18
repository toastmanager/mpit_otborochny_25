import matplotlib.pyplot as plt
import pandas as pd


def investigate_artifacts(df, save_path="visualizations/"):
    """
    Глубокий анализ артефактов данных: отрицательный стаж и отмены с низким бидом
    """
    import os

    os.makedirs(save_path, exist_ok=True)

    # Преобразуем даты
    df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])
    df["driver_reg_date"] = pd.to_datetime(df["driver_reg_date"])
    df["tender_timestamp"] = pd.to_datetime(df["tender_timestamp"])

    # Вычисляем стаж
    df["driver_experience_days"] = (
        df["order_timestamp"] - df["driver_reg_date"]
    ).dt.days

    # Находим артефакты
    negative_experience = df[df["driver_experience_days"] < 0]
    low_bid_canceled = df[
        (df["is_done"] == "cancel") & (df["price_bid_local"] <= df["price_start_local"])
    ]

    print("=" * 80)
    print("РАССЛЕДОВАНИЕ АРТЕФАКТОВ ДАННЫХ")
    print("=" * 80)

    # 1. Анализ временных паттернов
    print("\n1. ВРЕМЕННОЙ АНАЛИЗ:")
    print("-" * 40)

    # Для отрицательного стажа
    if not negative_experience.empty:
        negative_experience["order_hour"] = negative_experience[
            "order_timestamp"
        ].dt.hour
        negative_experience["order_date"] = negative_experience[
            "order_timestamp"
        ].dt.date
        negative_experience["order_month"] = negative_experience[
            "order_timestamp"
        ].dt.to_period("M")

        print("Отрицательный стаж:")
        print(
            f"• Период заказов: {negative_experience['order_timestamp'].min()} - {negative_experience['order_timestamp'].max()}"
        )
        print(
            f"• Период регистрации водителей: {negative_experience['driver_reg_date'].min()} - {negative_experience['driver_reg_date'].max()}"
        )
        print(
            f"• Дней с первой проблемы: {(df['order_timestamp'].max() - negative_experience['order_timestamp'].min()).days}"
        )

    # Для низких бидов
    if not low_bid_canceled.empty:
        low_bid_canceled["order_hour"] = low_bid_canceled["order_timestamp"].dt.hour
        low_bid_canceled["order_date"] = low_bid_canceled["order_timestamp"].dt.date

        print("\nНизкие биды:")
        print(
            f"• Период заказов: {low_bid_canceled['order_timestamp'].min()} - {low_bid_canceled['order_timestamp'].max()}"
        )

    # 2. Анализ конкретных водителей с отрицательным стажем
    print("\n2. АНАЛИЗ ПРОБЛЕМНЫХ ВОДИТЕЛЕЙ:")
    print("-" * 40)

    if not negative_experience.empty:
        problem_drivers = (
            negative_experience.groupby("driver_id")
            .agg(
                {
                    "order_id": "count",
                    "driver_experience_days": ["min", "max", "mean"],
                    "driver_reg_date": "first",
                    "order_timestamp": ["min", "max"],
                    "is_done": lambda x: (x == "cancel").sum(),
                }
            )
            .round(2)
        )

        problem_drivers.columns = [
            "total_orders",
            "min_exp",
            "max_exp",
            "mean_exp",
            "reg_date",
            "first_order",
            "last_order",
            "canceled_orders",
        ]
        problem_drivers = problem_drivers.sort_values("total_orders", ascending=False)

        print(f"Всего проблемных водителей: {len(problem_drivers)}")
        print("\nТоп-5 самых активных проблемных водителей:")
        for i, (driver_id, row) in enumerate(problem_drivers.head().iterrows()):
            print(
                f"{i + 1}. Водитель {driver_id}: {row['total_orders']} заказов, "
                f"стаж от {row['min_exp']} до {row['max_exp']} дн., "
                f"отмен: {row['canceled_orders']}"
            )

    # 3. Анализ пересечения артефактов
    print("\n3. ПЕРЕСЕЧЕНИЕ АРТЕФАКТОВ:")
    print("-" * 40)

    # Заказы, которые имеют оба артефакта
    double_artifact = negative_experience[
        negative_experience["order_id"].isin(low_bid_canceled["order_id"])
    ]
    print(f"Заказы с ОБОИМИ артефактами: {len(double_artifact)}")

    if not double_artifact.empty:
        print("Примеры заказов с двумя артефактами:")
        for _, row in double_artifact.head(3).iterrows():
            print(
                f"• Заказ {row['order_id']}: водитель {row['driver_id']}, "
                f"стаж {row['driver_experience_days']} дн., "
                f"бид {row['price_bid_local']} ≤ старт {row['price_start_local']}"
            )

    # 4. Анализ технических причин
    print("\n4. ТЕХНИЧЕСКИЕ ПРИЧИНЫ:")
    print("-" * 40)

    # Проверяем согласованность временных меток
    df["tender_delay_seconds"] = (
        df["tender_timestamp"] - df["order_timestamp"]
    ).dt.total_seconds()

    suspicious_times = df[df["tender_delay_seconds"] < 0]
    print(f"Заказы с tender_timestamp < order_timestamp: {len(suspicious_times)}")

    # Анализ рейтингов проблемных водителей
    if not negative_experience.empty:
        print("\nРейтинги водителей с отрицательным стажем:")
        print(f"• Минимальный рейтинг: {negative_experience['driver_rating'].min()}")
        print(f"• Максимальный рейтинг: {negative_experience['driver_rating'].max()}")
        print(f"• Средний рейтинг: {negative_experience['driver_rating'].mean():.2f}")
        print(
            f"• Медианный рейтинг: {negative_experience['driver_rating'].median():.2f}"
        )

    # 5. Визуализация временных паттернов
    if not negative_experience.empty:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Распределение по часам
        hourly_dist = negative_experience["order_hour"].value_counts().sort_index()
        ax1.bar(hourly_dist.index, hourly_dist.values, color="#e74c3c", alpha=0.7)
        ax1.set_xlabel("Час дня")
        ax1.set_ylabel("Количество заказов")
        ax1.set_title("Распределение заказов с отриц. стажем по часам")
        ax1.grid(True, alpha=0.3)

        # Распределение по датам
        daily_dist = negative_experience["order_date"].value_counts().sort_index()
        ax2.plot(
            daily_dist.index,
            daily_dist.values,
            marker="o",
            color="#e74c3c",
            linewidth=2,
        )
        ax2.set_xlabel("Дата")
        ax2.set_ylabel("Количество заказов")
        ax2.set_title("Динамика заказов с отриц. стажем по дням")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

        # Сравнение рейтингов
        ax3.hist(
            negative_experience["driver_rating"],
            bins=20,
            alpha=0.7,
            color="#e74c3c",
            label="Отриц. стаж",
        )
        ax3.hist(
            df[df["driver_experience_days"] >= 0]["driver_rating"],
            bins=20,
            alpha=0.5,
            color="#3498db",
            label="Нормальный стаж",
        )
        ax3.set_xlabel("Рейтинг водителя")
        ax3.set_ylabel("Количество")
        ax3.set_title("Сравнение распределения рейтингов")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Задержка тендера
        ax4.hist(df["tender_delay_seconds"] / 60, bins=50, alpha=0.7, color="#9b59b6")
        ax4.set_xlabel("Задержка тендера (минуты)")
        ax4.set_ylabel("Количество заказов")
        ax4.set_title("Распределение задержек тендера")
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(-10, 60)  # Ограничиваем для наглядности

        plt.tight_layout()
        plt.savefig(
            f"{save_path}11_artifact_investigation.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    # 6. Глубокий анализ данных о регистрации
    print("\n5. АНАЛИЗ ДАТ РЕГИСТРАЦИИ:")
    print("-" * 40)

    if not negative_experience.empty:
        # Проверяем, есть ли водители с будущей датой регистрации
        future_reg_drivers = negative_experience[
            negative_experience["driver_reg_date"]
            > negative_experience["order_timestamp"]
        ]
        print(
            f"Водители с датой регистрации в будущем: {future_reg_drivers['driver_id'].nunique()}"
        )

        # Анализ самых больших расхождений
        largest_discrepancies = negative_experience.nlargest(
            5, "driver_experience_days"
        )
        print("\nСамые большие расхождения (наиболее 'отрицательный' стаж):")
        for _, row in largest_discrepancies.iterrows():
            print(
                f"• Водитель {row['driver_id']}: заказ {row['order_timestamp']}, "
                f"регистрация {row['driver_reg_date']}, стаж {row['driver_experience_days']} дн."
            )

    # 7. Анализ системных паттернов
    print("\n6. СИСТЕМНЫЕ ПАТТЕРНЫ:")
    print("-" * 40)

    # Проверяем, связаны ли артефакты с определенными платформами
    if "platform" in df.columns:
        platform_analysis = (
            df.groupby("platform")
            .agg({"driver_id": "nunique", "order_id": "count"})
            .reset_index()
        )

        platform_analysis["negative_exp_ratio"] = platform_analysis["platform"].apply(
            lambda x: len(negative_experience[negative_experience["platform"] == x])
            / len(df[df["platform"] == x])
            if len(df[df["platform"] == x]) > 0
            else 0
        )

        platform_analysis["low_bid_ratio"] = platform_analysis["platform"].apply(
            lambda x: len(low_bid_canceled[low_bid_canceled["platform"] == x])
            / len(df[df["platform"] == x])
            if len(df[df["platform"] == x]) > 0
            else 0
        )

        print("Распределение артефактов по платформам:")
        for _, row in platform_analysis.iterrows():
            print(
                f"• {row['platform']}: отриц. стаж {row['negative_exp_ratio'] * 100:.2f}%, "
                f"низкие биды {row['low_bid_ratio'] * 100:.2f}%"
            )

    # 8. Выводы и рекомендации
    print("\n" + "=" * 80)
    print("ВЫВОДЫ И РЕКОМЕНДАЦИИ")
    print("=" * 80)

    conclusions = []

    if len(negative_experience) > 0:
        conclusions.append(
            "❶ ОТРИЦАТЕЛЬНЫЙ СТАЖ: Проблема с датами регистрации водителей"
        )
        conclusions.append(
            "  ▸ Возможные причины: некорректная синхронизация времени между системами"
        )
        conclusions.append(
            "  ▸ Рекомендация: аудит процесса регистрации водителей и синхронизации времени"
        )

    if len(low_bid_canceled) > 0:
        conclusions.append("❷ НИЗКИЕ БИДЫ: Отмены при биде ≤ стартовой цене")
        conclusions.append(
            "  ▸ Возможные причины: технические ошибки, манипуляции, баги в алгоритме ценообразования"
        )
        conclusions.append(
            "  ▸ Рекомендация: анализ логики расчета цен и условий отмен"
        )

    if len(double_artifact) > 0:
        conclusions.append(
            "❸ ПЕРЕСЕЧЕНИЕ АРТЕФАКТОВ: Найдены заказы с обеими проблемами"
        )
        conclusions.append(
            "  ▸ Возможные причины: системные сбои в определенных сценариях"
        )
        conclusions.append("  ▸ Рекомендация: детальный разбор конкретных кейсов")

    if len(suspicious_times) > 0:
        conclusions.append(
            "❹ ВРЕМЕННЫЕ АНОМАЛИИ: Нарушена последовательность временных меток"
        )
        conclusions.append(
            "  ▸ Возможные причины: проблемы с часовыми поясами, баги в логике"
        )
        conclusions.append(
            "  ▸ Рекомендация: проверка корректности временных меток в системе"
        )

    for conclusion in conclusions:
        print(conclusion)

    return {
        "negative_experience": negative_experience,
        "low_bid_canceled": low_bid_canceled,
        "double_artifact": double_artifact,
        "suspicious_times": suspicious_times,
        "problem_drivers": problem_drivers if not negative_experience.empty else None,
    }


df = pd.read_csv("./train.csv")
# Запуск расследования
artifacts_analysis = investigate_artifacts(df)
