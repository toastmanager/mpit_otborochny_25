import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("./train.csv")


def create_negative_experience_visualizations(df, save_path="visualizations/"):
    """
    Создает визуализации для анализа водителей с отрицательным стажем
    """
    import os

    os.makedirs(save_path, exist_ok=True)

    # Преобразуем даты в правильный формат
    df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])
    df["driver_reg_date"] = pd.to_datetime(df["driver_reg_date"])
    df["tender_timestamp"] = pd.to_datetime(df["tender_timestamp"])

    # Вычисляем стаж водителя на момент заказа (в днях)
    df["driver_experience_days"] = (
        df["order_timestamp"] - df["driver_reg_date"]
    ).dt.days

    # Находим водителей с отрицательным стажем
    negative_experience = df[df["driver_experience_days"] < 0]
    negative_exp_canceled = negative_experience[
        negative_experience["is_done"] == "cancel"
    ]

    # Подготовка данных для визуализаций
    total_drivers = df["driver_id"].nunique()
    negative_exp_drivers = negative_experience["driver_id"].nunique()
    total_negative_orders = len(negative_experience)
    negative_canceled_orders = len(negative_exp_canceled)

    # 1. Круговая диаграмма - распределение водителей по стажу
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Распределение водителей
    experience_status = ["Нормальный стаж", "Отрицательный стаж"]
    experience_counts = [total_drivers - negative_exp_drivers, negative_exp_drivers]
    colors_exp = ["#66b3ff", "#ff9999"]

    ax1.pie(
        experience_counts,
        labels=experience_status,
        autopct="%1.1f%%",
        colors=colors_exp,
        startangle=90,
    )
    ax1.set_title("Распределение водителей по стажу", fontsize=14, fontweight="bold")

    # Распределение заказов
    order_status = ["Обычные заказы", "Заказы с отриц. стажем"]
    order_counts = [len(df) - total_negative_orders, total_negative_orders]
    colors_orders = ["#4ecdc4", "#ff6b6b"]

    ax2.pie(
        order_counts,
        labels=order_status,
        autopct="%1.1f%%",
        colors=colors_orders,
        startangle=90,
    )
    ax2.set_title(
        "Распределение заказов по стажу водителей", fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(
        f"{save_path}5_driver_experience_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    # 2. Столбчатая диаграмма - ключевые метрики
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics = [
        "Всего водителей",
        "Водители с отриц. стажем",
        "Всего заказов",
        "Заказы с отриц. стажем",
        "Отмены с отриц. стажем",
    ]
    values = [
        total_drivers,
        negative_exp_drivers,
        len(df),
        total_negative_orders,
        negative_canceled_orders,
    ]

    colors_metrics = ["#3498db", "#e74c3c", "#3498db", "#e74c3c", "#f39c12"]

    bars = ax.bar(metrics, values, color=colors_metrics, alpha=0.8)
    ax.set_title(
        "Ключевые метрики по водителям с отрицательным стажем",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_ylabel("Количество", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    # Добавляем значения на столбцы
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(values) * 0.01,
            f"{value}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        f"{save_path}6_negative_experience_metrics.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    # 3. Распределение отрицательного стажа
    if not negative_experience.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Гистограмма стажа
        ax1.hist(
            negative_experience["driver_experience_days"],
            bins=20,
            color="#e74c3c",
            alpha=0.7,
            edgecolor="black",
        )
        ax1.axvline(
            x=negative_experience["driver_experience_days"].mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Среднее: {negative_experience['driver_experience_days'].mean():.1f} дн.",
        )
        ax1.set_xlabel("Отрицательный стаж (дни)", fontsize=12)
        ax1.set_ylabel("Количество заказов", fontsize=12)
        ax1.set_title(
            "Распределение отрицательного стажа водителей",
            fontsize=14,
            fontweight="bold",
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot стажа
        ax2.boxplot(negative_experience["driver_experience_days"], vert=False)
        ax2.set_xlabel("Отрицательный стаж (дни)", fontsize=12)
        ax2.set_title("Box plot отрицательного стажа", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{save_path}7_experience_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    # 4. Статусы заказов у водителей с отрицательным стажем
    if not negative_experience.empty:
        fig, ax = plt.subplots(figsize=(10, 6))

        status_counts = negative_experience["is_done"].value_counts()
        colors_status = ["#e74c3c", "#2ecc71"]  # cancel, done

        bars = ax.bar(
            status_counts.index, status_counts.values, color=colors_status, alpha=0.8
        )
        ax.set_title(
            "Статусы заказов у водителей с отрицательным стажем",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_ylabel("Количество заказов", fontsize=12)
        ax.set_xlabel("Статус заказа", fontsize=12)

        # Добавляем значения на столбцы
        for bar, value in zip(bars, status_counts.values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{value}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            f"{save_path}8_negative_exp_order_status.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    # 5. Сравнение показателей отмен
    if not negative_experience.empty:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Считаем проценты отмен
        overall_cancel_rate = len(df[df["is_done"] == "cancel"]) / len(df) * 100
        negative_exp_cancel_rate = (
            negative_canceled_orders / total_negative_orders * 100
            if total_negative_orders > 0
            else 0
        )

        categories = ["Общий показатель", "Водители с отриц. стажем"]
        cancel_rates = [overall_cancel_rate, negative_exp_cancel_rate]

        bars = ax.bar(categories, cancel_rates, color=["#3498db", "#e74c3c"], alpha=0.8)
        ax.set_title("Сравнение процента отмен заказов", fontsize=16, fontweight="bold")
        ax.set_ylabel("Процент отмен (%)", fontsize=12)

        # Добавляем значения на столбцы
        for bar, rate in zip(bars, cancel_rates):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            f"{save_path}9_cancel_rate_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    # 6. Анализ по платформам для водителей с отрицательным стажем
    if "platform" in df.columns and not negative_experience.empty:
        fig, ax = plt.subplots(figsize=(12, 6))

        platform_stats = (
            negative_experience.groupby("platform")
            .agg(
                {
                    "driver_id": "nunique",
                    "order_id": "count",
                    "is_done": lambda x: (x == "cancel").sum(),
                }
            )
            .reset_index()
        )

        platform_stats.columns = [
            "platform",
            "unique_drivers",
            "total_orders",
            "canceled_orders",
        ]

        x = np.arange(len(platform_stats))
        width = 0.25

        ax.bar(
            x - width,
            platform_stats["unique_drivers"],
            width,
            label="Уникальные водители",
            alpha=0.8,
        )
        ax.bar(
            x, platform_stats["total_orders"], width, label="Всего заказов", alpha=0.8
        )
        ax.bar(
            x + width,
            platform_stats["canceled_orders"],
            width,
            label="Отмененные заказы",
            alpha=0.8,
        )

        ax.set_xlabel("Платформа", fontsize=12)
        ax.set_ylabel("Количество", fontsize=12)
        ax.set_title(
            "Статистика по платформам для водителей с отрицательным стажем",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(platform_stats["platform"])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{save_path}10_platform_stats_negative_exp.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    # Сводная таблица с основными цифрами
    print("=" * 70)
    print("СВОДКА ПО ВОДИТЕЛЯМ С ОТРИЦАТЕЛЬНЫМ СТАЖЕМ")
    print("=" * 70)
    print(f"Всего уникальных водителей: {total_drivers:,}")
    print(
        f"Водителей с отрицательным стажем: {negative_exp_drivers:,} ({negative_exp_drivers / total_drivers * 100:.1f}%)"
    )
    print(f"Всего заказов: {len(df):,}")
    print(
        f"Заказов с отрицательным стажем: {total_negative_orders:,} ({total_negative_orders / len(df) * 100:.1f}%)"
    )
    print(f"Отмененных заказов с отриц. стажем: {negative_canceled_orders:,}")

    if total_negative_orders > 0:
        print(
            f"Процент отмен у водителей с отриц. стажем: {negative_canceled_orders / total_negative_orders * 100:.1f}%"
        )
        print(f"Общий процент отмен по всем заказам: {overall_cancel_rate:.1f}%")

        if not negative_experience.empty:
            print("\nСтатистика по отрицательному стажу:")
            print(
                f"• Минимальный стаж: {negative_experience['driver_experience_days'].min():.0f} дн."
            )
            print(
                f"• Максимальный стаж: {negative_experience['driver_experience_days'].max():.0f} дн."
            )
            print(
                f"• Средний стаж: {negative_experience['driver_experience_days'].mean():.1f} дн."
            )
            print(
                f"• Медианный стаж: {negative_experience['driver_experience_days'].median():.1f} дн."
            )

    # Примеры проблемных записей
    if not negative_experience.empty:
        print("\nПримеры заказов с отрицательным стажем:")
        sample_records = negative_experience[
            [
                "order_id",
                "driver_id",
                "driver_reg_date",
                "order_timestamp",
                "driver_experience_days",
                "is_done",
            ]
        ].head(3)
        for _, row in sample_records.iterrows():
            print(
                f"• Заказ {row['order_id']}: водитель {row['driver_id']}, "
                f"стаж {row['driver_experience_days']:.0f} дн., статус: {row['is_done']}"
            )


# Использование функции
create_negative_experience_visualizations(df)
