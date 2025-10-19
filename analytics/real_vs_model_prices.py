import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Настройка стиля для бизнес-презентаций
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

df = pd.read_csv("./train.csv")


def create_cancelation_visualizations(df, save_path="visualizations/"):
    """
    Создает визуализации для анализа отмененных заказов
    """
    import os

    os.makedirs(save_path, exist_ok=True)

    # Подготовка данных
    canceled_orders = df[df["is_done"] == "cancel"]
    low_bid_canceled = canceled_orders[
        canceled_orders["price_bid_local"] <= canceled_orders["price_start_local"]
    ]

    total_orders = len(df)
    total_canceled = len(canceled_orders)
    low_bid_canceled_count = len(low_bid_canceled)

    # 1. Круговая диаграмма - распределение статусов заказов
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    status_counts = df["is_done"].value_counts()
    colors = ["#ff9999", "#66b3ff"]
    ax1.pie(
        status_counts.values,
        labels=status_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    ax1.set_title("Распределение статусов заказов", fontsize=14, fontweight="bold")

    # 2. Круговая диаграмма - отмененные заказы по условию цены
    if total_canceled > 0:
        canceled_labels = ["Бид ≤ Старт\nцена", "Бид > Старт\nцена"]
        canceled_sizes = [
            low_bid_canceled_count,
            total_canceled - low_bid_canceled_count,
        ]
        colors_canceled = ["#ff6b6b", "#4ecdc4"]
        ax2.pie(
            canceled_sizes,
            labels=canceled_labels,
            autopct="%1.1f%%",
            colors=colors_canceled,
            startangle=90,
        )
        ax2.set_title(
            "Отмененные заказы:\nСравнение цены бида и стартовой цены",
            fontsize=14,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(f"{save_path}1_status_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 3. Столбчатая диаграмма - ключевые метрики
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ["Всего заказов", "Отменено", "Отменено (бид ≤ старт)"]
    values = [total_orders, total_canceled, low_bid_canceled_count]
    colors_metrics = ["#3498db", "#e74c3c", "#f39c12"]

    bars = ax.bar(metrics, values, color=colors_metrics, alpha=0.8)
    ax.set_title(
        "Ключевые метрики по отмененным заказам", fontsize=16, fontweight="bold"
    )
    ax.set_ylabel("Количество заказов", fontsize=12)

    # Добавляем значения на столбцы
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{value}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{save_path}2_key_metrics.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 4. Распределение разницы цен для отмененных заказов
    if not low_bid_canceled.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        low_bid_canceled["price_diff"] = (
            low_bid_canceled["price_bid_local"] - low_bid_canceled["price_start_local"]
        )

        # Гистограмма разницы цен
        ax1.hist(
            low_bid_canceled["price_diff"],
            bins=15,
            color="#e74c3c",
            alpha=0.7,
            edgecolor="black",
        )
        ax1.axvline(
            x=0, color="red", linestyle="--", linewidth=2, label="Нулевая разница"
        )
        ax1.set_xlabel("Разница цены (бид - старт)", fontsize=12)
        ax1.set_ylabel("Количество заказов", fontsize=12)
        ax1.set_title(
            "Распределение разницы цен\nдля отмененных заказов с бидом ≤ старту",
            fontsize=14,
            fontweight="bold",
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot разницы цен
        ax2.boxplot(low_bid_canceled["price_diff"], vert=False)
        ax2.set_xlabel("Разница цены (бид - старт)", fontsize=12)
        ax2.set_title("Box plot разницы цен", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_path}3_price_difference.png", dpi=300, bbox_inches="tight")
        plt.show()

    # 5. Сравнительная диаграмма по платформам
    if "platform" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))

        platform_cancel_stats = (
            df.groupby("platform")
            .apply(
                lambda x: pd.Series(
                    {
                        "total_orders": len(x),
                        "canceled_orders": len(x[x["is_done"] == "cancel"]),
                        "low_bid_canceled": len(
                            x[
                                (x["is_done"] == "cancel")
                                & (x["price_bid_local"] <= x["price_start_local"])
                            ]
                        ),
                    }
                )
            )
            .reset_index()
        )

        if not platform_cancel_stats.empty:
            x = np.arange(len(platform_cancel_stats))
            width = 0.25

            ax.bar(
                x - width,
                platform_cancel_stats["total_orders"],
                width,
                label="Всего заказов",
                alpha=0.8,
            )
            ax.bar(
                x,
                platform_cancel_stats["canceled_orders"],
                width,
                label="Отменено",
                alpha=0.8,
            )
            ax.bar(
                x + width,
                platform_cancel_stats["low_bid_canceled"],
                width,
                label="Отменено (бид ≤ старт)",
                alpha=0.8,
            )

            ax.set_xlabel("Платформа", fontsize=12)
            ax.set_ylabel("Количество заказов", fontsize=12)
            ax.set_title(
                "Статистика отмен по платформам", fontsize=16, fontweight="bold"
            )
            ax.set_xticks(x)
            ax.set_xticklabels(platform_cancel_stats["platform"])
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                f"{save_path}4_platform_stats.png", dpi=300, bbox_inches="tight"
            )
            plt.show()

    # 6. Сводная таблица с основными цифрами
    print("=" * 60)
    print("СВОДКА ПО ОТМЕНЕННЫМ ЗАКАЗАМ")
    print("=" * 60)
    print(f"Всего заказов в данных: {total_orders:,}")
    print(
        f"Отмененных заказов: {total_canceled:,} ({total_canceled / total_orders * 100:.1f}%)"
    )
    print(f"Отменено с бидом ≤ стартовой цене: {low_bid_canceled_count:,}")
    if total_canceled > 0:
        print(
            f"Доля проблемных отмен: {low_bid_canceled_count / total_canceled * 100:.1f}%"
        )

    if not low_bid_canceled.empty:
        print("\nСтатистика по разнице цен (бид - старт):")
        print(f"• Минимальная разница: {low_bid_canceled['price_diff'].min():.0f}")
        print(f"• Максимальная разница: {low_bid_canceled['price_diff'].max():.0f}")
        print(f"• Средняя разница: {low_bid_canceled['price_diff'].mean():.2f}")
        print(f"• Медианная разница: {low_bid_canceled['price_diff'].median():.2f}")


# Использование функции
create_cancelation_visualizations(df)
