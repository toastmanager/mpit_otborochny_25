import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def analyze_driver_statistics(df, save_path="visualizations/"):
    """
    Комплексный анализ статистики по водителям
    """
    import os

    os.makedirs(save_path, exist_ok=True)

    # Преобразуем даты
    df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])
    df["driver_reg_date"] = pd.to_datetime(df["driver_reg_date"])

    print("=" * 80)
    print("КОМПЛЕКСНАЯ СТАТИСТИКА ПО ВОДИТЕЛЯМ")
    print("=" * 80)

    # БАЗОВАЯ СТАТИСТИКА
    total_drivers = df["driver_id"].nunique()
    total_orders = len(df)

    print(f"\n📊 БАЗОВЫЕ ПОКАЗАТЕЛИ:")
    print(f"• Уникальных водителей: {total_drivers:,}")
    print(f"• Всего заказов: {total_orders:,}")
    print(f"• Среднее заказов на водителя: {total_orders / total_drivers:.1f}")

    # СТАТИСТИКА ПО АКТИВНОСТИ ВОДИТЕЛЕЙ
    driver_activity = (
        df.groupby("driver_id")
        .agg(
            {
                "order_id": "count",
                "is_done": lambda x: (x == "done").sum(),
                "driver_rating": "mean",
                "driver_reg_date": "first",
                "order_timestamp": ["min", "max"],
                "platform": lambda x: x.mode()[0] if len(x) > 0 else "unknown",
            }
        )
        .reset_index()
    )

    driver_activity.columns = [
        "driver_id",
        "total_orders",
        "completed_orders",
        "avg_rating",
        "reg_date",
        "first_order",
        "last_order",
        "main_platform",
    ]

    driver_activity["completion_rate"] = (
        driver_activity["completed_orders"] / driver_activity["total_orders"] * 100
    ).round(2)
    driver_activity["canceled_orders"] = (
        driver_activity["total_orders"] - driver_activity["completed_orders"]
    )
    driver_activity["cancel_rate"] = (
        driver_activity["canceled_orders"] / driver_activity["total_orders"] * 100
    ).round(2)

    # Анализ длительности работы
    driver_activity["work_duration_days"] = (
        driver_activity["last_order"] - driver_activity["first_order"]
    ).dt.days
    driver_activity["orders_per_day"] = (
        driver_activity["total_orders"] / driver_activity["work_duration_days"]
    ).round(2)

    print(f"\n🎯 СТАТИСТИКА АКТИВНОСТИ:")
    print(
        f"• Самый активный водитель: {driver_activity['total_orders'].max():,} заказов"
    )
    print(
        f"• Среднее заказов на водителя: {driver_activity['total_orders'].mean():.1f}"
    )
    print(f"• Медиана заказов: {driver_activity['total_orders'].median():.1f}")

    # СЕГМЕНТАЦИЯ ВОДИТЕЛЕЙ ПО АКТИВНОСТИ
    print(f"\n📈 СЕГМЕНТАЦИЯ ВОДИТЕЛЕЙ ПО АКТИВНОСТИ:")

    activity_segments = {
        "Высокая (>50 заказов)": len(
            driver_activity[driver_activity["total_orders"] > 50]
        ),
        "Средняя (11-50 заказов)": len(
            driver_activity[
                (driver_activity["total_orders"] >= 11)
                & (driver_activity["total_orders"] <= 50)
            ]
        ),
        "Низкая (2-10 заказов)": len(
            driver_activity[
                (driver_activity["total_orders"] >= 2)
                & (driver_activity["total_orders"] <= 10)
            ]
        ),
        "Единичная (1 заказ)": len(
            driver_activity[driver_activity["total_orders"] == 1]
        ),
    }

    for segment, count in activity_segments.items():
        percentage = (count / total_drivers) * 100
        print(f"• {segment}: {count:,} водителей ({percentage:.1f}%)")

    # СТАТИСТИКА ПО РЕЙТИНГАМ
    print(f"\n⭐ СТАТИСТИКА ПО РЕЙТИНГАМ:")
    print(f"• Средний рейтинг: {driver_activity['avg_rating'].mean():.3f}")
    print(f"• Медианный рейтинг: {driver_activity['avg_rating'].median():.3f}")
    print(f"• Минимальный рейтинг: {driver_activity['avg_rating'].min():.3f}")
    print(f"• Максимальный рейтинг: {driver_activity['avg_rating'].max():.3f}")

    # Водители с идеальным рейтингом
    perfect_rating_drivers = len(driver_activity[driver_activity["avg_rating"] == 5.0])
    print(
        f"• Водителей с рейтингом 5.0: {perfect_rating_drivers:,} ({perfect_rating_drivers / total_drivers * 100:.1f}%)"
    )

    # СТАТИСТИКА ПО ПЛАТФОРМАМ
    if "platform" in df.columns:
        print(f"\n📱 РАСПРЕДЕЛЕНИЕ ПО ПЛАТФОРМАМ:")
        platform_stats = (
            df.groupby("platform")
            .agg({"driver_id": "nunique", "order_id": "count"})
            .reset_index()
        )

        platform_stats.columns = ["platform", "unique_drivers", "total_orders"]
        platform_stats["drivers_percentage"] = (
            platform_stats["unique_drivers"] / total_drivers * 100
        ).round(1)
        platform_stats["orders_per_driver"] = (
            platform_stats["total_orders"] / platform_stats["unique_drivers"]
        ).round(1)

        for _, row in platform_stats.iterrows():
            print(
                f"• {row['platform']}: {row['unique_drivers']:,} водителей ({row['drivers_percentage']}%)"
            )

    # АНАЛИЗ ЭФФЕКТИВНОСТИ
    print(f"\n🎯 ЭФФЕКТИВНОСТЬ ВОДИТЕЛЕЙ:")
    print(
        f"• Средний процент выполнения: {driver_activity['completion_rate'].mean():.1f}%"
    )
    print(f"• Средний процент отмен: {driver_activity['cancel_rate'].mean():.1f}%")

    # Топ водителей по эффективности
    top_efficient = driver_activity[driver_activity["total_orders"] >= 5].nlargest(
        5, "completion_rate"
    )
    print(f"\n🏆 ТОП-5 САМЫХ ЭФФЕКТИВНЫХ ВОДИТЕЛЕЙ (от 5+ заказов):")
    for i, (_, row) in enumerate(top_efficient.iterrows(), 1):
        print(
            f"{i}. ID {row['driver_id']}: {row['completion_rate']}% выполнено, "
            f"{row['total_orders']} заказов, рейтинг {row['avg_rating']:.2f}"
        )

    # АНАЛИЗ ПО ВРЕМЕНИ РАБОТЫ
    print(f"\n⏰ СТАТИСТИКА ПО ВРЕМЕНИ РАБОТЫ:")
    print(
        f"• Средняя длительность работы: {driver_activity['work_duration_days'].mean():.1f} дней"
    )
    print(
        f"• Медианная длительность: {driver_activity['work_duration_days'].median():.1f} дней"
    )
    print(f"• Среднее заказов в день: {driver_activity['orders_per_day'].mean():.2f}")

    # АНАЛИЗ ПРОБЛЕМНЫХ ВОДИТЕЛЕЙ
    problematic_drivers = driver_activity[
        (driver_activity["cancel_rate"] > 50) & (driver_activity["total_orders"] >= 5)
    ]

    print(f"\n⚠️  ПРОБЛЕМНЫЕ ВОДИТЕЛИ (отмена >50%, от 5+ заказов):")
    print(f"• Количество: {len(problematic_drivers):,}")
    if len(problematic_drivers) > 0:
        avg_problem_rating = problematic_drivers["avg_rating"].mean()
        print(f"• Средний рейтинг проблемных водителей: {avg_problem_rating:.2f}")

    # ДЕТАЛЬНЫЙ АНАЛИЗ НОВЫХ ВОДИТЕЛЕЙ
    current_date = df["order_timestamp"].max()
    driver_activity["days_since_registration"] = (
        current_date - driver_activity["reg_date"]
    ).dt.days

    new_drivers = driver_activity[driver_activity["days_since_registration"] <= 30]
    print(f"\n🆕 НОВЫЕ ВОДИТЕЛИ (зарегистрированы в последние 30 дней):")
    print(f"• Количество: {len(new_drivers):,}")
    if len(new_drivers) > 0:
        print(f"• Среднее заказов: {new_drivers['total_orders'].mean():.1f}")
        print(f"• Средний рейтинг: {new_drivers['avg_rating'].mean():.2f}")

    # ВИЗУАЛИЗАЦИИ
    create_driver_visualizations(driver_activity, df, save_path)

    return {
        "driver_activity": driver_activity,
        "total_drivers": total_drivers,
        "total_orders": total_orders,
        "activity_segments": activity_segments,
        "platform_stats": platform_stats if "platform" in df.columns else None,
        "problematic_drivers": problematic_drivers,
        "new_drivers": new_drivers,
    }


def create_driver_visualizations(driver_activity, df, save_path):
    """
    Создает визуализации для статистики водителей
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # График 1: Распределение количества заказов на водителя
    ax1.hist(
        driver_activity["total_orders"],
        bins=50,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    ax1.set_xlabel("Количество заказов на водителя")
    ax1.set_ylabel("Количество водителей")
    ax1.set_title(
        "Распределение водителей по количеству заказов", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")  # Логарифмическая шкала для лучшей визуализации

    # График 2: Распределение рейтингов
    ax2.hist(
        driver_activity["avg_rating"].dropna(),
        bins=30,
        alpha=0.7,
        color="lightgreen",
        edgecolor="darkgreen",
    )
    ax2.axvline(
        driver_activity["avg_rating"].mean(),
        color="red",
        linestyle="--",
        label=f"Среднее: {driver_activity['avg_rating'].mean():.2f}",
    )
    ax2.axvline(
        driver_activity["avg_rating"].median(),
        color="blue",
        linestyle="--",
        label=f"Медиана: {driver_activity['avg_rating'].median():.2f}",
    )
    ax2.set_xlabel("Средний рейтинг водителя")
    ax2.set_ylabel("Количество водителей")
    ax2.set_title("Распределение рейтингов водителей", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # График 3: Процент выполнения заказов
    ax3.hist(
        driver_activity["completion_rate"],
        bins=30,
        alpha=0.7,
        color="gold",
        edgecolor="darkorange",
    )
    ax3.axvline(
        driver_activity["completion_rate"].mean(),
        color="red",
        linestyle="--",
        label=f"Среднее: {driver_activity['completion_rate'].mean():.1f}%",
    )
    ax3.set_xlabel("Процент выполненных заказов (%)")
    ax3.set_ylabel("Количество водителей")
    ax3.set_title(
        "Распределение процента выполнения заказов", fontsize=14, fontweight="bold"
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # График 4: Связь рейтинга и эффективности
    sample_size = min(1000, len(driver_activity))
    sample_data = driver_activity.sample(sample_size, random_state=42)

    scatter = ax4.scatter(
        sample_data["completion_rate"],
        sample_data["avg_rating"],
        c=sample_data["total_orders"],
        cmap="viridis",
        alpha=0.6,
        s=50,
    )
    ax4.set_xlabel("Процент выполнения заказов (%)")
    ax4.set_ylabel("Средний рейтинг")
    ax4.set_title(
        "Связь рейтинга и эффективности водителей\n(цвет = количество заказов)",
        fontsize=14,
        fontweight="bold",
    )
    ax4.grid(True, alpha=0.3)

    # Цветовая шкала для количества заказов
    plt.colorbar(scatter, ax=ax4, label="Количество заказов")

    plt.tight_layout()
    plt.savefig(f"{save_path}18_driver_statistics.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ДОПОЛНИТЕЛЬНЫЕ ВИЗУАЛИЗАЦИИ
    # Круговая диаграмма сегментов активности
    fig, ax = plt.subplots(figsize=(10, 8))

    activity_segments = {
        "Высокая (>50)": len(driver_activity[driver_activity["total_orders"] > 50]),
        "Средняя (11-50)": len(
            driver_activity[
                (driver_activity["total_orders"] >= 11)
                & (driver_activity["total_orders"] <= 50)
            ]
        ),
        "Низкая (2-10)": len(
            driver_activity[
                (driver_activity["total_orders"] >= 2)
                & (driver_activity["total_orders"] <= 10)
            ]
        ),
        "Единичная (1)": len(driver_activity[driver_activity["total_orders"] == 1]),
    }

    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
    wedges, texts, autotexts = ax.pie(
        activity_segments.values(),
        labels=activity_segments.keys(),
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    ax.set_title("Сегментация водителей по активности", fontsize=16, fontweight="bold")
    plt.savefig(f"{save_path}19_driver_segments.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_driver_summary_report(analysis_results):
    """
    Печатает сводный отчет по водителям
    """
    print("\n" + "=" * 80)
    print("📋 СВОДНЫЙ ОТЧЕТ ПО ВОДИТЕЛЯМ")
    print("=" * 80)

    stats = analysis_results["driver_activity"].describe()

    print(f"\n🎯 КЛЮЧЕВЫЕ МЕТРИКИ:")
    print(f"• Всего водителей: {analysis_results['total_drivers']:,}")
    print(f"• Всего заказов: {analysis_results['total_orders']:,}")
    print(
        f"• Заказов на водителя: {analysis_results['total_orders'] / analysis_results['total_drivers']:.1f}"
    )
    print(
        f"• Средний рейтинг: {analysis_results['driver_activity']['avg_rating'].mean():.2f}"
    )
    print(
        f"• Средний % выполнения: {analysis_results['driver_activity']['completion_rate'].mean():.1f}%"
    )

    print(f"\n💡 РЕКОМЕНДАЦИИ ДЛЯ БИЗНЕСА:")

    # Анализ для рекомендаций
    high_activity_drivers = len(
        analysis_results["driver_activity"][
            analysis_results["driver_activity"]["total_orders"] > 50
        ]
    )
    high_activity_percentage = (
        high_activity_drivers / analysis_results["total_drivers"]
    ) * 100

    if high_activity_percentage < 20:
        print("• ❗ Мало высокоактивных водителей - рассмотреть программу мотивации")

    if analysis_results["driver_activity"]["completion_rate"].mean() < 80:
        print("• ❗ Низкий процент выполнения - улучшить систему матчинга")

    if analysis_results["driver_activity"]["avg_rating"].mean() < 4.5:
        print("• ❗ Средний рейтинг низкий - улучшить качество сервиса")

    if (
        len(analysis_results["problematic_drivers"])
        > analysis_results["total_drivers"] * 0.1
    ):
        print("• ❗ Много проблемных водителей - пересмотреть процесс онбординга")

    print("• ✅ Рекомендуется программа лояльности для топ-водителей")
    print("• ✅ Внедрить систему геймификации для повышения активности")


# Запуск анализа
if __name__ == "__main__":
    df = pd.read_csv("./train.csv")
    driver_analysis = analyze_driver_statistics(df)
    print_driver_summary_report(driver_analysis)
