import pandas as pd
import matplotlib.pyplot as plt


def analyze_peak_hours(df, save_path="visualizations/"):
    """
    Анализирует часы пик на основе данных о заказах
    """
    import os

    os.makedirs(save_path, exist_ok=True)

    # Преобразуем временные метки
    df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])
    df["tender_timestamp"] = pd.to_datetime(df["tender_timestamp"])

    # Извлекаем час и день недели
    df["order_hour"] = df["order_timestamp"].dt.hour
    df["order_dayofweek"] = df[
        "order_timestamp"
    ].dt.dayofweek  # 0=понедельник, 6=воскресенье
    df["order_date"] = df["order_timestamp"].dt.date
    df["order_day_name"] = df["order_timestamp"].dt.day_name()

    # Дни недели на русском
    day_names_ru = {
        "Monday": "Понедельник",
        "Tuesday": "Вторник",
        "Wednesday": "Среда",
        "Thursday": "Четверг",
        "Friday": "Пятница",
        "Saturday": "Суббота",
        "Sunday": "Воскресенье",
    }
    df["order_day_name_ru"] = df["order_day_name"].map(day_names_ru)

    print("=" * 60)
    print("АНАЛИЗ ЧАСОВ ПИК")
    print("=" * 60)

    # 1. Анализ по часам
    hourly_stats = (
        df.groupby("order_hour")
        .agg(
            {
                "order_id": "count",
                "is_done": lambda x: (x == "done").sum(),
                "driver_id": "nunique",
            }
        )
        .reset_index()
    )

    hourly_stats.columns = [
        "hour",
        "total_orders",
        "completed_orders",
        "unique_drivers",
    ]
    hourly_stats["completion_rate"] = (
        hourly_stats["completed_orders"] / hourly_stats["total_orders"] * 100
    ).round(2)
    hourly_stats["orders_per_driver"] = (
        hourly_stats["total_orders"] / hourly_stats["unique_drivers"]
    ).round(2)

    # Преобразуем total_orders в int для корректного форматирования
    hourly_stats["total_orders_int"] = hourly_stats["total_orders"].astype(int)
    hourly_stats["completed_orders_int"] = hourly_stats["completed_orders"].astype(int)

    # Определяем часы пик
    peak_hours = hourly_stats.nlargest(3, "total_orders")
    low_hours = hourly_stats.nsmallest(3, "total_orders")

    print("\n📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"• Всего заказов: {len(df):,}")
    print(f"• Период анализа: {df['order_date'].min()} - {df['order_date'].max()}")
    print(f"• Уникальных дней: {df['order_date'].nunique()}")

    print("\n🏆 ТОП-3 ЧАСА ПИК:")
    for _, row in peak_hours.iterrows():
        print(
            f"• {int(row['hour']):02d}:00 - {int(row['total_orders']):,} заказов "
            f"({row['completion_rate']}% выполнено)"
        )

    print("\n📉 ТОП-3 НАИМЕНЕЕ АКТИВНЫХ ЧАСА:")
    for _, row in low_hours.iterrows():
        print(
            f"• {int(row['hour']):02d}:00 - {int(row['total_orders']):,} заказов "
            f"({row['completion_rate']}% выполнено)"
        )

    # 2. Анализ по дням недели
    daily_stats = (
        df.groupby(["order_dayofweek", "order_day_name_ru"])
        .agg(
            {
                "order_id": "count",
                "is_done": lambda x: (x == "done").sum(),
                "driver_id": "nunique",
            }
        )
        .reset_index()
    )

    daily_stats.columns = [
        "dayofweek",
        "day_name",
        "total_orders",
        "completed_orders",
        "unique_drivers",
    ]
    daily_stats["completion_rate"] = (
        daily_stats["completed_orders"] / daily_stats["total_orders"] * 100
    ).round(2)
    daily_stats = daily_stats.sort_values("dayofweek")

    # Преобразуем в int для корректного форматирования
    daily_stats["total_orders_int"] = daily_stats["total_orders"].astype(int)

    peak_day = daily_stats.loc[daily_stats["total_orders"].idxmax()]
    low_day = daily_stats.loc[daily_stats["total_orders"].idxmin()]

    print("\n📅 АНАЛИЗ ПО ДНЯМ НЕДЕЛИ:")
    print(
        f"• Самый загруженный день: {peak_day['day_name']} - {int(peak_day['total_orders']):,} заказов"
    )
    print(
        f"• Самый спокойный день: {low_day['day_name']} - {int(low_day['total_orders']):,} заказов"
    )

    # 3. Детальный анализ часов пик
    print("\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ЧАСОВ ПИК:")
    for hour in peak_hours["hour"]:
        hour_data = df[df["order_hour"] == hour]
        hour_cancel_rate = round(
            (len(hour_data[hour_data["is_done"] == "cancel"]) / len(hour_data) * 100), 2
        )

        print(f"\n⏰ Час {int(hour):02d}:00:")
        print(f"  • Всего заказов: {len(hour_data):,}")
        print(f"  • Процент отмен: {hour_cancel_rate}%")
        print(f"  • Уникальных водителей: {hour_data['driver_id'].nunique()}")
        if "price_bid_local" in df.columns:
            avg_bid = hour_data["price_bid_local"].mean()
            print(f"  • Средний бид: {avg_bid:.0f}")

    # 4. Визуализации
    create_peak_hours_visualizations(df, hourly_stats, daily_stats, save_path)

    # 5. Рекомендации
    print("\n💡 РЕКОМЕНДАЦИИ:")
    print(
        f"• Пиковые часы: {', '.join([f'{int(h):02d}:00' for h in peak_hours['hour']])}"
    )
    print("• Рекомендуется увеличить количество водителей в эти часы")
    print(
        f"• Наименее загруженные часы: {', '.join([f'{int(h):02d}:00' for h in low_hours['hour']])}"
    )

    return {
        "hourly_stats": hourly_stats,
        "daily_stats": daily_stats,
        "peak_hours": peak_hours,
        "low_hours": low_hours,
    }


def create_peak_hours_visualizations(df, hourly_stats, daily_stats, save_path):
    """
    Создает визуализации для анализа часов пик
    """
    # 1. Распределение заказов по часам
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # График 1: Количество заказов по часам
    ax1.bar(
        hourly_stats["hour"],
        hourly_stats["total_orders"],
        color="lightcoral",
        alpha=0.7,
        edgecolor="darkred",
    )
    ax1.set_xlabel("Час дня")
    ax1.set_ylabel("Количество заказов")
    ax1.set_title("Распределение заказов по часам дня", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Подсвечиваем часы пик
    peak_hours = hourly_stats.nlargest(3, "total_orders")
    for hour in peak_hours["hour"]:
        ax1.axvspan(
            hour - 0.4,
            hour + 0.4,
            alpha=0.3,
            color="red",
            label="Час пик" if hour == peak_hours["hour"].iloc[0] else "",
        )

    # График 2: Процент выполнения по часам
    ax2.plot(
        hourly_stats["hour"],
        hourly_stats["completion_rate"],
        marker="o",
        linewidth=2,
        color="green",
        markersize=6,
    )
    ax2.fill_between(
        hourly_stats["hour"], hourly_stats["completion_rate"], alpha=0.3, color="green"
    )
    ax2.set_xlabel("Час дня")
    ax2.set_ylabel("Процент выполнения (%)")
    ax2.set_title(
        "Процент выполненных заказов по часам", fontsize=14, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    # График 3: Заказы по дням недели
    days_order = [
        "Понедельник",
        "Вторник",
        "Среда",
        "Четверг",
        "Пятница",
        "Суббота",
        "Воскресенье",
    ]
    daily_stats_sorted = daily_stats.set_index("day_name").loc[days_order].reset_index()

    bars = ax3.bar(
        daily_stats_sorted["day_name"],
        daily_stats_sorted["total_orders"],
        color="skyblue",
        alpha=0.7,
        edgecolor="navy",
    )
    ax3.set_xlabel("День недели")
    ax3.set_ylabel("Количество заказов")
    ax3.set_title(
        "Распределение заказов по дням недели", fontsize=14, fontweight="bold"
    )
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3)

    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(daily_stats_sorted["total_orders"]) * 0.01,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # График 4: Тепловая карта активности (час × день недели)
    heatmap_data = df.pivot_table(
        index="order_day_name_ru",
        columns="order_hour",
        values="order_id",
        aggfunc="count",
        fill_value=0,
    )

    # Сортируем дни недели в правильном порядке
    days_order_ru = [
        "Понедельник",
        "Вторник",
        "Среда",
        "Четверг",
        "Пятница",
        "Суббота",
        "Воскресенье",
    ]
    heatmap_data = heatmap_data.reindex(days_order_ru)

    im = ax4.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
    ax4.set_xlabel("Час дня")
    ax4.set_ylabel("День недели")
    ax4.set_title(
        "Тепловая карта активности\n(заказы по дням и часам)",
        fontsize=14,
        fontweight="bold",
    )
    ax4.set_xticks(range(24))
    ax4.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax4.set_yticks(range(len(days_order_ru)))
    ax4.set_yticklabels(days_order_ru)

    # Добавляем цветовую шкалу
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"{save_path}12_peak_hours_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


# Дополнительный анализ: сегментация часов
def analyze_hour_segments(df):
    """
    Анализирует сегменты дня
    """
    print("\n🎯 СЕГМЕНТАЦИЯ ДНЯ:")

    # Определяем сегменты дня
    morning = df[(df["order_hour"] >= 6) & (df["order_hour"] < 12)]
    afternoon = df[(df["order_hour"] >= 12) & (df["order_hour"] < 18)]
    evening = df[(df["order_hour"] >= 18) & (df["order_hour"] < 24)]
    night = df[(df["order_hour"] >= 0) & (df["order_hour"] < 6)]

    segments = {
        "Утро (06:00-11:59)": morning,
        "День (12:00-17:59)": afternoon,
        "Вечер (18:00-23:59)": evening,
        "Ночь (00:00-05:59)": night,
    }

    for name, segment in segments.items():
        if len(segment) > 0:
            cancel_rate = round(
                (len(segment[segment["is_done"] == "cancel"]) / len(segment) * 100), 2
            )
            print(f"• {name}: {len(segment):,} заказов, отмены: {cancel_rate}%")


# Запуск анализа
if __name__ == "__main__":
    df = pd.read_csv("./train.csv")
    peak_analysis = analyze_peak_hours(df)
    analyze_hour_segments(df)
