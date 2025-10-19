import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_absolute_low_bid_cancels(df, save_path="visualizations/"):
    """
    Анализирует абсолютное количество отклоненных низких бидов в часы пик
    """
    import os

    os.makedirs(save_path, exist_ok=True)

    # Преобразуем временные метки
    df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])
    df["order_hour"] = df["order_timestamp"].dt.hour

    print("=" * 70)
    print("АНАЛИЗ АБСОЛЮТНОГО КОЛИЧЕСТВА ОТКЛОНЕННЫХ НИЗКИХ БИДОВ")
    print("=" * 70)

    # 1. Определяем часы пик
    hourly_orders = df.groupby("order_hour").size()
    peak_threshold = hourly_orders.quantile(0.75)
    peak_hours_list = hourly_orders[hourly_orders >= peak_threshold].index.tolist()
    df["is_peak_hour"] = df["order_hour"].isin(peak_hours_list)

    # 2. Определяем проблемные заказы
    df["is_low_bid_cancel"] = (df["is_done"] == "cancel") & (
        df["price_bid_local"] <= df["price_start_local"]
    )

    # 3. Анализ абсолютных количеств
    hourly_absolute = (
        df.groupby("order_hour")
        .agg(
            {
                "order_id": "count",
                "is_low_bid_cancel": "sum",
                "is_done": lambda x: (x == "cancel").sum(),  # Все отмены
            }
        )
        .reset_index()
    )

    hourly_absolute.columns = ["hour", "total_orders", "low_bid_cancels", "all_cancels"]
    hourly_absolute["is_peak"] = hourly_absolute["hour"].isin(peak_hours_list)

    # 4. Суммарная статистика
    total_low_bid_cancels = hourly_absolute["low_bid_cancels"].sum()
    peak_low_bid_cancels = hourly_absolute[hourly_absolute["is_peak"]][
        "low_bid_cancels"
    ].sum()
    off_peak_low_bid_cancels = hourly_absolute[~hourly_absolute["is_peak"]][
        "low_bid_cancels"
    ].sum()

    print("\n📊 АБСОЛЮТНЫЕ ПОКАЗАТЕЛИ:")
    print(f"• Всего отклоненных низких бидов: {total_low_bid_cancels:,}")
    print(f"• В часы пик: {peak_low_bid_cancels:,}")
    print(f"• Вне часов пик: {off_peak_low_bid_cancels:,}")
    print(
        f"• Доля в часы пик: {peak_low_bid_cancels / total_low_bid_cancels * 100:.1f}%"
    )

    # 5. Сравнение с ожидаемым распределением
    total_peak_orders = hourly_absolute[hourly_absolute["is_peak"]][
        "total_orders"
    ].sum()
    total_off_peak_orders = hourly_absolute[~hourly_absolute["is_peak"]][  # noqa: F841
        "total_orders"
    ].sum()

    expected_peak_cancels = total_low_bid_cancels * (total_peak_orders / len(df))
    actual_vs_expected = peak_low_bid_cancels - expected_peak_cancels

    print("\n📈 СРАВНЕНИЕ С ОЖИДАЕМЫМ РАСПРЕДЕЛЕНИЕМ:")
    print(f"• Ожидалось в часы пик: {expected_peak_cancels:.0f}")
    print(f"• Фактически в часы пик: {peak_low_bid_cancels:,}")
    print(
        f"• Разница: {actual_vs_expected:+.0f} ({actual_vs_expected / expected_peak_cancels * 100:+.1f}%)"
    )

    # 6. Топ часов по абсолютному количеству проблем
    print("\n🏆 ТОП-5 ЧАСОВ ПО КОЛИЧЕСТВУ ОТКЛОНЕННЫХ НИЗКИХ БИДОВ:")
    top_absolute = hourly_absolute.nlargest(5, "low_bid_cancels")
    for _, row in top_absolute.iterrows():
        peak_status = "ПИК" if row["is_peak"] else "не пик"
        percentage_of_total = row["low_bid_cancels"] / total_low_bid_cancels * 100
        print(
            f"• {int(row['hour']):02d}:00 - {int(row['low_bid_cancels']):,} случаев "
            f"({percentage_of_total:.1f}% от всех проблем, {peak_status})"
        )

    # 7. Анализ плотности проблем
    print("\n📋 ПЛОТНОСТЬ ПРОБЛЕМ:")
    peak_hours_count = len(peak_hours_list)
    off_peak_hours_count = 24 - peak_hours_count

    problems_per_peak_hour = peak_low_bid_cancels / peak_hours_count
    problems_per_off_peak_hour = off_peak_low_bid_cancels / off_peak_hours_count

    print(f"• Часов пик: {peak_hours_count}")
    print(f"• Проблем в час пик: {problems_per_peak_hour:.1f}")
    print(f"• Проблем в непиковый час: {problems_per_off_peak_hour:.1f}")
    print(f"• Соотношение: {problems_per_peak_hour / problems_per_off_peak_hour:.1f}:1")

    # 8. Визуализации абсолютных показателей
    create_absolute_visualizations(df, hourly_absolute, save_path)

    # 9. Статистическая значимость
    from scipy.stats import mannwhitneyu

    peak_data = hourly_absolute[hourly_absolute["is_peak"]]["low_bid_cancels"]
    off_peak_data = hourly_absolute[~hourly_absolute["is_peak"]]["low_bid_cancels"]

    stat, p_value = mannwhitneyu(peak_data, off_peak_data, alternative="two-sided")

    print("\n📊 СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ:")
    print(f"• U-тест Манна-Уитни: p-value = {p_value:.4f}")

    if p_value < 0.05:
        print("✅ Статистически значимая разница в абсолютных количествах")
        if peak_data.mean() > off_peak_data.mean():
            print("• В часы пик СИЛЬНО БОЛЬШЕ абсолютных случаев проблем")
        else:
            print("• В часы пик СИЛЬНО МЕНЬШЕ абсолютных случаев проблем")
    else:
        print("❌ Нет статистически значимой разницы в абсолютных количествах")

    return {
        "hourly_absolute": hourly_absolute,
        "total_low_bid_cancels": total_low_bid_cancels,
        "peak_low_bid_cancels": peak_low_bid_cancels,
        "off_peak_low_bid_cancels": off_peak_low_bid_cancels,
        "problems_per_peak_hour": problems_per_peak_hour,
        "problems_per_off_peak_hour": problems_per_off_peak_hour,
    }


def create_absolute_visualizations(df, hourly_absolute, save_path):
    """
    Создает визуализации для абсолютных показателей
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # График 1: Абсолютное количество проблем по часам
    colors = ["red" if peak else "blue" for peak in hourly_absolute["is_peak"]]
    bars = ax1.bar(
        hourly_absolute["hour"],
        hourly_absolute["low_bid_cancels"],
        color=colors,
        alpha=0.7,
        edgecolor="black",
    )
    ax1.set_xlabel("Час дня")
    ax1.set_ylabel("Количество отклоненных низких бидов")
    ax1.set_title(
        "АБСОЛЮТНОЕ количество проблем по часам\n(красный - часы пик)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(hourly_absolute["low_bid_cancels"]) * 0.01,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Легенда
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", alpha=0.7, label="Часы пик"),
        Patch(facecolor="blue", alpha=0.7, label="Вне часов пик"),
    ]
    ax1.legend(handles=legend_elements)

    # График 2: Накопительное распределение
    hourly_sorted = hourly_absolute.sort_values("low_bid_cancels", ascending=False)
    cumulative_percentage = (
        hourly_sorted["low_bid_cancels"].cumsum()
        / hourly_sorted["low_bid_cancels"].sum()
        * 100
    )

    ax2.plot(
        range(1, len(hourly_sorted) + 1),
        cumulative_percentage,
        marker="o",
        linewidth=2,
        color="purple",
    )
    ax2.axhline(y=80, color="red", linestyle="--", alpha=0.7, label="80% проблем")
    ax2.set_xlabel("Количество часов (отсортировано по проблемам)")
    ax2.set_ylabel("Накопительный процент проблем (%)")
    ax2.set_title(
        "Накопительное распределение проблем по часам", fontsize=14, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Находим, за сколько часов набирается 80% проблем
    hours_for_80 = len(cumulative_percentage[cumulative_percentage <= 80])
    ax2.axvline(
        x=hours_for_80,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"{hours_for_80} часов для 80% проблем",
    )

    # График 3: Сравнение пиковых и непиковых часов
    peak_data = hourly_absolute[hourly_absolute["is_peak"]]["low_bid_cancels"]
    off_peak_data = hourly_absolute[~hourly_absolute["is_peak"]]["low_bid_cancels"]

    categories = ["Пиковые часы", "Непиковые часы"]
    values = [peak_data.sum(), off_peak_data.sum()]
    colors_comparison = ["red", "blue"]

    bars_comp = ax3.bar(
        categories, values, color=colors_comparison, alpha=0.7, edgecolor="black"
    )
    ax3.set_ylabel("Общее количество проблем")
    ax3.set_title(
        "Сравнение общего количества проблем\nв пиковые vs непиковые часы",
        fontsize=14,
        fontweight="bold",
    )
    ax3.grid(True, alpha=0.3)

    # Добавляем значения на столбцы
    for bar, value in zip(bars_comp, values):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(values) * 0.01,
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # График 4: Соотношение заказов и проблем
    ax4.scatter(
        hourly_absolute["total_orders"],
        hourly_absolute["low_bid_cancels"],
        c=hourly_absolute["is_peak"].astype(int),
        cmap="RdYlBu",
        s=100,
        alpha=0.7,
    )
    ax4.set_xlabel("Общее количество заказов в час")
    ax4.set_ylabel("Количество отклоненных низких бидов")
    ax4.set_title(
        "Зависимость: заказы vs абсолютные проблемы\n(красный - часы пик)",
        fontsize=14,
        fontweight="bold",
    )
    ax4.grid(True, alpha=0.3)

    # Линия тренда
    z = np.polyfit(
        hourly_absolute["total_orders"], hourly_absolute["low_bid_cancels"], 1
    )
    p = np.poly1d(z)
    ax4.plot(
        hourly_absolute["total_orders"],
        p(hourly_absolute["total_orders"]),
        "r--",
        alpha=0.8,
        label=f"Тренд: y = {z[0]:.3f}x + {z[1]:.2f}",
    )

    # Аннотации для выбросов
    for _, row in hourly_absolute.nlargest(3, "low_bid_cancels").iterrows():
        ax4.annotate(
            f"{int(row['hour']):02d}:00\n{int(row['low_bid_cancels'])}",
            (row["total_orders"], row["low_bid_cancels"]),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    ax4.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}16_absolute_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Дополнительная визуализация: временная шкала
    fig, ax = plt.subplots(figsize=(15, 6))

    # Создаем временной ряд
    time_series = df[df["is_low_bid_cancel"]].copy()
    time_series["time_group"] = time_series["order_timestamp"].dt.floor("H")
    hourly_timeline = time_series.groupby("time_group").size()

    ax.plot(
        hourly_timeline.index,
        hourly_timeline.values,
        linewidth=1,
        alpha=0.7,
        color="red",
    )
    ax.scatter(
        hourly_timeline.index, hourly_timeline.values, s=20, color="darkred", alpha=0.8
    )

    ax.set_xlabel("Время")
    ax.set_ylabel("Количество проблем в час")
    ax.set_title(
        "Временной ряд: отклоненные низкие биды по часам",
        fontsize=14,
        fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}17_timeline_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


# Запуск анализа
if __name__ == "__main__":
    df = pd.read_csv("./train.csv")
    absolute_results = analyze_absolute_low_bid_cancels(df)
