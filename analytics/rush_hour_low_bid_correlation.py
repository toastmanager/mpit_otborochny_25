import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, pointbiserialr


def analyze_correlation_low_bid_peak_hours(df, save_path="visualizations/"):
    """
    Анализирует корреляцию между отклоненными бидами ниже базовой стоимости и часами пик
    """
    import os

    os.makedirs(save_path, exist_ok=True)

    # Преобразуем временные метки
    df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])
    df["order_hour"] = df["order_timestamp"].dt.hour

    print("=" * 70)
    print("КОРРЕЛЯЦИЯ: ОТКЛОНЕННЫЕ БИДЫ НИЖЕ СТАРТОВОЙ ЦЕНЫ И ЧАСЫ ПИК")
    print("=" * 70)

    # 1. Определяем часы пик
    hourly_orders = df.groupby("order_hour").size()
    peak_threshold = hourly_orders.quantile(0.75)  # Верхние 25% как часы пик
    df["is_peak_hour"] = df["order_hour"].isin(
        hourly_orders[hourly_orders >= peak_threshold].index
    )

    # 2. Определяем проблемные заказы (отклоненные с низким бидом)
    df["is_low_bid_cancel"] = (df["is_done"] == "cancel") & (
        df["price_bid_local"] <= df["price_start_local"]
    )

    # 3. Базовая статистика
    total_orders = len(df)
    peak_hour_orders = len(df[df["is_peak_hour"]])
    low_bid_cancel_orders = len(df[df["is_low_bid_cancel"]])

    print("\n📊 БАЗОВАЯ СТАТИСТИКА:")
    print(f"• Всего заказов: {total_orders:,}")
    print(
        f"• Заказов в часы пик: {peak_hour_orders:,} ({peak_hour_orders / total_orders * 100:.1f}%)"
    )
    print(
        f"• Отклоненных заказов с низким бидом: {low_bid_cancel_orders:,} ({low_bid_cancel_orders / total_orders * 100:.1f}%)"
    )

    # 4. Анализ распределения по часам
    hourly_analysis = (
        df.groupby("order_hour")
        .agg({"order_id": "count", "is_low_bid_cancel": "sum", "is_peak_hour": "first"})
        .reset_index()
    )

    hourly_analysis.columns = ["hour", "total_orders", "low_bid_cancels", "is_peak"]
    hourly_analysis["cancel_rate"] = (
        hourly_analysis["low_bid_cancels"] / hourly_analysis["total_orders"] * 100
    ).round(2)
    hourly_analysis["is_peak"] = hourly_analysis["is_peak"].astype(bool)

    # 5. Статистические тесты
    print("\n📈 СТАТИСТИЧЕСКИЙ АНАЛИЗ:")

    # Создаем таблицу сопряженности
    contingency_table = pd.crosstab(df["is_peak_hour"], df["is_low_bid_cancel"])

    print("Таблица сопряженности:")
    print(contingency_table)

    # Тест хи-квадрат
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print("\n• Тест хи-квадрат:")
    print(f"  Chi2 = {chi2:.4f}")
    print(f"  p-value = {p_value:.4f}")
    print(f"  Степени свободы = {dof}")

    # Точечно-бисериальная корреляция
    if len(df[df["is_low_bid_cancel"]]) > 0:
        correlation, p_corr = pointbiserialr(
            df["is_peak_hour"].astype(int), df["is_low_bid_cancel"].astype(int)
        )
        print("\n• Точечно-бисериальная корреляция:")
        print(f"  Коэффициент корреляции = {correlation:.4f}")
        print(f"  p-value = {p_corr:.4f}")

    # 6. Сравнение rates
    peak_low_bid_rate = (
        len(df[(df["is_peak_hour"]) & (df["is_low_bid_cancel"])])
        / len(df[df["is_peak_hour"]])
        * 100
    )
    off_peak_low_bid_rate = (
        len(df[(~df["is_peak_hour"]) & (df["is_low_bid_cancel"])])
        / len(df[~df["is_peak_hour"]])
        * 100
    )

    print("\n📊 СРАВНЕНИЕ ПРОЦЕНТОВ:")
    print(f"• В часы пик: {peak_low_bid_rate:.2f}% заказов - отклоненные низкие биды")
    print(
        f"• Вне часов пик: {off_peak_low_bid_rate:.2f}% заказов - отклоненные низкие биды"
    )
    print(f"• Разница: {abs(peak_low_bid_rate - off_peak_low_bid_rate):.2f}%")

    # 7. Детальный анализ по часам
    peak_hours = hourly_analysis[hourly_analysis["is_peak"]]
    off_peak_hours = hourly_analysis[~hourly_analysis["is_peak"]]

    print("\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ПО ЧАСАМ:")
    print(f"• Часы пик: {list(peak_hours['hour'])}")
    print(
        f"• Средний процент проблем в часы пик: {peak_hours['cancel_rate'].mean():.2f}%"
    )
    print(
        f"• Средний процент проблем вне часов пик: {off_peak_hours['cancel_rate'].mean():.2f}%"
    )

    # Топ часов по проблемам
    top_problem_hours = hourly_analysis.nlargest(5, "cancel_rate")
    print("\n🏆 ТОП-5 ЧАСОВ ПО ПРОЦЕНТУ ОТКЛОНЕННЫХ НИЗКИХ БИДОВ:")
    for _, row in top_problem_hours.iterrows():
        peak_status = "ПИК" if row["is_peak"] else "не пик"
        print(
            f"• {int(row['hour']):02d}:00 - {row['cancel_rate']}% проблем ({peak_status})"
        )

    # 8. Визуализации
    create_correlation_visualizations(df, hourly_analysis, save_path)

    # 9. Выводы
    print("\n💡 ВЫВОДЫ И ИНТЕРПРЕТАЦИЯ:")

    if p_value < 0.05:  # type: ignore
        print("✅ Статистически значимая связь обнаружена (p < 0.05)")
        if peak_low_bid_rate > off_peak_low_bid_rate:
            print(
                f"• В часы пик процент отклоненных низких бидов ВЫШЕ на {peak_low_bid_rate - off_peak_low_bid_rate:.2f}%"
            )
            print("• Возможные причины: повышенная конкуренция, спешка водителей")
        else:
            print(
                f"• В часы пик процент отклоненных низких бидов НИЖЕ на {off_peak_low_bid_rate - peak_low_bid_rate:.2f}%"
            )
            print(
                "• Возможные причины: больше водителей онлайн, лучшее соответствие спроса/предложения"
            )
    else:
        print("❌ Статистически значимой связи не обнаружено (p ≥ 0.05)")
        print("• Часы пик не влияют на вероятность отклонения низких бидов")

    if abs(correlation) > 0.1 and p_corr < 0.05:  # type: ignore
        strength = (
            "слабая"
            if abs(correlation) < 0.3  # type: ignore
            else "умеренная"
            if abs(correlation) < 0.5  # type: ignore
            else "сильная"
        )
        direction = "положительная" if correlation > 0 else "отрицательная"  # type: ignore
        print(
            f"• {strength.capitalize()} {direction} корреляция: r = {correlation:.3f}"
        )

    return {
        "contingency_table": contingency_table,
        "chi2_test": (chi2, p_value, dof),
        "correlation": (correlation, p_corr) if "correlation" in locals() else None,
        "hourly_analysis": hourly_analysis,
        "peak_low_bid_rate": peak_low_bid_rate,
        "off_peak_low_bid_rate": off_peak_low_bid_rate,
    }


def create_correlation_visualizations(df, hourly_analysis, save_path):
    """
    Создает визуализации для анализа корреляции
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # График 1: Процент проблем по часам с выделением пиковых часов
    colors = ["red" if peak else "blue" for peak in hourly_analysis["is_peak"]]
    ax1.bar(
        hourly_analysis["hour"],
        hourly_analysis["cancel_rate"],
        color=colors,
        alpha=0.7,
        edgecolor="black",
    )
    ax1.set_xlabel("Час дня")
    ax1.set_ylabel("Процент отклоненных низких бидов (%)")
    ax1.set_title(
        "Процент отклоненных низких бидов по часам\n(красный - часы пик)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    # Добавляем легенду
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", alpha=0.7, label="Часы пик"),
        Patch(facecolor="blue", alpha=0.7, label="Вне часов пик"),
    ]
    ax1.legend(handles=legend_elements)

    # График 2: Сравнение распределения в пиковые и непиковые часы
    peak_data = hourly_analysis[hourly_analysis["is_peak"]]["cancel_rate"]
    off_peak_data = hourly_analysis[~hourly_analysis["is_peak"]]["cancel_rate"]

    box_data = [peak_data, off_peak_data]
    ax2.boxplot(box_data, labels=["Часы пик", "Вне часов пик"])
    ax2.set_ylabel("Процент отклоненных низких бидов (%)")
    ax2.set_title(
        "Сравнение распределения процента проблем", fontsize=14, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)

    # График 3: Тепловая карта - заказы vs проблемы
    heatmap_data = (
        df.groupby(["order_hour", "is_low_bid_cancel"]).size().unstack(fill_value=0)
    )
    heatmap_data_percent = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100

    im = ax3.imshow(heatmap_data_percent, cmap="RdYlBu_r", aspect="auto")
    ax3.set_xlabel("Тип заказа")
    ax3.set_ylabel("Час дня")
    ax3.set_title(
        "Распределение типов заказов по часам (%)\n", fontsize=14, fontweight="bold"
    )
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Обычные", "Низкий бид+отмена"])
    ax3.set_yticks(range(24))
    ax3.set_yticklabels([f"{h:02d}" for h in range(24)])

    # Добавляем значения в ячейки
    for i in range(heatmap_data_percent.shape[0]):
        for j in range(heatmap_data_percent.shape[1]):
            ax3.text(
                j,
                i,
                f"{heatmap_data_percent.iloc[i, j]:.1f}%",
                ha="center",
                va="center",
                fontweight="bold",
                color="white" if heatmap_data_percent.iloc[i, j] > 50 else "black",
            )

    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # График 4: Scatter plot - активность vs проблемы
    ax4.scatter(
        hourly_analysis["total_orders"],
        hourly_analysis["cancel_rate"],
        c=hourly_analysis["is_peak"].astype(int),
        cmap="RdYlBu",
        s=100,
        alpha=0.7,
    )
    ax4.set_xlabel("Общее количество заказов")
    ax4.set_ylabel("Процент отклоненных низких бидов (%)")
    ax4.set_title(
        "Зависимость проблем от общей активности", fontsize=14, fontweight="bold"
    )
    ax4.grid(True, alpha=0.3)

    # Добавляем аннотации для выбросов
    for _, row in hourly_analysis.nlargest(3, "cancel_rate").iterrows():
        ax4.annotate(
            f"{int(row['hour']):02d}:00",
            (row["total_orders"], row["cancel_rate"]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(f"{save_path}14_correlation_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Дополнительная визуализация: временные тренды
    if len(df) > 1000:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Скользящее среднее по часам
        hourly_trend = (
            df.groupby("order_hour")
            .agg({"is_low_bid_cancel": "mean", "order_id": "count"})
            .reset_index()
        )

        ax.plot(
            hourly_trend["order_hour"],
            hourly_trend["is_low_bid_cancel"] * 100,
            marker="o",
            linewidth=2,
            color="red",
            label="Процент проблем",
        )
        ax.set_xlabel("Час дня")
        ax.set_ylabel("Процент отклоненных низких бидов (%)", color="red")
        ax.tick_params(axis="y", labelcolor="red")
        ax.grid(True, alpha=0.3)

        # Второй график - общая активность
        ax2 = ax.twinx()
        ax2.bar(
            hourly_trend["order_hour"],
            hourly_trend["order_id"],
            alpha=0.3,
            color="blue",
            label="Общее кол-во заказов",
        )
        ax2.set_ylabel("Количество заказов", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")

        ax.set_title(
            "Сравнение процента проблем и общей активности по часам",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.savefig(f"{save_path}15_trend_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()


# Запуск анализа
if __name__ == "__main__":
    df = pd.read_csv("./train.csv")
    correlation_results = analyze_correlation_low_bid_peak_hours(df)
