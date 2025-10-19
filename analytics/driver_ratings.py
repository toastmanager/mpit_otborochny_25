import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def calculate_rating_stats(df):
    """
    Рассчитывает среднее значение рейтинга и медианное отклонение
    """
    # Проверяем, есть ли колонка с рейтингом
    if "driver_rating" not in df.columns:
        print("❌ Колонка 'driver_rating' не найдена в данных")
        return None

    # Убираем возможные пропущенные значения
    ratings = df["driver_rating"].dropna()

    if len(ratings) == 0:
        print("❌ Нет данных о рейтингах для анализа")
        return None

    # Основные статистики
    mean_rating = ratings.mean()
    median_rating = ratings.median()

    # Медианное отклонение (Median Absolute Deviation - MAD)
    mad = np.median(np.abs(ratings - median_rating))

    # Дополнительные статистики для контекста
    min_rating = ratings.min()
    max_rating = ratings.max()
    std_rating = ratings.std()

    print("=" * 50)
    print("СТАТИСТИКА РЕЙТИНГОВ ВОДИТЕЛЕЙ")
    print("=" * 50)
    print(f"📊 Общее количество оценок: {len(ratings):,}")
    print(f"⭐ Средний рейтинг: {mean_rating:.4f}")
    print(f"📈 Медианный рейтинг: {median_rating:.4f}")
    print(f"🎯 Медианное отклонение (MAD): {mad:.4f}")
    print(f"📉 Стандартное отклонение: {std_rating:.4f}")
    print(f"🔽 Минимальный рейтинг: {min_rating:.4f}")
    print(f"🔼 Максимальный рейтинг: {max_rating:.4f}")
    print(f"📐 Диапазон: {max_rating - min_rating:.4f}")

    # Анализ распределения
    print("\n📋 РАСПРЕДЕЛЕНИЕ РЕЙТИНГОВ:")
    print(f"• 25-й перцентиль: {ratings.quantile(0.25):.4f}")
    print(f"• 75-й перцентиль: {ratings.quantile(0.75):.4f}")
    print(
        f"• IQR (межквартильный размах): {ratings.quantile(0.75) - ratings.quantile(0.25):.4f}"
    )

    # Процент водителей с максимальным рейтингом
    perfect_ratings = len(ratings[ratings == 5.0])
    perfect_percentage = (perfect_ratings / len(ratings)) * 100
    print(
        f"• Водителей с рейтингом 5.0: {perfect_ratings:,} ({perfect_percentage:.2f}%)"
    )

    # Сравнение MAD со стандартным отклонением
    print("\n📊 СРАВНЕНИЕ МЕР РАССЕЯНИЯ:")
    print(f"• MAD / STD: {mad / std_rating:.4f}")
    print("• MAD более устойчив к выбросам, чем стандартное отклонение")

    return {
        "mean": mean_rating,
        "median": median_rating,
        "mad": mad,
        "std": std_rating,
        "min": min_rating,
        "max": max_rating,
        "count": len(ratings),
        "perfect_ratings_count": perfect_ratings,
        "perfect_ratings_percentage": perfect_percentage,
    }


# Дополнительная функция для сравнения статистик между разными группами
def compare_rating_groups(df, group_column=None):
    """
    Сравнивает статистики рейтинга между разными группами
    """
    if group_column and group_column not in df.columns:
        print(f"❌ Колонка '{group_column}' не найдена")
        return

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ СТАТИСТИК РЕЙТИНГОВ ПО ГРУППАМ")
    print("=" * 60)

    if group_column:
        groups = df[group_column].unique()
        for group in groups:
            group_data = df[df[group_column] == group]
            group_ratings = group_data["driver_rating"].dropna()

            if len(group_ratings) > 0:
                print(f"\n🏷️  Группа: {group}")
                print(f"   Количество: {len(group_ratings):,}")
                print(f"   Средний: {group_ratings.mean():.4f}")
                print(f"   Медиана: {group_ratings.median():.4f}")
                print(
                    f"   MAD: {np.median(np.abs(group_ratings - group_ratings.median())):.4f}"
                )
    else:
        # Сравнение по статусу заказа
        if "is_done" in df.columns:
            statuses = df["is_done"].unique()
            for status in statuses:
                status_data = df[df["is_done"] == status]
                status_ratings = status_data["driver_rating"].dropna()

                if len(status_ratings) > 0:
                    print(f"\n📊 Статус: {status}")
                    print(f"   Количество: {len(status_ratings):,}")
                    print(f"   Средний: {status_ratings.mean():.4f}")
                    print(f"   Медиана: {status_ratings.median():.4f}")
                    print(
                        f"   MAD: {np.median(np.abs(status_ratings - status_ratings.median())):.4f}"
                    )


def plot_rating_distribution(df):
    """
    Визуализирует распределение рейтингов
    """
    ratings = df["driver_rating"].dropna()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Гистограмма
    ax1.hist(ratings, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    ax1.axvline(
        ratings.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Среднее: {ratings.mean():.3f}",
    )
    ax1.axvline(
        ratings.median(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Медиана: {ratings.median():.3f}",
    )
    ax1.set_xlabel("Рейтинг")
    ax1.set_ylabel("Количество")
    ax1.set_title("Распределение рейтингов водителей")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(ratings, vert=False)
    ax2.set_xlabel("Рейтинг")
    ax2.set_title("Box plot рейтингов")
    ax2.grid(True, alpha=0.3)

    # Плотность распределения
    sns.kdeplot(ratings, ax=ax3, fill=True, color="lightcoral")
    ax3.axvline(
        ratings.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Среднее: {ratings.mean():.3f}",
    )
    ax3.axvline(
        ratings.median(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Медиана: {ratings.median():.3f}",
    )
    ax3.set_xlabel("Рейтинг")
    ax3.set_ylabel("Плотность")
    ax3.set_title("Плотность распределения рейтингов")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # QQ plot для проверки нормальности
    from scipy import stats

    stats.probplot(ratings, dist="norm", plot=ax4)
    ax4.set_title("Q-Q plot (проверка нормальности)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rating_distribution_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


# Запуск анализа
if __name__ == "__main__":
    df = pd.read_csv("./train.csv")
    # Основная статистика
    rating_stats = calculate_rating_stats(df)

    # Сравнение по группам
    compare_rating_groups(
        df, group_column="platform"
    )  # Можно заменить на другую колонку
    compare_rating_groups(df)  # Сравнение по статусу заказа

    # Визуализация
    plot_rating_distribution(df)

    # Интерпретация результатов
    print("\n" + "=" * 50)
    print("ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 50)

    if rating_stats:
        mad_std_ratio = rating_stats["mad"] / rating_stats["std"]

        print(f"📈 Средний рейтинг: {rating_stats['mean']:.4f}")
        print(f"🎯 Медианное отклонение (MAD): {rating_stats['mad']:.4f}")

        if rating_stats["mad"] < 0.1:
            print("• MAD очень маленький → рейтинги очень стабильные")
        elif rating_stats["mad"] < 0.3:
            print("• MAD небольшой → рейтинги достаточно стабильные")
        else:
            print("• MAD значительный → есть разброс в рейтингах")

        if mad_std_ratio < 0.8:
            print("• MAD значительно меньше STD → возможны выбросы")
        elif mad_std_ratio > 1.2:
            print("• MAD больше STD → необычное распределение")
        else:
            print("• MAD и STD близки → распределение близко к нормальному")

        if rating_stats["perfect_ratings_percentage"] > 50:
            print(
                f"• Высокий процент максимальных оценок ({rating_stats['perfect_ratings_percentage']:.1f}%) → возможен bias в оценках"
            )
