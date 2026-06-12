"""
Программный модуль для учебной практики: генерация, очистка,
количественный анализ и визуализация данных.

Установка дополнительных библиотек выполняется в командной строке:
    python -m pip install --upgrade pip
    python -m pip install numpy pandas matplotlib

Запуск программы:
    python practice_program_code.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "practice_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_data(size: int = 1000, seed: int = 42) -> pd.Series:
    """Генерирует исходный набор целочисленных данных."""
    np.random.seed(seed)
    numbers = np.random.randint(-10000, 10001, size=size)
    return pd.Series(numbers, name="Исходные данные")


def add_digital_noise(series: pd.Series) -> pd.Series:
    """Добавляет цифровой мусор для проверки алгоритмов очистки."""
    noisy = series.astype("object").copy()
    noisy.loc[10] = None
    noisy.loc[20] = "мусор"
    noisy.loc[30] = "  1500  "
    noisy.loc[40] = series.iloc[0]
    noisy.loc[41] = series.iloc[0]
    noisy.loc[50] = 50000
    return noisy


def clean_data(series: pd.Series) -> pd.Series:
    """Очищает данные: переводит в числа, удаляет пропуски, дубликаты и выбросы."""
    cleaned = pd.to_numeric(series, errors="coerce")
    cleaned = cleaned.dropna()
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned[(cleaned >= -10000) & (cleaned <= 10000)]
    cleaned = cleaned.reset_index(drop=True).astype(int)
    cleaned.name = "Очищенные данные"
    return cleaned


def calculate_statistics(series: pd.Series) -> pd.DataFrame:
    """Рассчитывает числовые характеристики набора данных."""
    counts = series.value_counts()
    duplicated_values = counts[counts > 1].count()
    stats = {
        "Количество элементов": series.count(),
        "Минимальное значение": series.min(),
        "Максимальное значение": series.max(),
        "Сумма значений": series.sum(),
        "Среднее значение": round(series.mean(), 2),
        "Медиана": round(series.median(), 2),
        "Первый квартиль Q1": round(series.quantile(0.25), 2),
        "Третий квартиль Q3": round(series.quantile(0.75), 2),
        "Дисперсия": round(series.var(), 2),
        "Среднеквадратическое отклонение": round(series.std(), 2),
        "Количество повторяющихся значений": duplicated_values,
        "Количество пропусков": int(series.isna().sum()),
    }
    return pd.DataFrame(stats.items(), columns=["Показатель", "Значение"])


def build_dataframe(series: pd.Series) -> pd.DataFrame:
    """Формирует DataFrame с исходными и отсортированными данными."""
    return pd.DataFrame({
        "Исходные данные": series.reset_index(drop=True),
        "По возрастанию": pd.Series(sorted(series)),
        "По убыванию": pd.Series(sorted(series, reverse=True)),
    })


def save_visualizations(series: pd.Series, dataframe: pd.DataFrame) -> None:
    """Сохраняет графики анализа данных в файлы PNG."""
    plt.figure(figsize=(12, 5))
    plt.plot(series.index, series.values, linewidth=1)
    plt.title("Линейный график исходных данных")
    plt.xlabel("Номер элемента")
    plt.ylabel("Значение")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "line_plot.png"), dpi=200)
    plt.close()

    rounded_series = np.round(series / 100) * 100
    plt.figure(figsize=(10, 5))
    plt.hist(rounded_series, bins=30, edgecolor="black")
    plt.title("Гистограмма распределения данных")
    plt.xlabel("Значение")
    plt.ylabel("Частота")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "histogram.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(dataframe.index, dataframe["По возрастанию"], label="По возрастанию")
    plt.plot(dataframe.index, dataframe["По убыванию"], label="По убыванию")
    plt.title("Сравнение отсортированных данных")
    plt.xlabel("Индекс")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sorted_comparison.png"), dpi=200)
    plt.close()


def run_tests(original: pd.Series, noisy: pd.Series, cleaned: pd.Series, dataframe: pd.DataFrame) -> None:
    """Выполняет тестирование программных модулей."""
    assert len(original) == 1000, "Ошибка генерации: неверное количество элементов"
    assert original.min() >= -10000 and original.max() <= 10000, "Ошибка генерации: выход за диапазон"
    assert noisy.isna().sum() > 0 or noisy.astype(str).str.contains("мусор").any(), "Не добавлен тестовый мусор"
    assert cleaned.isna().sum() == 0, "Очистка не удалила пропуски"
    assert cleaned.duplicated().sum() == 0, "Очистка не удалила дубликаты"
    assert cleaned.min() >= -10000 and cleaned.max() <= 10000, "Очистка не удалила выбросы"
    assert list(dataframe.columns) == ["Исходные данные", "По возрастанию", "По убыванию"], "Ошибка структуры DataFrame"
    assert dataframe["По возрастанию"].is_monotonic_increasing, "Ошибка сортировки по возрастанию"
    assert dataframe["По убыванию"].is_monotonic_decreasing, "Ошибка сортировки по убыванию"


def main() -> None:
    original_series = generate_data()
    noisy_series = add_digital_noise(original_series)
    cleaned_series = clean_data(noisy_series)
    statistics_df = calculate_statistics(original_series)
    analysis_df = build_dataframe(original_series)

    original_series.to_csv(os.path.join(OUTPUT_DIR, "generated_series.csv"), index=False, encoding="utf-8-sig")
    noisy_series.to_csv(os.path.join(OUTPUT_DIR, "noisy_series.csv"), index=False, encoding="utf-8-sig")
    cleaned_series.to_csv(os.path.join(OUTPUT_DIR, "cleaned_series.csv"), index=False, encoding="utf-8-sig")
    statistics_df.to_csv(os.path.join(OUTPUT_DIR, "statistics_report.csv"), index=False, encoding="utf-8-sig")
    analysis_df.to_csv(os.path.join(OUTPUT_DIR, "result_dataframe.csv"), index=False, encoding="utf-8-sig")

    save_visualizations(original_series, analysis_df)
    run_tests(original_series, noisy_series, cleaned_series, analysis_df)

    print("Первые 10 сгенерированных значений:")
    print(original_series.head(10).to_string())
    print("\nЧисловые характеристики:")
    print(statistics_df.to_string(index=False))
    print("\nПервые 10 строк DataFrame:")
    print(analysis_df.head(10).to_string(index=False))
    print("\nТестирование завершено успешно. Все файлы сохранены в папку practice_results.")


if __name__ == "__main__":
    main()
