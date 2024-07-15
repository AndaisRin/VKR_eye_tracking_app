import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_process_data(file_path):
    # Загрузка данных из Excel-файла
    data = pd.read_excel(file_path)

    # Преобразование всех столбцов к числовому типу данных
    data = data.apply(pd.to_numeric, errors='coerce')

    # Замена значений NaN на средние значения по каждому столбцу
    data_filled = data.apply(lambda x: x.fillna(x.mean()), axis=0)

    return data, data_filled


def calculate_statistics(data_filled):
    # Расчет статистических показателей
    mean_values = data_filled.mean()
    median_values = data_filled.median()
    std_dev_values = data_filled.std()
    variance_values = data_filled.var()
    coefficient_of_variation = std_dev_values / mean_values

    # Создание расширенной таблицы с результатами
    extended_summary_table = pd.DataFrame({
        'Среднее значение': mean_values,
        'Медианное значение': median_values,
        'Стандартное отклонение': std_dev_values,
        'Дисперсия': variance_values,
        'Коэффициент вариации': coefficient_of_variation
    })

    return extended_summary_table


def save_extended_summary_table(table, output_path):
    # Сохранение расширенной таблицы в файл
    table.to_excel(output_path, index=False)


def plot_mean_median(mean_values, median_values, output_path):
    # Построение графика средних и медианных значений
    plt.figure(figsize=(14, 7))
    plt.plot(mean_values, label='Средние значения')
    plt.plot(median_values, label='Медианные значения')
    plt.xlabel('Номер действия')
    plt.ylabel('Время (с)')
    plt.title('Средние и медианные значения временных характеристик по действиям')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def plot_boxplots(data_filled, output_path):
    # Построение диаграмм размаха (boxplots) для каждого действия
    plt.figure(figsize=(20, 10))
    data_filled.boxplot()
    plt.xlabel('Номер действия')
    plt.ylabel('Время (с)')
    plt.title('Диаграммы размаха временных характеристик по действиям')
    plt.savefig(output_path)
    plt.close()


def plot_horizontal_boxplot(data, output_path):
    # Горизонтальный Box plot для распределения времени выполнения каждого действия
    time_data = data.drop(columns=['Команда'])
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=time_data.T, orient='h')
    plt.title('Распределение времени выполнения каждого действия')
    plt.ylabel('Действия')
    plt.xlabel('Время выполнения (с)')
    plt.yticks(ticks=range(len(data['Команда'])), labels=data['Команда'])
    plt.savefig('boxplot.png')


def main():
    # Путь к файлам
    file_path1 = 'Книга2_1.xlsx'
    file_path2 = 'Книга2_нужное_общее_1.xlsx'

    # Обработка первого файла
    data, data_filled = load_and_process_data(file_path1)
    extended_summary_table = calculate_statistics(data_filled)
    save_extended_summary_table(extended_summary_table, "Расширенная_таблица.xlsx")
    plot_mean_median(extended_summary_table['Среднее значение'], extended_summary_table['Медианное значение'],
                     'Средние_и_медианные_значения.png')
    plot_boxplots(data_filled, 'Диаграммы_размаха.png')

    # Обработка второго файла
    data2 = pd.read_excel(file_path2)
    missing_values = data2.isnull().sum()
    data_types = data2.dtypes
    print("Missing values:\n", missing_values)
    print("\nData types:\n", data_types)

    # Убираем первый столбец с описанием действий
    plot_horizontal_boxplot(data2, 'boxplot.png')


if __name__ == "__main__":
    main()
