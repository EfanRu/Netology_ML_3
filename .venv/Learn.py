from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from MyLogisticRegression import MyLogisticRegression
import matplotlib.pyplot as plt
import time

iris = load_iris()

# Для вывода всех столбцов
pd.set_option('display.max_columns', None)

# Считываем данные
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Добавляем целевую переменную (метки классов)
data['target'] = iris.target
data['species'] = iris.target_names[iris.target]

# Удаляем строки с species = 'setosa' и приводим к бинарным значениям 'target'
data = data[data['species'] != 'setosa']
data['target'] = data['target'] - 1

print('\nИзучаем данные:')
print(data.info())
print(data.head())
print(data.describe())
print('\nИзучаем уникальные итоговые значения:')
print(data['target'].unique())
print('\nПроверяем на нулевые значения:')
print(data[data['target'].isna()].head())

X = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
print('\nВыделяем признаки:')
print(X.head())

Y = data['target']
print('\nВыделяем результат:')
print(Y.head())

# разделяем тестовые данные как 80/20 %
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=11)

# Тестируем оба оптимизатора
optimizers = ['gradient_descent', 'rmsprop', 'nadam']
results = {}

for optimizer in optimizers:
    print(f"\n=== Тестируем {optimizer} ===")

    # Создаем модель с выбранным оптимизатором
    start_time = time.time()
    model = MyLogisticRegression(optimizer=optimizer)

    # Обучаем модель
    model.fit(X_train, Y_train)

    # Оцениваем точность
    model_score = model.score(X_test, Y_test)
    print(f'Точность {optimizer}: {model_score:.4f}')

    # Сохраняем результаты
    end_time = time.time()
    results[optimizer] = {
        'model': model,
        'score': model_score,
        'loss_history': model.loss_history,
        'time': end_time - start_time
    }

# Сравниваем графики потерь
plt.figure(figsize=(12, 6))

for optimizer in optimizers:
    loss_history = results[optimizer]['loss_history']
    plt.plot(loss_history, label=f'{optimizer} (final loss: {loss_history[-1]:.4f})')

plt.xlabel('Итерации')
plt.ylabel('Функция потерь')
plt.title('Сравнение методов оптимизации')
plt.legend()
plt.grid(True)
plt.savefig('optimization_comparison.png', dpi=300, bbox_inches='tight')
print("График сохранен как 'optimization_comparison.png'")

# Выводим итоговое сравнение
print("\n=== ИТОГОВОЕ СРАВНЕНИЕ ===")
for optimizer in optimizers:
    score = results[optimizer]['score']
    final_loss = results[optimizer]['loss_history'][-1]
    time_work = results[optimizer]['time']
    print(f"{optimizer}:\tТочность = {score:.4f},\tФинальные потери = {final_loss:.4f},\tВремя обучения = {time_work:.4f} секунд")
