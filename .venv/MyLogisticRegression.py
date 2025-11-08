import numpy as np
import pandas as pd


class MyLogisticRegression:
    def __init__(self, learning_rate=0.1, max_iter=1000, tol=1e-4, fit_intercept=True, optimizer='gradient_descent'):
        """
        Параметры:
        learning_rate: скорость обучения
        max_iter: максимальное количество итераций
        tol: критерий остановки (изменение функции потерь)
        fit_intercept: добавлять ли intercept (свободный член)
        optimizer: метод оптимизации ('gradient_descent' или 'rmsprop')
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.weights = None
        self.loss_history = []
        self.optimizer = optimizer
        self.s = None # Скользящее среднее квадратов градиентов
        self.beta = 0.9 # параметр для RMSProp (коэффициент затухания)
        self.beta2 = 0.999 # параметр для второго момента (нецентрированная дисперсия)
        self.epsilon = 1e-15 # чсло близкое к 0, но не 0, чтобы не получить log(0)
        # Для адаптивных методов
        self.m = None  # Первый момент (среднее)
        self.v = None  # Второй момент (дисперсия)
        self.t = 0  # Счетчик итераций

    def _sigmoid(self, z):
        """Сигмоидная функция"""
        # Для избежания переполнения
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _add_intercept(self, X):
        """Добавляет столбец единиц для intercept"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _compute_loss(self, y, y_pred):
        """Вычисляет log loss"""
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def _gradient_descent_step(self, X, y, y_pred):
        """Шаг градиентного спуска"""
        error = y_pred - y
        gradient = np.dot(X.T, error) / len(y)
        self.weights -= self.learning_rate * gradient
        return gradient

    def _rmsprop_step(self, X, y, y_pred):
        """Шаг RMSProp оптимизации"""
        error = y_pred - y
        gradient = np.dot(X.T, error) / len(y)

        # Инициализируем s если это первая итерация
        if self.s is None:
            self.s = np.zeros_like(gradient)

        # Обновляем скользящее среднее квадратов градиентов
        self.s = self.beta * self.s + (1 - self.beta) * (gradient ** 2)

        # Обновляем веса с адаптивной скоростью обучения
        adaptive_lr = self.learning_rate / (np.sqrt(self.s) + self.epsilon)
        self.weights -= adaptive_lr * gradient

        return gradient

    def _nadam_step(self, X, y, y_pred):
        """
        Шаг Nadam оптимизации
        Сочетает Nesterov acceleration и Adam
        """
        error = y_pred - y
        gradient = np.dot(X.T, error) / len(y)

        # Инициализация моментов при первой итерации
        if self.m is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)

        self.t += 1  # Увеличиваем счетчик итераций

        # Обновляем моменты
        self.m = self.beta * self.m + (1 - self.beta) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        # Корректируем моменты (bias correction)
        m_hat = self.m / (1 - self.beta ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Nesterov momentum: смотрим немного вперед
        momentum_correction = self.beta * m_hat + (1 - self.beta) * gradient / (1 - self.beta ** self.t)

        # Обновляем веса
        self.weights -= self.learning_rate * momentum_correction / (np.sqrt(v_hat) + self.epsilon)

        return gradient

    def fit(self, X, y):
        """
        Обучение модели

        Параметры:
        X: признаки (n_samples, n_features)
        y: целевая переменная (n_samples,)
        """
        # Преобразуем в numpy массивы
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Добавляем intercept если нужно
        if self.fit_intercept:
            X = self._add_intercept(X)

        # Инициализируем веса
        n_features = X.shape[1]
        self.weights = np.random.normal(0, 0.1, n_features)

        print(f"Используется оптимизатор: {self.optimizer}")

        # Градиентный спуск
        for i in range(self.max_iter):
            # Прямое распространение
            z = np.dot(X, self.weights)
            y_pred = self._sigmoid(z)

            # Выбираем метод оптимизации
            if self.optimizer == 'rmsprop':
                self._rmsprop_step(X, y, y_pred)
            elif self.optimizer == 'nadam':
                gradient = self._nadam_step(X, y, y_pred)
            else: # gradient_descent по умолчанию
                self._gradient_descent_step(X, y, y_pred)

            # Вычисляем loss и проверяем критерий остановки
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # Проверка сходимости
            if i > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                print(f"Сходимость достигнута на итерации {i}")
                break

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        """Возвращает вероятности принадлежности к классу 1"""
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.fit_intercept:
            X = self._add_intercept(X)

        z = np.dot(X, self.weights)
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Возвращает предсказанные классы"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def score(self, X, y):
        """Вычисляет accuracy"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_params(self):
        """Возвращает параметры модели"""
        return {
            'weights': self.weights,
            'loss_history': self.loss_history
        }