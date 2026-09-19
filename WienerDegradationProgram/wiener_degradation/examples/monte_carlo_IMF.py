import numpy as np
import sys
import os

# Добавляем текущую директорию в путь для импорта ваших модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from wiener_degradation import (WienerModel,LinearTrend, PowerTrend, PowerWithCovariateTrend, LinearWithCovariateTrend, Wiener_IMF)



# ==============================================================================
# 2. ФУНКЦИИ ДЛЯ МОНТЕ-КАРЛО
# ==============================================================================
def generate_data(model, n_units, time_points, covariate_values):
    """Генерация деградационных данных для Монте-Карло."""
    times = []
    deltas = []
    covs = []

    for i in range(n_units):
        c = covariate_values[i % len(covariate_values)]
        t_path = time_points

        # ИСПРАВЛЕНО: начинаем с X(0) = 0
        x_path = [0.0]
        current_x = 0.0

        for j in range(len(t_path) - 1):
            t_prev, t_curr = t_path[j], t_path[j+1]
            ro_prev = model.f_ro(t_prev, c)
            ro_curr = model.f_ro(t_curr, c)
            delta_ro = ro_curr - ro_prev

            increment = np.random.normal(
                loc=model.mu * delta_ro,
                scale=model.sigma * np.sqrt(delta_ro)
            )
            current_x += increment
            x_path.append(current_x)

        times.append(t_path)
        # Теперь len(x_path) = len(t_path), а len(np.diff(x_path)) = len(t_path) - 1
        deltas.append(np.diff(x_path))
        covs.append(c)

    return times, deltas, [None] * n_units, covs

def numerical_score(model, data, theta, epsilon=1e-5):
    """Вычисляет вектор оценки (градиент log-likelihood) численным методом."""
    score = np.zeros_like(theta)
    l0 = model.log_likelihood(theta)

    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon

        # Защита от недопустимых параметров (sigma > 0, mu > 0)
        if theta_plus[0] <= 0 or theta_minus[0] <= 0:
            score[i] = 0.0
            continue

        l_plus = model.log_likelihood(theta_plus)
        l_minus = model.log_likelihood(theta_minus)
        score[i] = (l_plus - l_minus) / (2 * epsilon)

    return score

def run_monte_carlo_test(model_name, trend_class, true_theta, n_units=50, n_simulations=2000):
    """Запуск МК теста для одной модели."""
    print(f"\n{'='*70}")
    print(f"ТЕСТ: {model_name.upper()}")
    print(f"Истинные параметры: {true_theta}")
    print(f"Симуляций: {n_simulations}, Единиц: {n_units}")
    print(f"{'='*70}")

    trend = trend_class()
    model = WienerModel(x=true_theta, trend=trend, z0=50.0)

    # Фиксируем план эксперимента для всех симуляций
    time_points = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    cov_values = [1.0, 2.0] if 'cov' in model_name else [0.0]

    scores = []
    theta_hats = []

    print("Генерация и оценка... (это может занять минуту)")
    for _ in range(n_simulations):
        data = generate_data(model, n_units, time_points, cov_values)
        model.data = data

        # 1. Вектор оценки при истинных параметрах
        s = numerical_score(model, data, true_theta)
        scores.append(s)

    scores = np.array(scores)

    # Эмпирическая ИМФ = Ковариация векторов оценки
    empirical_fim = np.cov(scores, rowvar=False)

    # Аналитическая ИМФ из вашего класса
    # Веса: равномерные, например, 0.5 и 0.5 для двух ковариат
    weights = [0.5, 0.5] if len(cov_values) == 2 else [1.0]
    times_list = [time_points.tolist() for _ in range(n_units)]

    imf_calculator = Wiener_IMF(model, n_units, cov_values, weights, times_list)
    analytical_fim = np.array(imf_calculator.getIMF())

    # Сравнение
    print("\n--- Сравнение матриц ---")
    print("Аналитическая ИМФ (Wiener_IMF):")
    print(np.round(analytical_fim, 4))

    print("\nЭмпирическая ИМФ (Ковариация Score-векторов):")
    print(np.round(empirical_fim, 4))

    # Относительная ошибка (по норме Фробениуса)
    norm_diff = np.linalg.norm(analytical_fim - empirical_fim)
    norm_analytical = np.linalg.norm(analytical_fim)
    rel_error = (norm_diff / norm_analytical) * 100

    print(f"\nОтносительная ошибка (норма Фробениуса): {rel_error:.2f}%")

    if rel_error < 5.0:
        print("✅ РЕЗУЛЬТАТ: Матрицы совпадают в пределах статистической погрешности Монте-Карло.")
    else:
        print("⚠️ РЕЗУЛЬТАТ: Обнаружено значимое расхождение! Требуется проверка аналитических формул.")

    return rel_error

# ==============================================================================
# 3. ЗАПУСК ТЕСТОВ ДЛЯ ВСЕХ 4 МОДЕЛЕЙ
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42) # Для воспроизводимости

    # Параметры: [sigma, mu] или [sigma, mu, gamma] или [sigma, mu, gamma, beta]
    # Внимание: sigma и mu должны быть > 0 согласно вашим проверкам в log_likelihood

    errors = []

    errors.append(run_monte_carlo_test(
        model_name="linear",
        trend_class=LinearTrend,
        true_theta=np.array([0.5, 1.0]),
        n_simulations=2000
    ))

    errors.append(run_monte_carlo_test(
        model_name="power",
        trend_class=PowerTrend,
        true_theta=np.array([0.5, 1.0, 1.2]),
        n_simulations=2000
    ))

    errors.append(run_monte_carlo_test(
        model_name="linear_covariate",
        trend_class=LinearWithCovariateTrend,
        true_theta=np.array([0.5, 1.0, 0.3]),
        n_simulations=2000
    ))

    errors.append(run_monte_carlo_test(
        model_name="power_covariate",
        trend_class=PowerWithCovariateTrend,
        true_theta=np.array([0.5, 1.0, 1.2, 0.3]),
        n_simulations=2000
    ))

    print(f"\n{'='*70}")
    print("ИТОГОВАЯ СВОДКА")
    print(f"{'='*70}")
    if max(errors) < 5.0:
        print("🎉 Все аналитические формулы в Wiener_IMF прошли проверку Монте-Карло!")
    else:
        print("🔴 Найдены расхождения. Необходимо проверить аналитические производные в imf.py для моделей с ошибкой > 5%.")