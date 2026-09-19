import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from wiener_degradation import (
    WienerModel, LinearTrend, PowerTrend,
    PowerWithCovariateTrend, LinearWithCovariateTrend, Wiener_IMF
)
def generate_data(model, n_units, time_points, covariate_values):
    """Генерация деградационных данных для Монте-Карло."""
    times = []
    deltas = []
    covs = []

    for i in range(n_units):
        c = covariate_values[i % len(covariate_values)]
        t_path = time_points

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

        if theta_plus[0] <= 0 or theta_minus[0] <= 0:
            score[i] = 0.0
            continue

        l_plus = model.log_likelihood(theta_plus)
        l_minus = model.log_likelihood(theta_minus)
        score[i] = (l_plus - l_minus) / (2 * epsilon)

    return score

# Импортируйте ваши функции generate_data, numerical_score и т.д.

def run_test_for_M(M_values, n_units=40):
    """Запуск теста для разных M."""
    true_theta = np.array([0.5, 1.0, 1.5, 0.5])
    time_points = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    cov_values = [1.0, 2.0]
    weights = [0.5, 0.5]

    trend = PowerWithCovariateTrend()
    model = WienerModel(x=true_theta, trend=trend, z0=20.0)
    times_list = [time_points.tolist() for _ in range(n_units)]

    # Аналитическая матрица
    model.set_params(true_theta)
    imf_calc = Wiener_IMF(model, n_units, cov_values, weights, times_list)
    analytical_fim = np.array(imf_calc.getIMF())

    results = []
    for M in M_values:
        scores = []
        for _ in range(M):
            data = generate_data(model, n_units, time_points, cov_values)
            model.data = data
            s = numerical_score(model, data, true_theta)
            scores.append(s)

        scores = np.array(scores)
        empirical_fim = np.cov(scores, rowvar=False)

        frob_error = np.linalg.norm(analytical_fim - empirical_fim) / np.linalg.norm(analytical_fim) * 100
        det_anal = np.linalg.det(analytical_fim)
        det_emp = np.linalg.det(empirical_fim)
        det_error = abs(det_anal - det_emp) / det_anal * 100

        results.append((M, frob_error, det_error))
        print(f"M={M}: Frobenius={frob_error:.2f}%, Det={det_error:.2f}%")

    return results

# Запуск анализа
M_values = [1000, 2000, 5000, 10000, 20000]
print("АНАЛИЗ ЗАВИСИМОСТИ ОШИБКИ ОТ ЧИСЛА СИМУЛЯЦИЙ M")
print("="*60)
results = run_test_for_M(M_values)

# Проверка, уменьшается ли ошибка как 1/sqrt(M)
print("\nАНАЛИЗ СКОРОСТИ СХОДИМОСТИ:")
for i in range(1, len(results)):
    M1, err1 = results[i-1][0], results[i-1][2]  # det error
    M2, err2 = results[i][0], results[i][2]
    expected_ratio = np.sqrt(M2/M1)
    actual_ratio = err1/err2
    print(f"M {M1}→{M2}: ожидаемое уменьшение {expected_ratio:.2f}x, фактическое {actual_ratio:.2f}x")