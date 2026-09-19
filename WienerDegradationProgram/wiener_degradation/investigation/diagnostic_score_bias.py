import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from wiener_degradation import (
    WienerModel, LinearTrend, PowerTrend,
    PowerWithCovariateTrend, LinearWithCovariateTrend, Wiener_IMF
)

# Мок тренда (как в ваших тестах)
class PowerCovTrend:
    name = 'power_covariate'
    def func(self, x, t, c=0):
        return (t ** x[2]) * np.exp(x[3] * c)
    def d_func_gamma(self, t, x, c=0):
        return x[1] * (t ** x[2]) * np.log(t) * np.exp(x[3] * c)
    def d_func_beta(self, t, x, c=0):
        return (t ** x[2]) * c * np.exp(x[3] * c)
    def d2_func_gamma(self, t, x, c=0):
        return x[1] * (t ** x[2]) * (np.log(t) ** 2) * np.exp(x[3] * c)
    def d2_func_beta(self, t, x, c=0):
        return (t ** x[2]) * (c ** 2) * np.exp(x[3] * c)
    def d2_func_gamma_beta(self, t, x, c=0):
        return x[1] * (t ** x[2]) * np.log(t) * c * np.exp(x[3] * c)

def generate_data(model, n_units, time_points, covariate_values):
    """Генерация данных без цензурирования."""
    times, deltas, covs = [], [], []
    for i in range(n_units):
        c = covariate_values[i % len(covariate_values)]
        t_path = time_points
        x_path = [0.0]
        current_x = 0.0

        for j in range(len(t_path) - 1):
            t_prev, t_curr = t_path[j], t_path[j+1]
            delta_ro = model.f_ro(t_curr, c) - model.f_ro(t_prev, c)
            increment = np.random.normal(
                loc=model.mu * delta_ro,
                scale=model.sigma * np.sqrt(delta_ro)
            )
            current_x += increment
            x_path.append(current_x)

        times.append(t_path)
        deltas.append(np.diff(x_path))
        covs.append(c)

    return times, deltas, [None]*n_units, covs

def numerical_score(model, data, theta, epsilon=1e-5):
    """Численный градиент log-likelihood."""
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

# Параметры из вашего теста
true_theta = np.array([0.5, 1.0, 1.5, 0.5])
n_units = 40
M = 10000  # Увеличим для точности
time_points = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
cov_values = [1.0, 2.0]

print(f"ДИАГНОСТИКА СМЕЩЕНИЯ SCORE-ВЕКТОРОВ")
print(f"Параметры: {true_theta}, Единиц: {n_units}, Симуляций: {M}")
print("="*60)

trend = PowerCovTrend()
model = WienerModel(x=true_theta, trend=trend, z0=20.0)

scores = []
for _ in range(M):
    data = generate_data(model, n_units, time_points, cov_values)
    model.data = data
    s = numerical_score(model, data, true_theta)
    scores.append(s)

scores = np.array(scores)
mean_score = np.mean(scores, axis=0)
std_score = np.std(scores, axis=0)

print(f"\nСреднее score-вектора (должно быть ~0):")
print(f"σ: {mean_score[0]:+.6f} ± {std_score[0]/np.sqrt(M):.6f}")
print(f"μ: {mean_score[1]:+.6f} ± {std_score[1]/np.sqrt(M):.6f}")
print(f"γ: {mean_score[2]:+.6f} ± {std_score[2]/np.sqrt(M):.6f}")
print(f"β: {mean_score[3]:+.6f} ± {std_score[3]/np.sqrt(M):.6f}")

# Проверка статистической значимости
z_scores = mean_score / (std_score / np.sqrt(M))
print(f"\nZ-статистики (|z| > 1.96 означает значимое смещение):")
print(f"σ: {z_scores[0]:+.2f}")
print(f"μ: {z_scores[1]:+.2f}")
print(f"γ: {z_scores[2]:+.2f}")
print(f"β: {z_scores[3]:+.2f}")

if np.any(np.abs(z_scores) > 1.96):
    print("\n🔴 Обнаружено СТАТИСТИЧЕСКИ ЗНАЧИМОЕ СМЕЩЕНИЕ!")
else:
    print("\n✅ Смещение не обнаружено.")