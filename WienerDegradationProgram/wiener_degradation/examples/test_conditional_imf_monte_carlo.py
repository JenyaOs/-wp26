import numpy as np
import sys
import os
from scipy.stats import norm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wiener_degradation import (WienerModel,LinearTrend, PowerTrend, PowerWithCovariateTrend, LinearWithCovariateTrend, Wiener_IMF,Wiener_ConditionIMF)


# ==============================================================================
# 2. ФУНКЦИИ ДЛЯ МОНТЕ-КАРЛО (УСЛОВНЫЕ)
# ==============================================================================
def generate_surviving_data(model, n_units, time_points, covariate_values, z):
    """Генерация данных ТОЛЬКО для выживших (не достигших порога z) единиц."""
    valid_times, valid_deltas, valid_covs = [], [], []
    attempts = 0
    max_attempts = n_units * 20 # Защита от бесконечного цикла

    while len(valid_times) < n_units and attempts < max_attempts:
        attempts += 1
        c = covariate_values[len(valid_times) % len(covariate_values)]
        t_path = time_points
        x_path = [0.0]
        current_x = 0.0
        failed = False

        for j in range(len(t_path) - 1):
            t_prev, t_curr = t_path[j], t_path[j+1]
            delta_ro = model.f_ro(t_curr, c) - model.f_ro(t_prev, c)

            increment = np.random.normal(
                loc=model.mu * delta_ro,
                scale=model.sigma * np.sqrt(delta_ro)
            )
            current_x += increment
            x_path.append(current_x)

            if current_x >= z:
                failed = True
                break

        if not failed:
            valid_times.append(t_path)
            valid_deltas.append(np.diff(x_path))
            valid_covs.append(c)

    return valid_times, valid_deltas, [None]*len(valid_times), valid_covs

def conditional_log_likelihood_exact(model, data, theta):
    """
    Точное условное логарифмическое правдоподобие (Ground Truth для численного дифференцирования).
    Использует точную формулу обратного гауссовского распределения.
    """
    ll_uncond = model.log_likelihood(theta)
    log_R_sum = 0.0

    for i in range(len(data[0])):
        t_last = data[0][i][-1]
        c = data[3][i]
        rho = model.f_ro(t_last, c)
        mu = theta[1]
        sigma = theta[0]
        z = model.z

        # ТОЧНАЯ формула функции надёжности (Inverse Gaussian Survival Function)
        C = (z - mu * rho) / (sigma * np.sqrt(rho))
        # Защита от переполнения экспоненты
        exp_arg = 2 * mu * z / (sigma**2)
        if exp_arg > 700:
            exp_term = np.exp(700)
        else:
            exp_term = np.exp(exp_arg)

        term1 = norm.cdf(C)
        term2 = exp_term * norm.cdf(C - 2 * z / (sigma * np.sqrt(rho)))

        R = term1 - term2
        R = max(R, 1e-300) # Защита от log(0)
        log_R_sum += np.log(R)

    return ll_uncond - log_R_sum

def numerical_conditional_score(model, data, theta, epsilon=1e-5):
    """Численный градиент условного логарифмического правдоподобия."""
    score = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon

        if theta_plus[0] <= 0 or theta_minus[0] <= 0:
            score[i] = 0.0
            continue

        l_plus = conditional_log_likelihood_exact(model, data, theta_plus)
        l_minus = conditional_log_likelihood_exact(model, data, theta_minus)
        score[i] = (l_plus - l_minus) / (2 * epsilon)

    return score

def run_conditional_monte_carlo_test(model_name, trend_class, true_theta, n_units=50, n_simulations=1000):
    print(f"\n{'='*70}")
    print(f"ТЕСТ (УСЛОВНЫЙ): {model_name.upper()}")
    print(f"Истинные параметры: {true_theta}")
    print(f"Симуляций (выживших): {n_simulations}, Единиц в симуляции: {n_units}")
    print(f"{'='*70}")

    trend = trend_class()
    model = WienerModel(x=true_theta, trend=trend, z0=50.0)

    time_points = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    cov_values = [1.0, 2.0] if 'cov' in model_name else [0.0]
    weights = [0.5, 0.5] if len(cov_values) == 2 else [1.0]

    scores = []

    print("Генерация выживших траекторий и оценка градиентов... (может занять 1-2 минуты)")
    for _ in range(n_simulations):
        data = generate_surviving_data(model, n_units, time_points, cov_values, model.z)
        model.data = data

        s = numerical_conditional_score(model, data, true_theta)
        scores.append(s)

    scores = np.array(scores)
    empirical_cond_fim = np.cov(scores, rowvar=False)

    # Аналитическая условная ИМФ из вашего класса
    times_list = [time_points.tolist() for _ in range(n_units)]
    imf_calc = Wiener_ConditionIMF(model, n_units, cov_values, weights, times_list, model.z)
    analytical_cond_fim = np.array(imf_calc.getIMF())

    print("\n--- Сравнение матриц ---")
    print("Аналитическая условная ИМФ (Wiener_ConditionIMF):")
    print(np.round(analytical_cond_fim, 4))

    print("\nЭмпирическая условная ИМФ (Ковариация условных Score-векторов):")
    print(np.round(empirical_cond_fim, 4))

    norm_diff = np.linalg.norm(analytical_cond_fim - empirical_cond_fim)
    norm_analytical = np.linalg.norm(analytical_cond_fim)
    rel_error = (norm_diff / norm_analytical) * 100 if norm_analytical > 0 else 100.0

    print(f"\nОтносительная ошибка (норма Фробениуса): {rel_error:.2f}%")

    if rel_error < 10.0: # Порог чуть выше, т.к. дисперсия условных оценок больше
        print("✅ РЕЗУЛЬТАТ: Матрицы совпадают.")
    else:
        print("🔴 РЕЗУЛЬТАТ: Обнаружено КРИТИЧЕСКОЕ расхождение! Формулы условной ИМФ неверны.")

    return rel_error

# ==============================================================================
# 3. ЗАПУСК ТЕСТОВ
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    errors = []

    # Примечание: для условного теста берем меньше симуляций (1000),
    # т.к. генерация с отбраковкой и численное дифференцирование сложнее.

    errors.append(run_conditional_monte_carlo_test(
        model_name="linear", trend_class=LinearTrend,
        true_theta=np.array([0.5, 1.0]), n_simulations=1000
    ))

    errors.append(run_conditional_monte_carlo_test(
        model_name="power", trend_class=PowerTrend,
        true_theta=np.array([0.5, 1.0, 1.2]), n_simulations=1000
    ))

    errors.append(run_conditional_monte_carlo_test(
        model_name="linear_covariate", trend_class=LinearWithCovariateTrend,
        true_theta=np.array([0.5, 1.0, 0.3]), n_simulations=1000
    ))

    errors.append(run_conditional_monte_carlo_test(
        model_name="power_covariate", trend_class=PowerWithCovariateTrend,
        true_theta=np.array([0.5, 1.0, 1.2, 0.3]), n_simulations=1000
    ))

    print(f"\n{'='*70}")
    print("ИТОГОВАЯ СВОДКА (УСЛОВНАЯ ИМФ)")
    print(f"{'='*70}")
    if max(errors) < 10.0:
        print("🎉 Условные аналитические формулы прошли проверку!")
    else:
        print("🔴 Найдены критические расхождения. Требуется замена формул F, d_C, d2_C и т.д. на точные (Inverse Gaussian).")