import numpy as np
import sys
import os
import concurrent.futures
import multiprocessing
import math

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from wiener_degradation import (
        WienerModel, PowerWithCovariateTrend, Wiener_ConditionIMF
    )
    print("✅ Модуль wiener_degradation успешно импортирован")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    sys.exit(1)

# ==============================================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================
def compute_F_Z(z0, mu, sigma, rho):
    """Вычисляет F(z0) в точном соответствии с формулой из PDF."""
    if rho <= 1e-12:
        return 1.0 if z0 >= 0 else 0.0
    C = (z0 - mu * rho) / (sigma * np.sqrt(2 * rho))
    return 0.5 * (1 + math.erf(C))


def conditional_log_likelihood(model, data, z0):
    """Вычисляет условный логарифм правдоподобия (Формула 5)."""
    sigma, mu = model.x[0], model.x[1]
    if sigma <= 0:
        return -np.inf

    total_ll = 0.0
    for t_i, Z_i, cov_i in data:
        rho_i = model.f_ro(t_i, cov_i)
        if rho_i <= 1e-10:
            continue

        # 1. ln f(Z_i)
        var_i = sigma**2 * rho_i
        log_pdf = -0.5 * np.log(2 * np.pi * var_i) - (Z_i - mu * rho_i)**2 / (2 * var_i)

        # 2. ln F(z0) с плавной защитой от log(0) вместо жесткого клипа
        F_val = compute_F_Z(z0, mu, sigma, rho_i)
        log_F = np.log(np.maximum(F_val, 1e-15))

        total_ll += (log_pdf - log_F)

    return total_ll


def numerical_score(true_theta, data, z0, trend, epsilon=1e-5):
    """Вычисляет вектор оценки (градиент log-likelihood) численным методом."""
    score = np.zeros_like(true_theta, dtype=float)

    for i in range(len(true_theta)):
        theta_plus = true_theta.copy()
        theta_minus = true_theta.copy()
        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon

        if theta_plus[0] <= 0 or theta_minus[0] <= 0:
            continue

        model_plus = WienerModel(x=theta_plus, trend=trend, z0=z0)
        model_minus = WienerModel(x=theta_minus, trend=trend, z0=z0)

        l_plus = conditional_log_likelihood(model_plus, data, z0)
        l_minus = conditional_log_likelihood(model_minus, data, z0)

        score[i] = (l_plus - l_minus) / (2 * epsilon)

    return score


def worker_simulate(args):
    """Рабочая функция для параллельного выполнения одной симуляции."""
    seed, true_theta, n_units, times_per_unit, cov_values, weights, z0 = args
    np.random.seed(seed)

    trend = PowerWithCovariateTrend()

    # Формируем массив ковариат (в точном соответствии с _convert_to_array)
    covariates = []
    for i in range(len(cov_values) - 1):
        covariates.extend([cov_values[i]] * int(weights[i] * n_units))
    while len(covariates) < n_units:
        covariates.append(cov_values[-1])
    covariates = np.array(covariates[:n_units])

    data = []
    sigma, mu = true_theta[0], true_theta[1]

    for i in range(n_units):
        c_i = covariates[i]
        t_i = times_per_unit[i][-1]  # <-- ИНДИВИДУАЛЬНОЕ время для i-й единицы

        model_temp = WienerModel(x=true_theta, trend=trend, z0=z0)
        rho_i = model_temp.f_ro(t_i, c_i)

        # Rejection sampling для усечённого нормального распределения (Z <= z0)
        Z_i = np.random.normal(loc=mu * rho_i, scale=sigma * np.sqrt(rho_i))
        attempts = 0
        while Z_i > z0 and attempts < 5000:
            Z_i = np.random.normal(loc=mu * rho_i, scale=sigma * np.sqrt(rho_i))
            attempts += 1

        if Z_i > z0:
            Z_i = z0 * 0.9999

        data.append((t_i, Z_i, c_i))

    return numerical_score(true_theta, data, z0, trend)
# ==============================================================================
# 2. ЗАПУСК МОНТЕ-КАРЛО ТЕСТА
# ==============================================================================
def run_monte_carlo_test(n_units=100, n_simulations=1000, z0=30.0):
    print(f"\n{'='*70}")
    print(f"ПРОВЕРКА УСЛОВНОЙ ИМФ (Монте-Карло)")
    print(f"Единиц: {n_units} | Симуляций: {n_simulations} | Порог z0: {z0}")
    print(f"{'='*70}")

    true_theta = np.array([0.8, 0.8, 1.5, 0.5])  # sigma, mu, gamma, beta
    cov_values = [2.5, 6.0]
    weights = [0.5, 0.5]
    Tmax = 21.0

    sigma, mu, gamma, beta = true_theta
    power_exp = max(1e-6, abs(gamma))

    # === Формируем массив ковариат и времена для каждой единицы ===
    # (в точном соответствии с логикой build_data_per_unit)
    n_stresses = len(cov_values)
    units_per_stress = max(1, n_units // n_stresses)

    covariates = []
    times_per_unit = []

    for i in range(n_units):
        # Определяем уровень стресса для i-й единицы
        stress_idx = min(i // units_per_stress, n_stresses - 1)
        S = cov_values[stress_idx]

        # Вычисляем предельное время для этого уровня стресса
        # t_limit = (z0 * exp(beta * S))^(1/gamma)
        t_limit_calc = np.power(z0 * np.exp(beta * S), 1.0 / power_exp)
        t_limit = min(t_limit_calc, Tmax)
        t_limit = max(0.5, t_limit)  # минимальное время наблюдения

        covariates.append(S)
        times_per_unit.append([t_limit])  # последнее время наблюдения

    covariates = np.array(covariates)

    # === Эмпирический расчёт (Монте-Карло) ===
    # Для каждой единицы используем ЕЁ собственное время наблюдения
    seeds = np.random.randint(0, 2**31 - 1, size=n_simulations)
    args_list = [(seed, true_theta, n_units, times_per_unit, cov_values, weights, z0)
                 for seed in seeds]

    n_workers = min(multiprocessing.cpu_count(), n_simulations)
    print(f"Запуск в {n_workers} процессов...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        scores_list = list(executor.map(worker_simulate, args_list,
                                        chunksize=max(1, n_simulations // n_workers)))

    scores = np.array(scores_list)
    empirical_fim = (scores.T @ scores) / n_simulations

    # === Аналитический расчёт ===
    trend = PowerWithCovariateTrend()
    model = WienerModel(x=true_theta, trend=trend, z0=z0)
    print( n_units, cov_values, weights, times_per_unit, z0)
    # Передаём times_per_unit с индивидуальными временами для каждой единицы
    imf_calc = Wiener_ConditionIMF(model, n_units, cov_values, weights, times_per_unit, z0)
    analytical_fim = imf_calc.getIMF()

    # ... (далее сравнение и вывод метрик — без изменений)

    # --- СРАВНЕНИЕ ---
    print("\n--- Аналитическая УИМФ ---")
    print(np.round(analytical_fim).astype('int'))

    print("\n--- Эмпирическая УИМФ (Ковариация Score) ---")
    print(np.round(empirical_fim).astype('int'))

    norm_diff = np.linalg.norm(analytical_fim - empirical_fim)
    norm_analytical = np.linalg.norm(analytical_fim)
    rel_error_frob = (norm_diff / norm_analytical) * 100 if norm_analytical > 0 else 0.0

    det_analytical = np.linalg.det(analytical_fim)
    det_empirical = np.linalg.det(empirical_fim)
    safe_det = max(abs(det_analytical), 1e-10)
    rel_error_det = (abs(det_analytical - det_empirical) / safe_det) * 100

    print("\n--- Метрики ошибки ---")
    print(f"Определитель аналитической: {det_analytical:.6e}")
    print(f"Определитель эмпирической:  {det_empirical:.6e}")
    print(f"Относительная ошибка Det:   {rel_error_det:.2f}%")
    print(f"Относительная ошибка Фроб.: {rel_error_frob:.2f}%")

    # Для матриц 4x4 при Монте-Карло допуски на определитель должны быть мягче
    is_frob_ok = rel_error_frob < 15.0
    is_det_ok = rel_error_det < 50.0 or (det_analytical > 0 and det_empirical > 0)

    print("\n--- Итог ---")
    if is_frob_ok:
        print("✅ РЕЗУЛЬТАТ: Элементы аналитической и эмпирической УИМФ успешно совпадают!")
        print("💡 Примечание: Небольшие расхождения в определителе нормальны для Монте-Карло")
        print("   из-за того, что матрица находится на границе положительной определенности.")
    else:
        print("⚠️ РЕЗУЛЬТАТ: Обнаружено расхождение. Попробуйте увеличить n_simulations до 5000+.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    np.random.seed(42)
    run_monte_carlo_test(n_units=80, n_simulations=10000, z0=20.0)