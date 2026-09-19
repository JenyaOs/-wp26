"""
ИССЛЕДОВАНИЕ КОРРЕКТНОСТИ АНАЛИТИЧЕСКИХ ФОРМУЛ ИМФ
Сравнение: Аналитическая ИМФ I(θ₀)  vs  Обратная эмпирическая ковариация Cov(θ̂)⁻¹
"""

import numpy as np
import sys
import os
import concurrent.futures
import multiprocessing
import time

# Настройка красивого вывода матриц (6 знаков после запятой, без научной нотации)
np.set_printoptions(precision=6, suppress=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from wiener_degradation import (
    WienerModel, PowerWithCovariateTrend, Wiener_IMF,
    ParameterEstimator
)

# ==============================================================================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
# ==============================================================================
FIXED_TIME_POINTS = list(range(0, 21, 2))
FIXED_COV_VALUES = [2.5, 6.0]
FIXED_WEIGHTS = [0.5, 0.5]

TRUE_THETA = np.array([0.5, 0.8, 1.5, 0.5])
Z0 = 30.0

N_SIMULATIONS = 10000  # Можно увеличить до 10000 для финального отчета
UNITS_TO_TEST = [80]

MAX_ERROR_FROB = 10.0  # Ужесточили допуск, так как сравниваем напрямую
MAX_ERROR_DET = 15.0


# ==============================================================================
# 1. БЫСТРАЯ ВЕКТОРИЗОВАННАЯ ГЕНЕРАЦИЯ ДАННЫХ
# ==============================================================================
def generate_data_fixed_plan_fast(n_units, time_points, cov_values, weights, true_theta, z0):
    trend = PowerWithCovariateTrend()
    model = WienerModel(x=true_theta, trend=trend, z0=z0)

    covariates = []
    for i in range(len(cov_values) - 1):
        covariates.extend([cov_values[i]] * int(weights[i] * n_units))
    while len(covariates) < n_units:
        covariates.append(cov_values[-1])
    covariates = np.array(covariates[:n_units])

    n_times = len(time_points)
    delta_ro_matrix = np.zeros((n_units, n_times - 1))
    for i in range(n_units):
        c = covariates[i]
        ro = np.array([model.f_ro(t, c) for t in time_points])
        delta_ro_matrix[i, :] = np.diff(ro)

    loc_matrix = model.mu * delta_ro_matrix
    scale_matrix = model.sigma * np.sqrt(np.maximum(delta_ro_matrix, 1e-10))
    increments = np.random.normal(loc=loc_matrix, scale=scale_matrix)

    return [time_points] * n_units, increments.tolist(), [None] * n_units, covariates.tolist()


# ==============================================================================
# 2. РАБОЧАЯ ФУНКЦИЯ
# ==============================================================================
def worker_simulate(args):
    seed, true_theta, n_units, z0 = args
    np.random.seed(seed)

    from wiener_degradation import WienerModel, PowerWithCovariateTrend, ParameterEstimator

    trend = PowerWithCovariateTrend()
    model = WienerModel(x=true_theta, trend=trend, z0=z0)

    data = generate_data_fixed_plan_fast(
        n_units, FIXED_TIME_POINTS, FIXED_COV_VALUES,
        FIXED_WEIGHTS, true_theta, z0
    )

    estimator = ParameterEstimator(model)
    try:
        rng = np.random.default_rng(seed)
        noise = 0  #rng.normal(0, true_theta * 0.1)
        x0 = np.maximum(true_theta + noise, 1e-3)

        theta_hat = estimator.estimate(x0, data)

        success = getattr(estimator, 'success', True)
        if not success or np.any(np.isnan(theta_hat)) or np.any(theta_hat <= 0):
            return np.full_like(true_theta, np.nan), False

        return np.array(theta_hat), True
    except Exception:
        return np.full_like(true_theta, np.nan), False


# ==============================================================================
# 3. ФУНКЦИЯ МОНТЕ-КАРЛО С ПРЯМЫМ СРАВНЕНИЕМ МАТРИЦ
# ==============================================================================
def run_monte_carlo_test(n_units):
    print(f"\n{'='*85}")
    print(f"ТЕСТ: n_units = {n_units} | Симуляций: {N_SIMULATIONS}")
    print(f"{'='*85}")

    seeds = np.random.randint(0, 2**31 - 1, size=N_SIMULATIONS)
    args_list = [(seed, TRUE_THETA, n_units, Z0) for seed in seeds]

    start_time = time.time()
    print(f"Запуск вычислений в {multiprocessing.cpu_count()} потоков...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results_list = list(executor.map(worker_simulate, args_list, chunksize=500))

    elapsed = time.time() - start_time
    print(f"Вычисления завершены за {elapsed:.1f} сек.")

    theta_hats_all = np.array([r[0] for r in results_list])
    success_flags = np.array([r[1] for r in results_list])

    n_success = np.sum(success_flags)
    success_rate = 100 * n_success / N_SIMULATIONS
    print(f"Сходимость MLE: {n_success}/{N_SIMULATIONS} ({success_rate:.1f}%)")

    theta_hats = theta_hats_all[success_flags]
    if len(theta_hats) < 100:
        print("❌ ОШИБКА: Недостаточно успешных симуляций")
        return None

    # --- 1. ЭМПИРИЧЕСКАЯ КОВАРИАЦИЯ И ЕЁ ОБРАТНАЯ МАТРИЦА ---
    empirical_cov_mle = np.cov(theta_hats, rowvar=False)
    mean_theta_hat = np.mean(theta_hats, axis=0)
    bias = np.abs(mean_theta_hat - TRUE_THETA) / TRUE_THETA * 100

    try:
        # ✅ КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Берем обратную матрицу от эмпирической ковариации
        empirical_fim = np.linalg.inv(empirical_cov_mle)
    except np.linalg.LinAlgError:
        print("❌ ОШИБКА: Эмпирическая ковариационная матрица вырождена.")
        return None

    # --- 2. АНАЛИТИЧЕСКАЯ ИМФ ---
    trend = PowerWithCovariateTrend()
    model = WienerModel(x=TRUE_THETA, trend=trend, z0=Z0)
    times_list = [FIXED_TIME_POINTS for _ in range(n_units)]

    imf_calculator = Wiener_IMF(model, n_units, FIXED_COV_VALUES, FIXED_WEIGHTS, times_list)
    analytical_fim = np.array(imf_calculator.getIMF())

    # ========================================================================
    # 📊 ЯВНЫЙ ВЫВОД ВСЕХ МАТРИЦ ДЛЯ ПРЯМОГО СРАВНЕНИЯ
    # ========================================================================
    print("\n" + "📊" + "="*83)
    print(f"ПРЯМОЕ СРАВНЕНИЕ МАТРИЦ (n = {n_units})")
    print("="*85)

    print("\n[1] АНАЛИТИЧЕСКАЯ ИМФ I(θ₀)  (из ваших формул):")
    print(np.round(analytical_fim, 4))
    sign_a, logdet_a = np.linalg.slogdet(analytical_fim)
    det_analytical = float(sign_a * np.exp(logdet_a))

    print("определите",det_analytical)


    print("\n[2] ОБРАТНАЯ ЭМПИРИЧЕСКАЯ КОВАРИАЦИЯ Cov(θ̂)⁻¹:")
    print(np.round(empirical_fim, 4))

    sign_e, logdet_e = np.linalg.slogdet(empirical_fim)
    det_empirical = float(sign_e * np.exp(logdet_e))
    print("определитель",det_empirical)

    print("\n[3] МАТРИЦА РАЗНОСТИ (Аналитическая I(θ₀)  −  Обратная Эмпирическая):")
    diff_matrix = analytical_fim - empirical_fim
    print(np.round(diff_matrix, 4))
    print("="*85 + "\n")

    # --- 3. СРАВНЕНИЕ (МЕТРИКИ) ---
    # Ошибка по норме Фробениуса между I(θ₀) и Cov⁻¹
    norm_diff = np.linalg.norm(diff_matrix)
    norm_analytical = np.linalg.norm(analytical_fim)
    rel_error_frob = (norm_diff / norm_analytical) * 100




    safe_det = max(abs(det_analytical), 1e-10)
    rel_error_det = (abs(det_analytical - det_empirical) / safe_det) * 100

    is_frob_ok = rel_error_frob < MAX_ERROR_FROB
    is_det_ok = rel_error_det < MAX_ERROR_DET
    is_unbiased = np.all(bias < 5.0)
    success = is_frob_ok and is_det_ok and is_unbiased

    print(f"📈 МЕТРИКИ СОВПАДЕНИЯ:")
    print(f"  • Ошибка по норме Фробениуса: {rel_error_frob:.2f}%  (допуск < {MAX_ERROR_FROB}%)")
    print(f"  • Ошибка определителя (Det):   {rel_error_det:.2f}%  (допуск < {MAX_ERROR_DET}%)")
    print(f"  • Максимальное смещение (Bias): {np.max(bias):.2f}%   (допуск < 5.0%)")
    print(f"\n  🏁 ИТОГОВЫЙ СТАТУС: {'✅ ПРОЙДЕН (Формулы корректны)' if success else '⚠️ ТРЕБУЕТ ПРОВЕРКИ'}")

    return {
        "n_units": n_units,
        "success_rate": success_rate,
        "rel_error_frob": rel_error_frob,
        "rel_error_det": rel_error_det,
        "det_analytical": det_analytical,
        "det_empirical": det_empirical,
        "bias": bias,
        "success": success
    }


# ==============================================================================
# 4. ЗАПУСК
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)

    print(f"🚀 ЗАПУСК ИССЛЕДОВАНИЯ ИМФ")
    print(f"Параметры: θ = {TRUE_THETA}, z₀ = {Z0}, Симуляций = {N_SIMULATIONS}")
    print(f"Метод: Прямое сравнение I(θ₀) и Cov(θ̂)⁻¹")

    results = []
    total_start = time.time()

    for n_units in UNITS_TO_TEST:
        res = run_monte_carlo_test(n_units)
        if res:
            results.append(res)

    total_elapsed = time.time() - total_start
    print(f"\n⏱️ Общее время выполнения: {total_elapsed/60:.1f} минут")

    # Итоговая сводная таблица
    print(f"\n{'='*100}")
    print(f"ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print(f"{'='*100}")
    print(f"{'n':<5} | {'Успех%':<8} | {'Det Аналит.':<14} | {'Det Эмпирич.⁻¹':<16} | {'Err Det%':<10} | {'Err Frob%':<10} | {'Статус'}")
    print("-" * 100)

    for res in results:
        status = "✅ OK" if res["success"] else "⚠️ FAIL"
        print(f"{res['n_units']:<5} | {res['success_rate']:<8.1f} | "
              f"{res['det_analytical']:<14.4e} | {res['det_empirical']:<16.4e} | "
              f"{res['rel_error_det']:<10.2f} | {res['rel_error_frob']:<10.2f} | {status}")
    print(f"{'='*100}")