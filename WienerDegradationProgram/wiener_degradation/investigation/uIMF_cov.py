import numpy as np

import os
import concurrent.futures
import multiprocessing

from scipy.optimize import minimize

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    """Вычисляет F(z0) = P(Z <= z0) через функцию erf."""
    if rho <= 1e-12:
        return 1.0 if z0 >= 0 else 0.0
    C = (z0 - mu * rho) / (sigma * np.sqrt(2 * rho))
    return 0.5 * (1 + math.erf(C))


def conditional_log_likelihood(theta, data, z0, trend):
    """
    Вычисляет условный логарифм правдоподобия.
    """
    sigma, mu = theta[0], theta[1]
    if sigma <= 1e-10:
        return -1e10

    total_ll = 0.0
    for t_i, Z_i, cov_i in data:
        # ВАЖНО: проверяем сигнатуру trend.f_ro
        try:
            rho_i = trend.f_ro(t_i, cov_i, theta)
        except TypeError:
            # Если сигнатура не совпадает, пробуем другой вариант
            rho_i = trend.f_ro(t_i, theta, cov_i)

        if rho_i <= 1e-12:
            continue

        # 1. ln f(Z_i)
        var_i = sigma ** 2 * rho_i
        log_pdf = -0.5 * np.log(2 * np.pi * var_i) - (Z_i - mu * rho_i) ** 2 / (2 * var_i)

        # 2. ln F(z0)
        F_val = compute_F_Z(z0, mu, sigma, rho_i)
        log_F = np.log(np.maximum(F_val, 1e-15))

        total_ll += (log_pdf - log_F)

    return total_ll


# ==============================================================================
# 2. ГЕНЕРАЦИЯ ДАННЫХ
# ==============================================================================
def generate_conditional_data(true_theta, n_units, times_per_unit, cov_values, weights, z0):
    """
    Генерирует данные из УСЛОВНОГО распределения (усечение сверху z0).
    """
    trend = PowerWithCovariateTrend()

    sigma, mu = true_theta[0], true_theta[1]

    # Формируем массив ковариат
    covariates = []
    for i in range(len(cov_values) - 1):
        covariates.extend([cov_values[i]] * int(weights[i] * n_units))
    while len(covariates) < n_units:
        covariates.append(cov_values[-1])
    covariates = np.array(covariates[:n_units])

    data = []
    n_truncated = 0  # Счётчик усечённых значений

    for i in range(n_units):
        c_i = covariates[i]
        t_i = times_per_unit[i][-1]

        # Проверяем сигнатуру trend.f_ro
        try:
            rho_i = trend.f_ro(t_i, c_i)
        except TypeError:
            rho_i = trend.f_ro(t_i, c_i)

        # Rejection sampling для усечённого нормального распределения (Z <= z0)
        Z_i = np.random.normal(loc=mu * rho_i, scale=sigma * np.sqrt(rho_i))
        attempts = 0
        while Z_i > z0 and attempts < 5000:
            Z_i = np.random.normal(loc=mu * rho_i, scale=sigma * np.sqrt(rho_i))
            attempts += 1

        if Z_i > z0:
            Z_i = z0 * 0.9999
            n_truncated += 1

        data.append((t_i, Z_i, c_i))

    return data, n_truncated


# ==============================================================================
# 3. ПОИСК ОЦЕНКИ МАКСИМАЛЬНОГО ПРАВДОПОДОБИЯ (MLE)
# ==============================================================================
def find_mle(true_theta, data, z0, verbose=False):
    """
    Находит оценку максимального правдоподобия theta_hat.
    """
    trend = PowerWithCovariateTrend()

    def neg_ll(theta):
        if theta[0] <= 1e-8:
            return 1e10
        ll = conditional_log_likelihood(theta, data, z0, trend)
        if np.isnan(ll) or np.isinf(ll):
            return 1e10
        return -ll

    # Начальное приближение — истинные параметры (без шума для стабильности)
    theta_init = true_theta.copy()
    theta_init[0] = max(0.1, theta_init[0])  # sigma > 0

    # Ограничения: sigma > 0
    bounds = [(1e-6, None), (None, None), (None, None), (None, None)]

    try:
        result = minimize(
            neg_ll,
            theta_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        if verbose:
            print(f"  Оптимизация: success={result.success}, fun={result.fun:.4f}, "
                  f"nit={result.nit}, message={result.message}")

        if result.success and result.fun < 1e9:
            return result.x
        else:
            if verbose:
                print(f"  ❌ Оптимизация не сошлась: {result.message}")
            return None
    except Exception as e:
        if verbose:
            print(f"  ❌ Исключение при оптимизации: {e}")
        return None


# ==============================================================================
# 4. РАБОЧАЯ ФУНКЦИЯ ДЛЯ ПАРАЛЛЕЛЬНОГО ВЫПОЛНЕНИЯ
# ==============================================================================
def worker_simulate(args):
    """Одна симуляция: генерация данных → поиск MLE."""
    seed, true_theta, n_units, times_per_unit, cov_values, weights, z0, verbose = args
    np.random.seed(seed)

    data, n_truncated = generate_conditional_data(true_theta, n_units, times_per_unit,
                                                  cov_values, weights, z0)

    if verbose:
        print(f"[Seed {seed}] Сгенерировано данных: {len(data)}, усечённых: {n_truncated}")

    # Если слишком много усечённых значений, данные неинформативны
    if n_truncated > n_units * 0.5:
        if verbose:
            print(f"  ⚠️ Слишком много усечённых значений ({n_truncated}/{n_units}), пропускаем")
        return None, n_truncated

    theta_hat = find_mle(true_theta, data, z0, verbose=verbose)
    return theta_hat, n_truncated


# ==============================================================================
# 5. ЗАПУСК МОНТЕ-КАРЛО ТЕСТА
# ==============================================================================
def run_monte_carlo_test(n_units=100, n_simulations=200, z0=30.0):
    print(f"\n{'='*70}")
    print(f"ПРОВЕРКА УСЛОВНОЙ ИМФ ЧЕРЕЗ ОБРАТНУЮ МАТРИЦУ (Монте-Карло)")
    print(f"Единиц: {n_units} | Симуляций: {n_simulations} | Порог z0: {z0}")
    print(f"{'='*70}")

    # ПАРАМЕТРЫ: подобраны так, чтобы mu*rho было ЗАМЕТНО МЕНЬШЕ z0
    # Для S=0: rho = t^1.0 * exp(0) = t. При t=10: mu*rho = 1.0*10 = 10 << 30
    # Для S=2: rho = t^1.0 * exp(0.2) ≈ 1.22*t. При t=10: mu*rho = 1.0*12.2 = 12.2 << 30
    true_theta = np.array([1.0, 1.0, 1.0, 0.2])  # sigma, mu, gamma, beta
    cov_values = [0.0, 2.0]
    weights = [0.5, 0.5]
    Tmax = 15.0  # Уменьшено время наблюдения

    sigma, mu, gamma, beta = true_theta
    power_exp = max(1e-6, abs(gamma))

    # === Формируем массив ковариат и индивидуальные времена для каждой единицы ===
    n_stresses = len(cov_values)
    units_per_stress = max(1, n_units // n_stresses)

    times_per_unit = []
    for i in range(n_units):
        stress_idx = min(i // units_per_stress, n_stresses - 1)
        S = cov_values[stress_idx]
        t_limit_calc = np.power(z0 * np.exp(beta * S), 1.0 / power_exp)
        t_limit = min(t_limit_calc, Tmax)
        t_limit = max(0.5, t_limit)
        times_per_unit.append([t_limit])

    # Проверяем ожидаемые значения mu*rho
    print("\n--- Проверка параметров ---")
    for stress_idx, S in enumerate(cov_values):
        t_test = times_per_unit[stress_idx * units_per_stress][-1]
        trend1 = PowerWithCovariateTrend()
        trend = WienerModel(x=true_theta, trend=trend1, z0=z0)

        rho_test = trend.f_ro(t_test, S)
        expected_Z = mu * rho_test
        print(f"  S={S}: t={t_test:.2f}, rho={rho_test:.2f}, E[Z]={expected_Z:.2f} (z0={z0})")
        if expected_Z > z0 * 0.8:
            print(f"  ⚠️ ВНИМАНИЕ: E[Z] близко к z0! Усечение будет сильным.")

    # === Запуск симуляций ===
    seeds = np.random.randint(0, 2**31 - 1, size=n_simulations)
    args_list = [(seed, true_theta, n_units, times_per_unit,
                  cov_values, weights, z0, True) for seed in seeds[:5]]  # Первые 5 с verbose

    # Остальные без verbose
    args_list += [(seed, true_theta, n_units, times_per_unit,
                   cov_values, weights, z0, False) for seed in seeds[5:]]

    n_workers = min(multiprocessing.cpu_count(), n_simulations)
    print(f"\nЗапуск в {n_workers} процессов...")
    print("⚠️  Внимание: каждая симуляция включает численную оптимизацию — это займёт время.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
         results = list(executor.map(worker_simulate, args_list,
                                    chunksize=max(1, n_simulations // n_workers)))

    # Разделяем результаты
    theta_hats = [r[0] for r in results]
    n_truncated_list = [r[1] for r in results]

    # === Фильтрация неудачных оптимизаций ===
    valid_thetas = [th for th in theta_hats if th is not None]
    n_failed = n_simulations - len(valid_thetas)

    avg_truncated = np.mean(n_truncated_list)
    print(f"\nУспешных оптимизаций: {len(valid_thetas)} из {n_simulations} "
          f"(неудачных: {n_failed})")
    print(f"Среднее число усечённых значений на симуляцию: {avg_truncated:.1f} из {n_units}")

    if len(valid_thetas) < 50:
        print("❌ Слишком мало успешных оптимизаций.")
        if avg_truncated > n_units * 0.3:
            print("💡 Причина: слишком сильное усечение. Уменьшите mu или увеличьте z0.")
        else:
            print("💡 Причина: проблемы с оптимизацией. Проверьте формулы тренда.")
        return

    theta_matrix = np.array(valid_thetas)

    # === Эмпирическая ковариация оценок параметров ===
    empirical_cov = np.cov(theta_matrix, rowvar=False)

    # === Аналитический расчёт ИМФ и её обратной ===
    trend = PowerWithCovariateTrend()
    model = WienerModel(x=true_theta, trend=trend, z0=z0)
    imf_calc = Wiener_ConditionIMF(model, n_units, cov_values, weights, times_per_unit, z0)
    analytical_fim = imf_calc.getIMF()

    try:
        analytical_inv_fim = np.linalg.inv(analytical_fim)
    except np.linalg.LinAlgError:
        print("❌ Аналитическая ИМФ вырождена, невозможно взять обратную матрицу.")
        return

    # === СРАВНЕНИЕ ===
    print("\n--- Аналитическая УИМФ I(θ) ---")
    print(np.round(analytical_fim, 2))

    print("\n--- Обратная аналитическая ИМФ I(θ)⁻¹ (теоретическая ковариация θ̂) ---")
    print(np.round(analytical_inv_fim, 4))

    print("\n--- Эмпирическая ковариация оценок θ̂ ---")
    print(np.round(empirical_cov, 4))

    # === Метрики ошибки ===
    norm_diff = np.linalg.norm(analytical_inv_fim - empirical_cov)
    norm_analytical = np.linalg.norm(analytical_inv_fim)
    rel_error_frob = (norm_diff / norm_analytical) * 100 if norm_analytical > 0 else 0.0

    det_analytical_inv = np.linalg.det(analytical_inv_fim)
    det_empirical = np.linalg.det(empirical_cov)
    safe_det = max(abs(det_analytical_inv), 1e-20)
    rel_error_det = (abs(det_analytical_inv - det_empirical) / safe_det) * 100

    # Сравнение средних оценок с истинными параметрами
    mean_theta = np.mean(theta_matrix, axis=0)
    rel_bias = np.abs(mean_theta - true_theta) / np.abs(true_theta) * 100

    print("\n--- Метрики ошибки ---")
    print(f"Определитель I(θ)⁻¹:            {det_analytical_inv:.6e}")
    print(f"Определитель эмпирической Cov:  {det_empirical:.6e}")
    print(f"Относительная ошибка Det:       {rel_error_det:.2f}%")
    print(f"Относительная ошибка Фроб.:     {rel_error_frob:.2f}%")

    print("\n--- Проверка несмещённости MLE ---")
    print(f"Истинные параметры θ:    {true_theta}")
    print(f"Среднее оценок θ̂:       {np.round(mean_theta, 4)}")
    print(f"Относительное смещение:  {np.round(rel_bias, 2)}%")

    # === Итог ===
    print("\n--- Итог ---")
    is_frob_ok = rel_error_frob < 25.0
    is_bias_ok = np.all(rel_bias < 10.0)

    if is_frob_ok and is_bias_ok:
        print("✅ РЕЗУЛЬТАТ: Условная ИМФ успешно проверена через обратную матрицу!")
        print("💡 Ковариация оценок MLE совпадает с I(θ)⁻¹ в пределах статистической точности.")
    else:
        print("⚠️ РЕЗУЛЬТАТ: Обнаружено расхождение.")
        if not is_frob_ok:
            print(f"   - Ошибка Фробениуса {rel_error_frob:.2f}% > 25%")
        if not is_bias_ok:
            print(f"   - Смещение оценок {np.max(rel_bias):.2f}% > 10%")
        print("💡 Рекомендации:")
        print("   - Увеличьте n_simulations до 500-1000")
        print("   - Увеличьте n_units до 200-500 для более точных MLE")


if __name__ == "__main__":
    #multiprocessing.freeze_support()
    #np.random.seed(42)

    run_monte_carlo_test(n_units=100, n_simulations=200, z0=30.0)