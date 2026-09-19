import numpy as np
import sys
import os
import concurrent.futures
import multiprocessing

# Добавляем текущую директорию в путь для импорта ваших модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from wiener_degradation import (
    WienerModel, LinearTrend, PowerTrend,
    PowerWithCovariateTrend, LinearWithCovariateTrend, Wiener_IMF
)

# ==============================================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (должны быть на верхнем уровне для pickle)
# ==============================================================================
def generate_data_fixed_plan_fast_limited(n_units, time_points, cov_values, weights, true_theta, z0):
    trend = PowerWithCovariateTrend()
    model = WienerModel(x=true_theta, trend=trend, z0=z0)

    # Формирование вектора ковариат
    covariates = []
    for i in range(len(cov_values) - 1):
        covariates.extend([cov_values[i]] * int(weights[i] * n_units))
    while len(covariates) < n_units:
        covariates.append(cov_values[-1])
    covariates = np.array(covariates[:n_units])

    n_times = len(time_points)

    # Вычисление дельт ro для каждого объекта
    delta_ro_matrix = np.zeros((n_units, n_times - 1))
    for i in range(n_units):
        c = covariates[i]
        ro = np.array([model.f_ro(t, c) for t in time_points])
        delta_ro_matrix[i, :] = np.diff(ro)

    # Параметры деградационного процесса
    loc_matrix = model.mu * delta_ro_matrix
    scale_matrix = model.sigma * np.sqrt(np.maximum(delta_ro_matrix, 1e-10))

    # 1. Генерация начальных деградационных приращений
    degradation_increments = np.random.normal(loc=loc_matrix, scale=scale_matrix)

    # 2. Проверка на достижение порога z0
    cum_degradation = np.cumsum(degradation_increments, axis=1)
    crossed_mask = cum_degradation >= z0
    has_crossed = np.any(crossed_mask, axis=1)

    # Находим индекс первого шага, на котором порог был превышен
    cross_indices = np.argmax(crossed_mask, axis=1)
    cross_indices = np.where(has_crossed, cross_indices, n_times - 1)

    times_list = []
    final_increments_list = []
    trails = []

    # 3. Обработка каждого объекта: разделение на деградацию и след
    for i in range(n_units):
        if has_crossed[i]:
            idx = cross_indices[i]

            # idx — индекс приращения, на котором порог достигнут
            # Приращение idx соответствует переходу от time_points[idx] к time_points[idx+1]
            # Значит, момент достижения порога — это time_points[idx+1]

            # Времена от начала до момента достижения порога включительно
            # Это time_points[0], time_points[1], ..., time_points[idx+1]
            end_time_idx = idx + 1  # индекс последнего времени (момент достижения порога)
            times_list.append(time_points[:end_time_idx + 1])

            # Деградационные приращения до момента превышения порога включительно
            # Это приращения 0, 1, ..., idx (всего idx+1 приращений)
            final_increments_list.append(degradation_increments[i, :idx + 1].tolist())

            # Переход к генерации следа объекта для оставшихся временных точек
            if idx + 1 < n_times - 1:
                trail_loc = np.zeros(n_times - 1 - (idx + 1))
                trail_scale = scale_matrix[i, idx + 1:]

                trail_incs = np.random.normal(loc=trail_loc, scale=trail_scale)
                trails.append(trail_incs.tolist())
            else:
                trails.append(None)
        else:
            # Если порог не достигнут, все времена и приращения
            times_list.append(list(time_points))
            final_increments_list.append(degradation_increments[i, :].tolist())
            trails.append(None)

    # Возвращаем индивидуальные времена, приращения, следы и ковариаты
    return times_list, final_increments_list, trails, covariates.tolist()

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

def worker_simulate(args):
    """
    Рабочая функция для параллельного выполнения одной симуляции.
    Вынесена на верхний уровень для корректной работы multiprocessing в Windows.
    """
    seed, true_theta, n_units, time_points, cov_values, z0 = args
    np.random.seed(seed)

    # Импортируем внутри, чтобы избежать проблем с pickle при передаче сложных объектов
    from wiener_degradation import WienerModel, PowerWithCovariateTrend

    trend = PowerWithCovariateTrend()
    model = WienerModel(x=true_theta, trend=trend, z0=z0)
    data = generate_data(model,n_units, time_points, cov_values )
    model.data = data

    return numerical_score(model, data, true_theta)


# ==============================================================================
# 2. ФУНКЦИЯ МОНТЕ-КАРЛО С ПАРАЛЛЕЛИЗАЦИЕЙ
# ==============================================================================
def run_monte_carlo_test(model_name, trend_class, true_theta, n_units=20, n_simulations=10000, z0=50.0):
    """Запуск МК теста для одной модели с параллельными вычислениями."""
    print(f"\n{'='*70}")
    print(f"ТЕСТ: {model_name.upper()} | Единиц: {n_units}")
    print(f"Истинные параметры: {true_theta}")
    print(f"Симуляций: {n_simulations} (распараллелено)")
    print(f"{'='*70}")

    time_points = list(range(0, 21, 1))
    cov_values =  [6.0, 2.5] if 'cov' in model_name else [0.0]
    weights = [0.5, 0.5] if len(cov_values) == 2 else [1.0]
    times_list = [time_points for _ in range(n_units)]

    # Генерация сидов для воспроизводимости в параллельных процессах
    seeds = np.random.randint(0, 2**31 - 1, size=n_simulations)
    args_list = [(seed, true_theta, n_units, time_points, cov_values, z0) for seed in seeds]

    print(f"Запуск вычислений в {multiprocessing.cpu_count()} потоков...")

    # Параллельное выполнение
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # chunksize оптимизирует передачу данных между процессами
        scores_list = list(executor.map(worker_simulate, args_list, chunksize=200))

    scores = np.array(scores_list)
    print("Вычисления завершены. Анализ результатов...")

    # Эмпирическая ИМФ = Ковариация векторов оценки
    empirical_fim = np.cov(scores, rowvar=False)

    # Аналитическая ИМФ
    trend = trend_class()
    model = WienerModel(x=true_theta, trend=trend, z0=z0)
    imf_calculator = Wiener_IMF(model, n_units, cov_values, weights, times_list)

    model.set_params(true_theta)
    analytical_fim = np.array(imf_calculator.getIMF())

    # --- СРАВНЕНИЕ МАТРИЦ ---
    print("\n--- Сравнение матриц ---")
    print("Аналитическая ИМФ:")
    print(np.round(analytical_fim, 4))
    print("\nЭмпирическая ИМФ (Ковариация Score):")
    print(np.round(empirical_fim, 4))

    # 1. Ошибка по норме Фробениуса
    norm_diff = np.linalg.norm(analytical_fim - empirical_fim)
    norm_analytical = np.linalg.norm(analytical_fim)
    rel_error_frob = (norm_diff / norm_analytical) * 100

    # 2. Сравнение ОПРЕДЕЛИТЕЛЕЙ (Детерминантов)
    det_analytical = np.linalg.det(analytical_fim)
    det_empirical = np.linalg.det(empirical_fim)

    # Защита от деления на ноль или очень маленьких чисел
    safe_det_analytical = max(abs(det_analytical), 1e-10)
    rel_error_det = (abs(det_analytical - det_empirical) / safe_det_analytical) * 100

    print("\n--- Сравнение определителей (Det) ---")
    print(f"Определитель аналитической ИМФ: {det_analytical:.6e}")
    print(f"Определитель эмпирической ИМФ:  {det_empirical:.6e}")
    print(f"Относительная ошибка определителя: {rel_error_det:.2f}%")

    print(f"\nОтносительная ошибка (норма Фробениуса): {rel_error_frob:.2f}%")

    is_frob_ok = rel_error_frob < 5.0
    is_det_ok = rel_error_det < 10.0 # Для определителя допуск чуть выше из-за накопления ошибок

    if is_frob_ok and is_det_ok:
        print("✅ РЕЗУЛЬТАТ: Матрицы и их определители совпадают в пределах статистической погрешности.")
    else:
        print("⚠️ РЕЗУЛЬТАТ: Обнаружено значимое расхождение! Требуется проверка формул.")

    return {
        "n_units": n_units,
        "rel_error_frob": rel_error_frob,
        "rel_error_det": rel_error_det,
        "det_analytical": det_analytical,
        "det_empirical": det_empirical,
        "success": is_frob_ok and is_det_ok
    }


# ==============================================================================
# 3. ЗАПУСК ТЕСТОВ ДЛЯ РАЗНЫХ РАЗМЕРОВ ВЫБОРКИ
# ==============================================================================
if __name__ == "__main__":
    # Важно для Windows: весь исполняемый код должен быть внутри __main__
    np.random.seed(42) # Базовый сид для генерации сидов процессов

    true_theta = np.array([0.5, 0.8, 1.5, 0.5])
    units_to_test = [10,20,40]
    n_simulations = 100
    z0 = 30.0

    print(f"НАЧАЛО МАСШТАБНОГО ТЕСТИРОВАНИЯ")
    print(f"Параметры: theta={true_theta}, z0={z0}, симуляций={n_simulations}")

    results = []
    for n_units in units_to_test:
        res = run_monte_carlo_test(
            model_name="power_covariate",
            trend_class=PowerWithCovariateTrend,
            true_theta=true_theta,
            n_units=n_units,
            n_simulations=n_simulations,
            z0=z0
        )
        results.append(res)

    # ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА
    print(f"\n\n{'='*80}")
    print("ИТОГОВАЯ СВОДКА ПО ОПРЕДЕЛИТЕЛЯМ И ОШИБКАМ")
    print(f"{'='*80}")
    print(f"{'Единиц':<10} | {'Det Аналит.':<15} | {'Det Эмпирич.':<15} | {'Ошибка Det (%)':<15} | {'Ошибка Фроб. (%)':<15} | {'Статус'}")
    print("-" * 80)

    all_success = True
    for res in results:
        status = "✅ OK" if res["success"] else "⚠️ FAIL"
        if not res["success"]:
            all_success = False

        print(f"{res['n_units']:<10} | "
              f"{res['det_analytical']:<15.4e} | "
              f"{res['det_empirical']:<15.4e} | "
              f"{res['rel_error_det']:<15.2f} | "
              f"{res['rel_error_frob']:<15.2f} | "
              f"{status}")

    print(f"{'='*80}")
    if all_success:
        print("🎉 Все аналитические формулы в Wiener_IMF успешно прошли проверку Монте-Карло!")
    else:
        print("🔴 Найдены расхождения. Необходимо проверить аналитические производные.")