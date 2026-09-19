"""Пример использования библиотеки wiener_degradation."""

import numpy as np
    from itertools import takewhile
from wiener_degradation import (
    WienerModel, LinearTrend, LinearWithCovariateTrend, PowerWithCovariateTrend, PowerTrend,
    DegradationDataGenerator, ParameterEstimator, ReliabilityCalculator, Wiener_IMF, Wiener_ConditionIMF
)

def get_params_to_IMF(model, count_obj, cov, weight, z0, maxTime, momentsCount):

    times_list, cov_list = [],[]

    count_per_group = [int(count_obj*weight[i]) for i in range(len(weight)-1)]
    count_per_group.append(count_obj - sum(count_per_group))

    times = np.linspace(0, maxTime, momentsCount)
    for i in range(len(cov)):
        #обрезка при достижении критического значенния показателя деградации

        t_max = min(maxTime, model.inv_f_ro(z0,cov[i]))
        times_cur  =  list(takewhile(lambda x: x <= t_max, times))
        for j in range(count_per_group[i]):
            times_list.append(times_cur)
            cov_list.append(cov[i])

    return times_list, cov_list

def get_params_to_IMF_times(model, count_obj, cov, weight, z0, times):

    times_list, cov_list = [],[]

    count_per_group = [int(count_obj*weight[i]) for i in range(len(weight)-1)]
    count_per_group.append(count_obj - sum(count_per_group))

    for i in range(len(cov)):
        #обрезка при достижении критического значенния показателя деградации
        print(model.inv_f_ro(z0,cov[i]))
        t_max = min(times[-1], model.inv_f_ro(z0,cov[i]))
        times_cur  =  list(takewhile(lambda x: x <= t_max, times))
        for j in range(count_per_group[i]):
            times_list.append(times_cur)
            cov_list.append(cov[i])

    return times_list, cov_list



def example_power_model_covariate(x_true, M_count, z0, times, cov, weight):
    """Пример для степенной модели."""
    print("\n" + "=" * 60)
    print("Степенная модель деградации с ковариатами")
    print("=" * 60)


    trend = PowerWithCovariateTrend()
    model = WienerModel(x=x_true, trend=trend, z0=z0)

    times_list, cov_list = get_params_to_IMF_times(model, M_count,cov, weight, z0, times)


    generator = DegradationDataGenerator(model)
    sample = generator.generate_power(M=M_count, x=x_true, z0=z0, covariate=cov_list,time=times_list)

    estimator = ParameterEstimator(model)
    x0 = [0.1, 0.5, 1.2, 0.9]
    estimated = estimator.estimate(x0, sample)


    print(f"Истинные параметры:   σ={sigma:.3f}, μ={mu:.3f}, γ={gamma:.3f},  beta={beta: 3f}")
    print(f"Оценённые параметры:  σ={estimated[0]:.3f}, μ={estimated[1]:.3f}, γ={estimated[2]:.3f}, beta={estimated[3]:.3f}")
    print(f"Относительные ошибки: {estimator.compute_errors(x_true, estimated)}")

    #Расчет ИМФ
    model.x=x_true
    imf_calculator = Wiener_IMF(model, M_count, cov, weight, times_list)
    analytical_fim = np.array(imf_calculator.getIMF())
    det_analytical = np.linalg.det(analytical_fim)

    # Вывод ИМФ
    print("Аналитическая ИМФ (Wiener_IMF):")
    print(analytical_fim.astype('int'))
    print("Определитель ИМФ:")
    print(det_analytical)
    print("log oпределитель ИМФ:")
    print(np.log(det_analytical))

    #Расчет УИМФ
    model.x=x_true
    condImf_calculator = Wiener_ConditionIMF(model, M_count,cov, weight, times_list, z0)
    analytical_fim = np.array(condImf_calculator.getIMF())
    det_analytical = np.linalg.det(analytical_fim)

    '''
    # Вывод УИМФ
    print("Условная аналитическая ИМФ (Wiener_ConditionIMF):")
    print(analytical_fim.astype('int'))
    print("Определитель УИМФ:")
    print(det_analytical)'''

    return det_analytical, np.log(det_analytical)




if __name__ == "__main__":
    sigma, mu, gamma, beta, z0 = 0.5, 0.8, 1.5, 0.8, 30
    x_true = [sigma, mu, gamma, beta]
    cov = [2.5, 6]
    weight = [0.5,0.5]

    M_count = 20
    times = np.linspace(0, 20, 21)

    det_analytical1, log_det_analytical1 = example_power_model_covariate(x_true, M_count, z0, times, cov, weight)

    #sigma, mu, gamma, beta, z0 = 0.5, 0.8, 1.5, 0.8, 30
    #x_true = [sigma, mu, gamma, beta]

    cov = [8, 1.3539]
    weight = [0.9, 0.1]
    z0 = z0
    M_count = 20
    times = np.linspace(0, 20, 21)#[0.000, 1.053, 2.105, 3.158, 4.211, 5.263, 6.316, 7.368, 8.421, 9.474, 10.526, 11.579, 12.632, 13.684, 14.737, 15.789, 16.842, 17.895, 18.947, 20.000]

    det_analytical2, log_det_analytical2 = example_power_model_covariate(x_true, M_count, z0, times, cov, weight)

    print("Сравнение")
    print(100*det_analytical2/det_analytical1)
    print(log_det_analytical2*100/log_det_analytical1-100)


    '''
    sigma, mu, gamma, beta, z0 = 0.5, 0.8, 1.5, 0.8, 20
    x_true = [sigma, mu, gamma, beta]
    cov = [2.5, 6]
    weight = [0.5,0.5]

    M_count = 20
    times = np.linspace(0, 20, 21)

    det_analytical1, log_det_analytical1 = example_power_model_covariate(x_true, M_count, z0, times, cov, weight)

    #sigma, mu, gamma, beta, z0 = 0.5, 0.8, 1.5, 0.8, 30
    #x_true = [sigma, mu, gamma, beta]

    cov = [7.9871 ,1.9972]
    weight = [0.6038, 0.3962]
    z0 = z0
    M_count = 20
    times = np.linspace(0, 20, 21)#[0.000, 1.053, 2.105, 3.158, 4.211, 5.263, 6.316, 7.368, 8.421, 9.474, 10.526, 11.579, 12.632, 13.684, 14.737, 15.789, 16.842, 17.895, 18.947, 20.000]

    det_analytical2, log_det_analytical2 = example_power_model_covariate(x_true, M_count, z0, times, cov, weight)

    print("Сравнение")
    print(100*det_analytical2/det_analytical1)
    print(log_det_analytical2*100/log_det_analytical1-100)'''

'''
[D_plan]
  z0 =30
  Статус: Success (Optimization terminated successfully)
  Оптимальные нагрузки: [7.9996 1.3539]
  Оптимальные веса:     [0.5553 0.4447]
  Моменты измерений по уровням нагрузки:
    Уровень 0 (S=7.9996): [0, 1, ..., 19] (20 точек)
    Уровень 1 (S=1.3539): [0, 1, ..., 14] (15 точек)
  Значение критерия:    -29.998488
  
  z0 = 20
  [D_plan]
  Статус: Success (Optimization terminated successfully)
  Оптимальные нагрузки: [7.9871 1.9972]
  Оптимальные веса:     [0.6038 0.3962]
  Моменты измерений по уровням нагрузки:
    Уровень 0 (S=7.9871): [0, 1, ..., 19] (20 точек)
    Уровень 1 (S=1.9972): [0, 1, ..., 13] (14 точек)
  Значение критерия:    -29.445147
'''
