import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wiener_degradation import (
    WienerModel, LinearTrend, PowerTrend,
    PowerWithCovariateTrend, LinearWithCovariateTrend, Wiener_IMF
)


# Тот же PowerCovTrend что и выше...

def analytical_I44(model, n, cov, weight, time):
    """Прямой расчёт I44 по формуле из Wiener_IMF."""
    time_list = time
    x = model.x
    covariates = []
    for i in range(len(cov) - 1):
        for _ in range(int(weight[i] * n)):
            covariates.append(cov[i])
    while len(covariates) < n:
        covariates.append(cov[-1])
    covariates = covariates[:n]

    s = 0
    for i in range(n):
        for j in range(len(time_list[i]) - 1):
            delta_ro = model.f_ro(time_list[i][j + 1], covariates[i]) - model.f_ro(time_list[i][j], covariates[i])
            der_ro = model.trend.d_func_beta(time_list[i][j + 1], x, covariates[i]) - model.trend.d_func_beta(time_list[i][j], x, covariates[i])
            if delta_ro != 0:
                s += np.power(der_ro, 2) * (0.5 / np.power(delta_ro, 2) + np.power(x[1] / x[0], 2) / delta_ro)
    return s

# Параметры
true_theta = np.array([0.5, 1.0, 1.5, 0.5])
n_units = 40
time_points = [0.0, 5.0, 10.0, 15.0, 20.0]
cov_values = [1.0, 2.0]
weights = [0.5, 0.5]

trend =  PowerWithCovariateTrend()
model = WienerModel(x=true_theta, trend=trend, z0=20.0)

times_list = [time_points for _ in range(n_units)]

# Расчёт через класс
imf_calc = Wiener_IMF(model, n_units, cov_values, weights, times_list)
I_matrix = np.array(imf_calc.getIMF())
I44_class = I_matrix[3, 3]

# Расчёт напрямую
I44_direct = analytical_I44(model, n_units, cov_values, weights, times_list)

print(f"ПРОВЕРКА ФОРМУЛЫ I44")
print(f"I44 из класса Wiener_IMF: {I44_class:.6f}")
print(f"I44 прямой расчёт:       {I44_direct:.6f}")
print(f"Разница:                 {abs(I44_class - I44_direct):.6f}")
print(f"Относительная ошибка:    {abs(I44_class - I44_direct)/I44_class*100:.6f}%")

if abs(I44_class - I44_direct) < 1e-10:
    print("✅ Формула I44 реализована корректно")
else:
    print("❌ Ошибка в реализации I44")