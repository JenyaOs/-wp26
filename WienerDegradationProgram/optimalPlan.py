"""
Модуль оптимального планирования деградационного эксперимента.
Использует классы IMF из библиотеки wiener_degradation.
"""

import numpy as np
from scipy.optimize import least_squares
from wiener_degradation import Wiener_IMF, Wiener_ConditionIMF


class OptimalPlanningProcedure:
    def __init__(self, _model, _n, _cov0, _weight0, _time0):
        # Преобразуем в list, чтобы избежать проблем с изменяемыми объектами
        self.covariate = list(_cov0)
        self.weight = list(_weight0)
        self.time = list(_time0)
        self.model = _model
        self.n = _n
        # Используем порог из модели, если он задан, иначе 50 (как в оригинале)
        self.z = getattr(_model, 'z', 50)
        self.arr = []
        self.arr_cov = []

    def getIMF_D(self):
        time_list = [self.time for _ in range(self.n)]
        imf = Wiener_IMF(self.model, self.n, self.covariate, self.weight, time_list)
        matrix = np.linalg.det(imf.getIMF())
        self.arr.append([list(self.time), matrix])
        return matrix

    def getConditionIMF_D(self):
        # Динамически создаем нулевую матрицу нужного размера (2x2, 3x3 или 4x4)
        dummy_imf = Wiener_IMF(self.model, self.n, self.covariate, self.weight, [self.time])
        matrix = np.zeros_like(dummy_imf.getIMF(), dtype=float)

        for i in range(len(self.covariate)):
            time_cur = [0]

            # Пороговое значение времени (логика из оригинального кода)
            threshold = np.power(self.z / self.model.x[1], 1.0) * np.exp(self.model.x[2] * np.log(self.covariate[i]))

            for j in range(len(self.time) - 1):
                if self.time[j + 1] <= threshold:
                    time_cur.append(self.time[j + 1])

            n_weighted = int(self.n * self.weight[i])

            if n_weighted > 0 and len(time_cur) > 1:
                time_list_cur = [time_cur for _ in range(n_weighted)]

                imf = Wiener_IMF(self.model, n_weighted, [self.covariate[i]], [1], time_list_cur)
                imfC = Wiener_ConditionIMF(self.model, n_weighted, [self.covariate[i]], [1], time_list_cur, self.z)

                matrix += imfC.getIMF() + imf.getIMF()

        # Защита от вырожденной матрицы при вычислении определителя
        try:
            return float(np.linalg.det(matrix))
        except np.linalg.LinAlgError:
            return 0.0

    def getLoads(self, cov):
        self.covariate = list(cov)
        imfC = self.getConditionIMF_D()
        self.arr_cov.append([list(cov), imfC])
        return -imfC

    def getWeight(self):
        new_weight = [0.1, 0.9]
        x = 1.0 / self.n
        step = 1.0 / self.n
        max_val = 0
        round_e = len(str(self.n))

        for i in range(int(self.n) - 2):
            weight_x = [round(x, round_e), round(1 - x, round_e)]
            self.weight = weight_x
            new_max = self.getIMF_D()
            if new_max > max_val:
                max_val = new_max
                new_weight = weight_x
            x += step
        return new_weight

    def getFireTime(self, time):
        fire = 0
        if time[0] < 0:
            fire += np.power(time[0], 2)
        for i in range(len(time) - 1):
            if time[i + 1] - time[i] < 0:
                return np.nan
        if time[-1] > 25:
            fire += np.power(time[-1] - 25, 2)
        return fire

    def getTime(self, time):
        self.time = list(time)
        return -self.getIMF_D()

    def directSearchD(self):
        IMF_0 = self.getIMF_D()
        conditionalIMF_0 = self.getConditionIMF_D()
        IMF_n, conditionalIMF_n = 0, 0

        print("Начальный план", self.getPlan())
        print(f"IMF_0: {IMF_0:.4e}, conditionalIMF_0: {conditionalIMF_0:.4e}")

        i = 0
        max_iterations = 20  # Защита от бесконечного цикла в веб-интерфейсе

        while (abs(IMF_0 - IMF_n) > 0.01 or abs(conditionalIMF_0 - conditionalIMF_n) > 0.01) and i < max_iterations:
            IMF_n, conditionalIMF_n = IMF_0, conditionalIMF_0

            # Оптимизация по ковариатам
            # В scipy least_squares bounds должны быть кортежем двух массивов: (lower_bounds, upper_bounds)
            num_covs = len(self.covariate)
            lower_bounds = [0.1] * num_covs  # Минимальная нагрузка (соответствует UI)
            upper_bounds = [5.0] * num_covs  # Максимальная нагрузка (соответствует UI)

            try:
                cov_n = least_squares(
                    self.getLoads,
                    self.covariate,
                    diff_step=0.2,
                    bounds=(lower_bounds, upper_bounds)
                )

                # Выбираем лучший результат из истории вычислений
                if len(self.arr_cov) > 0:
                    best_idx = np.argmax([item[1] for item in self.arr_cov])
                    self.covariate = list(self.arr_cov[best_idx][0])
                else:
                    self.covariate = list(cov_n.x)

                self.arr_cov = []
            except Exception as e:
                print(f"Предупреждение при оптимизации ковариат: {e}")

            IMF_0 = self.getIMF_D()
            conditionalIMF_0 = self.getConditionIMF_D()
            i += 1
            print(f"Шаг: {i}, План: {self.getPlan()}")
            print(f"IMF: {IMF_0:.4e}, Cond IMF: {conditionalIMF_0:.4e}")

        print("Оптимальный план", self.getPlan())
        return IMF_0, conditionalIMF_0

    def getPlan(self):
        return self.covariate, self.weight, self.time