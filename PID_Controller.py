# ПИД-контроллер для квадрокоптера
# импорт библиотек
import numpy as np
import math


class PID_Controller():
    def __init__(self, Kp, Kd, Ki, Ki_sat, dt):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.Ki_sat = Ki_sat
        self.dt = dt
        # Интегральная сумма
        self.int = [0., 0., 0.]

    def control_update(self, pos_error, vel_error):
        # Обновление интегральной составляющей
        self.int += pos_error * self.dt
        # Предотвращение накопления ошибки (windup)
        # т.к. накопленная интегральная ошибка может требовать невозможных значений тяги,
        # Интегральная составляющая будет продолжать накапливаться, даже когда система уже близка к целевому значению
        over_mag = np.argwhere(np.array(self.int) > np.array(self.Ki_sat))
        if over_mag.size != 0:
            for i in range(over_mag.size):
                mag = abs(self.int[over_mag[i][0]])  # Получаем величину для определения знака (направления)
                # Ограничиваем значение до максимального, сохраняя знак
                self.int[over_mag[i][0]] = (self.int[over_mag[i][0]] / mag) * self.Ki_sat[over_mag[i][0]]
        # Расчет управляющего воздействия (желаемого ускорения)
        des_acc = self.Kp * pos_error + self.Ki * self.int + self.Kd * vel_error
        return des_acc