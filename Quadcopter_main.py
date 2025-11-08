# Импорт необходимых библиотек
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import atexit
import signal
import sys
import os
import subprocess

# Импорт классов квадрокоптера и контроллера
from Quadcopter import Quadcopter
from PID_Controller import PID_Controller
from PID_Controller import PID_Controller
from Wind_Model import Wind_Model

# Параметры времени симуляции
sim_start = 0  # начальное время симуляции (сек)
sim_end = 20  # конечное время симуляции (сек)
dt = 0.01  # шаг по времени (сек)
time_index = np.arange(sim_start, sim_end + dt, dt)

# Цель
r_ref = np.array([5., 3., 9.])  # желаемая позиция [x, y, z] в инерциальной системе (метры)

# Начальные условия позиции
pos = [0., 0., 0.]  # начальная позиция [x, y, z] в инерциальной системе (метры)
vel = np.array([0., 0., 0.])  # начальная скорость [x, y, z] (м/с)
ang = np.array([0., 0., 0.])  # начальные углы Эйлера [крен, тангаж, рыскание] (градусы)

# Добавление случайных начальных угловых скоростей
deviation = 0  # величина начального возмущения (град/сек)
random_set = np.array([random.random(), random.random(), random.random()])
ang_vel = np.deg2rad(2 * deviation * random_set - deviation)  # начальная угловая скорость [крен, тангаж, рыскание]
ang_vel_init = ang_vel.copy()  # сохранение для последующего отображения

wind_model = Wind_Model(
    dt=dt,
    steady=np.array([16.0, 0.0, 0]),             # постоянный ветер, м/с
    gm_sigma=np.array([0.4, 0.4, 0.2]),            # шумовая составляющая (СКО)
    gm_tau=np.array([2.0, 2.0, 3.0]),              # «инерция» шума
    gust_events= None,  # примеры порыва ступеньки
    pulse_rate=0.3,
    pulse_amp_range=(0.5, 2.0),
    pulse_dur_range=(0.2, 0.8),
    seed=42
)
gravity = 9.8  # ускорение свободного падения (м/с²)

# Коэффициенты позиционного контроллера
Kp_pos = [.95, .95, 15.]  # пропорциональные [x,y,z]
Kd_pos = [1.8, 1.8, 15.]  # дифференциальные [x,y,z]
Ki_pos = [0.2, 0.2, 1.0]  # интегральные [x,y,z]
Ki_sat_pos = 1.1 * np.ones(3)  # насыщение интегральной составляющей (предотвращение перерегулирования)

# Коэффициенты углового контроллера
Kp_ang = [7, 7, 25.]  # пропорциональные [x,y,z]
Kd_ang = [3.7, 3.7, 9.]  # дифференциальные [x,y,z]
Ki_ang = [0.1, 0.1, 0.1]  # интегральные [x,y,z]
Ki_sat_ang = 0.1 * np.ones(3)  # насыщение интегральной составляющей

# Создание объектов квадрокоптера и контроллеров
quadcopter = Quadcopter(pos, vel, ang, ang_vel, r_ref, dt)
pos_controller = PID_Controller(Kp_pos, Kd_pos, Ki_pos, Ki_sat_pos, dt)
angle_controller = PID_Controller(Kp_ang, Kd_ang, Ki_ang, Ki_sat_ang, dt)

# Инициализация массивов для хранения результатов
total_error = []
position_total = []
total_thrust = []


def initialize_results(res_array, num):
    """Инициализация массива результатов заданного размера"""
    for i in range(num):
        res_array.append([])


# Инициализация массивов для различных параметров
position = [];
initialize_results(position, 3)
velocity = [];
initialize_results(velocity, 3)
angle = [];
initialize_results(angle, 3)
angle_vel = [];
initialize_results(angle_vel, 3)
motor_thrust = [];
initialize_results(motor_thrust, 4)
body_torque = [];
initialize_results(body_torque, 3)
wind_log = [[] for _ in range(3)]  # >>> ВЕТЕР: логирование

for k, t_cur in enumerate(time_index):
    # >>> ВЕТЕР: обновить ветер и передать в модель
    w_vec = wind_model.step(t_cur)
    quadcopter.set_wind(w_vec)

    # Контур позиции
    pos_error = quadcopter.calc_pos_error(quadcopter.pos)
    vel_error = quadcopter.calc_vel_error(quadcopter.vel)
    des_acc = pos_controller.control_update(pos_error, vel_error)

    # Коррекция по Z
    des_acc[2] = (gravity + des_acc[2]) / (math.cos(quadcopter.angle[0]) * math.cos(quadcopter.angle[1]))
    thrust_needed = quadcopter.mass * des_acc[2]

    mag_acc = np.linalg.norm(des_acc)
    if mag_acc == 0:
        mag_acc = 1
    ang_des = [math.asin(-des_acc[1] / mag_acc / math.cos(quadcopter.angle[1])),
               math.asin(des_acc[0] / mag_acc),
               0]
    mag_angle_des = np.linalg.norm(ang_des)
    if mag_angle_des > quadcopter.max_angle:
        ang_des = (np.array(ang_des) / mag_angle_des) * quadcopter.max_angle

    quadcopter.angle_ref = ang_des
    ang_error = quadcopter.calc_ang_error(quadcopter.angle)
    ang_vel_error = quadcopter.calc_ang_vel_error(quadcopter.ang_vel)
    tau_needed = angle_controller.control_update(ang_error, ang_vel_error)

    quadcopter.des2speeds(thrust_needed, tau_needed)
    quadcopter.step()

    # Запись данных
    position_total.append(np.linalg.norm(quadcopter.pos))
    for i in range(3):
        position[i].append(quadcopter.pos[i])
        velocity[i].append(quadcopter.vel[i])
        angle[i].append(np.rad2deg(quadcopter.angle[i]))
        angle_vel[i].append(np.rad2deg(quadcopter.ang_vel[i]))
        body_torque[i].append(quadcopter.tau[i])
        wind_log[i].append(w_vec[i])
    for i in range(4):
        motor_thrust[i].append(quadcopter.speeds[i] * quadcopter.kt)
    total_thrust.append(quadcopter.kt * np.sum(quadcopter.speeds))
    total_error.append(np.linalg.norm(quadcopter.pos_ref - quadcopter.pos))






def total_plot():
    """3D график траектории"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(position[0], position[1], position[2])
    ax.set_title('3D траектория')
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Y (м)')
    ax.set_zlabel('Z (м)')
    plt.show()

def save_sig(filename, time, data, label):
    """
    Сохраняет данные в файл .sig для sinus
    Формат:
    #label
    t1  data1
    t2  data2
    ...
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"#{label}\n")
        for t, val in zip(time, data):
            f.write(f"{t:.6f} {val:.6f}\n")



# Создадим папку для файлов, чтобы не захламлять текущую директорию
os.makedirs('sinus_data', exist_ok=True)

sig_files = []  # список файлов для запуска sinus
windows = []    # список списков файлов для каждого окна

# 1. Положение по XY — 2 линии в одном окне
pos_xy_files = []
save_sig('sinus_data/pos_x.sig', time_index, position[0], 'X(t)')
save_sig('sinus_data/pos_y.sig', time_index, position[1], 'Y(t)')
pos_xy_files.extend(['sinus_data/pos_x.sig', 'sinus_data/pos_y.sig'])
windows.append(pos_xy_files)

# 2. Высота
save_sig('sinus_data/height.sig', time_index, position[2], 'Z(t)')
windows.append(['sinus_data/height.sig'])

# 3. Линейные скорости (Vx, Vy, Vz) — 3 линии в одном окне
vel_files = []
for i, label in enumerate(['Vx(t)', 'Vy(t)', 'Vz(t)']):
    filename = f'sinus_data/vel_{i}.sig'
    save_sig(filename, time_index, velocity[i], label)
    vel_files.append(filename)
windows.append(vel_files)

# 4. Тяга моторов — 4 линии в одном окне
motor_files = []
for i in range(4):
    filename = f'sinus_data/motor_{i}.sig'
    save_sig(filename, time_index, motor_thrust[i], f'Motor_{i+1}(t)')
    motor_files.append(filename)
windows.append(motor_files)

# 5. Моменты (Mx, My, Mz) — 3 линии в одном окне
torque_files = []
for i, label in enumerate(['Mx(t)', 'My(t)', 'Mz(t)']):
    filename = f'sinus_data/torque_{i}.sig'
    save_sig(filename, time_index, body_torque[i], label)
    torque_files.append(filename)
windows.append(torque_files)

# 6. Углы Эйлера — 3 линии в одном окне
angle_files = []
for i, label in enumerate(['Roll(t)', 'Pitch(t)', 'Yaw(t)']):
    filename = f'sinus_data/angle_{i}.sig'
    save_sig(filename, time_index, angle[i], label)
    angle_files.append(filename)
windows.append(angle_files)

# 7. Угловые скорости — 3 линии в одном окне
ang_vel_files = []
for i, label in enumerate(['ωx(t)', 'ωy(t)', 'ωz(t)']):
    filename = f'sinus_data/ang_vel_{i}.sig'
    save_sig(filename, time_index, angle_vel[i], label)
    ang_vel_files.append(filename)
windows.append(ang_vel_files)

# 8. Ошибка позиции (разница между желаемой и текущей позицией)
error_x = r_ref[0] - np.array(position[0])
error_y = r_ref[1] - np.array(position[1])
error_z = r_ref[2] - np.array(position[2])
total_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)

save_sig('sinus_data/error_x.sig', time_index, error_x, 'Error X(t)')
save_sig('sinus_data/error_y.sig', time_index, error_y, 'Error Y(t)')
save_sig('sinus_data/error_z.sig', time_index, error_z, 'Error Z(t)')
save_sig('sinus_data/total_error.sig', time_index, total_error, 'Total Error(t)')

# Добавляем окно с ошибками (все компоненты и общая ошибка)
error_files = [
    'sinus_data/error_x.sig',
    'sinus_data/error_y.sig',
    'sinus_data/error_z.sig',
    'sinus_data/total_error.sig'
]
windows.append(error_files)

# Подготовка аргументов для запуска sinus.exe:
# Формируем строку вида:
# .\sinus.exe pos_x.sig pos_y.sig -n height.sig -n vel_0.sig vel_1.sig vel_2.sig -n motor_0.sig motor_1.sig motor_2.sig motor_3.sig ...
wind_files = []
for i, label in enumerate(['Wind_X(t)', 'Wind_Y(t)', 'Wind_Z(t)']):
    fn = f'sinus_data/wind_{i}.sig'
    save_sig(fn, time_index, wind_log[i], label)
    wind_files.append(fn)
windows.append(wind_files)

cmd = ['sinus.exe']

for i, group in enumerate(windows):
    if i > 0:
        cmd.append('-n')  # разделяем окна
    cmd.extend(group)

all_sig_files = []
for group in windows:
    all_sig_files.extend(group)
# Запускаем sinus
print('Запускаем sinus с командой:')
print(' '.join(cmd))
sinus_process = subprocess.Popen(cmd)
def cleanup():
    print('\nЗавершаем sinus и удаляем .sig файлы...')
    # Завершаем процесс sinus, если он ещё жив
    if sinus_process.poll() is None:
        sinus_process.terminate()
        try:
            sinus_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sinus_process.kill()
    # Удаляем файлы
    for f in all_sig_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Не удалось удалить {f}: {e}")

# Регистрируем функцию очистки при нормальном выходе
atexit.register(cleanup)

# Обработка Ctrl+C (SIGINT) чтобы вызвать cleanup и выйти
def signal_handler(sig, frame):
    print('Получен сигнал прерывания (Ctrl+C)')
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)



# Вывод графиков внутри Python
total_plot()

cleanup()