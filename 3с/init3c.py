import  numpy as np
from conversions3c import *

dt = float(0.1) # шаг моделирования РК4
tFinal = 1000.0 # время моделирования
t = np.arange(0.0, tFinal, dt) # массив эпох
mu = 398600.4418 # гравитационный параметр km^3 / sec^2
Jtense = np.diag([3.0,4.0,2.0])*1e-6 # тензор инерции кг*kм^2

# получаем начальные радиус-вектор и скорость в ИСО
[rIF0, velIF0] = orbital2Cartesian(7e3, 0, 15, 30, 20, 30, mu)

# кватернион ориентации, задаёт переход из ИСК в ССК
quat0 = np.array([1.0, 0, 0, 0], dtype= float)

# полная угловая скорость в ССК в начальный момент
omegaBF0 = np.array([0.001, 0.001, 0.001], dtype=float) # рад/с

# начальный фазовый вектор [rx, ry, rx, vx, vy, vz, q0, qi, qj, qk, omega1, omega2, omega3]
phaseVec = np.array([*rIF0, *velIF0, *quat0, *omegaBF0], dtype=float)
