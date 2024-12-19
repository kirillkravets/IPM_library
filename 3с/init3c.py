import  numpy as np
from orbitalToCartesian import orbitalToCartesian
from quaternion import Quaternion

# IF - ИСК, BF - ССК, OF - ОСК
# переводит вектор vecIF из ИСК в ОСК
def IFtoOF(rIF, vIF, vecIF):
    cVec = np.cross(rIF, vIF)  # вектор момента импульса
    # орты связанной системы координат
    e3 = rIF / np.linalg.norm(rIF)
    e2 = cVec / np.linalg.norm(cVec)
    e1 = np.cross(e2, e3)
    # матрица перехода из ОСК в ИСК
    matrixOBF2IF = np.array([e1, e2, e3], dtype=float)
    vecOF = matrixOBF2IF.dot(vecIF)
    return vecOF

# параметры моделирования
dt = float(1.0) # шаг моделирования РК4
tFinal = 1000.0 # время моделирования
t = np.arange(0.0, tFinal, dt) # массив эпох

# орбитальные элементы для задания начальных данных r, v
a0 = 7*1e3 # большая полуось (km)
ecc0 = 0.0 # эксцентреситет - круговая орбита
trueAnomaly0 = np.deg2rad(15) # истинная аномалия
raan0 = np.deg2rad(0) # долгота восходящего узла
inc0 = np.deg2rad(90)  # наклонение
aop0 = np.deg2rad(0) # аргумент перицентра

# гравитационный параметр
mu = 398600.4418 # km^3 / sec^2

# тензор инерции
Jtense = np.diag([3.0,4.0,2.0])*1e-6 # кг*kм^2

# получаем начальные радиус-вектор и скорость в ИСО
[rIF0, velIF0] = orbitalToCartesian(a0, ecc0, trueAnomaly0, raan0, inc0, aop0, mu)

# кватернион ориентации, задаёт переход из ИСК в ССК
# отсылается к классу Quaternion  в файле quaternion.py
quat0 = Quaternion(np.array([1.0, 0, 0, 0], dtype= float))

# полная угловая скорость в ССК в начальный момент
omegaBF0 = np.array([0.001, 0.001, 0.001], dtype=float) # рад/с

# начальный фазовый вектор [rx, ry, rx, vx, vy, vz, q0, qi, qj, qk, omega1, omega2, omega3]
phaseVec = np.array([*rIF0, *velIF0, *quat0.q, *omegaBF0], dtype=float)
