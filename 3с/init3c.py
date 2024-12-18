import  numpy as np
from orbitalToCartesian import orbitalToCartesian
from quaternion import Quaternion

def IFtoOF(rIF, vIF, vecIF):
    cVec = np.cross(rIF, vIF)  # вектор момента импульса
    # орты связанной системы координат
    e3 = rIF / np.linalg.norm(rIF)
    e2 = cVec / np.linalg.norm(cVec)
    e1 = np.cross(e2, e3)
    # матрица перехода из ОСК в ИСК
    matrixOBF2IF = np.array([e1, e2, e3], dtype=float)
    vecOF = np.dot(matrixOBF2IF, vecIF)
    return vecOF


dt = float(1.0) # шаг моделирования РК4
tFinal = 10000.0 # время моделирования
t = np.arange(0.0, tFinal, dt) # массив эпох

a0 = 7000 # большая полуось (км)
ecc0 = 0.5 # эксцентреситет
trueAnomaly0 = np.deg2rad(15) # истинная аномалия
raan0 = np.deg2rad(45) # долгота восходящего узла
inc0 = np.deg2rad(40)  # наклонение
aop0 = np.deg2rad(120) # аргумент перицентра
mu = 398600.4418 # km^3 / sec^2 # гравитационный параметр

# тензор инерции
Jtense = np.array([[0.4, 0, 0], [0, 0.5, 0], [0, 0, 0.3]], dtype=float)
# получаем начальные радиус-вектор и скорость в ИСО
[rIF0, velIF0] = orbitalToCartesian(a0, ecc0, trueAnomaly0, raan0, inc0, aop0, mu)

# ориентация КА относительно ОСК
# отсылается к классу Quaternion  в файле quaternion.py
quat0 = Quaternion(np.random.normal(0, 1, 4))

# начальная относительная угловая скорость в ССК задаётся случайным образом
omegaRelBF = np.array([0.001, np.random.normal(np.pi / 90, np.pi / 900), 0.001])
# (переносная) угловая скорость движения по орбите в начальный момент в ИСО
omegaIForb0 = np.cross(rIF0, velIF0) / np.linalg.norm(rIF0)**2
# (переносная) угловая скорость движения по орбите в начальный момент в ССК
omegaBForb0 = quat0.IF2BF(omegaIForb0)
# полная угловая скорость в ССК
omegaBF0 = omegaBForb0 + omegaRelBF

# начальный фазовый вектор [rx, ry, rx, vx, vy, vz, q0, qi, qj, qk, omega1, omega2, omega3]
phaseVec = np.array([*rIF0, *velIF0, *quat0.q, *omegaBF0], dtype=float)
