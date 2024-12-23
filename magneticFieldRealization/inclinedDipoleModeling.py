import numpy as np
from init import *
from magneticFieldRealization.conversions import IF2BF
import matplotlib.pyplot as plt
# IF - ИСК, BF - ССК, OF - ОСК

# Начальная долгота вектора направления магнитного поля Земли
latM0 = np.deg2rad(79)
# начальная широта вектора направления магнитного поля Земли
lonM0 = np.deg2rad(71)
# сидерический период Земли
tauE = 23 * 60 * 60 + 56 * 60 + 4 # сек
# угловая скорость вращения Земли
omegaE = 2 * np.pi / tauE

# Решаем уравнение с помощью метода Рунге-Кутты четвертого порядка
def RK4Model(y0, t, dt, vecFunction):
    # y - список размерности size(y0) x size(t),
    # vecFunction - векторное Лямбда-выражение модели движения
    # где y0 - вектор начальных данных, например, size[r, v, q, omega] = 12
    y = np.array([y0 for i in range(len(t))], dtype=float)
    y[0] = y0 # начальное условие
    for i in range(1, len(t)):
        k1 = vecFunction(y[i-1], t[i-1])
        k2 = vecFunction(y[i-1] + dt/2 * k1, t[i-1])
        k3 = vecFunction(y[i-1] + dt/2 * k2, t[i-1])
        k4 = vecFunction(y[i-1] + dt * k3, t[i-1])

        y[i] = y[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y # решение модели - список из фазовых векторов в каждый момент времени t

def quatMulvec(quat: np.ndarray[float], vec):
    '''
    осуществляет кватернионное умножение с вектором: quat * vec
    :param quat: кватернион size 4x1
    :param vec: трёхмерный вектор size 3x1
    :return: кватернион size 4x1
    '''

    q0, qVec = quat[0], quat[1:]
    qNew0 = -np.dot(qVec, vec)
    qNewVec = q0 * vec + np.cross(qVec, vec)
    return np.concatenate(([qNew0], qNewVec))

def quatMulQuat(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])

def calcGravMoment(rIF, quatBF: np.ndarray[float]):
    '''
    Подсчитывает гравитационный момент Земли в ССК
    :param rIF: радиус-вектор КА в ИСК (size 3x1)
    :param quatBF: кватернион ориентации из ИСК в ССК (size 4x1)
    :return:
    '''

    rC  = np.linalg.norm(rIF) # расстояние до центра масс КА
    rBF = IF2BF(rIF, quatBF) # радиус-вектор КА в CСК
    JrBF = Jtense.dot(rBF)

    return 3 * mu / rC**5 * np.cross(rBF, JrBF)

def calcMagneticFieldIF(rVecIF, time):
    '''
    считает вектор магнитной индукции поля Земли
    :param kIF: вектор направления магнитного диполя Земли
    :param rVecIF: положение аппарата в ИСК
    :param quat: кватернион ориентации аппарата
    :param timeMoment: время, прошедшее с начала моделирования
    :return: вектор магнитной индукции в ССК
    '''

    kIF = np.array([np.cos(lonM0 + omegaE * time) * np.sin(latM0),
                   np.sin(lonM0 + omegaE * time) * np.sin(latM0),
                   np.cos(latM0)])

    r = np.linalg.norm(rVecIF)
    BvecIF = -M / r**5 * (kIF * r - 3 * np.dot(kIF, rVecIF) * rVecIF / r**2)
    return BvecIF

def calcMagneticMomentBF(satMagnMomOF, rIF, vIF, quatBF, time):
    '''
    Подсчитывает магнитный момент Земли в ССК, действующий на КА
    :param satMagnMomOF: собственный дипольный момент КА в ОСК
    :param kIF:
    :param quatBF: кватернион ориентации ССК относительно ИСК
    :param rIF: радиус-вектор ЦМ КА в ИСК
    :param vIF: вектор скорости ЦМ КА в ИСК
    :param t: время с начала моделирования
    :return: магнитный момент в ССК
    '''
    satMagnMomBF = IF2BF(OF2IF(rIF, vIF, satMagnMomOF), quatBF)

    r = np.linalg.norm(rIF)
    # подсчитываем магнитное поле Земли
    BvecIF = calcMagneticFieldIF(rIF, time)
    BvecBF = IF2BF(BvecIF, quatBF)
    return np.cross(satMagnMomBF, BvecBF)

# y = np.array([rIF, vIF, qBF, omegaBf])
# y[0:3]  == rIF радиус-вектор ИСК
# y[3:6]  == vIF скорость ИСК
# y[6:10] == qBF кватернион ориентации ССК
# y[10:12] == omegaBF полная угловая скорость ССК
# tMom - время с начала моделирования
# уравнение движения, передаваемое в РК4

motionEquation = lambda y, tMom: np.array([
    # изменение радиус вектора dr_dt = v
    *y[3:6],
    # изменение скорости, задача 2х тел dv_dt = -mu * rVec / r**3
    *(-mu * y[0:3] / np.linalg.norm(y[0:3])**3),
    # формула Эйлера для кватернионов dq_dt = 1/2 q * omega
    *(0.5 * quatMulvec(y[6:10], y[10:])),
    # изменение полной угловой скорости
    # domega_dt = (J^{-1})([J(omega) x omega] + 3mu/r**3([A(rIF) x J(A(rIF))])
    *(np.linalg.inv(Jtense)).dot(
        np.cross(Jtense.dot(y[10:]), y[10:]) + calcGravMoment(y[0:3], y[6:10])
        + calcMagneticMomentBF(satMagnMomentOF, y[0:3], y[3:6], y[6:10], tMom))
], dtype=float)

# интегрирование уравнений движения
ySolution = RK4Model(phaseVec, t, dt, motionEquation)
# положение КА на каждый момент времени
rIF = ySolution[:, 0:3]
# кватернион ориентации на каждый момент времени
quatBF = ySolution[:, 6:10]

# магнитное поле, полученное в результате интегрирования уравнений движения
BincIF = np.zeros_like(rIF)
BincBF = np.zeros_like(rIF)

# оценка магнитного поля
BincIFest = np.zeros_like(rIF)
BincBFest = np.zeros_like(rIF)

# кватернион ориентации магнитного датчика в ДСК относительно ССК
quatM = np.array([0.1, 0.1, 0.0, 0.2], dtype=float)
quatM /= np.linalg.norm(quatM)
# кватернион шума (связан с расположением датчика)
deltaQuat = np.array([np.random.normal(0.0, 0.01) for i in range(4)])

for i in range(len(t)):
    BincIF[i] = calcMagneticFieldIF(rIF[i], t[i])
    QuatGen = quatMulQuat(quatBF[i], quatM)
    QuatGen = quatMulQuat(QuatGen, deltaQuat)
    QuatGen = quatMulQuat(QuatGen, np.concatenate(([quatM[0]], -quatM[1:])))
    # измеренное магнитное поле в СО датчика (зашумляем)
    BincSFmeas = IF2SF(BincIF[i], QuatGen) + np.random.normal(0.0, 1e-6, size=BincIF[i].shape)
    # получаем искомую оценку магнитного поля в ИСК
    BincIFest[i] = SF2IF(BincSFmeas, QuatGen)

projection = ['x', 'y', 'z']
for i in range(3):
    plt.plot(t, BincIF[:, i], label = 'B{} real'.format(projection[i]))
    plt.plot(t, BincIFest[:, i], label = 'B{} measured'.format(projection[i]))
    plt.legend()
    plt.title('B = func(t)')
    plt.ylabel('B, Тл')
    plt.xlabel('t, сек')
    plt.minorticks_on()
    plt.grid(which="both", axis='both')
    plt.show()
    plt.savefig("inclinedMagnDipole/B{}".format(projection[i]))
    plt.close()

# ax = plt.figure().add_subplot(projection='3d')
# # Prepare arrays x, y, z
# x = ySolution[:, 0]
# y = ySolution[:, 1]
# z = ySolution[:, 2]
# ax.plot(np.round(x, 1), np.round(y,1), np.round(z, 1), label='parametric curve')
# plt.legend()
# plt.show()
# plt.close()

