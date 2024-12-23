import matplotlib.pyplot as plt
import numpy as np

from init import *
from rkSolver import RK4Model


def calcMagneticFieldIF(kIF, rVecIF):
    '''
    считает вектор магнитной индукции поля Земли
    :param kIF:
    :param rVecIF: положение аппарата в ИСК
    :param quat: кватернион ориентации аппарата
    :return: вектор магнитной индукции в ССК
    '''
    r = np.linalg.norm(rVecIF)
    BvecIF = -M / r**5 * (kIF * r - 3 * np.dot(kIF, rVecIF) * rVecIF / r**2)
    return BvecIF

def calcMagneticMomentBF(satMagnMomOF, kIF, rIF, vIF, quatBF):
    '''
    Подсчитывает магнитный момент Земли, действующий на КА
    :param satMagnMomOF: собственный дипольный момент КА в ОСК
    :param kIF:
    :param quatBF: кватернион ориентации ССК относительно ИСК
    :param rIF: радиус-вектор ЦМ КА в ИСК
    :param vIF: вектор скорости ЦМ КА в ИСК
    :return: магнитный момент в ССК
    '''
    satMagnMomBF = IF2BF(OF2IF(rIF, vIF, satMagnMomOF), quatBF)

    r = np.linalg.norm(rIF)
    # подсчитываем магнитное поле Земли
    BvecIF = - M / r ** 5 * (kIF * r - 3 * np.dot(kIF, rIF) * rIF)
    BvecBF = IF2BF(BvecIF, quatBF)
    return np.cross(satMagnMomBF, BvecBF)


def calcGravMoment(rIF, quatBF: np.ndarray[float]):
    '''
    :param rIF: радиус-вектор КА в ИСК (size 3x1)
    :param quatBF: кватернион ориентации из ИСК в ССК (size 4x1)
    :return:
    '''

    rC  = np.linalg.norm(rIF) # расстояние до центра масс КА
    rBF = IF2BF(rIF, quatBF) # радиус-вектор КА в CСК
    JrBF = Jtense.dot(rBF)

    return 3 * mu / rC**5 * np.cross(rBF, JrBF)


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


# y = np.array([rIF, vIF, qBF, omegaBf])
# y[0:3]  == rIF радиус-вектор ИСК
# y[3:6]  == vIF скорость ИСК
# y[6:10] == qBF кватернион ориентации ССК
# y[10:]  == omegaBF полная угловая скорость ССК
# звездочка перед массивом означает распаковку массива
# *[1,2,3] <=> 1, 2, 3

# уравнение движения, передаваемое в РК4
motionEquation = lambda y: np.array([
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
        + calcMagneticMomentBF(satMagnMomentOF, kDirectIF, y[0:3], y[3:6], y[6:10]))
], dtype=float)

# интегрирование уравнений движения
ySolution = RK4Model(phaseVec, t, dt, motionEquation)
rIF = ySolution[:, 0:3]
# кватернион ориентации на каждый момент времени
quatOrientation = ySolution[:, 6:10]

BdirectIFmeas = np.array([np.zeros(3, dtype=float) for _ in range(len(t))])
BdirectBFmeas = np.array([np.zeros(3, dtype=float) for _ in range(len(t))])
BdirectIF = np.array([np.zeros(3, dtype=float) for _ in range(len(t))])
BdirectBF = np.array([np.zeros(3, dtype=float) for _ in range(len(t))])

for i in range(len(t)):
    BdirectIF[i] = calcMagneticFieldIF(kDirectIF, rIF[i])
    BdirectBF[i] = IF2BF(BdirectIF[i], quatOrientation[i])

    BdirectIFmeas[i] = BdirectIF[i] + np.random.normal(0, 1e-6, np.shape(BdirectIF[i]))
    BdirectBFmeas[i] = BdirectBF[i] + np.random.normal(0, 1e-6, np.shape(BdirectBF[i]))



index = np.array(['x', 'y', 'z'], dtype=str)
linestyle = ['solid', 'dotted', 'dashed']


for i in range(len(index)):

    fig, ax = plt.subplots()
    ax.plot(t, BdirectIF[:, i], linestyle = 'solid', label=r"$Breal{} = function(t)$".format(index[i]))
    ax.plot(t, BdirectIFmeas[:, i], linestyle='dotted', label=r"$Bmeas{} = function(t)$".format(index[i]))
    # ax.plot(t, [np.linalg.norm(BdirectIF[j]) for j in range(len(t))], label = r'|B|')
    ax.legend(loc=2) # upper left corner
    ax.set_ylabel(r'B, Тл', fontsize=8)
    ax.set_xlabel(r'$t$', fontsize=8)
    ax.set_title('Проекции вектора магнитной индукции\n для модели прямого диполя')
    plt.minorticks_on()
    plt.grid(which='both')
    plt.savefig('dirrectMagnDipole/dirrectDipoleB{}'.format(i))
    plt.show()
    plt.close()

ax = plt.figure().add_subplot(projection='3d')
# Prepare arrays x, y, z
x = ySolution[:, 0]
y = ySolution[:, 1]
z = ySolution[:, 2]
ax.plot(np.round(x, 1), np.round(y,1), np.round(z, 1), label='parametric curve')
plt.legend()
plt.savefig('orbitMotion')
plt.close()
