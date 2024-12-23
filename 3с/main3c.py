import matplotlib.pyplot as plt
import numpy as np

from rkSolver3c import RK4Model
from init3c import *

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
# лямбда-выражение, передаваемое в РК4, в качестве функции производной фазового вектора
gravMomentMoving = lambda y: np.array([
    # изменение радиус вектора dr_dt = v
    *y[3:6],
    # изменение скорости, задача 2х тел dv_dt = -mu * rVec / r**3
    *(-mu * y[0:3] / np.linalg.norm(y[0:3])**3),
    # формула Эйлера для кватернионов dq_dt = 1/2 q * omega
    *(0.5 * quatMulvec(y[6:10], y[10:])),
    # изменение полной угловой скорости
    # domega_dt = (J^{-1})([J(omega) x omega] + 3mu/r**3([A(rIF) x J(A(rIF))])
    *(np.linalg.inv(Jtense)).dot(
        np.cross(Jtense.dot(y[10:]), y[10:]) + calcGravMoment(y[0:3], y[6:10]))
], dtype=float)

# интегрирование уравнений движения
ySolution = RK4Model(phaseVec, t, dt, gravMomentMoving)

# орты положения равновесия ССК относительно ОСК
E1 = np.array([0, 0, 1], dtype=float)
E2 = np.array([0, 1, 0], dtype=float)
E3 = np.cross(E1, E2)

# интеграл Якоби h = 1/2 (omegaRel, J(omegaRel)) - 1/2 * omega0**2[(E2BF, J(E2BF)) - 3(E3BF, J(E3BF))]
JacobiIntegral = np.zeros(len(t))
JacobiIntegral1 = np.zeros(len(t))
omegaRelBF = np.array([np.zeros(3) for i in range(len(t))], dtype=float)

for i in range(len(t)):
    rVecIF = ySolution[i, 0:3]
    vVecIF = ySolution[i, 3:6]
    quatBF = ySolution[i, 6:10] # кватернион ориентации ССК относительно ИСК
    omegaBF = ySolution[i, 10:] # абсолютная угловая скорость в ССК

    # вектор приведённого момента импульса
    cVec = np.cross(rVecIF, vVecIF)
    # расстояние до центра масс КА
    Rc = np.linalg.norm(rVecIF)
    # переносная угловая скорость в ИСК
    omegaOrbIF = cVec / Rc**2
    # переносная угловая скорость в CСК
    omegaOrbBF = IF2BF(omegaOrbIF, quatBF)
    # квадрат модуля переносной скорости
    omega0sqr2 = mu / Rc**3
    # относительная угловая скорость в ССК
    omegaRelBF[i] = omegaBF - omegaOrbBF

    #орты ОСК в проекциях на ССК
    E2BF, E3BF = [IF2BF(OF2IF(rVecIF, vVecIF, Ei), quatBF) for Ei in [E2, E3]]
    JacobiIntegral1[i] = 0.5 * (
        np.dot(omegaRelBF[i], Jtense.dot(omegaRelBF[i])) + omega0sqr2 * (
            -np.dot(E2BF, Jtense.dot(E2BF) + 3 * np.dot(E3BF, Jtense.dot(E3BF))))
    )

    JacobiIntegral[i] = 0.5 * (
            np.dot(omegaRelBF[i], Jtense.dot(omegaRelBF[i])) + omega0sqr2 * (
        -np.dot(E2, Jtense.dot(E2) + 3 * np.dot(E3, Jtense.dot(E3))))
    )

plt.plot(t, JacobiIntegral, label = 'with E')
plt.plot(t, JacobiIntegral1, label = 'with EBF')

plt.legend()
plt.title('Jacobi Integral = func(t)')
plt.ylabel('Jacobi Integral')
plt.xlabel('t')
plt.minorticks_on()
plt.grid(which="both", axis='both')
plt.savefig('graphs3c/JacobiIntegral3c')
plt.show()
plt.close()

for i in range(3):
    omega = omegaRelBF[:, i]
    plt.plot(t, np.round(omega, 8), label = r'$\omega_{} $ rel'.format(i + 1))
plt.plot(t, [np.linalg.norm(ySolution[i, 10:]) for i in range(len(t))], label = r'$\omega_{abs}$')
plt.legend()
plt.title(r'$\omega$ = func(t)')
plt.ylabel(r'$\omega$')
plt.xlabel('t')
plt.minorticks_on()
plt.grid(which="both", axis='both')
plt.show()
plt.savefig("graphs3c/angleVelocity3c")
plt.close()

ax = plt.figure().add_subplot(projection='3d')
# Prepare arrays x, y, z
x = ySolution[:, 0]
y = ySolution[:, 1]
z = ySolution[:, 2]
ax.plot(np.round(x, 1), np.round(y,1), np.round(z, 1), label='parametric curve')
plt.legend()
plt.savefig('graphs3c/orbit')
plt.close()
