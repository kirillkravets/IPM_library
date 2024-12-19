import matplotlib.pyplot as plt
import numpy as np

from rkSolver3c import RK4Model
from init3c import *

def gravMoment(rIF, Jtense, quatBF, mu):
    # rIF - радиус-вектор КА в ССК (np.ndarray dim3)
    # Jtense - тензор инерции (np.ndarray dim 3x3)
    # quat - кватернион ориентации из ИСК в ССК (np.ndarray dim4)

    rC  = np.linalg.norm(rIF)
    rBF = Quaternion(quatBF).IF2BF(rIF)
    JrBF = Jtense.dot(rBF)
    return 3 * mu / rC**5 * np.dot(rBF, JrBF)

# y = np.array([rIF, vIF, qBF, omegaBf])
# y[0:3] == rIF
# y[3:6] == vIF
# y[6:10] == qBF.q
# y[10:] == omegaBf

gravMomentMoving = lambda y: np.array([
    # изменение радиус вектора dr_dt = v
    *y[3:6],
    # изменение скорости, задача 2х тел dv_dt = -mu * rVec / r**3
    *(-mu * y[0:3] / np.linalg.norm(y[0:3])**3),
    # формула Эйлера для кватернионов dq_dt = 1/2[omegaBF x q]
    *(0.5 * (Quaternion(y[6:10]).quatMulVec(y[10:]))),
    # изменение полной угловой скорости
    # domega_dt = (J^{-1})([J(omega) x omega] + 3mu/r**3([A(rIF) x J(A(rIF))])
    # J - тензор инерции, A(rIF) = rBF = Qconj * rIF * Q, где Q - кватернион
    *(np.linalg.inv(Jtense)).dot(
        np.cross(Jtense.dot(y[10:]), y[10:]) + gravMoment(y[0:3], Jtense, y[6:10], mu)
            )
], dtype=float)

# интегрирование уравнений движения
ySolution = RK4Model(phaseVec, t, dt, gravMomentMoving)
# массив абсолютной угловой скорости в ССК
omegaBF = ySolution[:, 10:]
rVecIF = ySolution[:, 0:3]
vVecIF = ySolution[:, 3:6]
cVecIF = [np.cross(rVecIF[i], vVecIF[i]) for i in range(len(t))]

quatBF = [Quaternion(ySolution[i, 6:10]) for i in range(len(t))]
omegaOrbIF = [cVecIF[i] / (np.linalg.norm(rVecIF[i]))**2 for i in range(len(t))]
omegaOrbBF = [quatBF[i].IF2BF(omegaOrbIF[i]) for i in range(len(t))]
omegaRelBF = omegaBF - omegaOrbBF
# модуль орбитальной угловой скорости
omega0 = np.array([np.sqrt(mu / np.linalg.norm(rVecIF[i])**3) for i in range(len(t))])

# орты ССК
e3 = [(rVecIF[i] / np.linalg.norm(rVecIF[i])) for i in range(len(t))]
e2 = [(cVecIF[i] / np.linalg.norm(cVecIF[i])) for i in range(len(t))]

# интеграл Якоби h = 1/2 (omegaRel, J(omegaRel)) - 1/2 * omega0**2(E2, J(E2)) + 3/2(E3, J(E3))
JacobiIntegral = np.zeros(len(t))
for i in range(len(t)):
    # радиус-вектор центра масс в ССК
    rBF = quatBF[i].IF2BF(rVecIF[i])
    e3BF = quatBF[i].IF2BF(e3[i])
    e2BF = quatBF[i].IF2BF(e2[i])
    # расстояние до центра масс
    Rc = np.linalg.norm(rBF)
    JacobiIntegral[i] = 0.5 * (np.dot(omegaRelBF[i], Jtense.dot(omegaRelBF[i])) +
                          omega0[i]**2 * (3 * np.dot(e3BF, Jtense.dot(e3BF)) - np.dot(Jtense.dot(e2BF), e2BF)))

kineticMoment = np.array([(np.cross(rVecIF[i], vVecIF[i]) + Jtense.dot(omegaBF[i]))  for i in range(len(t))])
for i in range(3):
    plt.plot(t, kineticMoment[:, i], label = 'kin_mom_{:}'.format(i))
plt.title('kineticMoment = func(t)')
plt.ylabel('kineticMoment')
plt.xlabel('t')
plt.minorticks_on()
plt.grid(which="both", axis='both')
plt.savefig('kineticMoment3c')
plt.close()

plt.plot(t, JacobiIntegral)
plt.title('JacobiIntegral = func(t)')
plt.ylabel('JacobiIntegral')
plt.xlabel('t')
plt.minorticks_on()
plt.grid(which="both", axis='both')
plt.savefig('JacobiIntegral3c')
plt.close()

for i in range(3):
    omega = omegaBF[:, i]
    plt.plot(t, np.round(omega, 8), label = 'omega_{}'.format(i))
    plt.legend()
plt.title('angle velocity = func(t)')
plt.ylabel('angle velocity')
plt.xlabel('t')
plt.minorticks_on()
plt.grid(which="both", axis='both')
plt.savefig("angleVelocity3c")
plt.close()

# plt.plot(t, np.round(omega0,5))
# plt.title('angle velovity orb= func(t)')
# plt.ylabel('angle velovity orb')
# plt.xlabel('t')
# plt.legend()
# plt.minorticks_on()
# plt.grid(which="both", axis='both')
# plt.show()
# plt.savefig('angleVelovityOrb3c.png')
# plt.close()
#

ax = plt.figure().add_subplot(projection='3d')
# Prepare arrays x, y, z
x = rVecIF[:, 0]
y = rVecIF[:, 1]
z = rVecIF[:, 2]
ax.plot(np.round(x, 1), np.round(y,1), np.round(z, 1), label='parametric curve')
plt.legend()
plt.show()
plt.close()
