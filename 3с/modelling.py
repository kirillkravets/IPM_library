import numpy as np
import matplotlib.pyplot as plt
from rkSolver import RK4Model
from init3c import *

# y = np.array([rIF, vIF, qBF, omegaBf])
# y[0:3] == rIF
# y[3:6] == vIF
# y[6:10] == qBF.q
# y[10:] == omegaBf
gravMomentMoving = lambda y: np.array([
    *y[3:6],
    *(-mu * y[0:3] / np.linalg.norm(y[0:3])**3),
    *(1.0 / 2 * (Quaternion(y[6:10]).vecMulQuat(y[10:]))),
    *(np.dot(
        np.linalg.inv(Jtense),
        np.cross(np.dot(Jtense, y[10:]), y[10:]) + (3 * mu / (np.linalg.norm(y[0:3]))**5) *
        np.cross(Quaternion(y[6:10]).IF2BF(y[0:3]), np.dot(Jtense, Quaternion(y[6:10]).IF2BF(y[0:3])))
    ))
    ], dtype=float)

ySolution = RK4Model(phaseVec, t, dt, gravMomentMoving)
omegaArr = ySolution[:, 10:]
# JacobiIntegral = 1 / 2 * np.dot(omegaArr, np.dot(Jtense, omegaArr)) +
#     3 / 2 *
rVecIF = ySolution[:, 0:3]
vVecIF = ySolution[:, 3:6]

kineticMoment = [np.linalg.norm(np.cross(rVecIF[i], vVecIF[i])) for i in range(len(t))]

plt.plot(t, kineticMoment)
plt.title('kineticMoment = func(t)')
plt.xlabel('kineticMoment')
plt.ylabel('t')
plt.minorticks_on()
plt.grid(which="both", axis='both')
plt.show()
plt.close()