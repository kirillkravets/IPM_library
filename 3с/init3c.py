import  numpy as np
from orbitalToCartesian import orbitalToCartesian
from quaternion import Quaternion
dt = float(1)
tFinal = 10000.0
t = np.arange(0.0, tFinal, dt)

a0 = 7000 # km
ecc0 = 0.5
trueAnomaly0 = np.deg2rad(15)
raan0 = np.deg2rad(45)
inc0 = np.deg2rad(40)
aop0 = np.deg2rad(120)
mu = 398600.4418 # km^3 / sec^2



Jtens = np.array([[0.4, 0, 0], [0, 0.5, 0], [0, 0, 0.3]], dtype=float)

[rIF0, velIF0] = orbitalToCartesian(a0, ecc0, trueAnomaly0, raan0, inc0, aop0, mu)

angleVel0 = np.cross(rIF0, velIF0) / np.linalg.norm(rIF0)**2

quat0 = Quaternion(np.random.normal(0, 1, 4))
print(rIF0, '\n', velIF0, '\n', angleVel0, '\n', quat0.getValue())

pulMomVec = np.cross(rIF0, velIF0)

e3 = rIF0 / np.linalg.norm(rIF0)
e2 = pulMomVec / np.linalg.norm(pulMomVec)
e1 = np.cross(e2, e3)
matrIF2OF= np.array([e1, e2, e3], dtype=float)

