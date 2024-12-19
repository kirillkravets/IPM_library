import numpy as np

inc0 = np.deg2rad(23)
raan0 = np.deg2rad(45)
aop0 = np.deg2rad(283)
a0 = 7000 # km
ecc0 = 0.5
trueAnomaly0 = np.deg2rad(15)
mu = 398600.4418 # km^3 / sec^2

dt = float(1)
tFinal = 10000.0
t = np.arange(0.0, tFinal, dt)

stateRV = np.zeros(6)



stateVec = []
