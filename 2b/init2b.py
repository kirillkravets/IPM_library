import numpy as np

inc0 = np.deg2rad(30)
raan0 = np.deg2rad(45)
aop0 = np.deg2rad(30)
a0 = 7000 # km
ecc0 = 0.5
trueAnomaly0 = np.deg2rad(15)
mu = 398600.4418 # km^3 / sec^2
Rearth = 6378.137
J2 = 1.08262668e-3

dt = float(1)
tFinal = 100000.0
t = np.arange(0.0, tFinal, dt)

stateRV = np.zeros(6)



stateVec = []
