import numpy as np

inc = np.deg2rad(40)
raan = np.deg2rad(45)
aop = np.deg2rad(120)
a = 7000 # km
ecc = 0.5
trueAnomaly = np.deg2rad(15)
mu = 398600.4418 # km^3 / sec^2

dt = float(1)
tFinal = 10000.0
t = np.arange(0.0, tFinal, dt)

stateRV = np.zeros(6)



stateVec = []
