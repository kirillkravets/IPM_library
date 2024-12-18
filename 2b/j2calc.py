import numpy as np

from init2b import *
import matplotlib.pyplot as plt
from rkSolver import RK4Model, RK4

def accelerationJ2(rVec):
    [rx, ry, rz]  = rVec
    r = np.linalg.norm(rVec)
    delta = 3 / 2 * J2 * mu * Rearth**2
    wx = - delta * rx / r**5 * (5 * rz**2 / r**2 - 1)
    wy = - delta * ry / r**5 * (5 * rz**2 / r**2 - 1)
    wz = - delta * rz / r**5 * (5 * rz**2 / r**2 - 3)
    wJ2  = np.array([wx, wy, wz])
    wCenter = - mu * rVec / r**3
    return wCenter + wJ2

def orbitalToCartesian(a, ecc, trueAnomaly):
    p = a * (1 - ecc ** 2)
    r = p / (1 + ecc * np.cos(trueAnomaly))
    trueAnomalyDot = np.sqrt(p * mu) / r**2
    rDot = - p * ecc * trueAnomalyDot * np.sin(trueAnomaly) / (1 + ecc * np.cos(trueAnomaly)) ** 2
    rVecOSC = np.array([r * np.cos(trueAnomaly), r * np.sin(trueAnomaly), 0])
    vxOSC = rDot * np.cos(trueAnomaly) - r * trueAnomalyDot * np.sin(trueAnomaly)
    vyOSC = rDot * np.sin(trueAnomaly) + r * trueAnomalyDot * np.cos(trueAnomaly)
    velVecOSC = np.array([vxOSC, vyOSC, 0])
    return np.concatenate((rVecOSC, velVecOSC), axis=None)

def cartesianToOrbital(stateRV, mu):

    rVec = stateRV[0:3] # radius vector in ISC
    r = np.linalg.norm(rVec)

    vVec = stateRV[3:] # velocity vector in ISC
    v = np.linalg.norm(vVec)

    cVec = np.cross(rVec, vVec) # moment of the pulse vector
    c = np.linalg.norm(cVec)

    fVec = np.cross(vVec, cVec) - mu * rVec / r # Laplace vector
    f = np.linalg.norm(fVec)

    p = c**2 / mu # orbit parameter

    ecc =  f / mu # eccentricity

    a = p / (1 - ecc**2) # large semi-axis

    zAxis = np.array([0, 0, 1.0], dtype=float)

    inc = np.acos(np.dot(cVec, zAxis)/ c) # inclination
    lVec = np.cross(zAxis, cVec)

    raan = np.rad2deg(np.arctan2(lVec[1], lVec[0])) # longitude of the ascending node

    clCross = np.cross(cVec, lVec)
    clVec  = clCross / np.linalg.norm(clCross)
    lVecNorm = lVec / np.linalg.norm(lVec)

    eccVec = 1 / mu * (np.cross(vVec, cVec) - mu * (rVec / r))
    # aop = (np.arctan2(eccVec[1], eccVec[0]) - raan) % (2 * np.pi) if ecc > 1e-8 else None  # argument of pericenter else NaN
    aop = np.rad2deg(np.arctan2(np.dot(clVec, fVec), np.dot(lVecNorm, fVec)))
    trueAnomaly = np.rad2deg((np.arccos(np.dot(eccVec/ecc, rVec/r))) % (2 * np.pi)) if ecc > 1e-8 else None

    return np.array([inc, raan, a, ecc, aop, trueAnomaly, c, f])

def OSCtoISC(raan, inc, aop, vectorOSC):
    rotRaan = np.array([[np.cos(raan), -np.sin(raan), 0],
                      [np.sin(raan), np.cos(raan), 0],
                      [0, 0, 1]])
    rotInc = np.array([[1, 0, 0],
                     [0, np.cos(inc), -np.sin(inc)],
                     [0, np.sin(inc), np.cos(inc)]])
    rotAop = np.array([[np.cos(aop), -np.sin(aop), 0],
                     [np.sin(aop), np.cos(aop), 0],
                     [0, 0, 1]])

    return rotRaan.dot(rotInc.dot(rotAop.dot(vectorOSC)))

stateOSC0 = orbitalToCartesian(a, ecc, trueAnomaly)

stateISC0 = np.zeros(6)
stateISC0[0:3] = OSCtoISC(raan, inc, aop, stateOSC0[0:3])
stateISC0[3:] = OSCtoISC(raan, inc, aop, stateOSC0[3:])

twoBodyModel = lambda stateVec: np.concatenate((stateVec[3:], accelerationJ2(stateVec[0:3])),axis=None)
stateRVarr = RK4Model(stateISC0, t, dt, twoBodyModel)
ax = plt.figure().add_subplot(projection='3d')

# Prepare arrays x, y, z
x = stateRVarr[:, 0]
y = stateRVarr[:, 1]
z = stateRVarr[:, 2]
ax.plot(x, y, z, label='parametric curve')
ax.legend()
plt.savefig('3dOrbitGraph.png')
plt.show()
plt.close()

orbitalElements = np.array([[0.0]*len(cartesianToOrbital(stateISC0, mu)) for i in  range(len(t))])

for i in range(len(t)):
    orbitalElements[i] = cartesianToOrbital(stateRVarr[i], mu)

orbitalTitle = ['inc', 'raan', 'a', 'ecc', 'aop', 'trueAnomaly', 'c', 'f']
for i in range(len(orbitalTitle)):
    plt.plot(t, np.round(orbitalElements[:, i],1))
    plt.title(orbitalTitle[i] + '= func(t)')
    plt.xlabel('t')
    plt.ylabel(orbitalTitle[i])
    plt.minorticks_on()
    plt.grid(which='both', axis='both')
    plt.savefig(orbitalTitle[i] + '2b.png')
    plt.close()

a = orbitalElements[:, 2]
ecc = orbitalElements[:, 3]
inc = orbitalElements[:, 0]

raanDot = 3/2 * J2 * (Rearth / (a * (1 - ecc**2)))**2 * np.sqrt(mu / a**3) * np.cos(inc)
aopDot =  -3/4 * J2 * (Rearth / (a * (1 - ecc**2)))**2 * np.sqrt(mu / a**3) * (5 * (np.cos(inc))**2 - 1)

raanAnalytical = np.zeros(len(raanDot))
aopAnalytical  = np.zeros(len(aopDot))

for i in range(len(t)):
    raanAnalytical[i] = raanDot[i] * i * dt
    aopAnalytical[i] = aopDot[i] * i * dt

titleAnalytical = ['raanTh', 'aopTh']
y = [raanAnalytical, aopAnalytical]
yDig = [orbitalElements[:, 1] - orbitalElements[0, 1], orbitalElements[:, 4] - orbitalElements[0, 4]]
for i in range(len(titleAnalytical)):
    plt.plot(t, np.rad2deg(y[i]))
    plt.plot(t, yDig[i])
    plt.title(titleAnalytical[i] + '= function(t)')
    plt.xlabel('t')
    plt.ylabel(titleAnalytical[i])
    plt.minorticks_on()
    plt.grid(which='both', axis='both')
    plt.savefig(titleAnalytical[i] + '.png')
    plt.close()