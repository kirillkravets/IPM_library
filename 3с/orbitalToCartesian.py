import numpy as np

def orbitalToCartesian(a, ecc, trueAnomaly, raan, inc, aop, mu):
    p = a * (1 - ecc ** 2)
    r = p / (1 + ecc * np.cos(trueAnomaly))
    trueAnomalyDot = np.sqrt(p * mu) / r**2
    rDot = - p * ecc * trueAnomalyDot * np.sin(trueAnomaly) / (1 + ecc * np.cos(trueAnomaly)) ** 2
    rVecOSC = np.array([r * np.cos(trueAnomaly), r * np.sin(trueAnomaly), 0])
    vxOSC = rDot * np.cos(trueAnomaly) - r * trueAnomalyDot * np.sin(trueAnomaly)
    vyOSC = rDot * np.sin(trueAnomaly) + r * trueAnomalyDot * np.cos(trueAnomaly)
    velVecOSC = np.array([vxOSC, vyOSC, 0])

    rotRaan = np.array([[np.cos(raan), -np.sin(raan), 0],
                        [np.sin(raan), np.cos(raan), 0],
                        [0, 0, 1]])
    rotInc = np.array([[1, 0, 0],
                       [0, np.cos(inc), -np.sin(inc)],
                       [0, np.sin(inc), np.cos(inc)]])
    rotAop = np.array([[np.cos(aop), -np.sin(aop), 0],
                       [np.sin(aop), np.cos(aop), 0],
                       [0, 0, 1]])
    rVecISC = rotRaan.dot(rotInc.dot(rotAop.dot(rVecOSC)))
    velVecISC = rotRaan.dot(rotInc.dot(rotAop.dot(velVecOSC)))
    return rVecISC, velVecISC

