import numpy as np

# переводит заданный набор элементов кеплеровой в декартовы переменные
def orbital2Cartesian(a, ecc, trueAnomaly, raan, inc, aop, mu):
    '''
        a:              большая полуось             (km)
        ecc:            эксцентреситет              (отн. ед.)
        trueAnomaly:    истинная аномалия           (degrees)
        raan:           долгота восходящего узла    (degrees)
        inc:            наклонение                  (degrees)
        aop:            аргумент перицентра         (degrees)
        mu:             гравитационный параметр     (км^3 / c^2)
    '''
    # переводим угловые элементы в радианы
    trueAnomaly, raan, inc, aop = [np.deg2rad(angleEl) for angleEl in [trueAnomaly, raan, inc, aop]]
    p = a * (1 - ecc ** 2) # параметр орбиты
    r = p / (1 + ecc * np.cos(trueAnomaly)) # расстояние до КА
    # находим производные истинной аномалии и расстояния
    trueAnomalyDot = np.sqrt(p * mu) / r**2
    rDot = - p * ecc * trueAnomalyDot * np.sin(trueAnomaly) / (1 + ecc * np.cos(trueAnomaly)) ** 2
    # задаём радиус-вектор и скорость в ОСК
    rVecOSC = np.array([r * np.cos(trueAnomaly), r * np.sin(trueAnomaly), 0])
    vxOSC = rDot * np.cos(trueAnomaly) - r * trueAnomalyDot * np.sin(trueAnomaly)
    vyOSC = rDot * np.sin(trueAnomaly) + r * trueAnomalyDot * np.cos(trueAnomaly)
    velVecOSC = np.array([vxOSC, vyOSC, 0])
    # определяем матрицы поворота 3-1-3 из ОСК в ИСК
    rotRaan = np.array([[np.cos(raan), -np.sin(raan), 0],
                        [np.sin(raan), np.cos(raan), 0],
                        [0, 0, 1]])
    rotInc = np.array([[1, 0, 0],
                       [0, np.cos(inc), -np.sin(inc)],
                       [0, np.sin(inc), np.cos(inc)]])
    rotAop = np.array([[np.cos(aop), -np.sin(aop), 0],
                       [np.sin(aop), np.cos(aop), 0],
                       [0, 0, 1]])
    # осуществляем переход в ИСК
    rVecISC = rotRaan.dot(rotInc.dot(rotAop.dot(rVecOSC)))
    velVecISC = rotRaan.dot(rotInc.dot(rotAop.dot(velVecOSC)))
    return rVecISC, velVecISC

# переводит вектор vecIF из ОСК в ИСК
def OF2IF(rIF, vIF, vecOF):
    '''
    все входные и выходные параметры имеют тип np.ndarray(size = 3x1)
    :param rIF: радиус-вектор КА в ИСК
    :param vIF: вектор скорости КА в ИСК
    :param vecOF: вектор в ОСК
    :return: вектор в ИСК
    '''
    cVec = np.cross(rIF, vIF)  # вектор момента импульса
    # орты ОСК, заданные в ИСК
    e3 = rIF / np.linalg.norm(rIF)
    e2 = cVec / np.linalg.norm(cVec)
    e1 = np.cross(e2, e3)
    # матрица перехода из ОСК в ИСК
    matrixOBF2IF = np.array([e1, e2, e3], dtype=float)

    return matrixOBF2IF.dot(vecOF)

def IF2BF(vecIF: np.ndarray[float], quatBF: np.ndarray[float]):
    '''
    :param vecIF: вектор в ИСК size = 3x1
    :param quatBF: кватернион ориентации КА в ССК size = 4x1
    :return: вектор в ССК size = 3x1
    '''
    q0, q1, q2, q3 = quatBF
    # преобразуем кватернион поворота в матрицу поворота
    RotMatr = np.array([
        [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]
    ], dtype=float)

    return RotMatr.dot(vecIF)

