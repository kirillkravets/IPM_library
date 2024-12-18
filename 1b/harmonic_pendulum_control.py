import  matplotlib.pyplot as plt
from init1b import *
from rkSolver import RK4Model

def pidControl(yMeas, yRef, kD, kI, kP, dTimeControl, N):
    [xMeas, derxMeas] = yMeas[-1]
    [xRef, derxRef] = yRef
    control = -kP * (xMeas - xRef) - kD * (derxMeas - derxRef)
    for i in range(N):
        control -= kI * (yMeas[i, 0] - yRef[0]) * dTimeControl
    return control

def LyapunovControl(yMeas, yRef, kLl, kLd, m, g, l):
    [xMeas, derxMeas] = yMeas
    [xRef, derxRef] = yRef
    return -kLl * (xMeas - xRef) - kLd * (derxMeas - derxRef) + g / l * np.sin(xMeas)

def HarmonicPendulum(y0, t, dt, yRef, dtControl, sigma, kP, kI, kD):

    stepCorrection = round(dtControl / dt)
    if dtControl / dt != stepCorrection:
        dtControl = stepCorrection * dt

    timeCorrection = round(t[-1] / dtControl)
    if t[-1] / dtControl != timeCorrection:
        t = np.arange(0, timeCorrection * dtControl, dt)

    y = np.array([np.array([0,0], dtype= float) for i in range(len(t))])
    u = np.array([0]*len(t), dtype=float)

    yMeas = np.array([np.array([0, 0], dtype= float) for i in range(timeCorrection)])

    for i in range(1, timeCorrection + 1):
        yMeas[i - 1] = y0 + np.random.normal(0, sigma)

        tRK4 = t[stepCorrection * (i - 1): stepCorrection * i:]
        control = LyapunovControl(yMeas[i - 1], yRef, kLl, kLd, m, g, l)
        #control = pidControl(yMeas[:i:], yRef, kD, kI, kP, dtControl, i)
        u[stepCorrection * (i - 1): stepCorrection * i:] = np.array([control]*stepCorrection)
        PendulumFunctionControl = lambda x: np.concatenate(([x[1], -g / l * np.sin(x[0]) + control ]), axis=None)
        y[stepCorrection * (i - 1): stepCorrection * i] = RK4Model(y0, tRK4, dt, PendulumFunctionControl)
        y0 = y[stepCorrection * i - 1]

    alpha = np.rad2deg(y[:, 0])
    omega = y[:, 1]

    fig = plt.figure(figsize=(10, 5))
    plt.xlabel('t, sec')
    plt.ylabel('control')
    plt.title('control = function(time)')
    plt.plot(t, u)
    plt.savefig('control1b')
    plt.show()

    fig = plt.figure(figsize=(10, 5))
    plt.xlabel('t, sec')
    plt.ylabel('alpha ')
    plt.title('angle = function(time)')
    plt.plot(t, alpha)
    plt.plot(t, np.array([np.rad2deg(yRef[0])] * len(t)))
    plt.savefig('angle1b')
    plt.show()
    #
    fig = plt.figure(figsize=(10, 5))
    plt.xlabel('t, sec')
    plt.ylabel('omega')
    plt.title('omega = function(time)')
    plt.plot(t, omega)
    plt.plot(t, np.array([yRef[1]] * len(t)))
    plt.savefig('omega1b')
    plt.show()

    fig = plt.figure(figsize=(10, 5))
    plt.xlabel('t, sec')
    plt.ylabel('angle measurements')
    plt.title('angleMeas = function(time)')
    x = np.arange(0, dtControl * timeCorrection, dtControl)
    plt.plot(x, np.rad2deg(yMeas[:, 0]))
    plt.plot(x, np.array([np.rad2deg(yRef[0])] * len(x)))
    plt.savefig('angleMeas1b')
    plt.show()

    fig = plt.figure(figsize=(10, 5))
    plt.xlabel('t, sec')
    plt.ylabel('angle velocity measurements')
    plt.title('omegaMeas = function(time)')
    plt.plot(x, yMeas[:, 1])
    plt.plot(x, np.array([yRef[1]] * len(x)))
    plt.savefig('omegaMeas1b')
    plt.show()

    fig = plt.figure(figsize=(10, 5))
    plt.xlabel('t, sec')
    plt.ylabel('relative angle ')
    plt.title('relAngle = function(time)')
    plt.plot(x, np.rad2deg(yRef[0] - yMeas[:, 0]))
    plt.savefig('angleRel1b')
    plt.show()

    fig = plt.figure(figsize=(10, 5))
    plt.xlabel('t, sec')
    plt.ylabel('relative angle velocity ')
    plt.title('relOmega = function(time)')
    plt.plot(x, np.array(yRef[1] - yMeas[:, 1]))
    plt.savefig('omegaRel1b')
    plt.show()

    return alpha[-1]

x = HarmonicPendulum(y0, t, dt, yRef, dtControl, sigma, kP, kI, kD)
