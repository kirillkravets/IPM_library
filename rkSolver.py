import  numpy as np

# Решаем уравнение с помощью метода Рунге-Кутты четвертого порядка
def RK4Model(y0, t, dt, vecFunction):
    y = np.array([y0 for i in range(len(t))], dtype=float)
    y[0] = y0
    for i in range(1, len(t)):
        k1 = vecFunction(y[i-1])
        k2 = vecFunction(y[i-1] + dt/2 * k1)
        k3 = vecFunction(y[i-1] + dt/2 * k2)

        k4 = vecFunction(y[i-1] + dt * k3)
        y[i] = y[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y

def RK4(y0, t, dt, vecFunction):
    y = y0
    k1 = vecFunction(y)
    k2 = vecFunction(y + dt/2 * k1)
    k3 = vecFunction(y + dt/2 * k2)
    k4 = vecFunction(y + dt * k3)
    y += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y