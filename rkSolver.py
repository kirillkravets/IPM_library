import  numpy as np

# Решаем уравнение с помощью метода Рунге-Кутты четвертого порядка
def RK4Model(y0, t, dt, vec2Function):
    y = np.array([[0, 0] for i in range(len(t))], dtype=float)
    y[0] = y0
    for i in range(1, len(t)):
        k1 = vec2Function(y[i-1])
        k2 = vec2Function(y[i-1] + dt/2 * k1)
        k3 = vec2Function(y[i-1] + dt/2 * k2)
        k4 = vec2Function(y[i-1] + dt * k3)
        y[i] = y[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y
