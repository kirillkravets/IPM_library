import numpy as np

dt = 0.001 #s
t = np.arange(0, 100, dt, dtype=float)
alpha = np.deg2rad(15) # rad
omega = 0.1 * np.pi # rad/s
y0 = [alpha, omega]
m = 0.1 # kg
l = 1 # m
g = 9.8 # m/s^2

PendulumFunction = lambda x: np.array([x[1], -g / l * np.sin(x[0])])



