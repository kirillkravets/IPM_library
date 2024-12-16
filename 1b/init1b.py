import numpy as np

dt = 0.01 #s
t = np.arange(0, 100, dt, dtype=float)
alpha = np.deg2rad(15) # rad
omega =  np.pi # rad/s
y0 = [alpha, omega]
m = 0.1 + np.random.normal(0, 0.005) # kg
l = 1 + np.random.normal(0, 0.05) # m
g = 9.8 + np.random.normal(0.05) # m/s^2

# Control parametres
sigma = np.array([np.deg2rad(0.5), np.deg2rad(0.03)])
kP = 10
kI = 10
kD = 1

kLl = 110
kLd = 20
dtControl = 5 * dt
yRef = np.array([np.deg2rad(70), 0])