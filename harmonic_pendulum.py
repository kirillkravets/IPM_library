from shutil import which

import  numpy as np
import  matplotlib.pyplot as plt
from init1a import m, l, g, y0, t, dt, PendulumFunction, omega, alpha
from rkSolver import RK4Model


def HarmonicPendulum(y0, t, dt, function):
    y = RK4Model(y0, t, dt, function)
    # print(np.round(np.rad2deg(y[:, 1a])))
    alpha = y[:, 0]
    omega = y[:, 1]

    fig = plt.figure(figsize=(10, 5))
    plt.xlabel('t, sec')
    plt.ylabel('angle, rad')
    plt.title('angle = function(time)')
    plt.plot(t, alpha)
    plt.minorticks_on()
    plt.grid(which='both', axis='both')
    plt.savefig('graphs 1a/angle')

    fig = plt.figure(figsize=(10, 5))
    plt.plot(t, omega)
    plt.xlabel('t, sec')
    plt.ylabel('omega, rad / sec')
    plt.title('omega = function(time)')
    plt.minorticks_on()
    plt.grid(which='both', axis='both')
    plt.savefig('graphs 1a/omega')


    E = l**2 * omega**2/ 2 - g * l * np.cos(alpha)
    fig = plt.figure(figsize=(10, 5))
    plt.xlabel('t, sec')
    plt.ylabel('Full Energy Dj/kg')
    plt.title('E = const(time)')
    plt.plot(t, E)
    plt.minorticks_on()
    plt.grid(which='both', axis='both')
    plt.savefig('graphs 1a/energy')

    fig = plt.figure(figsize=(10, 5))
    plt.plot(alpha, omega)
    plt.xlabel('alpha, rad')
    plt.ylabel('omega, rad / sec')
    plt.title('omega = function(alpha)')
    plt.minorticks_on()
    plt.grid( which='both', axis='both')
    plt.savefig('graphs 1a/phase_picture')
    print('Harmonic Penulum modeliing finished succesfully!')
    return 0

x = HarmonicPendulum(y0, t, dt, PendulumFunction)
