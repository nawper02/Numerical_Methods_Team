# KIN BLANDFORD, AUSTIN NEFF
# LAB 08

# Test Commit


import numpy as np
import matplotlib.pyplot as plt
from rk4 import rk4

odefun = lambda t, y, params: train_motion(t, y, params)


def train_motion(t, y, params):
    """
    t: current time (seconds)
    y: current state [position, velocity]
    params: dictionary of parameters that influence motion
    returns: dydt, [velocity, acceleration]
    """

    g = params["g"]
    p = params["p"]
    m = params["m"]
    Crr = params["Crr"]
    Cd = params["Cd"]
    Fp = params["Fp"]
    A = params["A"]

    acceletation = (Fp - (0.5 * p * Cd * A * (y[1] ** 2)) - (m * g * Crr)) / m
    velocity = y[1]

    dydt = [velocity, acceletation]

    return dydt


def moving_train():
    h = 0.01
    tspan = np.arange(0.0, 2, h)
    t, y = rk4(odefun, tspan, y0, h, params)

    plt.plot(t, y[:, 0], '-b', label='Position')
    plt.title('Simulation of a moving train -- position')
    plt.ylabel('Position (m)')
    plt.xlabel('Time (s)')
    plt.savefig("position.pdf")

    plt.figure()
    plt.plot(t, y[:, 1], '-r', label='Velocity')
    plt.title('Simulation of a moving train -- velocity')
    plt.ylabel('Velocity (m/s), Position (m)')
    plt.xlabel('Time (s)')
    plt.savefig("velocity.pdf")

    plt.legend(loc='best')

    plt.show()


if __name__ == "__main__":
    params = {"g": 9.81, "p": 1, "m": 10, "Crr": 0.002, "Cd": 0.4, "Fp": 1, "A": 0.05}
    y0 = [0, 0]
    moving_train()
