# KIN BLANDFORD, AUSTIN NEFF, HYRUM COLEMAN
# LAB 08'

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
    Ls = params["Ls"]
    Rw = params["Rw"]
    Rg = params["Rg"]
    Rp = params["Rp"]
    Mw = params["Mw"]
    Pg = params["Pg"]
    Csf = params["Csf"]

    # Length of acceleration phase
    La = (Ls * Rw) / Rg

    # If in acceleration phase
    if y[0] < La:
        # Is accelerating
        accel = True
    else:
        # Is decelerating
        accel = False

    # For housekeeping
    term_1 = (Rg * Pg * np.pi * Rp * Rp) / Rw
    term_2 = (p * Cd * A * (y[1] ** 2)) / 2
    term_3 = m * g * Crr
    sum_masses = m + Mw

    # if in acceleration phase, solve accelerating equations
    if accel:
        acceleration = (term_1 - term_2 - term_3) / sum_masses
        velocity = y[1]

    # if not in acceleration phase, solve deceleration equations
    if not accel:
        acceleration = (- term_2 - term_3) / m
        velocity = y[1]

    if velocity < 0:
        velocity = 0
        acceleration = 0

    dydt = [velocity, acceleration]

    Ft = ((Rg * Fp) / Rw) - (Mw * acceleration)
    if Ft < ((Csf * m * g) / 2):
        print(f"Wheel Slip Criterion Violated at x = {y[0]}!!!")

    return dydt


def main():
    h = 0.01
    tspan = np.arange(0.0, 10, h)
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
    params = {"g": 9.81, "p": 1.0, "m": 10.0, "Crr": 0.03, "Cd": 0.8, "Fp": 1, "A": 0.05, "Ls": 0.1, "Rw": 0.025,
              "Rg": 0.01, "Rp": 0.01, "Mw": 0.1, "Pg": 100e3, "Csf": 0.7}
    """
        • Piston stroke length: 0:1 m (Ls)
        • Wheel radius: 2:5 cm (Rw) (Converted to m)
        • Gear radius: 1:0 cm (Rg) (Converted to m)
        • Piston radius: 1:0 cm (Rp) (Converted to m)
        • Acceleration of gravity: 9:8 m/s2 (g)
        • Wheel mass: 0:1 kg (Mw)
        • Tank gauge pressure: 100 kPa (Pg) (Converted to Pa)
        • Air density: 1:0 kg/m3 (p)
        • Train mass: 10 kg (m)
        • Total frontal area of train: 0:05 m2 (A)
        • Coefβicient of static friction: 0.7 (Csf)
        • Drag coefβicient: 0.8 (Cd)
        • Rolling resistance coefβicient: 0.03 (Crr)
    """
    y0 = [0, 0] # pos, vel, accel(?)
    main()
