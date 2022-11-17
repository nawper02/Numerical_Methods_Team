# KIN BLANDFORD, AUSTIN NEFF, HYRUM COLEMAN
# LAB 08
import numpy as np
import matplotlib.pyplot as plt
from rk4 import rk4
from scipy.optimize import Bounds, minimize

# TODO:
#  - Run optimization -- may have to adapt cost (train motion) to not use dictionary, but list instead


class Slipped(Exception):
    pass


def train_motion(t, y, params):
    """
    t: current time (seconds)
    y: current state [position, velocity]
    params: dictionary of parameters that influence motion
    returns: dydt, [velocity, acceleration]
    """

    # Unpack parameters
    if type(params) == dict:
        Lt = params["Lt"]
        Rt = params["Rt"]
        P0 = params["P0"]
        Rg = params["Rg"]
        Ls = params["Ls"]
        Rp = params["Rp"]
        density = params["density"]
    if type(params) == list:
        Lt = params[0]
        Rt = params[1]
        P0 = params[2]
        Rg = params[3]
        Ls = params[4]
        Rp = params[5]
        density = params[6]
    if type(params) == np.ndarray:
        Lt = params[0]
        Rt = params[1]
        P0 = params[2]
        Rg = params[3]
        Ls = params[4]
        Rp = params[5]
        density = params[6]

    # Compute V0
    V0 = Lt * np.pi * Rp * Rp

    # Compute mass of train
    pp = 1250
    Lp = 1.5 * Ls
    mp = pp * np.pi * pow(Rp, 2) * Lp
    mt = density * Lp * np.pi * ((Rt**2) - pow(Rt/1.15, 2))
    m = mp + mt

    # Area of the piston head
    Ap = np.pi * Rp * Rp

    # Compute frontal area of train
    A = np.pi * Rt * Rt

    # Compute propulsion force
    # Shouldn't subtract Patm BECAUSE we already have gaUge pressure
    Fp = (((P0 * V0) / (V0 + Ap * (Rg / Rw) * y[0]))) * Ap

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
    #term_1 = (Rg * 70000 * Ap) / Rw
    term_1_seg_1 = Ap * Rg / Rw
    term_1_seg_2 = (P0 * V0) / (V0 + term_1_seg_1 * y[0])

    term_1 = term_1_seg_1 * (term_1_seg_2)
    #term_1_exponential= (Rg * (P0 * np.exp(-0.1 * t)) * Ap) / Rw
    term_2 = (p * Cd * A * (y[1] ** 2)) / 2
    term_3 = m * g * Crr
    sum_masses = m + Mw

    # if in acceleration phase, solve accelerating equations
    if accel:
        acceleration = (term_1 - term_2 - term_3) / sum_masses
        velocity = y[1]
    # if not in acceleration phase, solve deceleration equations
    else:
        acceleration = (- term_2 - term_3) / m
        velocity = y[1]

    if velocity < 0:
        acceleration = 0
        velocity = 0

    dydt = [velocity, acceleration]

    Ft = ((Rg * Fp) / Rw) - (Mw * acceleration)
    if Ft > ((Csf * m * g) / 2):
        raise Slipped
        #pass

    return dydt


def cost(params):
    try:
        h = 0.01
        tspan = np.arange(0.0, 10, h)
        t, y = rk4(train_motion, tspan, y0, h, params)
    except Slipped:
        return 101
    if max(y[:, 0]) < 10:
        return 99
    else:
        for index, position in enumerate(y[:, 0]):
            if position >= 10:
                print(f"\tRun complete, finish time: {t[index]}")
                if max(y[:, 0]) > 12.5:
                    return 105
                return t[index]


def fun_der(x, a):
    # DUMMY CODE
    dx = 2 * (x[0] - 1)
    dy = 2 * (x[1] - a)
    return np.array([dx, dy])


def fun_hess(x):
    # DUMMY CODE
    dx = 2
    dy = 2
    return np.zeros((7, 7))


def optimize():
    # Create Bounds
    #               Lower Bounds                          Upper Bounds
    #               Lt    Rt    P0     Rg     Ls   Rp    dens   Lt    Rt   P0     Rg    Ls   Rp    dens
    bounds = Bounds([0.2, 0.05, 70000, 0.002, 0.1, 0.02, 1200], [0.3, 0.2, 200000, 0.01, 0.5, 0.04, 8940])

    # Initial Parameters
    #             Lt     Rt     P0     Rg     Ls   Rp     dens
    x0 = np.array([0.25, 0.115, 150000, 0.005, 0.3, 0.032, 8000])


    #x0 = np.array([0.25, 0.1, 70e3, 0.01, 0.1, 0.01, 2700])
    # Optimized parameters:
    # 	Lt: 0.25
    # 	Rt: 0.1249595627400744
    # 	P0: 70001.1378918479
    # 	Rg: 0.006000000432202432
    # 	Ls: 0.29618432890077256
    # 	Rp: 0.029999916034144307

    # May not need jac or hess for this
    # res = minimize(cost, x0, method='trust-constr', hess=fun_hess, options={'verbose': 1}, bounds=bounds)

    print("Starting optimization...")
    #res = minimize(cost, x0, method='trust-constr', hess=fun_hess, options={'verbose': 1, 'gtol': 1e-16}, bounds=bounds)
    res = minimize(cost, x0, method='trust-constr', options={'verbose': 1}, bounds=bounds)
    #res = minimize(cost, x0, method='Nelder-Mead', options={'disp': True, 'maxiter': 1000, 'adaptive': True}, bounds=bounds)
    print("Optimization complete.")
    return res


def main():
    res = optimize()
    print(f"Optimized parameters:")
    print(f"\tLt: {res.x[0]}")
    print(f"\tRt: {res.x[1]}")
    print(f"\tP0: {res.x[2]}")
    print(f"\tRg: {res.x[3]}")
    print(f"\tLs: {res.x[4]}")
    print(f"\tRp: {res.x[5]}")
    print(f"\tdensity: {res.x[6]}")
    print(f"Optimized cost (time): {res.fun}")
    print(f"Copyable: {res.x.tolist()}")

    h = 0.01
    tspan = np.arange(0.0, 10, h)
    try:
        t, y = rk4(train_motion, tspan, y0, h, res.x)
        #x0 = [0.25, 0.115, 70000, 0.005, 0.3, 0.032, 8000]
        #t, y = rk4(train_motion, tspan, y0, h, x0)
    except Slipped:
        print("Train slipped")

    plt.plot(t, y[:, 0], '-b', label='Position')
    plt.title('Simulation of a moving train -- position')
    plt.ylabel('Position (m)')
    plt.xlabel('Time (s)')
    plt.legend(loc='best')
    plt.savefig("position.pdf")

    plt.figure()
    plt.plot(t, y[:, 1], '-r', label='Velocity')
    plt.title('Simulation of a moving train -- velocity')
    plt.ylabel('Velocity (m/s), Position (m)')
    plt.xlabel('Time (s)')
    plt.legend(loc='best')
    plt.savefig("velocity.pdf")

    plt.show()


if __name__ == "__main__":
    #params = {"g": 9.81, "p": 1.0, "m": 10.0, "Crr": 0.03, "Cd": 0.8, "Fp": 1, "A": 0.05, "Ls": 0.1, "Rw": 0.025,
    #          "Rg": 0.01, "Rp": 0.01, "Mw": 0.1, "Pg": 100e3, "Csf": 0.7, "P0": 70e3, "Lt": 0.25, "mat_dens": 2700}

    # Initialize design parameters (to optimize)
    design_params = {"Lt": 0.25, "Rt": 0.1, "density": 2700, "P0": 70e3, "Rg": 0.01, "Ls": 0.1, "Rp": 0.01}
    #                     Lt    Rt    dens   P0   Rg    Ls   Rp
    design_params_list = [0.25, 0.1, 2700, 70e3, 0.01, 0.1, 0.01]

    # Constants (Global)
    g = 9.81            # m/s^2
    Patm = 101325.0     # Pa
    p = 1.0             # kg/m^3
    Cd = 0.8            # -
    Crr = 0.03          # -
    Csf = 0.7           # -
    Rw = 0.025          # m
    Mw = 0.1            # kg

    """
        • Acceleration of gravity: 9:8 m/s2 (g)
        • Air density: 1:0 kg/m3 (p)
        • Train mass: 10 kg (m)
        • Rolling resistance coefficient: 0.03 (Crr)
        • Drag coefficient: 0.8 (Cd)
        • I have no idea what "Fp" is supposed to be, but it would go here if I knew what it was
        • Total frontal area of train: 0:05 m2 (A)
        • Piston stroke length: 0:1 m (Ls)
        • Wheel radius: 2:5 cm (Rw) (Converted to m)
        • Gear radius: 1:0 cm (Rg) (Converted to m)
        • Piston radius: 1:0 cm (Rp) (Converted to m)
        • Wheel mass: 0:1 kg (Mw)
        • Tank gauge pressure: 100 kPa (Pg) (Converted to Pa)
        • Coefficient of static friction: 0.7 (Csf)
        • Initial tank pressure: 70 kPa (P0) (Converted to Pa)
        • Tank length: 0:25 m (Lt)
        • Material density: 2700 kg/m3 (mat_dens)
        
        Material Density (kg/m3)
            PVC 1400
            acrylic 1200
            galvanized steel 7700
            stainless steel 8000
            titanium 4500
            copper 8940
            aluminum 2700
            
        Parameter - Symbol - Range of Values Units
            length of train - Lt - (0.2, 0.3) m
            radius of tank - Rt - (0.05, 0.2) m
            density of train material (choose)
            initial tank gauge pressure - P0 - (70000, 200000) Pa
            pinion gear radius - Rg - (0.002, 0.01) m
            length of piston stroke - Ls - (0.1, 0.5) m
            radius of piston - Rp - (0.02, 0.04) m
    """
    y0 = [0, 0] # pos, vel
    main()
