# KIN BLANDFORD, AUSTIN NEFF, HYRUM COLEMAN
# LAB 08

import numpy as np
import matplotlib.pyplot as plt
from rk4 import rk4
from scipy.optimize import Bounds, minimize

# TODO:
#  - Run optimization -- may have to adapt cost (train motion) to not use dictionary, but list instead
#  - Decrease random region based off of previous results
#  - Run optimization again



class Slipped(Exception):
    pass


class Res(object):
    def __init__(self, params, time):
        self.x = params
        self.fun = time


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
                #print(f"\tRun complete, finish time: {t[index]}")
                if max(y[:, 0]) > 12.5:
                    return 105
                return t[index]


def create_options_from_bounds(bounds, num_steps):
    options = np.linspace(bounds[0], bounds[1], num_steps)
    return options


def random_param(bounds):
    return np.random.uniform(bounds[0], bounds[1])


def optimize():
    # Create Bounds
    #               Lower Bounds                          Upper Bounds
    #               Lt    Rt    P0     Rg     Ls   Rp    dens   Lt    Rt   P0     Rg    Ls   Rp    dens
    bounds = Bounds([0.2, 0.05, 70000, 0.002, 0.1, 0.02, 1200], [0.3, 0.2, 200000, 0.01, 0.5, 0.04, 8940])

    # Create bounds for each parameter
    Lt_bounds = (0.2, 0.3)
    Rt_bounds = (0.05, 0.2)
    P0_bounds = (70000, 200000)
    Rg_bounds = (0.002, 0.01)
    Ls_bounds = (0.1, 0.5)
    Rp_bounds = (0.02, 0.04)
    density_bounds = (1200, 8940)

    time_list = []
    params_list = []
    best_params = None
    best_time = None
    for trial in range(1000):
        # Ramdomize Parameters

        Lt = random_param(Lt_bounds)
        Rt = random_param(Rt_bounds)
        P0 = random_param(P0_bounds)
        Rg = random_param(Rg_bounds)
        Ls = random_param(Ls_bounds)
        Rp = random_param(Rp_bounds)
        dens = random_param(density_bounds)

        #                  Lt  Rt  P0  Rg  Ls  Rp  dens
        params = np.array([Lt, Rt, P0, Rg, Ls, Rp, dens])

        time_list.append(cost(params))
        params_list.append(params)

        if time_list[-1] == min(time_list):
            best_params = params_list[-1]
            best_time = time_list[-1]

            print(f"Best parameters so far:")
            print(f"\tLt: {best_params[0]}")
            print(f"\tRt: {best_params[1]}")
            print(f"\tP0: {best_params[2]}")
            print(f"\tRg: {best_params[3]}")
            print(f"\tLs: {best_params[4]}")
            print(f"\tRp: {best_params[5]}")
            print(f"\tdensity: {best_params[6]}")
            print(f"Optimized cost (time): {best_time}")
            print(f"Copyable: {best_params.tolist()}\n")
    else:
        res = Res(best_params, best_time)

    return res


def local_optimize(params):
    # Create local bounds for each parameter
    dist = 0.1

    Lt = params[0]
    Rt = params[1]
    P0 = params[2]
    Rg = params[3]
    Ls = params[4]
    Rp = params[5]
    dens = params[6]

    Lt_s = dist * Lt
    Rt_s = dist * Rt
    P0_s = dist * P0
    Rg_s = dist * Rg
    Ls_s = dist * Ls
    Rp_s = dist * Rp
    dens_s = dist * dens

    Lt_bounds = (Lt-Lt_s, Lt+Lt_s)
    Rt_bounds = (Rt-Rt_s, Rt+Rt_s)
    P0_bounds = (P0-P0_s, P0+P0_s)
    Rg_bounds = (Rg-Rg_s, Rg+Rg_s)
    Ls_bounds = (Ls-Ls_s, Ls+Ls_s)
    Rp_bounds = (Rp-Rp_s, Rp+Rp_s)
    density_bounds = (dens-dens_s, dens+dens_s)

    time_list = []
    params_list = []
    best_params = None
    best_time = None
    for trial in range(1000):
        # Ramdomize Parameters

        Lt = random_param(Lt_bounds)
        Rt = random_param(Rt_bounds)
        P0 = random_param(P0_bounds)
        Rg = random_param(Rg_bounds)
        Ls = random_param(Ls_bounds)
        Rp = random_param(Rp_bounds)
        dens = random_param(density_bounds)

        #                  Lt  Rt  P0  Rg  Ls  Rp  dens
        params = np.array([Lt, Rt, P0, Rg, Ls, Rp, dens])

        time_list.append(cost(params))
        params_list.append(params)

        if time_list[-1] == min(time_list):
            best_params = params_list[-1]
            best_time = time_list[-1]

            print(f"Best parameters so far:")
            print(f"\tLt: {best_params[0]}")
            print(f"\tRt: {best_params[1]}")
            print(f"\tP0: {best_params[2]}")
            print(f"\tRg: {best_params[3]}")
            print(f"\tLs: {best_params[4]}")
            print(f"\tRp: {best_params[5]}")
            print(f"\tdensity: {best_params[6]}")
            print(f"Optimized cost (time): {best_time}")
            print(f"Copyable: {best_params.tolist()}\n")
    else:
        res = Res(best_params, best_time)

    return res


def main():
    res = optimize()
    print("Completed coarse optimization, beginning local optimization")
    res = local_optimize(res.x)
    print("Local optimization complete.")

    print(f"Final parameters:")
    print(f"\tLt: {res.x[0]}")
    print(f"\tRt: {res.x[1]}")
    print(f"\tP0: {res.x[2]}")
    print(f"\tRg: {res.x[3]}")
    print(f"\tLs: {res.x[4]}")
    print(f"\tRp: {res.x[5]}")
    print(f"\tdensity: {res.x[6]}")
    print(f"Copyable: {list(res.x)}")
    print(f"Optimized cost (time): {res.fun}\n")

    h = 0.01
    tspan = np.arange(0.0, 10, h)
    t, y = rk4(train_motion, tspan, y0, h, res.x)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(t, y[:, 0], '-b', label='Position')
    ax2.plot(t, y[:, 1], 'r--', label='Velocity')
    ax1.plot(res.fun, 10, 'go', label='Finish Time')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax2.set_ylabel('Velocity (m/s)')
    plt.title('Simulation of train -- position and velocity vs time')
    fig.legend()

    plt.savefig("combined.pdf")

    plt.show()


if __name__ == "__main__":
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
    # 5.72
    # [0.22759787898721132, 0.1179539530478348, 170742.99973942252, 0.0030649285122781172, 0.1945437293158895, 0.02969740379346559, 5508.651741337037]
    # 5.659
    # [0.20057940644612615, 0.09296024791629555, 70816.23598171223, 0.004088326139245712, 0.2995078683326422, 0.03554130090484798, 3911.1736625844583]
    # 5.67
    # [0.29782652950466626, 0.1278568956275774, 98542.27223684711, 0.0055843550349363585, 0.3654223978744685, 0.029086120013843354, 2204.6409017118917]
    # 5.73
    # [0.29167760039360136, 0.16287376035864523, 196240.11538343632, 0.004053528909644406, 0.3328770360491726, 0.03338539866008866, 3835.046508285344]
    # 5.72
    # [0.2923267012567824, 0.19065723161207404, 135736.80144742876, 0.0023804678835232292, 0.16654835147538863, 0.02868447929729715, 1630.0399232167974]
    # 5.58
    # [0.29342833084413933, 0.16037627405090937, 132086.6856449127, 0.004518855095953315, 0.30093142628474456, 0.024947758891551694, 1469.2146204772644]