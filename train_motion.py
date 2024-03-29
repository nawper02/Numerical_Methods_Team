# KIN BLANDFORD, AUSTIN NEFF, HYRUM COLEMAN
# LAB 08


import numpy as np

# Initialize global constants
g = 9.81  # m/s^2
Patm = 101325.0  # Pa
p = 1.0  # kg/m^3
Cd = 0.8  # -
Crr = 0.03  # -
Csf = 0.7  # -
Rw = 0.025  # m
Mw = 0.1  # kg
y0 = [0, 0]  # pos, vel


# Function that evaluates the motion of the train at a given state
def train_motion(t, y, params):
    """
    t: current time (seconds)
    y: current state [position, velocity]
    params: dictionary of parameters that influence motion
    returns: dydt, [velocity, acceleration]
    """

    # Unpack parameters
    if type(params) == dict:
        Lt = params["Lt"]['value']
        Rt = params["Rt"]['value']
        P0 = params["P0"]['value']
        Rg = params["Rg"]['value']
        Ls = params["Ls"]['value']
        Rp = params["Rp"]['value']
        dens = params["dens"]['value']
    elif type(params) == list or type(params) == tuple:
        Lt = params[0]
        Rt = params[1]
        P0 = params[2]
        Rg = params[3]
        Ls = params[4]
        Rp = params[5]
        dens = params[6]
    elif type(params) == np.ndarray:
        Lt = params[0]
        Rt = params[1]
        P0 = params[2]
        Rg = params[3]
        Ls = params[4]
        Rp = params[5]
        dens = params[6]
    else:
        raise TypeError("params must be a dictionary, list, or numpy array :)")

    # Compute V0
    V0 = Lt * np.pi * Rp * Rp

    # Compute mass of train
    pp = 1250
    Lp = 1.5 * Ls
    mp = pp * np.pi * pow(Rp, 2) * Lp
    mt = dens * Lp * np.pi * ((Rt ** 2) - pow(Rt / 1.15, 2))
    m = mp + mt

    # Area of the piston head
    Ap = np.pi * Rp * Rp

    # Compute frontal area of train
    A = np.pi * Rt * Rt

    # Compute propulsion force
    # Shouldn't subtract Patm BECAUSE we already have gaUge pressure
    Fp = (P0 * V0 / V0 + Ap * (Rg / Rw) * y[0]) * Ap

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
    term_1_seg_1 = Ap * Rg / Rw
    term_1_seg_2 = (P0 * V0) / (V0 + term_1_seg_1 * y[0])

    term_1 = term_1_seg_1 * term_1_seg_2
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

    slipFlag = False
    Ft = ((Rg * Fp) / Rw) - (Mw * acceleration)
    if Ft > ((Csf * m * g) / 2):
        slipFlag = True

    return dydt, slipFlag
