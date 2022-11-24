# KIN BLANDFORD, AUSTIN NEFF, HYRUM COLEMAN
# LAB 08

import numpy as np
from train_motion import train_motion
from rk4 import rk4
import csv

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


class Res(object):
    def __init__(self, params, time):
        # store params into dictionary x
        self.x = {"Lt": params[0], "Rt": params[1], "P0": params[2], "Rg": params[3], "Ls": params[4], "Rp": params[5],
                  "dens": params[6]}
        self.time = time
        self.list = [self.x["Lt"], self.x["Rt"], self.x["P0"], self.x["Rg"], self.x["Ls"], self.x["Rp"], self.x["dens"]]


def print_table_3(params):
    # Unpack parameters
    if type(params) == dict:
        Lt = params["Lt"]
        Rt = params["Rt"]
        P0 = params["P0"]
        Rg = params["Rg"]
        Ls = params["Ls"]
        Rp = params["Rp"]
        dens = params["dens"]
    elif type(params) == list:
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
        raise TypeError("params must be a dictionary, list, or numpy array")

    # Compute mass of train
    pp = 1250
    Lp = 1.5 * Ls
    mp = pp * np.pi * pow(Rp, 2) * Lp
    mt = dens * Lp * np.pi * ((Rt ** 2) - pow(Rt / 1.15, 2))
    m = mp + mt  # total mass of train in kg

    # Compute Volume of Tank
    V0 = Lt * np.pi * Rp * Rp

    # Compute Height of Train
    Ht = (2 * Rt) + Rw

    # Compute Frontal Area of Train
    Af = 2 * np.pi * Rt * Rt

    # Find closest material to optimal density
    materials = {"PVC": 1400.0, "Acrylic": 1200.0, "Galvanized Steel": 7700.0, "Stainless Steel": 8000.0,
                 "Titanium": 4500.0, "Copper": 8940.0, "Aluminum": 2700.0}
    closest_material = min(materials, key=lambda x: abs(materials[x] - dens))

    # Print Table 3

    # same as the above code, but without rounding
    print("Table 3: Train Physical Quantities")
    print(f"\tLength of Train: {Lt} m")
    print(f"\tOuter Diameter of train: {2 * Rt} m")
    print(f"\tHeight of train: {Ht} m")
    print(f"\tMaterial of train: {closest_material}")
    print(f"\tTotal Mass of Train: {m} kg")
    print(f"\tTrain frontal area: {Af} m^2")
    print(f"\tInitial Pressure: {P0} Pa")
    print(f"\tInitial tank volume: {V0} m^3")
    print(f"\tPinion Gear Radius: {Rg} m")
    print(f"\tLength of stroke: {Ls} m")
    print(f"\tTotal length of piston: {Lp} m")
    print(f"\tDiameter of piston: {2 * Rp} m")
    print(f"\tMass of piston: {mp} kg")

def run_race_simulation(params):
    random_cost = np.random.uniform(99.0, 999.0)
    h = 0.01
    tspan = np.arange(0.0, 10, h)
    t, y = rk4(train_motion, tspan, y0, h, params)
    if y is True:  # if train slips, return a large time
        return random_cost  # 100
    if max(y[:, 0]) < 10:  # if train doesn't reach the finish line, return a large time
        return random_cost  # 99
    if 2 * params[1] > 0.2: # if the trains width exceeds 0.2m, return a large time
        return random_cost
    # if the trains height exceeds 0.23 m, return a large time
    if (2 * params[1]) + Rw > 0.23:
        return random_cost
    else:
        for index, position in enumerate(y[:, 0]):
            if position >= 10:
                if max(y[:, 0]) > 12.5:  # if train goes too far, return a large time
                    return random_cost  # 105
                return t[index]
            else:
                pass


def random_param(bounds):
    return np.random.uniform(bounds[0], bounds[1])


def random_density():
    dens_options = [1400.0, 1200.0, 7700.0, 8000.0, 4500.0, 8940.0, 2700.0]
    # return a random choice from dens_options
    return np.random.choice(dens_options)


def optimize(params_bounds, num_trials):
    best_params = None
    best_time = None
    for trial in range(num_trials):
        # Randomize Parameters
        Lt = random_param(params_bounds["Lt"])
        Rt = random_param(params_bounds["Rt"])
        P0 = random_param(params_bounds["P0"])
        Rg = random_param(params_bounds["Rg"])
        Ls = random_param(params_bounds["Ls"])
        Rp = random_param(params_bounds["Rp"])
        dens = random_density()

        params = np.array([Lt, Rt, P0, Rg, Ls, Rp, dens])
        raceTime = run_race_simulation(params)

        # if this is the first trial, set best_params and best_time
        if best_params is None:
            best_params = params
            best_time = raceTime

        # if this is not the first trial, compare to best_params and best_time
        if raceTime < best_time:
            best_params = params
            best_time = raceTime
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


def create_bounds_in_range(var, dist, bounds):
    lower, upper = bounds
    var_s = dist * var
    if var - var_s < lower:
        res_lower = lower
    else:
        res_lower = var - var_s
    if var + var_s > upper:
        res_upper = upper
    else:
        res_upper = var + var_s
    bounds = (res_lower, res_upper)
    return bounds


def local_optimize(params, num_trials, dist, params_bounds):
    # Create local bounds for each parameter
    Lt_bounds = create_bounds_in_range(params['Lt'], dist, params_bounds['Lt'])

    Rt_bounds = create_bounds_in_range(params['Rt'], dist, params_bounds['Rt'])
    P0_bounds = create_bounds_in_range(params['P0'], dist, params_bounds['P0'])
    Rg_bounds = create_bounds_in_range(params['Rg'], dist, params_bounds['Rg'])
    Ls_bounds = create_bounds_in_range(params['Ls'], dist, params_bounds['Ls'])
    Rp_bounds = create_bounds_in_range(params['Rp'], dist, params_bounds['Rp'])
    dens_bounds = create_bounds_in_range(params['dens'], dist, params_bounds['dens'])

    best_params = None
    best_time = None

    for trial in range(num_trials):
        Lt = random_param(Lt_bounds)
        Rt = random_param(Rt_bounds)
        P0 = random_param(P0_bounds)
        Rg = random_param(Rg_bounds)
        Ls = random_param(Ls_bounds)
        Rp = random_param(Rp_bounds)
        dens = random_density()

        params = np.array([Lt, Rt, P0, Rg, Ls, Rp, dens])
        raceTime = run_race_simulation(params)

        # if this is the first trial, set best_params and best_time
        if best_params is None:
            best_params = params
            best_time = raceTime

        # if this is not the first trial, compare to best_params and best_time
        if raceTime < best_time:
            best_params = params
            best_time = raceTime
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


def exhaustive_search(params, params_bounds):
    # unpack params_bounds
    Lt_bounds = params_bounds['Lt']
    Rt_bounds = params_bounds['Rt']
    P0_bounds = params_bounds['P0']
    Rg_bounds = params_bounds['Rg']
    Ls_bounds = params_bounds['Ls']
    Rp_bounds = params_bounds['Rp']
    dens_bounds = params_bounds['dens']

    num_spaces = 6

    Lt_range = np.linspace(Lt_bounds[0], Lt_bounds[1], num_spaces)
    Rt_range = np.linspace(Rt_bounds[0], Rt_bounds[1], num_spaces)
    P0_range = np.linspace(P0_bounds[0], P0_bounds[1], num_spaces)
    Rg_range = np.linspace(Rg_bounds[0], Rg_bounds[1], num_spaces)
    Ls_range = np.linspace(Ls_bounds[0], Ls_bounds[1], num_spaces)
    Rp_range = np.linspace(Rp_bounds[0], Rp_bounds[1], num_spaces)
    # dens_range = np.linspace(dens_bounds[0], dens_bounds[1], num_spaces)
    dens_range = [1400.0, 1200.0, 7700.0, 8000.0, 4500.0, 8940.0, 2700.0]

    best_params = None
    best_time = None
    iters = 0

    for lt in Lt_range:
        print(f"Percent done: {(iters / pow(num_spaces, 7)) * 100}%")
        for rt in Rt_range:
            for p0 in P0_range:
                for rg in Rg_range:
                    if iters != 0:
                        res = Res(best_params, best_time)
                        with open('race_times.csv', 'a', newline='') as file:
                            writer = csv.writer(file, delimiter=',')
                            writer.writerow(
                                [res.time, res.x['Lt'], res.x['Rt'], res.x['P0'], res.x['Rg'], res.x['Ls'], res.x['Rp'],
                                 res.x['dens']])
                    for ls in Ls_range:
                        for rp in Rp_range:
                            for dens in dens_range:
                                params = np.array([lt, rt, p0, rg, ls, rp, dens])
                                raceTime = run_race_simulation(params)
                                # if this is the first trial, set best_params and best_time
                                if best_params is None:
                                    best_params = params
                                    best_time = raceTime

                                # if this is not the first trial, compare to best_params and best_time
                                if raceTime < best_time:
                                    best_params = params
                                    best_time = raceTime
                                    # print(f"Best parameters so far:")
                                    # print(f"\tLt: {best_params[0]}")
                                    # print(f"\tRt: {best_params[1]}")
                                    # print(f"\tP0: {best_params[2]}")
                                    # print(f"\tRg: {best_params[3]}")
                                    # print(f"\tLs: {best_params[4]}")
                                    # print(f"\tRp: {best_params[5]}")
                                    # print(f"\tdensity: {best_params[6]}")
                                    # print(f"Optimized cost (time): {best_time}")
                                    # print(f"Copyable: {best_params.tolist()}\n")
                                iters += 1

    else:
        res = Res(best_params, best_time)
    return res
