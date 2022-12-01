# KIN BLANDFORD, AUSTIN NEFF, HYRUM COLEMAN
# LAB 08

import numpy as np
from train_motion import train_motion
from rk4 import rk4
import csv
import random
import itertools
import multiprocessing

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


# Object to store the results of the optimization
class Res(object):
    def __init__(self, params, race_time):
        self.x = params
        self.time = race_time
        self.list = [self.x['Lt']['value'], self.x['Rt']['value'],
                     self.x['P0']['value'], self.x['Rg']['value'],
                     self.x['Ls']['value'], self.x['Rp']['value'],
                     self.x['dens']['value']]


# Helper method to print table 3 for the memo
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

    # Find the closest material to optimal density
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


# Method that takes in a list of parameters and returns the time it takes to complete the race.
# if the race is not completed,
def run_race_simulation(params, returnVec=False):
    h = 0.01
    tspan = np.arange(0.0, 10, h)
    t, y = rk4(train_motion, tspan, y0, h, params)
    if y is True:  # if train slips, return a large time
        return 100, params
    if max(y[:, 0]) < 10:  # if train doesn't reach the finish line, return a large time
        return 101, params
    else:
        for index, position in enumerate(y[:, 0]):
            if position >= 10:
                if max(y[:, 0]) > 12.5:  # if train goes too far, return a large time
                    return 102, params
                if returnVec:
                    return t, params, y
                return t[index], params
            else:
                pass


# Helper method to return a random value within the bounds of the parameter
def random_param(bounds):
    return np.random.uniform(bounds[0], bounds[1])


# Helper method to choose a random density from the list of densities
def random_density():
    dens_options = [1400.0, 1200.0, 7700.0, 8000.0, 4500.0, 8940.0, 2700.0]
    # return a random choice from dens_options
    return np.random.choice(dens_options)


# Method to perform random optimization on the global scale
def random_search(params, num_trials, best_time=None, best_params=None):

    for trial in range(num_trials):
        # Randomize Parameters
        for idx, key in enumerate(params):
            if key != 'dens':
                params[key]['value'] = random_param(params[key]['bounds'])
            else:
                params[key]['value'] = params[key]['bounds'][random.randint(0, len(params[key]['bounds']) - 1)]

        # Run Simulation

        raceTime, yVec = run_race_simulation(params)

        # If this is the first trial, set best_params and best_time
        if best_params is None:
            best_params = params
            best_time = raceTime

        # If this is not the first trial, compare to best_params and best_time
        if raceTime < best_time:
            best_params = params
            best_time = raceTime

        if trial % int(num_trials / 4) == 0:
            print(f"Trial {trial} of {num_trials} complete.")
            print(f"Best parameters so far:")

            if type(best_params) == dict:
                for idx, key in enumerate(best_params):
                    print(f"\t{key} = {best_params[key]['value']}")
            elif type(best_params) == list:
                for idx, key in enumerate(best_params):
                    print(f"\t{idx} = {best_params[idx]}")
            else:
                print("Something went wrong.")
            print(f"Optimized cost (time): {best_time}")

    else:
        res = Res(best_params, best_time)
    return res


def exhaustive_search(params, num, best_time=None, best_params=None):
    num_spaces = num

    for idx, key in enumerate(params):
        if key != 'dens':
            params[key]['range'] = np.linspace(params[key]['bounds'][0], params[key]['bounds'][1], num_spaces)
        else:
            params[key]['range'] = params[key]['bounds']
    iters = 0

    with multiprocessing.Pool() as pool:
        for res in pool.imap_unordered(run_race_simulation, itertools.product(*[params[key]['range'] for key in params])):
            race_time, res_params = res
            iters += 1

            if best_params is None:
                best_params = res_params
                best_time = race_time

            # If this is not the first trial, compare to best_params and best_time
            if race_time < best_time and race_time < 7:
                print(f"New best parameters found!")
                best_params = res_params
                best_time = race_time
                iterPercent = iters / (num_spaces ** 7) * 100

                with open('race_times.csv', 'a', newline='') as file:
                    writer = csv.writer(file, delimiter=',')
                    writer.writerow([best_time, best_params[0], best_params[1], best_params[2], best_params[3],
                                     best_params[4], best_params[5], best_params[6], iterPercent])

            if iters % int(num_spaces ** 7 / 100) == 0:
                roundedPercent = round(iters / (num_spaces ** 7) * 100, 3)
                print(f"Roughly {roundedPercent}% complete.")
        else:
            for idx, key in enumerate(params):
                params[key]['value'] = best_params[idx]
            res = Res(best_params, best_time)
        return res


# Helper method to create a tuple of parameter bounds in a range around the parameter
# Accounts for strict boundaries
def create_bounds_in_range(var, bounds, dist):
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


# Method to perform random optimization on the local scale
def optimize(method, params, num, dist=1, best_time=None, best_params=None):
    # Create local bounds for each parameter

    if dist != 1:
        for idx, key in enumerate(params):
            if key != 'dens':
                params[key]['bounds'] = create_bounds_in_range(params[key]['value'], params[key]['bounds'], dist)

    res = method(params, num, best_time, best_params)

    return res
