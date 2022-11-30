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


class Res(object):
    def __init__(self, params, race_time):
        self.x = params
        self.time = race_time
        self.list = [self.x['Lt']['value'], self.x['Rt']['value'],
                     self.x['P0']['value'], self.x['Rg']['value'],
                     self.x['Ls']['value'], self.x['Rp']['value'],
                     self.x['dens']['value']]


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


def random_param(bounds):
    return np.random.uniform(bounds[0], bounds[1])


def random_search(params, num_trials):

    best_params = None
    best_time = None
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
            for idx, key in enumerate(best_params):
                print(f"\t{key} = {best_params[key]['value']}")
            print(f"Optimized cost (time): {best_time}")

    else:
        res = Res(best_params, best_time)
    return res


def exhaustive_search(params, num):
    num_spaces = num

    for idx, key in enumerate(params):
        if key != 'dens':
            params[key]['range'] = np.linspace(params[key]['bounds'][0], params[key]['bounds'][1], num_spaces)
        else:
            params[key]['range'] = params[key]['bounds']

    best_params = None
    best_time = None
    iters = 0

    # q:

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


def optimize(method, params, num, dist=1):
    # Create local bounds for each parameter
    for idx, key in enumerate(params):
        if key != 'dens':
            params[key]['bounds'] = create_bounds_in_range(params[key]['value'], params[key]['bounds'], dist)

    res = method(params, num)

    return res
