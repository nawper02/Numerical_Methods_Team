# KIN BLANDFORD, AUSTIN NEFF, HYRUM COLEMAN
# LAB 08

import numpy as np
import matplotlib.pyplot as plt
import csv
from rk4 import rk4
from train_motion import train_motion
from optimize_method import optimize, local_optimize, exhaustive_search

# TODO:
#  - Run optimization -- may have to adapt cost (train motion) to not use dictionary, but list instead
#  - Decrease random region based off of previous results
#  - Run optimization again


def main():
    # Initialize bounds

    Lt_bounds = (0.2, 0.3)
    Rt_bounds = (0.05, 0.2)
    P0_bounds = (70000, 200000)
    Rg_bounds = (0.002, 0.01)
    Ls_bounds = (0.1, 0.5)
    Rp_bounds = (0.02, 0.04)
    dens_bounds = (1200, 8940)

    # Initialize params dict to be in between bounds

    params = {
        "Lt": (Lt_bounds[0] + Lt_bounds[1]) / 2,
        "Rt": (Rt_bounds[0] + Rt_bounds[1]) / 2,
        "P0": P0_bounds[1],
        "Rg": (Rg_bounds[0] + Rg_bounds[1]) / 2,
        "Ls": (Ls_bounds[0] + Ls_bounds[1]) / 2,
        "Rp": (Rp_bounds[0] + Rp_bounds[1]) / 2,
        "dens": (dens_bounds[0] + dens_bounds[1]) / 2
    }

    # Make params_bounds dictionary

    params_bounds = {
        "Lt": Lt_bounds,
        "Rt": Rt_bounds,
        "P0": P0_bounds,
        "Rg": Rg_bounds,
        "Ls": Ls_bounds,
        "Rp": Rp_bounds,
        "dens": dens_bounds}

    num_trials = 2888
    num_spaces = 11

    # Run optimization

    new_method = True

    if new_method:
        print("Running scipy.optimize.minimize")
        res = exhaustive_search(params, params_bounds)
    else:
        print("Running Randomized Optimization")
        res = optimize(params_bounds, num_trials)
        print("Completed coarse optimization, beginning local optimization with dist = 0.5")
        res = local_optimize(res.x, num_trials, .5, params_bounds)
        print(".5 local optimization complete, beginning local optimization with dist = 0.1")
        res = local_optimize(res.x, num_trials, .1, params_bounds)
        print(".1 local optimization complete, beginning local optimization with dist = 0.01")
        res = local_optimize(res.x, num_trials, .01, params_bounds)

    # Print optimization results

    print(f"Final parameters:")
    print(f"\tLt: {res.x['Lt']}")
    print(f"\tRt: {res.x['Rt']}")
    print(f"\tP0: {res.x['P0']}")
    print(f"\tRg: {res.x['Rg']}")
    print(f"\tLs: {res.x['Ls']}")
    print(f"\tRp: {res.x['Rp']}")
    print(f"\tdensity: {res.x['dens']}")
    print(f"Final Optimized time: {res.time}")
    print(f"Copyable: {res.list}")

    # Run final simulation

    h = 0.01
    tspan = np.arange(0.0, 10, h)
    t, y = rk4(train_motion, tspan, y0, h, res.x)

    # Plot results

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax2.plot(t, y[:, 0], '-b', label='Position')
    ax1.plot(t, y[:, 1], 'r--', label='Velocity')
    ax2.plot(res.time, 10, 'go', label='Finish Time')

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)', color='b')
    ax1.set_ylabel('Velocity (m/s)', color='r')
    ax2.set_ylim([0, 1.1*max(y[:, 0])])
    ax1.set_ylim([0, 1.1*max(y[:, 1])])
    plt.title('Train Motion')
    fig.legend()

    plt.xlim(0, max(t))

    # Save and show plot

    # First save finish time and its associated parameters to a csv file

    with open('race_times.csv', 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow([res.time, res.x['Lt'], res.x['Rt'], res.x['P0'], res.x['Rg'], res.x['Ls'], res.x['Rp'], res.x['dens']])

    # Then save the plot as a pdf

    plt.savefig("combined.pdf")
    plt.show()

    print("Done")


if __name__ == "__main__":

    # Initialize global constants
    g = 9.81            # m/s^2
    Patm = 101325.0     # Pa
    p = 1.0             # kg/m^3
    Cd = 0.8            # -
    Crr = 0.03          # -
    Csf = 0.7           # -
    Rw = 0.025          # m
    Mw = 0.1            # kg
    y0 = [0, 0]  # pos, vel

    # Run main
    main()

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
