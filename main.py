# KIN BLANDFORD, AUSTIN NEFF, HYRUM COLEMAN
# NUMERICAL METHODS DESIGN PROJECT
"""
This program simulates the motion of a train on a track. It also finds the optimum
parameters for the train to reach the distance in the fastest possible time..
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import csv
from rk4 import rk4
from train_motion import train_motion
from optimize_method import Res, print_table_3
from optimize_method import optimize, exhaustive_search, random_search, run_race_simulation


def main():
    # Initialize parameter bounds

    Lt_bounds = (0.2, 0.3)
    Rt_bounds = (0.05, 0.2)
    P0_bounds = (70000, 200000)
    Rg_bounds = (0.002, 0.01)
    Ls_bounds = (0.1, 0.5)
    Rp_bounds = (0.02, 0.04)
    dens_bounds = (1400, 1200, 7700, 8000, 4500, 8940, 2700)
    # Initialize params dict to be in between bounds

    # Variable to control the number of trials to be simulated

    params = {"Lt": {"bounds": Lt_bounds, "value": 0.25},
              "Rt": {"bounds": Rt_bounds, "value": 0.125},
              "P0": {"bounds": P0_bounds, "value": 100000},
              "Rg": {"bounds": Rg_bounds, "value": 0.006},
              "Ls": {"bounds": Ls_bounds, "value": 0.3},
              "Rp": {"bounds": Rp_bounds, "value": 0.03},
              "dens": {"bounds": dens_bounds, "value": 4500}
              }
    num_trials = 2888
    # Run optimization

    # Two boolean options to control the behavior of the program
    # New method: runs exhaustive search
    # Use specific params: runs the simulation with a set of given parameters and does not perform optimization

    do_exhaustive = False
    use_specific_params = True
    num_spaces = 5

    if not use_specific_params:
        print("Do you want to find new times, or refine old times?")
        print("1. Find new times")
        print("2. Refine old times")
        choice = input("Enter 1 or 2: ")
        if choice == "1":
            print("Find new times selected")
            do_exhaustive = True

            print("How many spaces do you want to search?")
            choice = input("Enter any number: ")
            num_spaces = int(choice)
        elif choice == "2":
            print("Refine old times selected")
            do_exhaustive = False

            # read csv file to get best time and params
            with open('race_times.csv', 'r') as f:
                reader = csv.reader(f)
                # search first column for lowest time, if a string is found, skip it
                best_time = 1000
                for row in reader:
                    try:
                        if float(row[0]) < best_time:
                            best_time = float(row[0])
                            best_params = row[1:]
                    except ValueError:
                        pass
            for idx, key in enumerate(params):
                params[key]["value"] = float(best_params[idx])
        else:
            print("Invalid input, exiting program")
            exit()

        if do_exhaustive:
            print("Running exhaustive search...")
            res = optimize(exhaustive_search, params, num_spaces)
        else:
            print("Running random search...")
            res = optimize(random_search, params, num_trials, dist=0.5, best_time=best_time, best_params=best_params)
            print(".5 local optimization complete, beginning local optimization with dist = 0.1")
            res = optimize(random_search, res.x, num_trials, dist=0.1, best_time=res.time, best_params=res.x)
            print(".1 local optimization complete, beginning local optimization with dist = 0.01")
            res = optimize(random_search, res.x, num_trials, dist=0.01, best_time=res.time, best_params=res.x)

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
        print_table_3(params)
    else:
        # The following params are presented in the memo
        #         Lt,           Rt,          P0,       Rg,    Ls,        Rp,            dens
        params = [0.2, 0.15500000000000003, 96000.0, 0.0028, 0.18, 0.026000000000000002, 1200]
        # With available parts, the following params are the best we can do

        time = run_race_simulation(params)
        res = Res(params, time)
        print_table_3(params)
        print(f"Final Optimized time: {res.time}")

    # Initialize step size, time, and simulate the motion of the train

    h = 0.01
    tspan = np.arange(0.0, 10, h)
    t, y = rk4(train_motion, tspan, y0, h, res.x)

    print(f"Final parameters:")
    for idx, key in enumerate(res.x):
        print(f"{key}: {res.x[key]['value']}")
    print(f"Final Optimized time: {res.time}")
    print(f"Copyable: {res.list}")

    # Run final simulation
    final_params = res.x
    t, y = run_race_simulation(final_params, returnVec=True)

    # Plot results

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax2.plot(t, y[:, 0], '-b', label='Position')
    ax1.plot(t, y[:, 1], 'r--', label='Velocity')
    ax2.plot(res.time, 10, 'go', label='Finish Time')
    ax2.axvline(res.time, color='g', linestyle='--', label='Finish Time')
    ax2.axhline(10, color='g', linestyle='--', label='Finish Position')
    ax2.axhline(12.5, color='b', linestyle='--', label='End of Track')

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)', color='b')
    ax1.set_ylabel('Velocity (m/s)', color='r')
    ax2.set_ylim([0, 1.1*max(y[:, 0])])
    ax1.set_ylim([0, 1.1*max(y[:, 1])])
    plt.title('Train Motion')
    fig.legend()

    plt.xlim(0, max(t))

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
    """
    Chosen Design Parameters
    Final Optimized time: 5.629999999999924
              Lt                   Rt                   P0                 Rg                   Ls                   Rp                   dens
    Copyable: [0.22846648696036723, 0.05466069781203025, 79287.9280621113, 0.005359712674866734, 0.4054846071410888, 0.030577956262483063, 8940.0]
    Table 3: Train Physical Quantities
        Length of Train: 0.25 m
        Outer Diameter of train: 0.2 m
        Height of train: 0.23 m
        Material of train: Titanium
        Total Mass of Train: 28.9006 kg
        Train frontal area: 0.0982 m^2
        Initial Pressure: 200000 Pa
        Initial tank volume: 0.0007 m^3
        Pinion Gear Radius: 0.006 m
        Length of stroke: 0.3 m
        Total length of piston: 0.45 m
        Diameter of piston: 0.06 m
        Mass of piston: 1.5904 kg
    """
