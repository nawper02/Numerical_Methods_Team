# KIN BLANDFORD, AUSTIN NEFF, HYRUM COLEMAN
# LAB 08

import numpy as np
import matplotlib.pyplot as plt
import csv
from optimize_method import optimize, exhaustive_search, random_search, run_race_simulation

# TODO:
#  - Use random search and exhaustive search to optimize the train's parameters
#  - Read csv file to get best time from previous runs


def main():
    # Initialize bounds

    Lt_bounds = (0.2, 0.3)
    Rt_bounds = (0.05, 0.2)
    P0_bounds = (70000, 200000)
    Rg_bounds = (0.002, 0.01)
    Ls_bounds = (0.1, 0.5)
    Rp_bounds = (0.02, 0.04)
    dens_bounds = (1400, 1200, 7700, 8000, 4500, 8940, 2700)
    # Initialize params dict to be in between bounds

    params = {"Lt": {"bounds": Lt_bounds, "value": 0.25},
              "Rt": {"bounds": Rt_bounds, "value": 0.125},
              "P0": {"bounds": P0_bounds, "value": 100000},
              "Rg": {"bounds": Rg_bounds, "value": 0.006},
              "Ls": {"bounds": Ls_bounds, "value": 0.3},
              "Rp": {"bounds": Rp_bounds, "value": 0.03},
              "dens": {"bounds": dens_bounds, "value": 4500}
              }
    num_trials = 2888
    num_spaces = 6

    # Run optimization

    new_method = True

    if new_method:
        print("Running exhaustive search...")
        res = optimize(exhaustive_search, params, num_spaces)
    else:
        print("Running random search...")
        res = optimize(random_search, params, num_trials)
        print("Completed coarse optimization, beginning local optimization with dist = 0.5")
        res = optimize(random_search, res.x, num_trials, dist=0.5)
        print(".5 local optimization complete, beginning local optimization with dist = 0.1")
        res = optimize(random_search, res.x, num_trials, dist=0.1)
        print(".1 local optimization complete, beginning local optimization with dist = 0.01")
        res = optimize(random_search, res.x, num_trials, dist=0.01)

    # Print optimization results

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
