# KIN BLANDFORD, AUSTIN NEFF, HYRUM COLEMAN
# LAB 08

import numpy as np
import matplotlib.pyplot as plt
from rk4 import rk4
from train_motion import train_motion
from optimize_method import optimize, local_optimize

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

    # Make params_bounds dictionary
    params_bounds = {
        "Lt": Lt_bounds,
        "Rt": Rt_bounds,
        "P0": P0_bounds,
        "Rg": Rg_bounds,
        "Ls": Ls_bounds,
        "Rp": Rp_bounds,
        "dens": dens_bounds}

    num_trials = 1000

    res = optimize(params_bounds, num_trials)
    print("Completed coarse optimization, beginning local optimization")
    res = local_optimize(res, num_trials, .01)
    print("Local optimization complete.")

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

    h = 0.01
    tspan = np.arange(0.0, 10, h)
    t, y = rk4(train_motion, tspan, y0, h, res.x)

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
    plt.savefig("combined.pdf")
    plt.show()


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

    # 5.72
    # [0.22759787898721132, 0.1179539530478348, 170742.99973942252, 0.0030649285122781172, 0.1945437293158895,
    # 0.02969740379346559, 5508.651741337037]

    # 5.659
    # [0.20057940644612615, 0.09296024791629555, 70816.23598171223, 0.004088326139245712, 0.2995078683326422,
    # 0.03554130090484798, 3911.1736625844583]

    # 5.67
    # [0.29782652950466626, 0.1278568956275774, 98542.27223684711, 0.0055843550349363585, 0.3654223978744685,
    # 0.029086120013843354, 2204.6409017118917]

    # 5.73
    # [0.29167760039360136, 0.16287376035864523, 196240.11538343632, 0.004053528909644406, 0.3328770360491726,
    # 0.03338539866008866, 3835.046508285344]

    # 5.72
    # [0.2923267012567824, 0.19065723161207404, 135736.80144742876, 0.0023804678835232292, 0.16654835147538863,
    # 0.02868447929729715, 1630.0399232167974]

    # 5.58
    # [0.29342833084413933, 0.16037627405090937, 132086.6856449127, 0.004518855095953315,
    # 0.30093142628474456, 0.024947758891551694, 1469.2146204772644]

    # unknown time
    # [0.29342833084413933, 0.16037627405090937, 132086.6856449127, 0.004518855095953315, 0.30093142628474456,
    # 0.024947758891551694, 1469.2146204772644]

    # 5.6699999999999235
    # [0.19739815303094518, 0.09346809312765124, 169667.59784325594, 0.004126413728920994, 0.3275365103149462,
    # 0.0327935037139186, 8462.017978586722]

    # 5.649999999999924
    # [0.1919382517333743, 0.0893634089702165, 97938.03198283388, 0.004948947886315307, 0.40893735489112343,
    # 0.028672510065196756, 3438.3636208460707]

    # 5.569999999999926
    # [0.27185975586418787, 0.07226214955804353, 105810.88670226459, 0.002660184276023843, 0.14686578163436,
    # 0.025537158491719152, 7083.541559707305]
