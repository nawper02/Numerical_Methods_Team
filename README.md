# Numerical_Methods_Team


This is the GitHub repository for Kin Blandford, Hyrum Coleman, and Austin Neff's Numerical Methods project.

# What's happening here?

The files in this repository optimizes 7 design parameters of a train in order to minimize the amount of time it takes for the train to travel 10 meters without exceeding 12 meters.
We implemented random search and exhaustive search and used multiprocessing to speed up our computation time.


# What files are in here?

`main.py`:
This script contains the initiation of our bounds and starting value in the params dictionary and the user input options for which optimization method they want to use. It also reads `race_times.csv` to get the best params from previous runs as a starting point.

`optimize_method.py`:
This script contains all of our optimization methods than `main.py` calls. This includes our random search and exhaustive search. It also includes our multiprocessing method.

`train_motion.py`:
This script contains the necessary equations of motion to simulate the motion of the train.

`rk4.py`:
This script is our rk4 method that solves the ODE's in `train_motion.py`
