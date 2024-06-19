import os
import sys
import numpy as np
import pybind11
import torch
import pickle as pkl
import time

sys.path.append("./build")
import a_star

with open("../assets/envs/cube_3_3_3_actions.pickle", "rb") as f:
    actions = np.array(pkl.load(f))

with open("../assets/tests/test_states.pickle", "rb") as f:
    test_states = pkl.load(f)
    
with open("../assets/tests/test_distance.pickle", "rb") as f:
    test_distance = pkl.load(f)

a_star.init_wyhash()
a_star.set_cube3_actions(actions)
# a_star.run_openmp_test()

t = 200
print("Distance: ", test_distance[t])
a_star.search_a(
    test_states[t], # state
    10_000_000, # limit size
    True # debug
)