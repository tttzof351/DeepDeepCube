import os
import sys
import numpy as np
import pybind11
import torch
import pickle as pkl
import time

sys.path.append("./build")
import cpp_a_star

with open("../assets/envs/cube_3_3_3_actions.pickle", "rb") as f:
    actions = np.array(pkl.load(f))

with open("../assets/tests/test_states.pickle", "rb") as f:
    test_states = pkl.load(f)
    
with open("../assets/tests/test_distance.pickle", "rb") as f:
    test_distance = pkl.load(f)

# a_star.run_openmp_test()

cpp_a_star.init_envs(actions)

t = 800
print("Distance: ", test_distance[t])
result = cpp_a_star.search_a(
    test_states[t], # state
    5_000_000, # limit size
    True # debug
)

print("Result actions: ", result.actions)
print("Result h_values: ", [np.round(h, 3) for h in result.h_values])
print("Result visit_nodes: ", result.visit_nodes)