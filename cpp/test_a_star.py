import os
import sys
import numpy as np
import pybind11
import torch
import pickle as pkl
import time
from catboost import CatBoostRegressor
import random

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

sys.path.append("./build")
sys.path.append("../")
import cpp_a_star
from cube3_game import Cube3Game

with open("../assets/tests/test_states.pickle", "rb") as f:
    test_states = pkl.load(f)
    
with open("../assets/tests/test_distance.pickle", "rb") as f:
    test_distance = pkl.load(f)

game = Cube3Game("../assets/envs/cube_3_3_3_actions.pickle")

# a_star.run_openmp_test()

cpp_a_star.init_envs(game.actions)
cpp_a_star.run_openmp_test()

# print("Start alloc test")
# cpp_a_star.test_allocation_dealocation()
# print("End alloc test")

model = CatBoostRegressor()
model.load_model("../assets/models/catboost_cube3.cb")

t = 400
# t = 931
state = test_states[t]

def heuristic(state):
    state = np.array(state).reshape(-1, game.space_size)
    out = model.predict(state)
    return out.tolist()

print("Heuristic_search_a:")
result = cpp_a_star.heuristic_search_a(
    heuristic,
    state, # state
    1_000_000, # limit size
    True # debug
)

print("Result actions: ", result.actions)
print("Result h_values: ", [np.round(h, 3) for h in result.h_values])
print("Result visit_nodes: ", result.visit_nodes)

print("=================")

print("catboost_search_a:")
result = cpp_a_star.catboost_search_a(
    state, # state
    1_000_000, # limit size
    True # debug
)

print("Result actions: ", result.actions)
print("Result h_values: ", [np.round(h, 3) for h in result.h_values])
print("Result visit_nodes: ", result.visit_nodes)
