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

with open("../assets/envs/cube_3_3_3_actions.pickle", "rb") as f:
    actions = np.array(pkl.load(f))

with open("../assets/tests/test_states.pickle", "rb") as f:
    test_states = pkl.load(f)
    
with open("../assets/tests/test_distance.pickle", "rb") as f:
    test_distance = pkl.load(f)

# a_star.run_openmp_test()

cpp_a_star.init_envs(actions)

game = Cube3Game("../assets/envs/cube_3_3_3_actions.pickle")

# print("Start alloc test")
# cpp_a_star.test_allocation_dealocation()
# print("End alloc test")


t = 931
state = test_states[t]
model = CatBoostRegressor()
model.load_model("../assets/models/catboost_cube3.cb")


start_predict = model.predict(state)
target_predict = 16
diff_predict = int(start_predict - target_predict)

start_path = np.random.choice(
    a=len(game.actions),
    size=diff_predict
)

print("start_predict:", start_predict)
print("diff predict:", diff_predict)
print("start_path:", start_path)

current_predict = start_predict
current_path = start_path

while current_predict > target_predict:
    random_pos = np.random.randint(0, len(current_path))
    random_action = np.random.randint(0, len(game.actions))

    new_path = current_path.copy()
    new_path[random_pos] = random_action

    current_state = state.copy()    
    for a in current_path:
        current_state = game.apply_action(state=current_state, action=a)

    new_predict = model.predict(current_state)
    epsilon = np.random.rand()
    
    if new_predict < current_predict or epsilon > 0.9:
        current_predict = new_predict
        current_path = new_path
        print("Prediction:", new_predict)

current_state = state.copy()    
for a in current_path:
    current_state = game.apply_action(state=current_state, action=a)

result = cpp_a_star.search_a(
    current_state, # state
    1_000_000, # limit size
    True # debug
)

print("Result actions: ", result.actions)
print("Result h_values: ", [np.round(h, 3) for h in result.h_values])
print("Result visit_nodes: ", result.visit_nodes)
