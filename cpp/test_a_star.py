import os
import sys
import numpy as np
import pybind11
import torch
import pickle as pkl
import time
from catboost import CatBoostRegressor
from accelerate import Accelerator
import random

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

sys.path.append("./build")
sys.path.append("../")
import cpp_a_star
from cube3_game import Cube3Game
from models import Cube3ResnetModel

with open("../assets/tests/test_states.pickle", "rb") as f:
    test_states = pkl.load(f)
    
with open("../assets/tests/test_distance.pickle", "rb") as f:
    test_distance = pkl.load(f)

game = Cube3Game("../assets/envs/cube_3_3_3_actions.pickle")

# a_star.run_openmp_test()

cpp_a_star.init_envs(game.actions)
cpp_a_star.run_openmp_test()

if False:
    cpp_a_star.check_hashes()

# t = 400
# t = 931
t = 622
state = test_states[t]

if False:
    catboost_model = CatBoostRegressor()
    catboost_model.load_model("../assets/models/catboost_cube3.cb")

    def catboost_heuristic(state):
        state = np.array(state).reshape(-1, game.space_size)
        out = catboost_model.predict(state)
        return out.tolist()

    print("Catboost Heuristic_search_a:")
    result = cpp_a_star.heuristic_search_a(
        catboost_heuristic,
        state, # state
        1_000_000, # limit size
        True # debug
    )

    print("Result actions: ", result.actions)
    print("Result h_values: ", [np.round(h, 3) for h in result.h_values])
    print("Result visit_nodes: ", result.visit_nodes)

print("=================")

if False:
    print("catboost_search_a:")
    result = cpp_a_star.catboost_search_a(
        state, # state
        1_000_000, # limit size
        True # debug
    )

    print("Result actions: ", result.actions)
    print("Result h_values: ", [np.round(h, 3) for h in result.h_values])
    print("Result visit_nodes: ", result.visit_nodes)

print("=================")

if False:
    print("resnet_search_a:")

    resnet_model = Cube3ResnetModel()
    resnet_model.load_state_dict(torch.load("../assets/models/Cube3ResnetModel.pt"))
    resnet_model = resnet_model.to_torchscript()
    accelerator = Accelerator()

    device = accelerator.device
    resnet_model = accelerator.prepare(resnet_model)
    resnet_model.eval()

    def pytorch_heuristic(state):
        state = np.array(state).reshape(-1, game.space_size).astype(np.int32)
        # print("state:", state.shape)
        state = torch.tensor(state).to(device)
        # print("state torch:", state)

        with torch.no_grad():
            output = resnet_model(state)
            output = output.detach().cpu().numpy()[:, 0].tolist()
            # print("output:", output)

            return output

    result = cpp_a_star.heuristic_search_a(
        pytorch_heuristic,
        state, # state
        1_000_000, # limit size
        True # debug
    )

    print("Result actions: ", result.actions)
    print("Result h_values: ", [np.round(h, 3) for h in result.h_values])
    print("Result visit_nodes: ", result.visit_nodes)

print("=================")

if True:
    print("catboost_parallel_search_a:")
    result = cpp_a_star.catboost_parallel_search_a(
        state, # state
        1_000_000, # limit size
        True # debug
    )

    print("Result actions: ", result.actions)
    print("Result h_values: ", [np.round(h, 3) for h in result.h_values])
    print("Result visit_nodes: ", result.visit_nodes)

