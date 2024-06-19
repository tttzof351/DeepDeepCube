import pickle as pkl
import torch
import numpy as np
import random
from tqdm import tqdm

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import time

import pandas as pd

from cube3_game import Cube3Game
from a_star import AStar

import argparse

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def train_catboost():
    game = Cube3Game("./assets/envs/cube_3_3_3_actions.pickle")

    max_distance = 30

    train_states, train_distances = game.get_random_states(
        n_states=10_000_000, 
        max_distance=max_distance
    )
    val_states, val_distances = game.get_random_states(
        n_states=1000, 
        max_distance=max_distance
    )    
    
    test_states, test_distances = game.get_random_states(
        n_states=1000, 
        max_distance=max_distance
    )
    with open("./assets/tests/test_states.pickle", "wb") as f:
         pkl.dump(test_states, f)

    with open("./assets/tests/test_distance.pickle", "wb") as f:
         pkl.dump(test_distances, f)

    print(f"train_states: {train_states.shape}; train_distances: {train_distances.shape}")
    model = CatBoostRegressor(
        verbose=False, 
        iterations=10_000,
        max_depth=8,
        use_best_model=True
    )
    model.fit(
         train_states,
         train_distances,
         verbose=True,
         eval_set=(val_states, val_distances)
    )

    test_predictions = model.predict(test_states)
    r2_test = r2_score(test_distances, test_predictions)
    mse_test = mean_squared_error(test_distances, test_predictions)

    print("r2_test:", r2_test)
    print("mse_test:", mse_test)

    print("test_distances:", test_distances[-500:-500+10])
    print("test_predictions", test_predictions[-500:-500+10])

    model.save_model(f"./assets/models/catboost_cube3.cb")
    model.save_model(f"./assets/models/catboost_cube3.cpp", format="CPP")    


def test_catboost():
    game = Cube3Game("./assets/envs/cube_3_3_3_actions.pickle")

    with open("./assets/tests/test_states.pickle", "rb") as f:
        test_states = pkl.load(f)

    with open("./assets/tests/test_distance.pickle", "rb") as f:
        test_distances = pkl.load(f)

    # print("test_states:", test_distances.shape)
    # print("test_distances:", test_distances.shape)

    model = CatBoostRegressor()
    model.load_model("./assets/models/catboost_cube3.cb")

    records = []
    for i in range(len(test_distances)):
        if i != 400:
            continue

        target_distance = test_distances[i]
        if target_distance > 0:
            print(f"Distance i={i}):", target_distance)
            start = time.time()
            state = test_states[i]
            a_star = AStar(
                game=game,
                heuristic=model,
                root_state=state,
                limit_size=100_000,
                verbose=True
            )

            target_node = a_star.search(game)
            visit_nodes = len(a_star.close)
            end = time.time()
            duration = np.round(end - start, 3)
            
            rec = {
                "i": i,
                "state": state,
                "distance": target_distance,
                "path": None,
                "path_h": None,
                "visit_nodes": visit_nodes,
                "duration": duration
            }
            if target_node is not None:
                solution_path = target_node.get_path()
                rec["path"] = [n.action for n in solution_path]
                rec["path_h"] = [np.round(n.h, 3) for n in solution_path]

                print("Target node:", target_node.state)
                print("visit_nodes:", visit_nodes)
                print(f"g_min: {np.round(target_node.g, 3)}; h_min: {np.round(target_node.h, 3)}; f_min: {np.round(target_node.f, 3)}")              
                print("Path actions:", [n.action for n in solution_path])
                print("Path h:", [np.round(n.h, 3) for n in solution_path])
            else:
                print("Can't find!")
            
            records.append(rec)
            df = pd.DataFrame(records)
            df.to_pickle("./assets/reports/report_dataframe.pkl")
                        
def benchmank_a_star():
    with open("test_states.pickle", "rb") as f:
        test_states = pkl.load(f)

    game = Cube3Game("./cube_3_3_3_actions.pickle")

    model = CatBoostRegressor()
    model.load_model("catboost_cube3.cb")

    state = test_states[332]

    start = time.time()
    a_star = AStar(
        game=game,
        heuristic=model,
        root_state=state,
        limit_size=100_000,
        verbose=True
    )
    target_node = a_star.search(game)
    visit_nodes = len(a_star.close)

    end = time.time()
    duration = np.round(end - start, 3)
    
    solution_path = target_node.get_path()
    print("Target node:", target_node.state)
    print("visit_nodes:", visit_nodes)
    print(f"g_min: {np.round(target_node.g, 3)}; h_min: {np.round(target_node.h, 3)}; f_min: {np.round(target_node.f, 3)}")              
    print("Path actions:", [n.action for n in solution_path])
    print("Path h:", [np.round(n.h, 3) for n in solution_path])

def benchmark_catboost():
    model = CatBoostRegressor()
    model.load_model("catboost_cube3.cb")

    start = time.time()

    for _ in range(10000):
        inp = np.arange(54)
        _ = model.predict(inp)

    end = time.time()
    
    duration = np.round(end - start, 3)
    print(f"Duration: {duration} sec")

def test_cube_env():
    game = Cube3Game("./assets/envs/cube_3_3_3_actions.pickle")

    with open("./assets/tests/test_states.pickle", "rb") as f:
        test_states = pkl.load(f)

    with open("./assets/tests/test_distance.pickle", "rb") as f:
        test_distances = pkl.load(f)

    print("test_distances[40]: ", test_distances[40])
    print("test_states[40]: ", test_states[40])

    root_state = test_states[40]
    for action in range(len(game.actions)):
        child_state = game.apply_action(
            state=root_state,
            action=action
        )
        print(f"action:{action}")
        print(child_state)

        if game.is_goal_by_state(child_state):
            print("Found!")
            break

    pass

def test_cpp():
    import os
    import sys
    sys.path.append("./assets/shared_libraries/macos")

    import cpp_a_star

    with open("./assets/envs/cube_3_3_3_actions.pickle", "rb") as f:
        actions = np.array(pkl.load(f))

    with open("./assets/tests/test_states.pickle", "rb") as f:
        test_states = pkl.load(f)
        
    with open("./assets/tests/test_distance.pickle", "rb") as f:
        test_distance = pkl.load(f)

    cpp_a_star.init_envs(actions)

    records = []
    for i in range(len(test_distance)):
        start = time.time()

        state = test_states[i]
        target_distance = test_distance[i]
        
        print("Distance: ", target_distance)
        result = cpp_a_star.search_a(
            state, # state
            10_000_000, # limit size
            True # debug
        )

        end = time.time()
        duration = np.round(end - start, 3)

        rec = {
            "i": i,
            "state": state,
            "target_distance": target_distance,
            "solution": result.actions,
            "h_values": [np.round(h, 3) for h in result.h_values],
            "visit_nodes": result.visit_nodes,
            "duration": duration
        }
        print(rec)
        records.append(rec)

        df = pd.DataFrame(records)
        df.to_pickle("./assets/reports/cpp_reports.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')
    args = parser.parse_args()

    if args.mode == "test":
        test_catboost()
    elif args.mode == "train":
        train_catboost()
    elif args.mode == "benchmark_a_star":
        benchmank_a_star()
    elif args.mode == "benchmark_catboost":
        benchmark_catboost()
    elif args.mode == "test_env":
        test_cube_env()
    elif args.mode == "test_cpp":
        test_cpp()
