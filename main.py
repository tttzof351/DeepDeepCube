import pickle as pkl
import torch
from accelerate import Accelerator
import numpy as np
import random
from tqdm import tqdm

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from numba import njit

from sklearn.metrics import root_mean_squared_error

import time

import pandas as pd

from cube3_game import Cube3Game
from a_star import AStar
from dataset import Cube3Dataset
from models import Cube3ResnetModel

import argparse

# seed = 42, 43 using for test
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

    with open("./assets/tests/val_states.pickle", "wb") as f:
         pkl.dump(val_states, f)

    with open("./assets/tests/val_distance.pickle", "wb") as f:
         pkl.dump(val_distances, f)

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
        # if i != 400:
        #     continue

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

def test_resnet():
    game = Cube3Game("./assets/envs/cube_3_3_3_actions.pickle")

    with open("./assets/tests/test_states.pickle", "rb") as f:
        test_states = pkl.load(f)

    with open("./assets/tests/test_distance.pickle", "rb") as f:
        test_distances = pkl.load(f)

    # print("test_states:", test_distances.shape)
    # print("test_distances:", test_distances.shape)

    model = Cube3ResnetModel()
    model.load_state_dict(torch.load("./assets/models/Cube3ResnetModel.pt"))
    
    model = torch.jit.script(model)
    model = torch.jit.trace(model, torch.randint(low=0, high=54, size=(2, 54)))
    # model = torch.compile(model)
    
    accelerator = Accelerator()
    device = accelerator.device
    model = accelerator.prepare(model)

    class HWrap:
        def __init__(self, model):
            self.model = model
            model.eval()
        
        def predict(self, state):
            with torch.no_grad():
                state = torch.tensor(np.array(state)).to(device)
                output = self.model(state)
                return output.detach().cpu().numpy()[:, 0]

    h_wrap = HWrap(model)
    
    records = []
    for i in range(409, len(test_distances)):
        # if i != 400:
        #     continue

        target_distance = test_distances[i]
        if target_distance > 0:
            print(f"Distance i={i}):", target_distance)
            start = time.time()
            state = test_states[i]
            a_star = AStar(
                game=game,
                heuristic=h_wrap,
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
            # df.to_pickle("./assets/reports/report_resnet_dataframe.pkl")

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

def test_cpp(
        path_test_states: str,
        path_test_distance: str,
        output_path: str
):
    import os
    import sys
    sys.path.append("./assets/shared_libraries/macos")

    import cpp_a_star

    with open("./assets/envs/cube_3_3_3_actions.pickle", "rb") as f:
        actions = np.array(pkl.load(f))

    with open(path_test_states, "rb") as f:
        test_states = pkl.load(f)
        
    with open(path_test_distance, "rb") as f:
        test_distance = pkl.load(f)

    cpp_a_star.init_envs(actions)

    records = []
    for i in range(0, len(test_distance)):
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
            "duration_sec": duration
        }
        print(rec)
        records.append(rec)

        df = pd.DataFrame(records)
        df.to_pickle(output_path)

def generate_test_1000():
    seed = 43

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    game = Cube3Game("./assets/envs/cube_3_3_3_actions.pickle")

    test_states, test_distances = game.get_random_states(
        n_states=20, 
        min_distance=1000,
        max_distance=1001
    )

    with open("./assets/tests/test_states_1000.pickle", "wb") as f:
         pkl.dump(test_states, f)

    with open("./assets/tests/test_distance_1000.pickle", "wb") as f:
         pkl.dump(test_distances, f)

def metropolis_a_star_cpp(
    path_test_states = "./assets/tests/test_states.pickle",
    path_test_distance = "./assets/tests/test_distance.pickle",
    output_path = "./assets/reports/metropolis_cpp_reports.pkl"        
):
    import os
    import sys
    sys.path.append("./assets/shared_libraries/macos")

    import cpp_a_star

    with open("./assets/envs/cube_3_3_3_actions.pickle", "rb") as f:
        actions = np.array(pkl.load(f))

    with open(path_test_states, "rb") as f:
        test_states = pkl.load(f)
        
    with open(path_test_distance, "rb") as f:
        test_distance = pkl.load(f)

    cpp_a_star.init_envs(actions)

    model = CatBoostRegressor()
    model.load_model("./assets/models/catboost_cube3.cb")
    game = Cube3Game("./assets/envs/cube_3_3_3_actions.pickle")

    records = []
    # for i in reversed(range(0, len(test_distance))):
    for i in range(0, len(test_distance)):
        start = time.time()

        state = test_states[i]
        target_distance = test_distance[i]

        start_predict = model.predict(state)

        if start_predict > 19:            
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
            
            metropolis_counter = 0
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
                    # print("Prediction:", new_predict)
                
                metropolis_counter += 1

            current_state = state.copy()    
            for a in current_path:
                current_state = game.apply_action(state=current_state, action=a)

            print("Distance: ", target_distance, "; Metropolis: ", metropolis_counter)
            result = cpp_a_star.search_a(
                current_state, # state
                10_000_000, # limit size
                True # debug
            )
            
            if len(result.actions) > 0:
                solution = [-1] + current_path.tolist() + result.actions[1:]
            else:
                solution = []

        else:
            result = cpp_a_star.search_a(
                state, # state
                10_000_000, # limit size
                True # debug
            )
            solution = result.actions
            metropolis_counter = 0
        
        end = time.time()
        duration = np.round(end - start, 3)

        rec = {
            "i": i,
            "state": state,
            "target_distance": target_distance,
            "solution": solution,
            "h_values": [np.round(h, 3) for h in result.h_values],
            "visit_nodes": result.visit_nodes,
            "duration_sec": duration,
            "metropolis_counter": metropolis_counter
        }
        print(rec)
        records.append(rec)

        df = pd.DataFrame(records)
        df.to_pickle(output_path)

def train_resnet():
    game = Cube3Game("./assets/envs/cube_3_3_3_actions.pickle")

    with open("./assets/tests/val_states.pickle", "rb") as f:
        val_states = pkl.load(f)

    with open("./assets/tests/val_distance.pickle", "rb") as f:
        val_distances = pkl.load(f)

    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(val_states), 
        torch.from_numpy(val_distances.astype(np.float32))
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False, 
        num_workers=1
    )
    
    training_dataset = Cube3Dataset(
        game=game,
        size=1_000_000
    )
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset, 
        batch_size=128,
        shuffle=True, 
        num_workers=2
    )
    model = Cube3ResnetModel()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    accelerator = Accelerator()
    device = accelerator.device    

    model, optimizer, training_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, training_dataloader, val_dataloader
    )

    mse_loss = torch.nn.MSELoss()

    print("Accelerator device:", device)

    global_i = 0
    rmse_accum_loss = 0
    print_count = 100
    val_count = 1000
    
    best_val_score = float("inf")

    while True:
        for data in training_dataloader:
            optimizer.zero_grad()
            model.train()

            states, targets = data

            outputs = model(states)

            loss = mse_loss(outputs, targets)
            accelerator.backward(loss)
            optimizer.step()

            rmse_accum_loss += np.sqrt(loss.item())
            global_i += 1
            
            if (global_i % print_count == 0):
                av_rmse_accum_loss = np.round(rmse_accum_loss / print_count, 3)

                print(f"{global_i}): train_rmse={av_rmse_accum_loss}")
                rmse_accum_loss = 0.0

            if (global_i % val_count == 0):
                model.eval()

                val_acc_rmse = 0
                val_count_batch = 0               
                with torch.no_grad():
                    for val_data in val_dataloader:
                        val_count_batch += 1
                        val_states, val_targets = val_data
                        val_targets = val_targets.unsqueeze(dim=1)

                        # print("val_states:", val_states.shape)
                        # print("val_targets:", val_targets.shape)
                        val_outputs = model(val_states)
                        # print("val_outputs:", val_outputs.shape)
                        val_loss = mse_loss(val_outputs, val_targets)                        
                        val_acc_rmse += np.sqrt(val_loss.item())

                val_acc_rmse = np.round(val_acc_rmse / val_count_batch, 4)
                print("==========================")
                print(f"{global_i}): val_rmse={val_acc_rmse}")

                if val_acc_rmse < best_val_score:
                    torch.save(model.state_dict(), "./assets/models/Cube3ResnetModel.pt")
                    best_val_score = val_acc_rmse
                    print(f"Saved model!")
                else:
                    print(f"Old model is best! val_acc_rmse={val_acc_rmse} > best_val_score={best_val_score}")


        #     break
        # break
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')
    args = parser.parse_args()

    if args.mode == "test_catboost":
        test_catboost()
    elif args.mode == "train_catboost":
        train_catboost()
    elif args.mode == "test_resnet":
        test_resnet()
    elif args.mode == "benchmank_a_star":
        benchmank_a_star()
    elif args.mode == "benchmark_catboost":
        benchmark_catboost()
    elif args.mode == "test_cube_env":
        test_cube_env()
    elif args.mode == "test_cpp":
        test_cpp(
            path_test_states = "./assets/tests/test_states.pickle",
            path_test_distance = "./assets/tests/test_distance.pickle",
            output_path = "./assets/reports/cpp_reports.pkl"
        )
    elif args.mode == "test_1000_cpp":
        test_cpp(
            path_test_states = "./assets/tests/test_states_1000.pickle",
            path_test_distance = "./assets/tests/test_distance_1000.pickle",
            output_path = "./assets/reports/cpp_reports_1000.pkl"
        )
    elif args.mode == "gen_test_1000":
        generate_test_1000()
    elif args.mode == "metropolis_a_star_1000":
        metropolis_a_star_cpp(
            path_test_states = "./assets/tests/test_states_1000.pickle",
            path_test_distance = "./assets/tests/test_distance_1000.pickle",
            output_path = "./assets/reports/cpp_metropolis_reports_1000.pkl"
        )
    elif args.mode == "metropolis_a_star":
        metropolis_a_star_cpp(
            path_test_states = "./assets/tests/test_states.pickle",
            path_test_distance = "./assets/tests/test_distance.pickle",
            output_path = "./assets/reports/cpp_metropolis_reports.pkl"
        )
    elif args.mode == "train_resnet":
        train_resnet()