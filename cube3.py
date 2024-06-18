import pickle as pkl
import torch
import numpy as np
import random
from tqdm import tqdm

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import heapq
import bisect
import time
from numba import njit

import pandas as pd

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class Cube3Game:
    def __init__(self, path_to_action: str):        
        with open(path_to_action, "rb") as f:
            self.actions = np.array(pkl.load(f))

        self.initial_state = np.arange(0, 54)
        self.hash_inital_state = hash(str(self.initial_state))

    def apply_action(self, state, action):
        return state[self.actions[action]]

    # def apply_all_actions(self, state):
    #     states = np.expand_dims(state, axis=0)             
    #     states = np.repeat(
    #         a=states, 
    #         repeats=len(self.actions),
    #         axis=0
    #     )
    #     for i in range(self.actions):
    #         states[i, :] = 
        

    def get_random_states(
            self, 
            n_states=10, 
            max_distance=30
        ):
            random_states = np.expand_dims(self.initial_state, axis=0)             
            random_states = np.repeat(
                a=random_states, 
                repeats=n_states,
                axis=0
            )

            num_random_disatnces = np.random.choice(
                 a=max_distance,
                 size=n_states
            )
            num_random_disatnces = np.sort(num_random_disatnces)            

            for i in tqdm(range(n_states)):
                state = random_states[i, :]
                distance = num_random_disatnces[i]
                path = np.random.choice(
                    a=len(self.actions),
                    size=distance
                )
                for action in path:
                    state = self.apply_action(state=state, action=action)
                
                random_states[i, :] = state

            return random_states, num_random_disatnces
    
    def is_goal_by_state(self, state):
        return np.array_equal(
            state,
            self.initial_state
        )
    
    def is_goal(self, state, state_hash):
        if state_hash == self.hash_inital_state:
            return self.is_goal_by_state(state=state)
        else:
            return False

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


class Node:
    def __init__(self, state: np.array, h):
        self.state = state
        self.state_hash = hash(str(state))

        # print("H:", h)
        self.h = h
        self.g = 0.0       
        self.f = None
        
        self.parent = None
        self.action = None
        
    def reset_f(self):
        self.f = self.h + 0.9 * self.g
        
    def get_path(self):
        node = self
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))
    
    def __eq__(self, other):
        if self.state_hash == other.state_hash:
            return np.array_equal(self.state, other.state)
        else:
            return False
        
    def __lt__(self, other):
        return self.f < other.f

class NodeQueue:
    def __init__(self):
        self.queue = []
        self.hashes = {}

    def insert(self, node):
        heapq.heappush(self.queue, node)
        self.hashes[node.state_hash] = node

    def pop_min_element(self):
        node = heapq.heappop(self.queue)
        # print("pop:", node.state)
        # print("pop:", node.state_hash)

        _ = self.hashes.pop(node.state_hash)
        
        return node
    
    def is_contains(self, node):
        return node.state_hash in self.hashes
    
    def __len__(self):
        return len(self.queue)
    
    def reset_q(self):
        heapq.heapify(self.queue)

class AStar:
    def __init__(
            self, 
            game,
            heuristic,
            root_state,
            limit_size = 100_000,
            verbose=False
        ):
        self.limit_size = limit_size        
        self.verbose = verbose
        self.heuristic = heuristic
        
        root_node = Node(
            state=root_state, 
            h=heuristic.predict(root_state)
        )
        root_node.reset_f()
        
        self.open = NodeQueue()
        self.close = NodeQueue()

        self.open.insert(root_node)

    def search(self, game):
        global_i = 0
        start = time.time()
        while len(self.open) > 0:
            best_node = self.open.pop_min_element()

            child_states = [game.apply_action(best_node.state, action) for action in range(len(game.actions))]
            child_hs = self.heuristic.predict(child_states)
            
            for action in range(len(game.actions)):
                child_state = child_states[action]
                child_h = child_hs[action]
                
                child_node = Node(
                    state=child_state, 
                    h=child_h
                )
                child_node.parent = best_node
                child_node.g = best_node.g + 1
                child_node.action = action
                child_node.reset_f()
                
                if game.is_goal(state=child_node.state, state_hash=child_node.state_hash):
                    return child_node, len(self.close)
                
                if self.close.is_contains(child_node):                    
                    continue
                elif self.open.is_contains(child_node):
                    prev_child_node = self.open.hashes[child_node.state_hash]
                    if prev_child_node.g > child_node.g:
                        prev_child_node.g = child_node.g
                        prev_child_node.parent = child_node.parent
                        prev_child_node.action = child_node.action                        
                        prev_child_node.reset_f()

                        # TODO: Need to return ?
                        # self.open.reset_q()
                    pass                
                else:
                    # path = child_node.get_path()
                    # print("path:", path)
                    self.open.insert(child_node)                

            self.close.insert(best_node)
            
            global_i += 1
            if global_i % 1000 == 0 and self.verbose:
                end = time.time()
                duration = np.round(end - start, 3)
                print(f"close len: {len(self.close)}; duration: {duration} s; g_min: {np.round(best_node.g, 3)}; h_min: {np.round(best_node.h, 3)}; f_min: {np.round(best_node.f, 3)}")
                start = end

            if global_i > self.limit_size:
                return None, len(self.close)
        
        return None, len(self.close)


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
        if i != 800:
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

            target_node, close_nodes = a_star.search(game)
            end = time.time()
            duration = np.round(end - start, 3)
            
            rec = {
                "i": i,
                "state": state,
                "distance": target_distance,
                "path": None,
                "path_h": None,
                "close_nodes": close_nodes,
                "duration": duration
            }
            if target_node is not None:
                solution_path = target_node.get_path()
                rec["path"] = [n.action for n in solution_path]
                rec["path_h"] = [np.round(n.h, 3) for n in solution_path]

                print("Target node:", target_node.state)
                print("close_nodes:", close_nodes)
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
    target_node, close_nodes = a_star.search(game)
    end = time.time()
    duration = np.round(end - start, 3)
    
    solution_path = target_node.get_path()
    print("Target node:", target_node.state)
    print("close_nodes:", close_nodes)
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

if __name__ == "__main__":
    # train_catboost()
    test_catboost()
    # benchmank_a_star()
    # benchmark_catboost()
    # test_cube_env()