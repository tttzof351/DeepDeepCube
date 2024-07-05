import numpy as np
import time

from numba import njit
import heapq
import bisect


class Node:
    def __init__(self, state: np.array, h):
        self.state = state
        self.state_hash = hash(str(state))

        # print("H:", h)
        self.h = h
        self.g = 0.0       
        self.f = None
        
        self.parent = None
        self.action = -1
        
    def reset_f(self):
        self.f = self.h + self.g
        
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
        
        h = heuristic.predict([root_state])[0]
        # print("h:", h)
        root_node = Node(
            state=root_state, 
            h=h
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
            # print("child_hs:", child_hs)
            
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
                    return child_node
                
                if self.close.is_contains(child_node):                    
                    continue
                elif self.open.is_contains(child_node):
                    #TODO: Need implementation
                    # prev_child_node = self.open.hashes[child_node.state_hash]
                    # if prev_child_node.g > child_node.g:
                    #     prev_child_node.g = child_node.g
                    #     prev_child_node.parent = child_node.parent
                    #     prev_child_node.action = child_node.action                        
                    #     prev_child_node.reset_f()

                        # TODO: Need to return ?
                        # self.open.reset_q()
                    pass                
                else:
                    self.open.insert(child_node)                

            self.close.insert(best_node)
            
            global_i += 1
            if global_i % 1000 == 0 and self.verbose:
                end = time.time()
                duration = np.round(end - start, 3)
                print(f"close len: {len(self.close)}; duration: {duration} s; g_min: {np.round(best_node.g, 3)}; h_min: {np.round(best_node.h, 3)}; f_min: {np.round(best_node.f, 3)}")
                start = end

            if global_i > self.limit_size:
                return None
        
        return None
