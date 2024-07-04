import numpy as np
import pickle as pkl
from tqdm import tqdm

class Cube3Game:
    def __init__(self, path_to_action: str):        
        with open(path_to_action, "rb") as f:
            self.actions = np.array(pkl.load(f))

        self.initial_state = np.arange(0, 54)
        self.hash_inital_state = hash(str(self.initial_state))

    def apply_action(self, state, action):
        return state[self.actions[action]]        

    def get_random_states(
            self, 
            n_states=10, 
            min_distance=0,
            max_distance=30,
            verbose=True
        ):
            random_states = np.expand_dims(self.initial_state, axis=0)             
            random_states = np.repeat(
                a=random_states, 
                repeats=n_states,
                axis=0
            )

            num_random_disatnces = np.random.choice(
                 a=np.arange(min_distance, max_distance),
                 size=n_states
            )
            num_random_disatnces = np.sort(num_random_disatnces)            

            if verbose:
                iterations = tqdm(range(n_states))
            else:
                iterations = range(n_states)
            
            for i in iterations:
                state = random_states[i, :]
                distance = num_random_disatnces[i]
                path = np.random.choice(
                    a=len(self.actions),
                    size=distance
                )
                for k, action in enumerate(path):
                    # print(f"{i}-{k}): {action}")
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
