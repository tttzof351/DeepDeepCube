import torch
import numpy as np
import random

from cube3_game import Cube3Game

class Cube3Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            game,
            size=10_000,
            min_distance=0,
            max_distance=30,
            seed=0                        
        ):
        self.game = game
        self.seed = seed
        self.size = size
        self.min_distance = min_distance
        self.max_distance = max_distance

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        state, target_distance =  self.game.get_random_states(
            n_states=1,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
            verbose=False
        )

        return state[0, :], target_distance.astype(np.float32)
    
if __name__ == "__main__":
    game = Cube3Game("./assets/envs/cube_3_3_3_actions.pickle")
    dataset = Cube3Dataset(game=game)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=100,
        shuffle=True, 
        num_workers=0
    )

    for data in dataloader:
        states, targets = data
        print("states:", states.shape)
        print("targets:", targets.shape)
        break