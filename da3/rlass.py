import gym
from gym import spaces
import numpy as np


class TreasureHuntEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def _init_(self):
        self.shape = (6, 6)
        self.n_states = np.prod(self.shape)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.n_states)
        self.player_pos = None
        self.treasure_pos = None
        self.obstacles = []
        self.enemies = []
        self.reset()

    def reset(self):
        self.player_pos = tuple(np.random.randint(0, self.shape[0], 2))
        self.treasure_pos = tuple(np.random.randint(0, self.shape[0], 2))
        self.obstacles = [tuple(np.random.randint(
            0, self.shape[0], 2)) for _ in range(6)]
        self.enemies = [tuple(np.random.randint(0, self.shape[0], 2))
                        for _ in range(5)]
        while self.player_pos == self.treasure_pos or self.player_pos in self.obstacles or self.player_pos in self.enemies:
            self.player_pos = tuple(np.random.randint(0, self.shape[0], 2))
        return self._get_obs()

    def step(self, action):
        if action == 0:  # move up
            new_pos = (self.player_pos[0] - 1, self.player_pos[1])
        elif action == 1:  # move down
            new_pos = (self.player_pos[0] + 1, self.player_pos[1])
        elif action == 2:  # move left
            new_pos = (self.player_pos[0], self.player_pos[1] - 1)
        elif action == 3:  # move right
            new_pos = (self.player_pos[0], self.player_pos[1] + 1)

        if not (0 <= new_pos[0] < self.shape[0] and 0 <= new_pos[1] < self.shape[1]):
            # player out of bounds
            reward = -1
        elif new_pos == self.treasure_pos:
            # player has reached the treasure
            reward = 1
        elif new_pos in self.obstacles:
            # player hit an obstacle
            reward = -1
        elif new_pos in self.enemies:
            # player hit an enemy
            reward = -1
        else:
            # valid move
            reward = 0
            self.player_pos = new_pos

        done = reward != 0
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.ravel_multi_index(self.player_pos, self.shape)

    def render(self, mode='human'):
    outfile = StringIO() if mode == 'ansi' else sys.stdout
    for i in range(self.shape[0]):
        for j in range(self.shape[1]):
            if self.player_pos == (i, j):
                output = "P "
            elif self.treasure_pos == (i, j):
                output = "T "
            elif (i, j) in self.obstacles:
                output = "O "
            elif (i, j) in self.enemies:
                output = "E "
            else:
                output = "- "
            outfile.write(output)
        outfile.write("\n")

    return outfile.getvalue()
