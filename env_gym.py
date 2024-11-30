import gym
from gym import spaces
import abc_py as abcPy
import numpy as np
import torch
import dgl
import graphExtractor as GE

class ABCEnvironment(gym.Env):
    """
    Gym-compatible environment for working with ABC synthesis tasks.
    """

    def __init__(self, aigfile, max_steps=20):
        super(ABCEnvironment, self).__init__()
        self._abc = abcPy.AbcInterface()
        self._aigfile = aigfile
        self.max_steps = max_steps

        # Initialize ABC environment
        self._abc.start()
        self._abc.read(self._aigfile)
        init_stats = self._abc.aigStats()
        self.init_num_and = float(init_stats.numAnd)
        self.init_lev = float(init_stats.lev)
        self._setup_reward_baseline()

        # Define action and observation space
        self.action_space = spaces.Discrete(self.num_actions())
        state_dim = self.dim_state()
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32),
            "graph": spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),  # Example for graph features
        })

    def _setup_reward_baseline(self):
        self.resyn2()
        self.resyn2()
        resyn2_stats = self._abc.aigStats()
        total_reward = self._stat_value(self._abc.aigStats()) - self._stat_value(resyn2_stats)
        self.reward_baseline = total_reward / self.max_steps

    def reset(self):
        self._abc.end()
        self._abc.start()
        self._abc.read(self._aigfile)
        self._last_stats = self._abc.aigStats()
        self._cur_stats = self._last_stats
        self.len_seq = 0
        self.last_act = -1
        return self._get_observation()

    def step(self, action):
        self._take_action(action)
        self.len_seq += 1

        observation = self._get_observation()
        reward = self._get_reward()
        done = self.len_seq >= self.max_steps
        return observation, reward, done, {}

    def close(self):
        self._abc.end()

    def _take_action(self, action_idx):
        if action_idx == 0:
            self._abc.balance(l=False)
        elif action_idx == 1:
            self._abc.rewrite(l=False)
        elif action_idx == 2:
            self._abc.refactor(l=False)
        elif action_idx == 3:
            self._abc.rewrite(l=False, z=True)
        elif action_idx == 4:
            self._abc.refactor(l=False, z=True)
        elif action_idx == 5:
            self._abc.end()
        else:
            raise ValueError("Invalid action index")

        self._last_stats = self._cur_stats
        self._cur_stats = self._abc.aigStats()

    def _get_observation(self):
        state_array = np.array([
            self._cur_stats.numAnd / self.init_num_and,
            self._cur_stats.lev / self.init_lev,
            self._last_stats.numAnd / self.init_num_and,
            self._last_stats.lev / self.init_lev
        ])
        step_array = np.array([self.len_seq / self.max_steps])
        combined = np.concatenate((state_array, step_array), axis=-1).astype(np.float32)
        graph = GE.extract_dgl_graph(self._abc)  # Example for DGL graph extraction
        print(graph)
        return {"state": combined, "graph": graph}

    def _get_reward(self):
        if self.len_seq >= self.max_steps:
            return 0
        return self._stat_value(self._last_stats) - self._stat_value(self._cur_stats) - self.reward_baseline

    def resyn2(self):
        self._abc.balance(l=False)
        self._abc.rewrite(l=False)
        self._abc.refactor(l=False)
        self._abc.balance(l=False)
        self._abc.rewrite(l=False, z=True)
        self._abc.refactor(l=False, z=True)
        self._abc.rewrite(l=False, z=True)

    def num_actions(self):
        return 6

    def dim_state(self):
        return 4 + self.num_actions() * 1 + 1

    def _stat_value(self, stat):
        return float(stat.numAnd) / float(self.init_num_and)