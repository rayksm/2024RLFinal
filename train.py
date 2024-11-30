import gym
from gym import spaces
import numpy as np
import torch
from torch import nn
import dgl
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from env_gym import ABCEnvironment # Replace with your actual module name

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the custom GCN model
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_len):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, hidden_size).to(device)
        self.conv2 = dgl.nn.GraphConv(hidden_size, hidden_size).to(device)
        self.conv3 = dgl.nn.GraphConv(hidden_size, hidden_size).to(device)
        self.conv4 = dgl.nn.GraphConv(hidden_size, out_len).to(device)

    def forward(self, g):
        g = g.to(device)
        g.ndata['feat'] = g.ndata['feat'].to(device)
        g = dgl.add_self_loop(g)

        h = self.conv1(g, g.ndata['feat'])
        h = torch.relu(h)
        h = self.conv2(g, h)
        h = torch.relu(h)
        h = self.conv4(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return torch.squeeze(hg)

# Define the custom fully connected model
class FcModel(nn.Module):
    def __init__(self, num_feats, out_chs):
        super(FcModel, self).__init__()
        self.fc1 = nn.Linear(num_feats, 32 - 4).to(device)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32).to(device)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(32, out_chs).to(device)
        self.gcn = GCN(6, 12, 4)

    def forward(self, x, graph):
        x = x.to(device)
        graph_state = self.gcn(graph)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(torch.cat((x, graph_state), 0))
        x = self.act2(x)
        x = self.fc3(x)
        return x

# Define the custom feature extractor for stable-baselines3
class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, policy_network, value_network):
        super(CustomExtractor, self).__init__(observation_space, features_dim)
        self.policy_network = policy_network
        self.value_network = value_network

    def forward(self, observations):
        graph = observations["graph"]  # Graph data from observations
        state = observations["state"].to(device).float()

        # Forward pass through policy and value networks
        policy_out = self.policy_network(state, graph)
        value_out = self.value_network(state, graph)

        # Combine outputs (optional, depending on usage)
        return torch.cat([policy_out, value_out], dim=-1)

# Wrap the `ABCEnvironment` with graph and state observations
class ABCObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ABCObservationWrapper, self).__init__(env)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(env.dim_state(),), dtype=np.float32),
            "graph": spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),  # Example shape for graph
        })

    def observation(self, obs):
        graph = self.generate_graph()  # Add logic to generate graph data
        return {"state": obs, "graph": graph}

    def generate_graph(self):
        # Replace this with your actual graph generation logic
        return np.random.random((10,))  # Example graph data as random numbers

# Training configuration
def make_env():

    env = ABCEnvironment("/home/rayksm/rlfinal/benchmarks/mcnc/Combinational/blif/dalu.blif")  # Adjust file path as needed
    return ABCObservationWrapper(env)

# Policy kwargs
env_instance = ABCEnvironment("/home/rayksm/rlfinal/benchmarks/mcnc/Combinational/blif/dalu.blif")
policy_kwargs = dict(
    features_extractor_class=CustomExtractor,
    features_extractor_kwargs=dict(
        features_dim = 256,  # Adjust as necessary
        policy_network=FcModel(env_instance.dim_state(), env_instance.num_actions()),  # Example dimensions
        value_network=FcModel(env_instance.dim_state(), 1),   # Example dimensions
    ),
    net_arch=[],  # No additional layers
)

# Create training and evaluation environments
train_env = DummyVecEnv([make_env for _ in range(4)])  # Parallel training environments
eval_env = DummyVecEnv([make_env])  # Single evaluation environment

# Define the PPO model
model = PPO(
    "MultiInputPolicy",  # Use MultiInputPolicy for dictionary observations
    train_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=1e-3,
)

# Train the PPO model
model.learn(total_timesteps=100000)

# Evaluate the PPO model
obs = eval_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)

print("Evaluation complete.")