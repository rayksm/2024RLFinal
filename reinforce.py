import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import bisect
import random
from dgl.nn.pytorch import GraphConv
import dgl
import mcts_pure as mcts

torch.manual_seed(2024)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attn_fc = nn.Linear(hidden_size, 1)  # Learnable weights for attention

    def forward(self, g, h):
        # Compute attention scores
        g.ndata['score'] = self.attn_fc(h)  # Shape: [num_nodes, 1]
        g.ndata['score'] = torch.softmax(g.ndata['score'], dim=0)  # Normalize across nodes

        # Weighted sum of node features
        g.ndata['weighted_feat'] = g.ndata['score'] * h  # Apply attention to node features
        hg = dgl.sum_nodes(g, 'weighted_feat')  # Aggregate to get graph-level representation
        return hg
    
class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, out_len):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size).to(device)
        self.conv2 = GraphConv(hidden_size, hidden_size).to(device)
        self.conv3 = GraphConv(hidden_size, hidden_size).to(device)
        self.conv4 = GraphConv(hidden_size, hidden_size).to(device)
        
        self.conv5 = GraphConv(hidden_size, int(hidden_size / 2)).to(device)
        self.conv6 = GraphConv(int(hidden_size / 2), int(hidden_size / 2)).to(device)

        self.conv7 = GraphConv(int(hidden_size / 2), out_len).to(device)



    def forward(self, g):
        g = g.to(device)
        g.ndata['feat'] = g.ndata['feat'].to(device)
        g = dgl.add_self_loop(g)

        h = self.conv1(g, g.ndata['feat'])
        h = torch.relu(h)
        h_res = h

        h = self.conv2(g, h)
        h = torch.relu(h)
        h = h + h_res

        h = self.conv3(g, h)
        h = torch.relu(h)
        h_res = h

        h = self.conv4(g, h)
        h = torch.relu(h)
        h = h + h_res
        
        h = self.conv5(g, h)
        h = torch.relu(h)
        h_res = h

        h = self.conv6(g, h)
        h = torch.relu(h)
        h = h + h_res

        h = self.conv7(g, h)
        h = torch.relu(h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        
        #hg = self.pool(g, h)
        #hg = self.fc(hg)
        
        return torch.squeeze(hg)
        

class FcModel(nn.Module):
    def __init__(self, numFeats, outChs):
        super(FcModel, self).__init__()
        self._numFeats = numFeats
        self._outChs = outChs

        self.fc1 = nn.Linear(numFeats, 32).to(device)
        self.gcn = GCN(6, 64, 16)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(32 + 16, 32).to(device)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(32, 32).to(device)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(32, 32).to(device)
        self.act4 = nn.ReLU()

        self.fc5 = nn.Linear(32, 32).to(device)
        self.act5 = nn.ReLU()

        self.fc6 = nn.Linear(32, outChs).to(device)

        #self.dropout1 = nn.Dropout(p=0.5)
        #self.dropout2 = nn.Dropout(p=0.5)
        #self.dropout3 = nn.Dropout(p=0.5)
        #self.dropout4 = nn.Dropout(p=0.5)

    def forward(self, x, graph):
        x = x.to(device)
        graph_state = self.gcn(graph)

        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(torch.cat((x, graph_state), dim = 0))
        x = self.act2(x)
        #x = self.dropout1(x)
        x_res = x

        x = self.fc3(x)
        x = self.act3(x)
        #x = self.dropout2(x)
        x = x + x_res

        x = self.fc4(x)
        x = self.act4(x)
        #x = self.dropout3(x)
        x_res = x

        x = self.fc5(x)
        x = self.act5(x)
        #x = self.dropout4(x)
        x = x + x_res
        
        x = self.fc6(x)
        return x


class FcModelGraph(nn.Module):
    def __init__(self, numFeats, outChs):
        super(FcModelGraph, self).__init__()
        self._numFeats = numFeats
        self._outChs = outChs
        
        self.fc1 = nn.Linear(numFeats, 32).to(device)
        self.gcn = GCN(6, 64, 16)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(32 + 16, 32).to(device)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(32, 32).to(device)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(32, 32).to(device)
        self.act4 = nn.ReLU()

        self.fc5 = nn.Linear(32, 32).to(device)
        self.act5 = nn.ReLU()

        self.fc6 = nn.Linear(32, outChs).to(device)

        #self.dropout1 = nn.Dropout(p=0.5)
        #self.dropout2 = nn.Dropout(p=0.5)
        #self.dropout3 = nn.Dropout(p=0.5)
        #self.dropout4 = nn.Dropout(p=0.5)

    def forward(self, x, graph):
        x = x.to(device)
        graph_state = self.gcn(graph)

        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(torch.cat((x, graph_state), dim = 0))
        x = self.act2(x)
        #x = self.dropout1(x)
        x_res = x

        x = self.fc3(x)
        x = self.act3(x)
        #x = self.dropout2(x)
        x = x + x_res

        x = self.fc4(x)
        x = self.act4(x)
        #x = self.dropout3(x)
        x_res = x

        x = self.fc5(x)
        x = self.act5(x)
        #x = self.dropout4(x)
        x = x + x_res
        
        x = self.fc6(x)
        return x
        #print("graph_state:", graph_state)
        #print("After x = ", x)
        #print()
        return x

# Policy Network
class PiApprox(object):
    def __init__(self, dimStates, numActs, alpha, network):
        self._dimStates = dimStates
        self._numActs = numActs
        self._alpha = alpha
        self._network = network(dimStates, numActs).to(device)
        self._old_network = network(dimStates, numActs).to(device)
        self._old_network.load_state_dict(self._network.state_dict())
        self._optimizer = torch.optim.Adam(self._network.parameters(), alpha, [0.9, 0.999])
        #self.tau = .5
        self.tau = 2 # temperature for gumbel_softmax # more random when tau > 1
        self.count_print = 0
    def load_model(self, path):
        self._network.load_state_dict(torch.load(path))
    
    def save_model(self, path):
        torch.save(self._network.state_dict(), path)

    def __call__(self, s, graph, phaseTrain=True):
        self._old_network.eval()
        s = s.to(device).float()
        out = self._old_network(s, graph)
        probs = F.softmax(out / self.tau, dim=-1)
        if phaseTrain:
            m = Categorical(probs)
            action = m.sample()

            if self.count_print % 100 == 0:
                print("Prob = ", probs)
            self.count_print += 1
        else:
            action = torch.argmax(out)
        return action.data.item()
    
    def update_old_policy(self):
        self._old_network.load_state_dict(self._network.state_dict())

    def update(self, s, graph, a, gammaT, delta, epsilon = 0.2):
        # PPO
        self._network.train()

        # now log_prob
        s = s.to(device).float()
        logits = self._network(s, graph)
        log_prob = torch.log_softmax(logits, dim=-1)[a]

        # old log_prob
        with torch.no_grad():
            old_logits = self._old_network(s, graph)
            old_log_prob = torch.log_softmax(old_logits, dim=-1)[a]

        #ratio
        ratio = torch.exp(log_prob - old_log_prob)

        # PPO clipping
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        loss = -torch.min(ratio * delta, clipped_ratio * delta)

        # gradient
        self._optimizer.zero_grad()
        loss.backward()

        if self.count_print % 100 == 1:
            total_norm = torch.nn.utils.clip_grad_norm_(self._network.parameters(), float('inf'))
            print(f"Policy Network Gradient norm before clipping: {total_norm}")
        #self.count_print += 1

        self._optimizer.step()

    def episode(self):
        pass

class Baseline(object):
    def __init__(self, b):
        self.b = b

    def __call__(self, s):
        return self.b

    def update(self, s, G):
        pass

# value network
"""
class BaselineVApprox(object):
    def __init__(self, dimStates, alpha, network):
        self._dimStates = dimStates
        self._alpha = alpha
        self._network = network(dimStates, 1).to(device)
        self._optimizer = torch.optim.Adam(self._network.parameters(), alpha, [0.9, 0.999])

    def __call__(self, state, graph):
        self._network.eval()
        state = state.to(device).float()
        return self.value(state, graph).data

    def value(self, state, graph):
        state = state.to(device).float()
        out = self._network(state, graph)  # Pass both state and graph
        return out

    def update(self, state, G, graph):
        self._network.train()
        state = state.to(device).float()
        vApprox = self.value(state, graph)  # Pass both state and graph
        loss = (torch.tensor([G], device=device) - vApprox[-1]) ** 2 / 2
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
"""
class BaselineVApprox(object):
    def __init__(self, dimStates, numActs, alpha, network):
        self._dimStates = dimStates
        self._numActs = numActs
        self._alpha = alpha
        self._network = network(dimStates + 1, 1).to(device)
        self._old_network = network(dimStates + 1, 1).to(device)
        self._old_network.load_state_dict(self._network.state_dict())
        self._optimizer = torch.optim.Adam(self._network.parameters(), alpha, [0.9, 0.999])
        self.count_print = 0
    
    def load_model(self, path):
        self._network.load_state_dict(torch.load(path))
    
    def save_model(self, path):
        torch.save(self._network.state_dict(), path)
    
    def __call__(self, state, action, graph):
        self._old_network.eval()
        state = state.to(device).float()
        return self.value(state, action, graph).data
    
    def maxvalue(self, state, graph):
        self._old_network.eval()
        state = state.to(device).float()
        
        # Initialize vmax with the value of the first action
        vmax = self.value(state, 0, graph).data
        
        for i in range(1, self._numActs):
            v_current = self.value(state, i, graph).data
            vmax = torch.max(vmax, v_current)
        
        return vmax
    
    def value(self, state, action, graph):
        state = state.to(device).float()
        action_tensor = torch.tensor([action], dtype=state.dtype, device=state.device)
        action_features = torch.cat((state, action_tensor), dim=-1)  # Concatenate state and action
        out = self._old_network(action_features, graph)  # Pass both combined features and graph
        return out
    
    def newvalue(self, state, action, graph):
        state = state.to(device).float()
        action_tensor = torch.tensor([action], dtype=state.dtype, device=state.device)
        action_features = torch.cat((state, action_tensor), dim=-1)  # Concatenate state and action
        out = self._network(action_features, graph)  # Pass both combined features and graph
        return out
    
    def update_old_policy(self):
        self._old_network.load_state_dict(self._network.state_dict())

    def update(self, state, action, G, graph):
        self._network.train()
        state = state.to(device).float()
        #action = action.to(device).float().unsqueeze(0)
        vApprox = self.newvalue(state, action, graph)  # Estimate Q-value
        loss = (torch.tensor([G], device=device) - vApprox[-1]) ** 2 / 2

        self._optimizer.zero_grad()
        loss.backward()
    
        if self.count_print % 100 == 0:
            total_norm = torch.nn.utils.clip_grad_norm_(self._network.parameters(), float('inf'))
            print(f"Value Network Gradient norm before clipping: {total_norm}")
        self.count_print += 1

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=10.0)  # Adjust max_norm as needed

        self._optimizer.step()


class Trajectory(object):
    """
    @brief The experience of a trajectory
    """
    def __init__(self, states, rewards, actions, value):
        self.states = states
        self.rewards = rewards
        self.actions = actions
        self.value = value
    def __lt__(self, other):
        return self.value < other.value




class Reinforce(object):
    def __init__(self, env, gamma, pi, baseline):
        self._env = env
        self._gamma = gamma
        self._pi = pi
        self._baseline = baseline
        self.memTrajectory = [] # the memorized trajectories. sorted by value
        self.memLength = 4
        self.sumRewards = []
        self.lenSeq = 0
        self.count_update = 0
        self.TRewards = 0

    def genTrajectory(self, phaseTrain=True):
        self._env.reset()
        state = self._env.state()
        term = False
        states, rewards, actions = [], [0], []
        while not term:
            action = self._pi(state[0], state[1], phaseTrain)
            term = self._env.takeAction(action)

            nextState = self._env.state()
            nextReward = self._env.reward()

            states.append(state)
            rewards.append(nextReward)
            actions.append(action)

            state = nextState

            if len(states) > 20:
                term = True

        return Trajectory(states, rewards, actions, self._env.curStatsValue())
    
    def episode(self, gen_traj = 10, phaseTrain=True):
        #trajectories = []
        #for _ in range(gen_traj):
        #    trajectory = self.genTrajectory(phaseTrain=phaseTrain) # Generate a trajectory of episode of states, actions, rewards
        #    trajectories.append(trajectory)
        
        self.lenSeq = 0
        self.updateTrajectory(gen_traj, phaseTrain)
        self._pi.episode()
        return self._env.returns(), self.TRewards
        #return [self._env._curstate]
    
    def updateTrajectory(self, gen_traj, phaseTrain=True):
        #TRewards = []
        #avgnodes = []
        #avgedges = []

        for gg in range(gen_traj):
            self._env.reset()
            state = self._env.state()
            term = 0
            states, rewards, actions = [], [0], []

            self.TRewards = 0
            #states = trajectory.states
            #rewards = trajectory.rewards
            #actions = trajectory.actions

            #bisect.insort(self.memTrajectory, trajectory) # memorize this trajectory
            #self.lenSeq = len(states) # Length of the episode

            #for tIdx in range(self.lenSeq):
            while term < 20:

                action = self._pi(state[0], state[1], phaseTrain)
                term = self._env.takeAction(action)

                nextState = self._env.state()
                nextReward = self._env.reward()

                #G = sum(self._gamma ** (k - tIdx - 1) * rewards[k] for k in range(tIdx + 1, self.lenSeq + 1))
                #G = nextReward + self._gamma * self._baseline.maxvalue(nextState[0], nextState[1])
                if term < 20:
                    next_action = self._pi(nextState[0], nextState[1], phaseTrain)
                    G = nextReward + self._gamma * self._baseline(nextState[0], next_action, nextState[1])
                else:
                    G = nextReward

                self.TRewards += nextReward
                #state = states[tIdx]
                #action = actions[tIdx]

                baseline = self._baseline(state[0], action, state[1])
                delta = G - baseline
                self._baseline.update(state[0], action, G, state[1])

                #self._pi.update(state[0], state[1], action, self._gamma ** tIdx, delta)
                self._pi.update(state[0], state[1], action, 1, delta)

                state = nextState

                self.lenSeq += 1
                self.count_update += 1
                #print(term)

            self._baseline.update_old_policy()
            if gg % 2 == 0:  # origin = 5
                self._pi.update_old_policy()

            #print("-----------------------------------------------")
            #print("Total Reward = ", self.TRewards)    
            #print(state[1].edata['feat'])
            #self.sumRewards.append(sum(rewards))
            #TRewards.append(sum(rewards))
            #avgnodes.append(state[1].num_nodes())
            #avgedges.append(state[1].num_edges())
        
        
        #print('Sum Reward = ', sum(TRewards) / gen_traj)
        #print(rewards)
        #print('Avg Nodes  = ', sum(avgnodes) / gen_traj)
        #print('Avg Edges  = ', sum(avgedges) / gen_traj, "\n")


    def replay(self):
        for idx in range(min(self.memLength, int(len(self.memTrajectory) / 10))):
            if len(self.memTrajectory) / 10 < 1:
                return
            upper = min(len(self.memTrajectory) / 10, 30)
            r1 = random.randint(0, upper)
            self.updateTrajectory(self.memTrajectory[idx])
