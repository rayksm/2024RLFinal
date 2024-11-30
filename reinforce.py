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
        #self.conv3 = GraphConv(hidden_size, hidden_size).to(device)
        self.conv4 = GraphConv(hidden_size, out_len).to(device)

        #self.pool = AttentionPooling(hidden_size)  # AttentionPooling layer
        #self.fc = nn.Linear(hidden_size, out_len)

    def forward(self, g):
        g = g.to(device)
        g.ndata['feat'] = g.ndata['feat'].to(device)
        g = dgl.add_self_loop(g)

        h = self.conv1(g, g.ndata['feat'])
        h = torch.relu(h)

        #h1 = h

        h = self.conv2(g, h)
        h = torch.relu(h)
        #h = self.conv3(g, h)
        #h = torch.relu(h)
        #h = h + h1
        
        h = self.conv4(g, h)
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

        self.fc1 = nn.Linear(numFeats, 128 - 64).to(device)
        self.gcn = GCN(6, 128, 64)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 64).to(device)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 32).to(device)
        self.act3 = nn.ReLU()


        self.fc4 = nn.Linear(32, outChs).to(device)

    def forward(self, x, graph):
        x = x.to(device)
        graph_state = self.gcn(graph)

        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(torch.cat((x, graph_state), 0))
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        return x


class FcModelGraph(nn.Module):
    def __init__(self, numFeats, outChs):
        super(FcModelGraph, self).__init__()
        self._numFeats = numFeats
        self._outChs = outChs
        
        self.fc1 = nn.Linear(numFeats, 128 - 64).to(device)
        self.gcn = GCN(6, 128, 64)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 64).to(device)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 32).to(device)
        self.act3 = nn.ReLU()


        self.fc4 = nn.Linear(32, outChs).to(device)

    def forward(self, x, graph):
        x = x.to(device)
        graph_state = self.gcn(graph)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(torch.cat((x, graph_state), 0))
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        return x

# Policy Network
class PiApprox(object):
    def __init__(self, dimStates, numActs, alpha, network):
        self._dimStates = dimStates
        self._numActs = numActs
        self._alpha = alpha
        self._network = network(dimStates, numActs).to(device)
        self._optimizer = torch.optim.Adam(self._network.parameters(), alpha, [0.9, 0.999])
        self.tau = 2
        #self.tau = .5 # temperature for gumbel_softmax # more random when tau > 1

    def __call__(self, s, graph, phaseTrain=True):
        self._network.eval()
        s = s.to(device).float()
        out = self._network(s, graph)
        probs = F.softmax(out / self.tau, dim=-1)

        if self.tau > 0.5:
            self.tau -= (1.5) / 200

        if phaseTrain:
            m = Categorical(probs)
            action = m.sample()
            #print(probs)
        else:
            action = torch.argmax(out)
        return action.data.item()

    def update(self, s, graph, a, gammaT, delta):
        self._network.train()
        s = s.to(device).float()
        prob = self._network(s, graph)
        logProb = torch.log_softmax(prob, dim=-1)
        loss = -gammaT * delta * logProb
        #print("Policy Loss = ", loss[a].item())
        self._optimizer.zero_grad()
        loss[a].backward()
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
        #print("Value Loss = ", loss.item())
        self._optimizer.zero_grad()
        loss.backward()
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

            if len(states) >= 20:
                term = True

        return Trajectory(states, rewards, actions, self._env.curStatsValue())
    
    def episode(self, gen_traj = 1, phaseTrain=True):
        trajectories = []
        for _ in range(gen_traj):
            trajectory = self.genTrajectory(phaseTrain=phaseTrain) # Generate a trajectory of episode of states, actions, rewards
            trajectories.append(trajectory)

        self.updateTrajectory(trajectories, gen_traj, phaseTrain)
        self._pi.episode()
        return self._env.returns()
    
    def updateTrajectory(self, trajectories, gen_traj, phaseTrain=True):
        TRewards = []
        #avgnodes = []
        #avgedges = []

        for trajectory in trajectories:
            states = trajectory.states
            rewards = trajectory.rewards
            actions = trajectory.actions
            
            bisect.insort(self.memTrajectory, trajectory) # memorize this trajectory
            self.lenSeq = len(states) # Length of the episode

            for tIdx in range(self.lenSeq):
                G = sum(self._gamma ** (k - tIdx - 1) * rewards[k] for k in range(tIdx + 1, self.lenSeq + 1))
                #if tIdx < self.lenSeq - 1:
                #    G = rewards[tIdx + 1] + self._gamma * self._baseline(states[tIdx + 1][0], states[tIdx + 1][1])
                #else:
                #    G = rewards[tIdx + 1]

                state = states[tIdx]
                action = actions[tIdx]

                baseline = self._baseline(state[0], state[1])
                delta = G - baseline
                self._baseline.update(state[0], G, state[1])

                self._pi.update(state[0], state[1], action, self._gamma ** tIdx, delta)
            #print(state[1])
            self.sumRewards.append(sum(rewards))
            TRewards.append(sum(rewards))
            #avgnodes.append(state[1].num_nodes())
            #avgedges.append(state[1].num_edges())
        
        #print("-----------------------------------------------")
        print('Sum Reward = ', sum(TRewards) / gen_traj)
        #print('Avg Nodes  = ', sum(avgnodes) / gen_traj)
        #print('Avg Edges  = ', sum(avgedges) / gen_traj, "\n")

    def replay(self):
        for idx in range(min(self.memLength, int(len(self.memTrajectory) / 10))):
            if len(self.memTrajectory) / 10 < 1:
                return
            upper = min(len(self.memTrajectory) / 10, 30)
            r1 = random.randint(0, upper)
            self.updateTrajectory(self.memTrajectory[idx])
