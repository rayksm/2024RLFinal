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
import torch.nn.init as init

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

        self.fc1 = nn.Linear(numFeats, 64).to(device)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(64, 64).to(device)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 64).to(device)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(64, 64).to(device)
        self.act4 = nn.ReLU()

        self.fc5 = nn.Linear(64, 32).to(device)
        self.act5 = nn.ReLU()

        self.fc6 = nn.Linear(32, outChs).to(device)

        self.gcn = GCN(7, 64, 16)
        #self.gcn = GCN(6, 64, 16)

        self.fcst = nn.Linear(64 + 16, 64).to(device)
        self.actst = nn.ReLU()

        # Custom initialization
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize each layer
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    init.constant_(module.bias, 0)

            
            elif isinstance(module, GCN):
                # If GCN has parameters, initialize them here
                for param in module.parameters():
                    if param.dim() > 1:
                        init.xavier_uniform_(param)
            

    def forward(self, x, graph):
        x = x.to(device)
        graph_state = self.gcn(graph)

        x = self.fc1(x)
        x = self.act1(x)
        x_res = x

        x = self.fcst(torch.cat((x, graph_state), dim = 0))
        x = self.actst(x)
        #x = self.dropout1(x)
        #x = self.bn1(x)

        x = self.fc2(x)
        x = self.act2(x)
        x = x + x_res

        x = self.fc3(x)
        x = self.act3(x)
        x_res = x

        x = self.fc4(x)
        x = self.act4(x)
        x = x + x_res

        x = self.fc5(x)
        x = self.act5(x)
        
        x = self.fc6(x)

        #print("graph_state:", graph_state)
        #print("After x = ", x.tolist())
        #print()
        return x

"""
class FcModelGraph(nn.Module):
    def __init__(self, numFeats, outChs):
        super(FcModelGraph, self).__init__()
        self._numFeats = numFeats
        self._outChs = outChs
        
        self.fc1 = nn.Linear(numFeats, 64).to(device)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(64, 64).to(device)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 64).to(device)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(64, 64).to(device)
        self.act4 = nn.ReLU()

        self.fc5 = nn.Linear(64, 32).to(device)
        self.act5 = nn.ReLU()

        self.fc6 = nn.Linear(32, outChs).to(device)

        self.gcn = GCN(7, 64, 16)
        #self.gcn = GCN(6, 64, 16)

        self.fcst = nn.Linear(64 + 16, 64).to(device)
        self.actst = nn.ReLU()

        # Custom initialization
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize each layer
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    init.constant_(module.bias, 0)

            
            elif isinstance(module, GCN):
                # If GCN has parameters, initialize them here
                for param in module.parameters():
                    if param.dim() > 1:
                        init.xavier_uniform_(param)
            

    def forward(self, x, graph):
        x = x.to(device)
        graph_state = self.gcn(graph)

        x = self.fc1(x)
        x = self.act1(x)
        x_res = x

        x = self.fcst(torch.cat((x, graph_state), dim = 0))
        x = self.actst(x)
        #x = self.dropout1(x)
        #x = self.bn1(x)

        x = self.fc2(x)
        x = self.act2(x)
        x = x + x_res

        x = self.fc3(x)
        x = self.act3(x)
        x_res = x

        x = self.fc4(x)
        x = self.act4(x)
        x = x + x_res

        x = self.fc5(x)
        x = self.act5(x)
        
        x = self.fc6(x)

        #print("graph_state:", graph_state)
        #print("After x = ", x.tolist())
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
        self.tau = 1 # temperature for gumbel_softmax # more random when tau > 1
        self.count_print = 0

        self.explore = 0
        self.exp_prob = torch.ones(numActs).to(device) * (self.explore / numActs)

    def load_model(self, path):
        self._network.load_state_dict(torch.load(path))
    
    def save_model(self, path):
        torch.save(self._network.state_dict(), path)

    def __call__(self, s, graph, phaseTrain=True, ifprint = False):
        self._old_network.eval()
        s = s.to(device).float()
        out = self._old_network(s, graph)
        probs = F.softmax(out / self.tau, dim=-1) * (1 - self.explore) + self.exp_prob
        #print(probs)
        if phaseTrain:
            m = Categorical(probs)
            action = m.sample()
            #if ifprint: print(f"{action.data.item()} ({out[action.data.item()].data.item():>6.3f})", end=" > ")
            if ifprint: print(f"{action.data.item()}", end=" > ")

            #if self.count_print % 25 == 0:
            #    print("Prob = ", probs)
            self.count_print += 1
        else:
            action = torch.argmax(out)
            if ifprint: print(f"{action.data.item()}", end=" > ")
        return action.data.item()
    
    def update_old_policy(self):
        self._old_network.load_state_dict(self._network.state_dict())

    def update(self, s, graph, a, gammaT, delta, vloss, epsilon = 0.1, beta = 0.1, vbeta = 0.01):
        # PPO
        self._network.train()

        # now log_prob
        s = s.to(device).float()
        logits = self._network(s, graph)
        log_prob = torch.log_softmax(logits / self.tau, dim=-1)[a]

        # old log_prob
        with torch.no_grad():
            old_logits = self._old_network(s, graph)
            old_log_prob = torch.log_softmax(old_logits / self.tau, dim=-1)[a]

        #ratio
        ratio = torch.exp(log_prob - old_log_prob)

        # entropy
        entropy = -torch.sum(F.softmax(logits / self.tau, dim=-1) * log_prob, dim=-1).mean()

        # PPO clipping
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        
        loss = -torch.min(ratio * delta, clipped_ratio * delta) - beta * entropy + vbeta * vloss
        #print("(Loss = ", loss.data.item(), end = ") ")
        #print(f"(Loss = {loss.data.item():.3f}", end=") || ")

        # gradient
        self._optimizer.zero_grad()
        loss.backward(retain_graph=True)

        #if self.count_print % 25 == 1:
        #total_norm = torch.nn.utils.clip_grad_norm_(self._network.parameters(), float('inf'))
        #print(f"Policy Network Gradient norm before clipping: {total_norm}")
        #self.count_print += 1
        #torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=100.0)
        #torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=10.0)

        self._optimizer.step()

    def episode(self):
        pass
"""

class FcModelGraph(nn.Module):
    def __init__(self, numFeats, outChs):
        super(FcModelGraph, self).__init__()
        self._numFeats = numFeats
        self._outChs = outChs
        
        self.fc1 = nn.Linear(numFeats, 64).to(device)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(64, 64).to(device)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 64).to(device)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(64, 64).to(device)
        self.act4 = nn.ReLU()

        self.fc5 = nn.Linear(64, 32).to(device)
        self.act5 = nn.ReLU()

        self.fc6 = nn.Linear(32, outChs).to(device)

        self.gcn = GCN(7, 64, 16)

        self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True).to(device)
        self.fcst = nn.Linear(64 + 16 + 16, 64).to(device)
        self.actst = nn.ReLU()

        # Custom initialization
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize each layer
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    init.constant_(module.bias, 0)

            
            elif isinstance(module, GCN):
                # If GCN has parameters, initialize them here
                for param in module.parameters():
                    if param.dim() > 1:
                        init.xavier_uniform_(param)
            

    def forward(self, x, graph, lastgraph):
        x = x.to(device)
        graph_state = self.gcn(graph)
        graph_state = graph_state.unsqueeze(0).unsqueeze(0)
        graph_state, lastgraph= self.lstm(graph_state, lastgraph)  
        graph_state = graph_state.squeeze(0).squeeze(0)  
        #lastgraph_state = self.gcn(lastgraph)

        x = self.fc1(x)
        x = self.act1(x)
        x_res = x

        x = self.fcst(torch.cat((x, graph_state), dim = 0))
        x = self.actst(x)
        #x = self.dropout1(x)
        #x = self.bn1(x)

        x = self.fc2(x)
        x = self.act2(x)
        x = x + x_res

        x = self.fc3(x)
        x = self.act3(x)
        x_res = x

        x = self.fc4(x)
        x = self.act4(x)
        x = x + x_res

        x = self.fc5(x)
        x = self.act5(x)
        
        x = self.fc6(x)

        #print("graph_state:", graph_state)
        #print("After x = ", x.tolist())
        #print()
        return x, lastgraph

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
        self.tau = 1 # temperature for gumbel_softmax # more random when tau > 1
        self.count_print = 0

        self.explore = 0
        self.exp_prob = torch.ones(numActs).to(device) * (self.explore / numActs)

    def load_model(self, path):
        self._network.load_state_dict(torch.load(path))
    
    def save_model(self, path):
        torch.save(self._network.state_dict(), path)

    def __call__(self, s, graph, lastgraph, phaseTrain=True, ifprint = False):
        self._old_network.eval()
        s = s.to(device).float()
        out, lastgraph = self._old_network(s, graph, lastgraph)
        probs = F.softmax(out / self.tau, dim=-1) * (1 - self.explore) + self.exp_prob
        #print(probs)
        if phaseTrain:
            m = Categorical(probs)
            action = m.sample()
            #if ifprint: print(f"{action.data.item()} ({out[action.data.item()].data.item():>6.3f})", end=" > ")
            if ifprint: print(f"{action.data.item()}", end=" > ")

            #if self.count_print % 25 == 0:
            #    print("Prob = ", probs)
            self.count_print += 1
        else:
            action = torch.argmax(out)
            if ifprint: print(f"{action.data.item()}", end=" > ")
        return action.data.item(), lastgraph
    
    def update_old_policy(self):
        self._old_network.load_state_dict(self._network.state_dict())

    def update(self, s, graph, lastgraph, oldlastgraph, a, gammaT, delta, vloss, epsilon=0.1, beta=0.1, vbeta=0.01):
        # PPO
        self._network.train()

        # Move inputs to device
        s = s.to(device).float()

        # Forward pass through current network
        logits, alastgraph = self._network(s, graph, lastgraph)  # Correct unpacking
        log_prob = torch.log_softmax(logits / self.tau, dim=-1)[a]

        # Forward pass through old network (no grad)
        with torch.no_grad():
            old_logits, oldlastgraph = self._old_network(s, graph, oldlastgraph)  # Correct unpacking
            old_log_prob = torch.log_softmax(old_logits / self.tau, dim=-1)[a]

        # Compute ratio
        ratio = torch.exp(log_prob - old_log_prob)

        # Compute entropy
        entropy = -torch.sum(F.softmax(logits / self.tau, dim=-1) * log_prob, dim=-1).mean()

        # PPO clipping
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

        # Compute loss
        loss = -torch.min(ratio * delta, clipped_ratio * delta) - beta * entropy + vbeta * vloss

        # Backpropagation
        self._optimizer.zero_grad()
        loss.backward(retain_graph=True)  # Set to False unless multiple backward passes are needed
        #loss.backward(retain_graph=False)  # Set to False unless multiple backward passes are needed

        # Apply gradient clipping
        #gradient_cut = 10000.0
        #total_norm = torch.nn.utils.clip_grad_norm_(self._network.parameters(), float('inf'))
        #if total_norm > gradient_cut: 
        #    print("Cut!", end=" ")
        #print(f"Value Network Gradient norm before clipping: {total_norm}", end=" ")

        #torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm = gradient_cut)  # Adjust max_norm as needed

        self._optimizer.step()

        # Detach hidden states to prevent gradient tracking
        #lastgraph = (lastgraph[0].detach(), lastgraph[1].detach())
        #oldlastgraph = (oldlastgraph[0].detach(), oldlastgraph[1].detach())

        # Return both hidden states
        #return lastgraph

    def init_hidden(self, batch_size=1):
        # This function initializes hidden state, not model parameters
        h_0 = torch.zeros(1, batch_size, 32, device=device)
        c_0 = torch.zeros(1, batch_size, 32, device=device)
        return (h_0, c_0)
    
    def episode(self):
        pass


class Baseline(object):
    def __init__(self, b):
        self.b = b

    def __call__(self, s):
        return self.b

    def update(self, s, G):
        pass

class RewardNormalizer:
    def __init__(self, alpha=0.01, epsilon=1e-8):
        # Initialize mean and variance as zero arrays of the given shape
        self.mean = np.zeros(20, dtype=np.float32)
        self.var = np.zeros(20, dtype=np.float32)
        self.alpha = alpha
        self.epsilon = epsilon
        self.threshold = 0.2

        self.newmean = np.zeros(20, dtype=np.float32)
        self.newvar = np.zeros(20, dtype=np.float32)
        self.k = 20 * np.ones(20, dtype=np.int32)

    def normalize(self, reward, step):

        # Update running mean
        self.mean[step] = (1 - self.alpha) * self.mean[step] + self.alpha * reward
        
        # Update running variance
        self.var[step] = (1 - self.alpha) * self.var[step] + self.alpha * (reward - self.mean[step])**2

        # Compute standard deviation
        std = np.sqrt(self.var[step]) + self.epsilon
        
        # Normalize the rewards
        normalized_reward = (reward - self.mean[step]) / std

        if self.k[step] > 0:
            self.k[step] -= 1
            # Update window running mean
            self.newmean[step] = (1 - self.alpha) * self.newmean[step] + self.alpha * reward
        
            # Update window running variance
            self.newvar[step] = (1 - self.alpha) * self.newvar[step] + self.alpha * (reward - self.newmean[step])**2
        else:
            self.newmean[step] = 0
            self.newvar[step] = 0
            self.k[step] = 20
        self.check_reset(step)

        return normalized_reward
    
    def check_reset(self, step):
        if self.k[step] < 10 and (self.newmean[step] - (1 + self.epsilon) * self.mean[step] > 0 or self.newmean[step] - (1 - self.epsilon) * self.mean[step] < 0):
            if self.var[step] > self.newvar[step]:
                
                self.mean[step] = self.newmean[step]
                self.var[step] = self.newvar[step]

                self.newmean[step] = 0
                self.newvar[step] = 0
                self.k[step] = 20

    
# value network
class BaselineVApprox(object):
    def __init__(self, dimStates, numActs, alpha, network):
        self._dimStates = dimStates
        self._numActs = numActs
        self._alpha = alpha
        self._network = network(dimStates, 1).to(device)
        self._old_network = network(dimStates, 1).to(device)
        self._old_network.load_state_dict(self._network.state_dict())
        self._optimizer = torch.optim.Adam(self._network.parameters(), alpha, [0.9, 0.999])
        self.count_print = 0
        self.vloss = 0
    
    def load_model(self, path):
        self._network.load_state_dict(torch.load(path))
    
    def save_model(self, path):
        torch.save(self._network.state_dict(), path)
    
    def __call__(self, state, graph):
        self._old_network.eval()
        state = state.to(device).float()
        #return self.value(state, action, graph).data
        return self.value(state, graph).data
    
    """
    def maxvalue(self, state, graph):
        self._old_network.eval()
        state = state.to(device).float()
        
        # Initialize vmax with the value of the first action
        vmax = self.value(state, 0, graph).data
        
        for i in range(1, self._numActs):
            v_current = self.value(state, i, graph).data
            vmax = torch.max(vmax, v_current)
        
        return vmax
    """

    def value(self, state, graph):
        self._old_network.eval()
        state = state.to(device).float()
        #action_tensor = torch.tensor([action], dtype=state.dtype, device=state.device)
        #action_features = torch.cat((state, action_tensor), dim=-1)  # Concatenate state and action
        #out = self._old_network(action_features, graph)  # Pass both combined features and graph
        out = self._old_network(state, graph)
        return out
    
    def newvalue(self, state, graph):
        self._network.eval()
        state = state.to(device).float()
        #action_tensor = torch.tensor([action], dtype=state.dtype, device=state.device)
        #action_features = torch.cat((state, action_tensor), dim=-1)  # Concatenate state and action
        #out = self._network(action_features, graph)  # Pass both combined features and graph
        out = self._network(state, graph)
        return out
    
    def update_old_policy(self):
        self._old_network.load_state_dict(self._network.state_dict())

    def update(self, state, action, G, graph):
        self._network.train()
        state = state.to(device).float()
        #action = action.to(device).float().unsqueeze(0)
        vApprox = self.newvalue(state, graph)  # Estimate Q-value
        loss = (torch.tensor([G], device=device) - vApprox[-1]) ** 2 / 2

        self.vloss = loss
        self._optimizer.zero_grad()
        loss.backward()
    
        #if self.count_print % 25 == 0:
        #    total_norm = torch.nn.utils.clip_grad_norm_(self._network.parameters(), float('inf'))
        #    print(f"Value Network Gradient norm before clipping: {total_norm}")
        self.count_print += 1

        # Apply gradient clipping
        gradient_cut = 1000.0
        #total_norm = torch.nn.utils.clip_grad_norm_(self._network.parameters(), float('inf'))
        #if total_norm > gradient_cut: 
        #    print("Cut!", end=" ")
        #print(f"Value Network Gradient norm before clipping: {total_norm}", end=" ")

        torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm = gradient_cut)  # Adjust max_norm as needed

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
        self.TNumAnd = 0
        self.rewardnormalizer = RewardNormalizer()

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
        return self.TNumAnd / gen_traj, self.TRewards / gen_traj
        #return [self._env._curstate]
    
    def updateTrajectory(self, gen_traj, phaseTrain=True):
        #TRewards = []
        #avgnodes = []
        #avgedges = []
        self.TRewards = 0
        self.TNumAnd = 0

        steplen = self._env.total_action_len
        update_time = 0
        for gg in range(gen_traj):
            self._env.reset()
            state = self._env.state()
            term = 0
            lastgraph = self._pi.init_hidden()
            states, advantages, Gs, actions, lastgraphs = [], [], [], [], []

            thiseporeward = 0
            #states = trajectory.states
            #rewards = trajectory.rewards
            #actions = trajectory.actions

            #bisect.insort(self.memTrajectory, trajectory) # memorize this trajectory
            #self.lenSeq = len(states) # Length of the episode

            #for tIdx in range(self.lenSeq):
            while term < steplen:

                action, lastgraph = self._pi(state[0], state[1], lastgraph, phaseTrain, 1)
                term = self._env.takeAction(action)

                nextState = self._env.state()
                nextReward = self.rewardnormalizer.normalize(self._env.reward(), term - 1)

                #G = sum(self._gamma ** (k - tIdx - 1) * rewards[k] for k in range(tIdx + 1, self.lenSeq + 1))
                #G = nextReward + self._gamma * self._baseline.maxvalue(nextState[0], nextState[1])
                if term < steplen:
                    #next_action = self._pi(nextState[0], nextState[1], phaseTrain, 0)
                    #G = nextReward + self._gamma * self._baseline(nextState[0], next_action, nextState[1])
                    g = nextReward + self._gamma * self._baseline.value(nextState[0], nextState[1])
                    #G = nextReward + self._gamma * self._baseline.maxvalue(nextState[0], nextState[1])
                else:
                    g = nextReward

                baseline = self._baseline(state[0], state[1])
                delta = g - baseline
                #delta = baseline
                #print("(The delta = ", delta.data.item(), ", baseline = ", baseline.data.item(), end=") ")
                #print(f"(Delta = {delta.data.item():.3f}, G = {G.item():.3f}, Baseline = {baseline.item():.3f}", end=") | ")
                #print(f"(The delta = {delta.data.item():.3f}, G = {nextReward:.3f}", end=") | ")
                #print(f"(The delta = {delta.data.item():.3f}, G + baseline = {G.item():.3f}", end=") | ")
                
                #if phaseTrain:
                #    self._baseline.update(state[0], action, g, state[1])
                #    self._pi.update(state[0], state[1], action, 1, delta)
                
                states.append(state)
                actions.append(action)
                advantages.append(delta)
                Gs.append(g)
                lastgraphs.append(lastgraph)
                #vlosses.append(self._baseline.vloss)

                state = nextState

                self.lenSeq += 1
                self.count_update += 1

                self.TRewards += nextReward
                thiseporeward += nextReward
                if term == steplen: self.TNumAnd += self._env.returns()[0]

                #print(term)
            #print(thiseporeward, self._env.statValue(state))
            print(f"|| TR = {thiseporeward:>7.3f}, RA = {int(self._env.curStatsValue()):4d}", end=" || \n")

            if phaseTrain:

                lastgraph = self._pi.init_hidden()
                for i in range(steplen):
                    state = states[i]
                    action = actions[i]
                    g = Gs[i]
                    oldlastgraph = lastgraphs[i]
                    lastgraph = (oldlastgraph[0].clone().detach(), oldlastgraph[1].clone().detach())
                    culmu_advantage = sum(advantages[k] for k in range(i, steplen))
                    
                    self._baseline.update(state[0], action, g, state[1])
                    #lastgraph = self._pi.update(state[0], state[1], lastgraph, oldlastgraph, action, 1, culmu_advantage, self._baseline.vloss.item())
                    self._pi.update(state[0], state[1], lastgraph, oldlastgraph, action, 1, culmu_advantage, self._baseline.vloss.item())


                update_time += 1
                if update_time % 2 == 0:  # origin = 5
                    self._baseline.update_old_policy()
                if update_time % 2 == 0:  # origin = 5
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
