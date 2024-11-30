import numpy as np
import bisect
import random

class MCTSNode:
    def __init__(self, state, graph, parent=None, action=None):
        self.state = state
        self.graph = graph
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0  # Total reward for this node

    def is_fully_expanded(self):
        return len(self.children) == len(self.get_possible_actions())

    def get_possible_actions(self):
        # Placeholder: Implement based on your environment
        return list(range(len(self.state[1])))

    def expand(self, action, next_state, next_graph):
        child = MCTSNode(next_state, next_graph, parent=self, action=action)
        self.children.append(child)
        return child

    def best_child(self, exploration_weight=1.0):
        choices_weights = [
            (child.value / (child.visits + 1e-5)) + exploration_weight * (2 * (child.visits + 1e-5)) ** 0.5
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]


class MCTS:
    def __init__(self, env, policy_network, gamma):
        self.env = env
        self.policy_network = policy_network
        self.gamma = gamma

    def search(self, root_state, root_graph, simulations=10):
        root = MCTSNode(root_state, root_graph)

        for _ in range(simulations):
            node = root
            while not node.is_fully_expanded() and node.children:
                node = node.best_child()

            if not node.is_fully_expanded():
                possible_actions = node.get_possible_actions()
                action = random.choice(possible_actions)
                next_state, reward, done, next_graph = self.env.simulate_action(node.state, action)
                node.expand(action, next_state, next_graph)

            self.backpropagate(node, reward)

        return root.best_child().action

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            reward *= self.gamma
            node = node.parent