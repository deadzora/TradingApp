
import numpy as np
class ContextualBandit:
    def __init__(self, model_dir, epsilon_start=0.15, epsilon_floor=0.03):
        self.epsilon = epsilon_start; self.epsilon_floor = epsilon_floor
    def select(self, x_vec):
        if np.random.rand() < self.epsilon: return np.random.choice(["HOLD","MR","MOM"]), self.epsilon/3
        return "MOM", 1-self.epsilon
    def decay_epsilon(self, decay=0.03):
        self.epsilon = max(self.epsilon - decay, self.epsilon_floor)
