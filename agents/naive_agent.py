from .agent_common import Agent
import numpy as np

class NaiveAgent(Agent):
    def play(self, go):
        return np.random.randint(3)