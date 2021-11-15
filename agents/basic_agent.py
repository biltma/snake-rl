from .agent_common import Agent
import numpy as np

class BasicAgent(Agent):
    def play(self, go):
        snake = go.snake
        target = go.target

        distances = snake.sense(target)[0:3]
        distances[np.where(distances < 1)] = go.game_size
        return np.argmin(distances)