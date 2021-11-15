from .agent_common import Agent
from collections import defaultdict
from snake import DIRECTIONS
import numpy as np

class QReplayAgent(Agent):
    def __init__(self,
                    epsilon=0.05,
                    epsilon_decay=0.95,
                    epsilon_min=0,
                    learning_rate=0.001,
                    learning_rate_decay=1,
                    learning_rate_min=0.0001,
                    discount_factor=0.1,
                    max_history=10000,
                    batch_size=32,
                    replay_rate=1):
        self.ep = epsilon
        self.ep_decay = epsilon_decay
        self.ep_min = epsilon_min
        self.lr = learning_rate
        self.lr_decay = learning_rate_decay
        self.lr_min = learning_rate_min
        self.df = discount_factor
        self.max_history = max_history
        self.batch_size = batch_size
        self.replay_rate = replay_rate
        
        self.frames = 0
        self.replay_buffer = []
        self.qtable = defaultdict(lambda: np.random.uniform(size=3))
    
    def episode_setup(self):
        self.prev_action = None
        self.prev_state = None
        self.prev_position = None
    
    def calculate_state_action(self, snake, target = None):
        if snake.has_died: return ("DEAD", 0)
        distances = snake.sense()
        target_distances = snake.sense(target)
        danger = (distances == 1).astype(int)
        target_ahead = (target_distances > 0).astype(int)
        state = f'{"".join(danger.astype(str))}{"".join(target_ahead.astype(str))}'
        action = np.argmax(self.qtable[state])
        return state, action
    
    def calculate_reward(self, snake, target = None):
        if snake.has_died: return -100
        if snake.has_eaten: return 100
        if np.sum(np.abs(snake.head.value - target)) < np.sum(np.abs(self.prev_position - target)):
            return 1
        return -1

    def update_qtable(self, s_next, a_best, r):
        # Update from current experience
        s, a = self.prev_state, self.prev_action
        temporal_diff = self.df * self.qtable[s_next][a_best] - self.qtable[s][a]
        increment = self.lr * (r + temporal_diff)
        self.qtable[s][a] += increment
        
    def replay_experience(self):
        # Replay experience
        samples = np.random.choice(len(self.replay_buffer), size=self.batch_size)
        for i in samples:
            # Bellman eq
            replay = self.replay_buffer[i]
            s, a, r = replay['state'], replay['action'], replay['reward']
            s_next = replay['next_state']
            a_best = np.argmax(self.qtable[s_next])
            temporal_diff = self.df * self.qtable[s_next][a_best] - self.qtable[s][a]
            increment = self.lr * (r + temporal_diff)
            self.qtable[s][a] += increment
    
    def remember_step(self, snake, s, a):
        self.prev_state = s
        self.prev_action = a
        self.prev_position = snake.head.value
    
    def save_experience(self, s, a, r, s_next):
        replay = {'state': s, 'action': a, 'reward': r, 'next_state': s_next}
        self.replay_buffer.insert(0, replay)
        if len(self.replay_buffer) > self.max_history:
            self.replay_buffer.pop(-1)
    
    def play(self, go):
        # Game
        if go.step == 0:
            self.episode_setup()
        snake = go.snake
        target = go.target

        # Get state, action
        # If reward from previous action, update qtable
        s, a = self.calculate_state_action(snake, target)
        if not go.step == 0:
            r = self.calculate_reward(snake, target)
            self.update_qtable(s, a, r)
            self.save_experience(self.prev_state, self.prev_action, r, s)
            if len(self.replay_buffer) >= self.batch_size and self.frames % self.replay_rate == 0:
                self.replay_experience()
                

        # Choose next action
        if np.random.uniform() < self.ep:
            a = np.random.randint(3)
        
        self.remember_step(snake, s, a)
        self.frames += 1
        return a
     
    def has_died(self, snake):
        # super().has_died(snake, True)
        s, a = self.calculate_state_action(snake)
        r = self.calculate_reward(snake)
        self.update_qtable(s, a, r)
        self.save_experience(self.prev_state, self.prev_action, r, s)
        if len(self.replay_buffer) >= self.batch_size and self.frames % self.replay_rate == 0:
                self.replay_experience()
        self.ep = max(self.ep * self.ep_decay, self.ep_min)
        self.lr = max(self.lr * self.lr_decay, self.lr_min)
