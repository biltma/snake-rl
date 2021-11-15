from .agent_common import Agent
from collections import defaultdict
from snake import DIRECTIONS
import tensorflow as tf
from tensorflow.keras import layers, activations, losses, optimizers
import numpy as np

def create_model():
    model = tf.keras.Sequential([
        layers.Dense(128, input_shape=(8,)),
        layers.Activation(activations.relu),
        layers.Dense(32),
        layers.Activation(activations.relu),
        layers.Dense(3),
        layers.Activation(activations.linear)
    ])
    return model

class DQAgent(Agent):
    def __init__(self,
                    epsilon=0.05,
                    epsilon_decay=1,
                    epsilon_min=0,
                    learning_rate=0.0001,
                    learning_rate_decay=1,
                    learning_rate_min=0.0001,
                    discount_factor=0.1,
                    max_history=10000,
                    batch_size=1000,
                    replay_rate=4,
                    copy_rate=2000):
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
        self.copy_rate = copy_rate
        
        self.frames = 0
        self.replay_buffer = []
        self.init_models()
    
    def init_models(self):
        self.loss = losses.Huber()
        self.optimizer = optimizers.Adam(learning_rate=self.lr)
        self.model = create_model()
        self.target_model = create_model()
    
    def episode_setup(self):
        self.prev_action = None
        self.prev_state = None
        self.prev_position = None
    
    # def calculate_state_action(self, snake, target = None):
    #     if snake.has_died: return (np.ones(snake.game_size ** 2 + 4), 0)
    #     game_map = np.zeros((snake.game_size, snake.game_size))
    #     coords = snake.get_all_positions()
    #     game_map[coords[0][1], coords[0][0]] = 1
    #     for c in coords[1:]:
    #         game_map[c[1], c[0]] = 0.5
    #     game_map[target[1], target[0]] = -1
    #     directions = [int(all(snake.direction == D)) for D in DIRECTIONS.ARR]
    #     state = np.concatenate((game_map.flatten(), directions))
    #     action = self.predict_action(state)
    #     return state, action

    def calculate_state_action(self, snake, target = None):
        if snake.has_died: return (np.zeros(8), 0)
        distances = snake.sense()
        target_distances = snake.sense(target)
        danger = (distances == 1).astype(int)
        target_ahead = (target_distances > 0).astype(int)
        state = np.concatenate((distances, target_distances))
        action = self.predict_action(state)
        return state, action
    
    def calculate_reward(self, snake, target = None):
        if snake.has_died: return -100
        if snake.has_eaten: return 100
        if np.sum(np.abs(snake.head.value - target)) < np.sum(np.abs(self.prev_position - target)):
            return 1
        return -1
    
    def predict_action(self, state):
        return np.argmax(self.model(np.expand_dims(state, axis=0)))

    def update_model(self, s, a, r):
        with tf.GradientTape() as tape:
            q_predicted = self.model(np.expand_dims(self.prev_state, axis=0), training=True)[:, self.prev_action]
            q_target = r + self.df * self.target_model(np.expand_dims(s, axis=0))[:, a]
            loss = self.loss(q_predicted, q_target)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    
    def replay_experience(self):
        if self.batch_size == 0: return
        # Replay experience
        samples = np.random.choice(len(self.replay_buffer), size=self.batch_size)
        states = np.array([self.replay_buffer[i]['state'] for i in samples])
        future_states = np.array([self.replay_buffer[i]['next_state'] for i in samples])
        rewards = np.array([self.replay_buffer[i]['reward'] for i in samples])
        future_rewards = np.max(self.target_model(future_states), axis=1)

        mask = np.zeros((self.batch_size, 3))
        for n, i in enumerate(samples):
            mask[n, self.replay_buffer[i]['action']] = 1
        mask = mask.astype(bool)
        with tf.GradientTape() as tape:
            q_predicted = self.model(states, training=True)
            q_predicted = q_predicted[mask]
            q_target = rewards + self.df * future_rewards
            loss = self.loss(q_predicted, q_target)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    
    def copy_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
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
            self.update_model(s, a, r)
            self.save_experience(self.prev_state, self.prev_action, r, s)
            if len(self.replay_buffer) >= self.batch_size and self.frames % self.replay_rate == 0:
                self.replay_experience()

        # Copy model if necessary
        if self.frames % self.copy_rate == 0: self.copy_model()

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
        self.save_experience(self.prev_state, self.prev_action, r, s)
        self.update_model(s, a, r)
        self.ep = max(self.ep * self.ep_decay, self.ep_min)
        self.lr = max(self.lr * self.lr_decay, self.lr_min)
