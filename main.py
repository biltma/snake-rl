from snake import GameObject
import numpy as np
import scipy as sc
import tensorflow as tf
from agents.debug_agent import DebugAgent
from agents.naive_agent import NaiveAgent
from agents.basic_agent import BasicAgent
from agents.input_agent import InputAgent
from agents.q_agent import QAgent
from agents.q_replay_agent import QReplayAgent
from agents.dq_agent import DQAgent
from matplotlib import pyplot as plt
from collections import defaultdict

def plot_scores(scores):
    for k, arr in scores.items():
        plt.plot(arr, label=k)
    plt.legend()
    plt.show()

def test_agents(go, agents, N, BATCHES, BATCH_SIZE):
    scores = defaultdict(lambda: np.zeros(BATCHES))
    for i in range(N):
        for name, Agent in agents.items():
            print(f"Starting {name}")
            agent = Agent()
            scores[name] += np.array(agent.train(lambda: go.play(agent), BATCHES, BATCH_SIZE, print_batch=False)) / N
        if True:
            print(f"Completed Batch {i}")
    plot_scores(scores)
    return scores

def play_agent(agent, go):
    return lambda: go.play(agent)

GAME_SIZE = 8
N = 10
BATCHES = 250
BATCH_SIZE = 1

go = GameObject(GAME_SIZE)

QAgentHigh = lambda: QAgent(discount_factor=0.9, learning_rate=0.0005)
QAgentLow = lambda: QAgent(discount_factor=0.1, learning_rate=0.0005)

agents = {"QAgentLow": QAgentLow, "QAgentHigh" : QAgentHigh}
test_agents(go, agents, N, BATCHES, BATCH_SIZE)


