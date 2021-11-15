from .agent_common import Agent
from .input_agent import InputAgent

class DebugAgent(Agent):
    def __init__(self, agent):
        self.input_agent = InputAgent()
        self.agent = agent
    
    def play(self, go):
        agent_action = self.agent.play(go)
        print("Agent Chose: " + str(agent_action))

        a = self.input_agent.play(go)
        self.agent.prev_action = a
        return a