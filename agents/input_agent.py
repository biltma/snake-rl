from .agent_common import Agent

class InputAgent(Agent):
    def play(self, go):
        i = None
        while i not in ['w', 'a', 'd']:
            i = input("Left/Forward/Right (a/w/d):\n")
        if i == 'a': return 0
        if i == 'w': return 1
        if i == 'd': return 2