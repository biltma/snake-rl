import numpy as np

class Agent:
    def play(self, snake, target):
        raise NotImplementedError

    def has_died(self, snake, print_result=False):
        if print_result:
            print(f"Snake has died! Score: {len(snake)}")
        
    def train(self, play, batches=10, batch_size=25, print_batch=True):
        scores = []
        for i in range(batches):
            batch = []
            for j in range(batch_size):
                batch.append(play())
            scores.append(np.mean(batch))
            if print_batch:
                print(f"Batch {i}: {scores[-1]}")
        return scores
    
    def evaluate(self, play, iters=50):
        return self.train(play, 1, iters)
