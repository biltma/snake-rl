import numpy as np

class Agent:
    def play(self, snake, target):
        raise NotImplementedError

    def has_died(self, snake, print_result=False):
        if print_result:
            print(f"Snake has died! Score: {len(snake)}")
        
    def train(self, play, batches=10, batch_size=25, print_batch=True):
        scores = []
        best_run = -np.inf
        worst_run = np.inf
        for i in range(batches):
            batch = []
            for j in range(batch_size):
                n = play()
                best_run = max(n, best_run)
                worst_run = min(n, worst_run)
                batch.append(n)
            scores.append(np.mean(batch))
            if print_batch:
                print(f"Batch {i}: {scores[-1]}")
        return {
            'scores': scores,
            'best': best_run,
            'worst': worst_run
        }
    
    def evaluate(self, play, iters=50):
        return self.train(play, 1, iters)
