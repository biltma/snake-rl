import numpy as np
from matplotlib import pyplot as plt, colors

C0 = np.array([0, 0, 0])
C1 = np.array([122/255, 66/255, 23/255])
C2 = np.array([20/255, 200/255, 223/255])

class DLLNode:
    def __init__(self, value, next = None, prev = None):
        self.value = value
        self.next = next
        self.prev = prev
    
    def __str__(self) -> str:
        return f'{self.value}'

class DIRECTIONS:
    N = np.array([0, 1])
    E = np.array([1, 0])
    S = np.array([0, -1])
    W = np.array([-1, 0])
    ARR = [N, E, S, W]

    CCW = np.array([[0, -1], [1, 0]])
    CW = np.array([[0, 1], [-1, 0]])

    def right(direction):
        return np.dot(DIRECTIONS.CW, direction)
    
    def left(direction):
        return np.dot(DIRECTIONS.CCW, direction)
    
    def behind(direction):
        return -1 * direction

class GameObject:
    def __init__(self, game_size):
        self.game_size = game_size

    def plot_snake(self):
        coords = self.snake.get_all_positions()
        img = np.zeros((self.game_size, self.game_size))
        img[coords[0][1], coords[0][0]] = 0.6
        for c in coords[1:]:
            img[c[1], c[0]] = 0.35
        img[self.target[1], self.target[0]] = 1
        plt.imshow(img, cmap='Greys', origin='lower')
        plt.pause(0.1)
    
    def spawn_target(self):
        while self.target is None:
            potential_target = np.random.randint((self.game_size, self.game_size))
            if not self.snake.crosses(potential_target):
                self.target = potential_target

    def play(self, agent, visualize=False):
        self.step = 0
        self.snake = Snake(self.game_size)
        self.target = None
        while not self.snake.has_died:
            # Create target
            if self.target is None:
                self.spawn_target()
                    
            if visualize:
                plt.close()
                self.plot_snake()

            # Get action
            action = agent.play(self)
            D = self.snake.direction
            if action == 0: D = DIRECTIONS.left(D)
            if action == 2: D = DIRECTIONS.right(D)
            self.snake.direction = D

            self.snake.extend(1)
            if any(self.snake.sense() == 0):
                self.snake.shorten(1)
                self.snake.has_died = True
            elif self.snake.crosses(self.target):
                self.snake.has_eaten = True
                self.target = None
            else:
                self.snake.has_eaten = False
                self.snake.shorten(1)
                if self.snake.crosses():
                    self.snake.has_died = True
            self.step += 1

        agent.has_died(self.snake)
        return len(self.snake)


class Snake:
    def __init__(self, game_size, start_position = None, start_direction = None):
        self.game_size = game_size
        self.start_position = start_position or np.array([0, 0])
        self.start_direction = start_direction or DIRECTIONS.N
        self.reset() # Reset Snake State
    
    def reset(self):
        self.head = DLLNode(self.start_position)
        self.tail = self.head
        self.direction = self.start_direction
        self.has_died = False
        self.has_eaten = False
    
    def sense(self, target=None):
        DF = self.direction
        DR = DIRECTIONS.right(DF)
        DL = DIRECTIONS.left(DF)
        DB = DIRECTIONS.behind(DF)
        bf = np.where(DF != 0)[0]
        lr = (bf + 1) % 2

        directions = [DL, DF, DR, DB]
        signs = np.sum(directions, axis=1)
        indeces = [lr, bf, lr, bf]
        distances = np.zeros(4)

        if target is not None:
            for i in range(len(distances)):
                idx = indeces[i]
                distances[i] = (target[idx] - self.head.value[idx]) * signs[i]
            return distances 
      
        for i in range(len(distances)):
            idx = indeces[i]
            distances[i] = self.game_size - self.head.value[idx] if signs[i] > 0 else self.head.value[idx] + 1
        
        head = self.head.next
        while head is not None:
            potential_distances = np.zeros(len(distances))
            for i in range(len(distances)):
                idx = indeces[i]
                potential_distances[i] = (head.value[idx] - self.head.value[idx]) * signs[i]
            for i in range(len(distances)):
                if potential_distances[i] > 0 and potential_distances[(i + 1) % 2] == 0:
                    distances[i] = min(distances[i], potential_distances[i])
            head = head.next
        return distances

    def crosses(self, target=None):
        # If target not defined, set target to current head and check for overlap with other heads
        head = self.head.next if target is None else self.head
        target = self.head.value if target is None else target
        
        while head is not None:
            if (target == head.value).all():
                return True
            head = head.next
        return False

    def extend(self, n = 1):
        for _ in range(n):
            new_head = DLLNode(self.head.value + self.direction, next=self.head)
            self.head.prev = new_head
            self.head = new_head
    
    def shorten(self, n = 1):
        if(self.tail == self.head):
            return False
        for _ in range(n):
            self.tail = self.tail.prev
            self.tail.next = None
    
    def get_all_positions(self):
        positions = []
        head = self.head
        while head is not None:
            positions.append(head.value)
            head = head.next
        return positions

    def __str__(self):
        s = str(self.head)
        head = self.head.next
        while head is not None:
            s += f',\n{head}'
            head = head.next
        return s
    
    def __len__(self):
        n = 1
        head = self.head.next
        while head is not None:
            n += 1
            head = head.next
        return n