import random as rnd
from collections import deque

class ReplayBuffer():
    def __init__(self, buffer_max_size):
        self.buffer_deque_0 = deque()
        self.buffer_deque_1 = deque()
        self.BUFFER_MAX_SIZE = buffer_max_size
        return

    def store_memory(self, experience: tuple):
        action = experience[1]

        if action == 0:
            self.buffer_deque_0.append(experience)
            if len(self.buffer_deque_0) > self.BUFFER_MAX_SIZE / 2: self.buffer_deque_0.popleft()
        elif action == 1:
            self.buffer_deque_1.append(experience)
            if len(self.buffer_deque_1) > self.BUFFER_MAX_SIZE / 2: self.buffer_deque_1.popleft()
        return

    def collect_memory(self, batch_size):
        a = rnd.sample(list(self.buffer_deque_0), batch_size // 2)
        a.extend(rnd.sample(list(self.buffer_deque_1), batch_size // 2))
        return a
        
    def erase_memory(self):
        self.buffer_deque_0.clear()
        self.buffer_deque_1.clear()
        return
    
    def __len__(self):
        return len(self.buffer_deque_0) + len(self.buffer_deque_1)