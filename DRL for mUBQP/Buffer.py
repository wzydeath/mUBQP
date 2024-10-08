from heapq import heappush, heappushpop, heapify, nlargest
from operator import itemgetter
import random
import numpy as np
import torch
from collections import deque

# 经验回放
class ExpReplayBuffer(object):

    def __init__(self, buffer_size):
        super().__init__()
        self.buffer = deque(maxlen=buffer_size)

    def push(self, graph, state, action, reward, next_state, done):
        self.buffer.append((graph, state, action, reward, next_state, done))

    def sample(self, bs):
        graph, state, action, reward, next_state, done = \
            zip(*random.sample(self.buffer, bs))
        return np.stack(graph, 0), list(state), np.stack(action, 0), \
               np.stack(reward, 0), list(next_state), \
               np.stack(done, 0).astype(np.float32)

    def __len__(self):
        return len(self.buffer)


# 优先经验回放
class Sample(tuple):
    def __lt__(self, x):
        #第0维为td值
        return self[0] < x[0]


class PrioritizedExpReplayBuffer(object):
    def __init__(self, buffer_size, alpha):
        self.buffer = []
        self.alpha = alpha
        self.buffer_size = buffer_size

    def heapify(self):
        # 建立小顶堆：以td为排序依据，td小的排前面
        heapify(self.buffer)

    def push(self, graph, state, action, reward, next_state, done):
        #设置样本的初始时序分差
        td = 1.0 if not self.buffer else \
            nlargest(1, self.buffer, key=itemgetter(0))[0][0]

        #向优先队列插入样本
        if len(self.buffer) < self.buffer_size:
            heappush(self.buffer, Sample((td, graph, state, action, reward, next_state, done)))
        else:
            heappushpop(self.buffer, Sample((td, graph, state, action, reward, next_state, done)))

    #设置样本的时序差分误差
    def set_td_value(self, index, value):
        for idx_s, idx_t in enumerate(index):  # enumerate 枚举，返回index, value
            self.buffer[idx_t] = Sample((value[idx_s], \
                                      *self.buffer[idx_t][1:]))

    def sample(self, bs, beta=1.0):
    #计算权重并归一化
        with torch.no_grad():
            weights = torch.tensor([val[0] for val in self.buffer])
            weights = weights.abs().pow(self.alpha)
            weights = weights / weights.sum()
            prob = weights.cpu().numpy()
            weights = (len(weights) * weights).pow(-beta)
        index = random.choices(range(len(weights)), weights=prob, k=bs)

    # 根据index返回训练样本
        _, graph, state, action, reward, next_state, done = \
            zip(*[self.buffer[i] for i in index])
        weights = [weights[i] for i in index]

        return np.stack(weights, 0).astype(np.float32), index, \
            np.stack(graph, 0), list(state), np.stack(action, 0), \
            np.stack(reward, 0), list(next_state), \
            np.stack(done, 0).astype(np.float32)

    def __len__(self):
        return len(self.buffer)