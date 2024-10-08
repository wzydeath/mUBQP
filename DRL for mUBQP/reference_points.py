from scipy.special import comb, perm
import torch
import numpy as np

class ReferencePoint(object):

    def __init__(self, obj_num, H):
        super().__init__()
        self.ref_size = int(comb(H + obj_num - 1, obj_num - 1))
        self.ref_points = np.zeros([self.ref_size, obj_num])
        self.single_rf = np.random.dirichlet(np.ones(obj_num), size=1)
        self.num = 0
        self.t_array = np.arange(0, H+1)
        self.temp = np.zeros(obj_num)
        self.generate_ref(0, 0, obj_num, H)
        self.seq = [i for i in range(self.ref_size)]
        self.get_path()

    def generate_ref(self, index, sum, obj_num, H):
        if index == obj_num:
            if sum == H:
                self.ref_points[self.num] = self.temp / H
                self.num += 1
            return
        if sum > H: return
        for i in range(H + 1):
            self.temp[index] = self.t_array[i]
            self.generate_ref(index + 1, sum + self.t_array[i], obj_num, H)

    def get_path(self):
        dist = []
        for i in range(self.ref_size):
            dist.append([np.linalg.norm(x=(self.ref_points[i] - self.ref_points[j]))
                         for j in range(self.ref_size)])

        def cal_fitness(ind):
            sum = 0
            for i in range(self.ref_size - 1):
                sum += dist[ind[i]][ind[i+1]]
            return sum

        def two_opt(route):
            min_cost = cal_fitness(route)
            org_cost = min_cost
            best_i, best_j = -1, -1
            for i in range(self.ref_size - 1):
                for j in range(i+2, self.ref_size-1):
                    r_begin, r_end = i+2, j
                    cost = org_cost - dist[route[i]][route[i + 1]] - dist[route[j]][route[j + 1]] \
                           + dist[route[i]][route[j]] + dist[route[i + 1]][route[j + 1]]
                    if cost < min_cost:
                        min_cost = cost
                        best_i, best_j = i, j
            if best_i != -1:
                route[best_i + 1], route[best_j] = route[best_j],  route[best_i + 1]
                if best_i + 2 < best_j:
                    route[best_i + 2: best_j].reverse()

        pre_fit = cal_fitness(self.seq)
        while True:
            two_opt(self.seq)
            now_fit = cal_fitness(self.seq)
            if now_fit == pre_fit: break
            pre_fit = now_fit

if __name__ == '__main__':
    #rf = ReferencePoint(2, 99)
    #print(rf.single_rf)
    rf = np.random.dirichlet(np.ones(2), size=64)
    print(rf.shape)
    point = torch.tensor(rf, dtype=torch.float32)
    print(point)
