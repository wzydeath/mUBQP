import numpy as np

class Chromosome:
    def __init__(self, obj_num, num, ind):
        self.ind = ind
        self.num = num
        self.f = np.zeros([obj_num], dtype=np.float32)
        self.obj_num = obj_num
        self.wf = 0

    def get_fitness(self, data):
        for k in range(self.obj_num):
            self.f[k] = 0
            for i in range(self.num):
                for j in range(self.num):
                    self.f[k] += data[k][i][j] * self.ind[i] * self.ind[j]
            # self.f[k] = np.dot(np.dot(self.ind, data[k]), self.ind)

    def compute_ws(self, ref_points, max_num, min_num):
        self.wf = 0
        for i in range(self.obj_num):
            self.wf += ref_points[i] * (self.f[i] - min_num[i]) / (max_num[i] - min_num[i])




