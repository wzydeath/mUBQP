import numpy as np
from env import Chromosome
import random
import copy
from pymoo.indicators.hv import Hypervolume
import sys
from time import time

data_file1 = sys.argv[1] # f'data/validate_data/{cust_num}/dist_mat_{cust_num}_0.txt'
data_file2 = sys.argv[2]
# GEN = int(sys.argv[2])
out_file = sys.argv[3]
node_num = 50
num = node_num + 1
obj_num = 2


N = 100
H = 99
max_capacity = 1
INF = float('inf')
pop = []


def get_data():
    data1 = np.loadtxt(data_file1)
    data2 = np.loadtxt(data_file2)
    data = []
    data.append(data1)
    data.append(data2)
    data = np.stack(data)
    # data = 2 * np.random.random([obj_num, num, num]) - 1
    # data[:,0,:] = 0
    # data[:,:,0] = 0
    return data.reshape([obj_num, num, num])


def init_pop(data):
    for i in range(N):
        ind = np.zeros([num], dtype=np.float32)
        for j in range(num):
            p = random.random()
            if p < 0.5:
                ind[j] = 1
        chrome = Chromosome(obj_num, num, ind)
        chrome.get_fitness(data)
        pop.append(chrome)
    return pop

def crossover(parent1, parent2, data):
    child = np.zeros([num], dtype=np.float32)
    for i in range(num):
        if parent1.ind[i] == parent2.ind[i]:
            child[i] = parent1.ind[i]
        else:
            child[i] = random.choice([0, 1])
    child_chrome = Chromosome(obj_num, num, child)
    child_chrome.get_fitness(data)
    return child_chrome

def mutation(chrome, data):
    rate = 1.0 / num
    for i in range(num):
        p = random.random()
        if p < rate:
            chrome.ind[i] = 1 - chrome.ind[i]
    chrome.get_fitness(data)
    return chrome


def is_better(f1, f2):
    better = False
    for i in range(obj_num):
        if f1[i] < f2[i] :
            return False
        elif f1[i] > f2[i]:
            better = True
    return better


def fast_nondominated_sort(pop):
    F = []
    n = np.zeros(2 * N, dtype=np.int32)
    S = [[] for _ in range(2 * N)]

    for i in range(len(pop)):
        for j in range(len(pop)):
            if i != j:
                if is_better(pop[i].f, pop[j].f):
                    S[i].append(j)
                elif is_better(pop[j].f, pop[i].f):
                    n[i] += 1
    temp = []
    for i in range(len(pop)):
        if n[i] == 0:
            temp.append(i)
    F.append(temp)
    index = 0
    while len(F[index]) != 0:
        temp = []
        size = len(F[index])
        for i in range(size):
            p = F[index][i]
            q_size = len(S[p])
            for j in range(q_size):
                q = S[p][j]
                n[q] -= 1
                if n[q] == 0:
                    temp.append(q)
        F.append(temp)
        index += 1
    return F


def find_max_min(sub_pop):
    max_obj_cd = [-INF] * obj_num
    min_obj_cd = [INF] * obj_num

    for chrome in sub_pop:
        for obj_index in range(obj_num):
            if chrome.f[obj_index] > max_obj_cd[obj_index]:
                max_obj_cd[obj_index] = chrome.f[obj_index]
            if chrome.f[obj_index] < min_obj_cd[obj_index]:
                min_obj_cd[obj_index] = chrome.f[obj_index]

    return max_obj_cd, min_obj_cd

def crowding_distance(sub_pop):
    max_obj_cd, min_obj_cd = find_max_min(sub_pop)
    for chrome in sub_pop:
        chrome.cd = 0.0
    for obj_index in range(obj_num):
        sub_pop.sort(key=lambda chrome: chrome.f[obj_index])
        sub_pop[0].cd = INF
        sub_pop[-1].cd = INF
        for i in range(1, len(sub_pop) - 1):
            sub_pop[i].cd += (sub_pop[i + 1].f[obj_index] - sub_pop[i - 1].f[obj_index]) / (
                        max_obj_cd[obj_index] - min_obj_cd[obj_index])
    sub_pop.sort(key=lambda chrome: chrome.cd, reverse=True)


def copy_pop(pop, F):
    tmp_pop = []
    index = 0
    while len(tmp_pop) + len(F[index]) <= N:
        for i in range(len(F[index])):
            tmp_pop.append(pop[F[index][i]])
        index += 1
    while len(tmp_pop) < N:
        sub_pop = []
        for i in range(len(F[index])):
            sub_pop.append((pop[F[index][i]]))
        crowding_distance(sub_pop)
        for i in range(len(F[index])):
            if len(tmp_pop) < N: tmp_pop.append(sub_pop[i])
            else: break
    return tmp_pop


def output(pop):
    for i in range(len(pop)):
        for j in range(obj_num):
            print('%.6f' % pop[i].f[j], end=' ')
        print('\n')

def process(GEN):
    start = time()
    data = get_data()
    pop = init_pop(data)
    childrens = []
    for iter in range(GEN):
        print(iter)
        for _ in range(N):
            parent1 = random.randint(0, N - 1)
            parent2 = random.randint(0, N - 1)
            while parent1 == parent2:
                parent2 = random.randint(0, N - 1)
            child = crossover(pop[parent1], pop[parent2], data)
            p = random.random()
            if p < 0.01:
                child = mutation(child, data)
            childrens.append(child)
        for child in childrens:
            pop.append(child)
        F = fast_nondominated_sort(pop)
        pop = copy_pop(pop, F)
        childrens.clear()
    output(pop)
    finish = time()
    file = open(f"./result/{node_num}/time.txt", 'a')
    file.write('\n')
    file.write(str((finish - start)))
    file.close()
    fit = np.zeros([N, obj_num])
    for i in range(N):
        for j in range(obj_num):
            fit[i][j] = pop[i].f[j]
    np.savetxt(out_file, fit, fmt='%.4f')
    # file = open(f"./result/{cust_num}/{GEN}/time.txt", 'a')
    # file.write('\n')
    # file.write(str((finish - start) ))
    # file.close()
    metric = Hypervolume(ref_point=np.array([150, 150]),
                         norm_ref_point=False,
                         zero_to_one=False
                         )
    # fit = np.zeros([N, obj_num])
    # for i in range(N):
    #     for j in range(obj_num):
    #         fit[i][j] = pop[i].f[j]
    # np.savetxt(sys.argv[4], fit, fmt='%.4f')
    hv = metric.do(-fit)
    # print(hv)
    file = open(f"./result/{node_num}//hv.txt", 'a')
    file.write('\n')
    file.write(str(hv))



if __name__ == '__main__':
    process(100)
