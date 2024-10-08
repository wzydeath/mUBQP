import numpy as np
from env import Chromosome
import copy
from scipy.special import comb, perm
from pymoo.indicators.hv import Hypervolume
from time import time
import sys
import random


obj_num = 2
data_file1 = sys.argv[1] # f'data/validate_data/{cust_num}/dist_mat_{cust_num}_0.txt'
data_file2 = sys.argv[2]
# GEN = int(sys.argv[2])
out_file = sys.argv[3]
node_num = 100
num = node_num + 1

N = 100
H = 99
max_capacity = 1
INF = float('inf')
T = 5
ref_size = int(comb(H + obj_num - 1, obj_num - 1))
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


def output(pop):
    for i in range(len(pop)):
        for j in range(obj_num):
            print('%.6f' % pop[i].f[j], end=' ')
        print('\n')


def generate_lambda():
    t_array = np.arange(0, H + 1)
    temp = np.zeros(obj_num)
    num = 0
    ref_points = np.zeros([ref_size, obj_num])

    def generate_ref(index, sum, obj_num, H):
        nonlocal num
        if index == obj_num:
            if sum == H:
                ref_points[num] = temp / H
                num += 1
            return
        if sum > H: return
        for i in range(H + 1):
            temp[index] = t_array[i]
            generate_ref(index + 1, sum + t_array[i], obj_num, H)
    generate_ref(0, 0, obj_num, H)
    return ref_points


def init_para():
    ref_points = generate_lambda()
    B = np.zeros([N, T], dtype=int)

    def calculate_neighbor(index, ref_points, B):
        Euclid = np.zeros(N, dtype=float)
        for i in range(N):
            Euclid[i] = np.sqrt(np.sum((ref_points[i] - ref_points[index]) ** 2))
        ind = list(range(N))
        ind.sort(key=lambda x: Euclid[x])
        for i in range(T):
            B[index][i] = ind[i]

    for i in range(N):
        calculate_neighbor(i, ref_points, B)

    return ref_points, B

def find_max_min(pop):
    max_obj = [-INF] * obj_num
    min_obj = [INF] * obj_num

    for chrome in pop:
        for obj_index in range(obj_num):
            if chrome.f[obj_index] > max_obj[obj_index]:
                max_obj[obj_index] = chrome.f[obj_index]
            if chrome.f[obj_index] < min_obj[obj_index]:
                min_obj[obj_index] = chrome.f[obj_index]

    return max_obj, min_obj

def process(GEN):
    start = time()
    data = get_data()
    pop = init_pop(data)
    ref_points, B = init_para()
    for iter in range(GEN):
        print(iter)
        for i in range(N):
            max_num, min_num = find_max_min(pop)
            parent1 = random.randint(0, T - 1)
            parent2 = random.randint(0, T - 1)
            while parent1 == parent2:
                parent2 = random.randint(0, T - 1)
            child = crossover(pop[B[i][parent1]], pop[B[i][parent2]], data)
            p = random.random()
            if p < 0.01:
                child = mutation(child, data)
            for j in range(T):
                child.compute_ws(ref_points[B[i][j]], max_num, min_num)
                pop[B[i][j]].compute_ws(ref_points[B[i][j]], max_num, min_num)
                if child.wf > pop[B[i][j]].wf:
                    pop[B[i][j]] = child
    finish = time()
    file = open(f"./result/{node_num}/time.txt", 'a')
    file.write('\n')
    file.write(str((finish - start)))
    file.close()
    fit = np.zeros([N, obj_num])
    for i in range(N):
        for j in range(obj_num):
            fit[i][j] = pop[i].f[j]
    np.savetxt(out_file,fit, fmt='%.4f')
    # 20: [15,3000]
    # 50: [30, 6000]
    # 100: [60, 9000]
    output(pop)
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


