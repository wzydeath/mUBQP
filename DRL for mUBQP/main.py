from config import Config, load_pkl, train_parser
from model import AttentionModel
import train
import torch
from data import Generator
from Buffer import ExpReplayBuffer
from torch import nn
import numpy as np
import sys, os
from time import time
from reference_points import ReferencePoint
from data import generate_data
import matplotlib.pyplot as plt
import random


# def get_costs(x, pi, rf):
#     obj_func = []
#     sum = []
#     for i in range(len(x)):
#         d = torch.gather(input=x[i], dim=1, index=pi[:, :, None].repeat(1, 1, 2))
#         obj = (torch.sum((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2), dim=1)
#                + (d[:, 0] - d[:, -1]).norm(p=2, dim=1))
#         obj_func.append(obj)
#         sum.append(rf[:, i] * torch.abs(obj - obj.min() + 1E-6))
#     result, _ = torch.max(torch.stack(sum, dim=0), dim=0)
#     return result

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def evaluate_multiobjective(data, model, point, decode_type='greedy'):

    with torch.no_grad():
        costs = []
        sum = 0
        for i in range(point.size(0)):
            batch = data[0].size(0)
            # point[i] = torch.tensor([1,0,0], dtype=torch.float32, device=device)
            eval_point = point[i].unsqueeze(0).repeat(batch, 1)
            # eval_point = point[i].unsqueeze(0)
            encoder_output = model.Encoder(data, eval_point)
            cost, f1, f2, _, _ = model.Decoder(data, encoder_output, eval_point,'greedy')
            # f1 = env.get_carnum()
            sum += cost.mean()
            # print(obj_cost)
            concat_f = torch.cat([f1.squeeze(-1),f2.squeeze(-1)],dim=-1)
            costs.append(concat_f)
        # costs.append(cost.mean())
        sum /= point.size(0)
        costs = torch.stack(costs,dim=1)
    return costs, sum



if __name__ == '__main__':

    H, obj_num = 99, 2
    batch_size = 2
    n_cust = 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = load_pkl(train_parser().path)
    eval_dataset = generate_data(device,batch_size, n_cust, obj_num)
    model = AttentionModel(cfg.embed_dim, cfg.n_encode_layers, cfg.n_heads, cfg.tanh_clipping)
    # model.load_state_dict(torch.load('./TSP20_train_epoch27.pt'))
    rf = ReferencePoint(obj_num, H)
    point = torch.tensor(np.stack(rf.ref_points), dtype=torch.float32).to(device)
    cost, _ = evaluate_multiobjective(eval_dataset, model, point)
    for i in range(batch_size):
        solution = cost[i].cpu().numpy()
        np.savetxt(f'result/output{i}.txt', solution, fmt='%.6f')
    #x = get_costs(eval_dataset, pi, point)
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(x[0],x[1],x[2])
    # model1.load_state_dict(torch.load('./TSP20_train_epoch999.pt', map_location=torch.device('cpu')))
    # cost, pi = model1.evaluate(eval_dataset, point)
    # x = get_costs(eval_dataset, pi, point)
    # pylot.scatter(x[0], x[1])
    plt.show()

    x = np.array(cost)
    print(x)
    #np.savetxt('result.txt', x.numpy(), fmt='%.6f')
    with open('result.txt','w') as file:
        #for i in range(len(cost)):
            #file.write(cost[i])
        np.savetxt(file, x, fmt='%.6f')

    # os.system('chcp 65001')
    # os.system(f'DRL_CSP.exe {data_file} {solution_file} > {result_file}')
    # t2 = time()
    # with open(time_file, 'w') as file:
    #     t = t2 - t1
    #     file.write(str(t))

