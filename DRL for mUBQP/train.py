import torch
import torch.nn as nn
import torch.optim as optim
from time import time
import numpy as np

from model import AttentionModel
from data import generate_data
from config import Config, load_pkl, train_parser
from reference_points import ReferencePoint

def get_costs_ws(x, pi, rf):
    sum = 0
    for i in range(len(x)):
        d = torch.gather(input=x[i], dim=1, index=pi[:, :, None].repeat(1, 1, 2))
        dist = (torch.sum((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2), dim=1)
                + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)) * rf[:, i]
        sum += dist
    return sum

def train(cfg, log_path=None):

    # H, obj_num = 13, 3
    # rf = ReferencePoint(obj_num, H)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AttentionModel(cfg.embed_dim, cfg.n_encode_layers, cfg.n_heads, cfg.tanh_clipping)
    model_copy = AttentionModel(cfg.embed_dim, cfg.n_encode_layers, cfg.n_heads, cfg.tanh_clipping)
    model_copy.load_state_dict(model.state_dict())
    model.to(device)
    model_copy.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # eval_dataset = [generate_data(device, cfg.batch, cfg.n_customer) for i in range(obj_num)]
    # eval_point = torch.tensor(rf.ref_points[0], dtype=torch.float32).unsqueeze(0).repeat(cfg.batch, 1).to(device)
    obj_num = 2

    t1 = time()
    flag = False
    batch_size = 64
    for epoch in range(300):
        #rf.seq = torch.randperm(rf.ref_size)
        #if epoch < 10: rf.seq = torch.ones(rf.ref_size, dtype=torch.int) * 0
        #else:  rf.get_path() # torch.randperm(rf.ref_size)#rf.get_path()
        for i in range(100):
            data = generate_data(device, batch_size, 10, obj_num)
            model.train()
            point = np.zeros(obj_num)
            point[0] = np.random.randint(0, 101)
            sum_p = point[0]
            for j in range(1, obj_num - 1):
                point[j] = np.random.randint(0, 101 - sum_p)
                sum_p += point[j]
            point[obj_num - 1] = 100 - sum_p
            point = point / 100
            point = torch.tensor(point, dtype=torch.float32).repeat(batch_size, 1).to(device)
            with torch.no_grad():
                bs,_, _,  _, _ = model(data, point, 'greedy')
            cost, f1, f2,  ll,  pi = model(data, point)
            loss = ((cost - bs) * ll).mean()


            # 损失函数的优化
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()

            #_, pi = model.evaluate(eval_dataset, eval_point)
            #cost = get_costs_ws(eval_dataset,pi,point)
            #ave_loss.append(loss.item())
            #ave_L.append(cost.mean().item())

            #if i % 1 == 0:
            if i % (cfg.batch_verbose) == 0:
                t2 = time()
                print('Epoch %d %i: Loss: %1.3f L: %1.3f, L1: %1.3f, L2: %1.3f, r1: %1.3f, r2: %1.3f, %dmin%dsec' % (
                    epoch,  i, loss, cost.mean(),f1.mean(), f2.mean(), point[0][0],point[0][1], (t2 - t1) // 60, (t2 - t1) % 60))
                t1 = time()
        flag = not flag
        # model_copy.load_state_dict(model.state_dict())
        # if epoch % 100 == 0:
        # torch.save(model.state_dict(), './model/epoch%s.pt' % (epoch))


if __name__ == '__main__':
    cfg = load_pkl(train_parser().path)
    train(cfg)

# device = 'cpu'
# dataset = Generator(device, cfg.batch, cfg.n_customer)
# cost, _ = model1.generate_sample(buffer, dataset.data, 'greedy')

