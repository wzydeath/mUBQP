import torch
import torch.nn as nn

from layers import MultiHeadAttention, DotProductAttention
from data import generate_data
from decoder_utils import TopKSampler, CategoricalSampler, Env
from numpy import random
import numpy as np


# def get_cost(x, pi):
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     sol = torch.zeros([x.size(0),x.size(1)], dtype=torch.float32).to(device)
#     val = torch.ones_like(pi, dtype=torch.float32).to(device)
#     sol = sol.scatter_(dim=1, index=pi, src=val).unsqueeze(1)
#     sum = torch.bmm(torch.bmm(sol, x), sol.permute(0, 2, 1))
#     return sum.squeeze(-1)

def get_cost(x, pi, rf):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    index = (pi == 0).int().argmax(dim=1)
    sol = torch.zeros_like(pi, device=device,dtype=torch.float32)
    row_index = torch.arange(pi.size(0)).unsqueeze(1)
    mask = torch.arange(pi.size(1)).to(device).unsqueeze(0)<index.unsqueeze(1)
    sol[row_index, pi*mask] = 1
    sol = sol.unsqueeze(1)
    f1 = torch.bmm(torch.bmm(sol, x[0]), sol.permute(0, 2, 1))
    f2 = torch.bmm(torch.bmm(sol, x[1]), sol.permute(0, 2, 1))
    # f2 = torch.zeros_like(f1, dtype=torch.float32,device=device)
    cost = f1 * rf[0,0]  + f2 * rf[0, 1]
    # cost = torch.max(rf[0,0] * torch.abs(f1-200), rf[0,1]*torch.abs(f2-200))
    return -cost.squeeze(), f1, f2, sol



class Decoder(nn.Module):
    def __init__(self, embed_dim=128, n_heads=8, clip=10., **kwargs):  # clip = 10.
        super().__init__(**kwargs)

        self.Wk1 = nn.Linear(embed_dim, embed_dim, bias=True)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk2 = nn.Linear(embed_dim, embed_dim, bias=False)
        #self.Wq_fixed = nn.Linear(embed_dim * 3, embed_dim, bias=False)
        self.Wout = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_step = nn.Linear(2+embed_dim, embed_dim, bias=False)
        self.init_node = nn.Parameter(torch.zeros([embed_dim]), requires_grad=True)
        self.lstm = nn.LSTM(embed_dim, embed_dim)

        self.MHA = MultiHeadAttention(n_heads=n_heads, embed_dim=embed_dim, need_W=False)
        self.SHA = DotProductAttention(clip=clip, return_logits=True, head_depth=embed_dim)
        #self.SHA_value = DotProductAttention(clip=None, return_logits=True, head_depth=embed_dim)
        # SHA ==> Single Head Attention, because this layer n_heads = 1 which means no need to spilt heads
        self.env = Env
        self.embed_dim = embed_dim
        self.W = nn.Linear(embed_dim, 1, bias=False)
        self.embed_dim = embed_dim


    def compute_static(self, node_embeddings):
        self.K1 = self.Wk1(node_embeddings)
        self.V = self.Wv(node_embeddings)
        self.K2 = self.Wk2(node_embeddings)

    def compute_dynamic(self, mask, step_context):
        #self.Q_fixed = self.Wq_fixed(step_context)
        #Q1 = self.Wq_step(torch.cat((step_context, rf_embedding.unsqueeze(1)),dim=-1))
        Q1 = self.Wq_step(step_context)
        #Q1 = self.Q_fixed + Q_step
        Q2 = self.MHA([Q1, self.K1, self.V], mask=mask)
        Q2 = self.Wout(Q2)
        logits = self.SHA([Q2, self.K2, None], mask=mask)
        return logits.squeeze(dim=1)

    # def get_costs(self, x, pi, rf):
    #     sum = 0
    #     for i in range(len(x)):
    #         d = torch.gather(input = x[i], dim = 1, index = pi[:,:,None].repeat(1,1,2))
    #         dist = (torch.sum((d[:, 1:] - d[:, :-1]).norm(p = 2, dim = 2), dim = 1)
    #             + (d[:, 0] - d[:,-1]).norm(p = 2, dim = 1))  * rf[:, i]
    #         sum += dist
    #     return sum
    # def forward(self, x, encoder_output, decode_type = 'sampling'):
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #     node_embeddings, graph_embedding = encoder_output
    #     logits = torch.sigmoid(self.W(node_embeddings))
    #     sol = torch.zeros_like(logits, dtype=torch.float32, device = device)
    #     sol[logits>0.5] = 1
    #     sum = torch.bmm(torch.bmm(sol.permute(0,2,1), x), sol)
    #     log_ps = torch.zeros([x.size(0)],dtype=torch.float32,device=device)
    #     # log_p = torch.log_softmax(logits, dim=-1)
    #     for i in range(x.size(0)):
    #         log_ps[i] = torch.sum(logits[i, logits[i]>0.5],dim=-1)
    #     return -sum, log_ps, sol

#     def forward(self, x, encoder_output, decode_type = 'sampling'):
#         node_embeddings, graph_embedding = encoder_output
#         h_x = self.W(node_embeddings)
#         h_x = torch.sigmoid(h_x).squeeze(-1)
#         out = torch.zeros([x.size(0), x.size(1)])
#         out[h_x>0.5] = 1
#         cost = -torch.bmm(torch.bmm(out, x), out.permute(0, 2, 1))

    # def forward(self, x, encoder_output, rf, decode_type = 'sampling'):
    #     edge_embeddings, node_embeddings = encoder_output
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #     selecter = {'greedy': TopKSampler(), 'sampling': CategoricalSampler()}.get(decode_type, None)
    #     batch_size = x[0].size(0)
    #     batch_idx = torch.arange(0, batch_size).unsqueeze(1).to(device)
    #     cur_node = self.init_node.unsqueeze(0).repeat(batch_size, 1)
    #     lstm_h = torch.zeros([1, batch_size, self.embed_dim]).to(device)
    #     lstm_c = torch.zeros([1, batch_size, self.embed_dim]).to(device)
    #     log_ps, tours, next_node = [], [], []
    #     step_context = torch.cat([rf, cur_node],dim=-1).unsqueeze(1)
    #     env = Env(x, node_embeddings)
    #     mask = env.create_mask_t1()
    #     is_end = torch.ones([batch_size], dtype=torch.bool, device=device)
    #
    #     # first node
    #     self.compute_static(node_embeddings)
    #     logits = self.compute_dynamic(mask, step_context)
    #     log_p = torch.log_softmax(logits, dim=-1)  # log(p)
    #     next_node = selecter(log_p, is_end)
    #     cur_mask, cur_node_embedding, is_end = env._get_step(next_node, is_end)
    #     mask = mask | cur_mask
    #     tours.append(next_node.squeeze(1))
    #     log_ps.append(log_p)
    #
    #     for i in range(env.n_nodes - 1):
    #         context, (lstm_h, lstm_c) = self.lstm(cur_node_embedding.permute(1, 0, 2), (lstm_h, lstm_c))
    #         step_context = torch.cat([rf.unsqueeze(1),context.permute(1, 0, 2)],dim=-1)
    #         candidate = edge_embeddings[batch_idx, next_node].squeeze(1)
    #         self.compute_static(candidate)
    #         logits = self.compute_dynamic(mask, step_context)
    #         log_p = torch.log_softmax(logits, dim=-1)  # log(p)
    #         next_node = selecter(log_p, is_end)
    #         cur_mask, cur_node_embedding, is_end = env._get_step(next_node, is_end)
    #         mask = mask | cur_mask
    #         tours.append(next_node.squeeze(1))
    #         log_ps.append(log_p)
    #     pi = torch.stack(tours, 1)
    #     cost, f1, f2, sol = get_cost(x, pi, rf)
    #     cost = cost * -1
    #     ll = env.get_log_likelihood(torch.stack(log_ps, 1), pi, sol)
    #
    #     return cost, f1, f2, ll, pi

    def forward(self, x, encoder_output, rf, decode_type = 'sampling'):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        node_embeddings, graph_embedding = encoder_output
        env = Env(x, node_embeddings)
        # mask, step_context, D = env._create_t1()
        mask = env.create_mask_t1()
        sol_embed = self.init_node.unsqueeze(0).repeat(env.batch, 1)
        batch_idx = torch.arange(0, env.batch).unsqueeze(1).to(device)
        selecter = {'greedy': TopKSampler(), 'sampling': CategoricalSampler()}.get(decode_type, None)
        log_ps, tours, next_node = [], [], []
        cost = torch.zeros([env.batch, 1], dtype=torch.float32, device=device)
        self.compute_static(node_embeddings)
        end = torch.ones_like(cost, dtype=torch.bool, device=device)
        step_context = torch.cat([rf, sol_embed], dim=1).unsqueeze(1)
        for i in range(env.n_nodes):
            logits = self.compute_dynamic(mask, step_context)
            log_p = torch.log_softmax(logits, dim=-1)  # log(p)
            next_node = selecter(log_p, end)
            cur_mask, cur_embedding, end = env._get_step(next_node, end)
            mask = mask | cur_mask

            delta1 = x[0][batch_idx, next_node].squeeze(1)[mask.squeeze(-1)].reshape(env.batch, -1)
            delta1 = torch.sum(delta1, dim=-1, keepdim=True)

            # delta2 = x[1][batch_idx, next_node].squeeze(1)[mask.squeeze(-1)].reshape(env.batch, -1)
            # delta2 = torch.sum(delta2, dim=-1, keepdim=True)

            # cost[end] = cost[end] - rf[0,0]*delta1[end] #- rf[0,1]*delta2[end]
            # sol_embed[end.squeeze(-1)] = sol_embed[end.squeeze(-1)] + cur_embedding[end.squeeze(-1)].squeeze(1)
            step_context = torch.cat([rf, cur_embedding.squeeze(1)], dim=1).unsqueeze(1)
            tours.append(next_node.squeeze(1))
            log_ps.append(log_p)


        pi = torch.stack(tours, 1)
        cost, f1, f2, sol = get_cost(x, pi, rf)
        # cost = cost * -1
        ll = env.get_log_likelihood(torch.stack(log_ps, 1), pi, sol)

        return cost, f1, f2, ll, pi



if __name__ == '__main__':
    batch, n_nodes, embed_dim = 5, 10, 128
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    obj_num = 2
    point = np.zeros(obj_num)
    point[0] = np.random.randint(0, 101)
    sum_p = point[0]
    for j in range(1, obj_num - 1):
        point[j] = np.random.randint(0, 101 - sum_p)
        sum_p += point[j]
    point[obj_num - 1] = 100 - sum_p
    point = point / 100
    point = torch.tensor(point, dtype=torch.float32).repeat(batch, 1).to(device)
    data = generate_data(device, batch, n_nodes, obj_num)
    edge_embeddings = torch.rand(batch, n_nodes+1, n_nodes+1, 128)
    node_embeddings = torch.rand(batch, n_nodes+1, 128)
    graph_embedding = torch.mean(node_embeddings, dim=1)
    print(node_embeddings.shape, graph_embedding.shape)
    decoder = Decoder()
    cost, f1, f2, ll, pi = decoder(data, (node_embeddings, graph_embedding), point)
    print(cost.mean(), f1.mean(), f2.mean())








# cnt = 0
# for i, k in decoder.state_dict().items():
# 	print(i, k.size(), torch.numel(k))
# 	cnt += torch.numel(k)
# print(cnt)

# ll.mean().backward()
# print(decoder.Wk1.weight.grad)
# https://discuss.pytorch.org/t/model-param-grad-is-none-how-to-debug/52634
