import torch
import torch.nn as nn
import math
import random
import numpy as np

class Env():
    def __init__(self, x, node_embeddings):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.device = 'cpu'
        #self.xy = torch.tensor(x).to(self.device)
        if node_embeddings is not None:
            self.node_embeddings = node_embeddings
            self.batch, self.n_nodes, self.embed_dim = node_embeddings.size()
            self.visited_customer = torch.zeros((self.batch, self.n_nodes, 1), dtype = torch.bool).to(self.device)

    def _get_step(self, next_node, end):
        one_hot = torch.eye(self.n_nodes).to(self.device)[next_node]		# torch.eye生成 n_nodes * n_nodes的对角线矩阵，对角线为1，其余为0
        # visited_mask 为next_node的mask
        visited_mask = one_hot.type(torch.bool).permute(0,2,1).to(self.device)  # (batch, n_nodes, 1) --> bool
        is_end = torch.ones_like(next_node, dtype=torch.bool)
        is_end = is_end.masked_fill(next_node == 0, 0)
        end = end & is_end
        # visited_mask = visited_mask | is_end.unsqueeze(-1)
        # all_true = torch.all(visited_mask.squeeze(-1), dim=1)
        # all_true_indices = torch.nonzero(all_true).squeeze()
        # visited_mask[all_true_indices,0,0] = False
        # visited_mask[:,0,0] = False
        prev_node_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = next_node[:,:,None]
                                           .repeat(1,1,self.embed_dim))
        return visited_mask, prev_node_embedding, end


    def _create_t1(self):
        mask_t1 = self.create_mask_t1()
        step_context_t1, D_t1 = self.create_context_D_t1()
        return mask_t1, step_context_t1, D_t1

    def create_mask_t1(self):
        mask_customer = self.visited_customer.to(self.device)
        mask_depot = torch.ones([self.batch, 1, 1], dtype = torch.bool).to(self.device)
        #return torch.cat([mask_depot, mask_customer], dim = 1)
        return mask_customer


    def get_log_likelihood(self, _log_p, pi, sol):
        """ _log_p: (batch, decode_step, n_nodes)
            pi: (batch, decode_step), predicted tour
        """
        log_p = torch.gather(input = _log_p, dim = 2, index = pi[:,:,None]).squeeze(-1)
        is_end = torch.ones([pi.size(0)],dtype=torch.long, device=self.device)
        sum =  torch.zeros([pi.size(0)],dtype=torch.float32, device=self.device)
        for i in range(sol.size(-1)):
            sum[is_end == 1] = sum[is_end == 1] + log_p[is_end == 1, i]
            is_end[pi[:,i] == 0] = 0
        return sum
        # return torch.sum(log_p.squeeze(-1), 1)   # log(p(x)) = log(p(x1)p(x2)...p(xn)) = log(p(x1)) + log(p(x2)) + ...log(p(xn))

class Sampler(nn.Module):
    """ args; logits: (batch, n_nodes)
        return; next_node: (batch, 1)
        TopKSampler <=> greedy; sample one with biggest probability
        CategoricalSampler <=> sampling; randomly sample one from possible distribution based on probability
    """
    def __init__(self, n_samples = 1, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples

class TopKSampler(Sampler):
    def forward(self, logits, is_end):
        nodes = torch.topk(logits, self.n_samples, dim = 1)[1]
        return nodes
# class ETopKSampler(Sampler):
#     def forward(self, logits, epsilon=0.1):
#         r = random.random()
#         if r > epsilon:
#             return torch.topk(logits, self.n_samples, dim = 1)[1]# == torch.argmax(log_p, dim = 1).unsqueeze(-1)
#         else:
#             # mask = torch.ones_like(logits) * math.sqrt(len(logits[0]))
#             # mask = mask.masked_fill(logits == -1e+10, 0)
#             # return torch.multinomial(mask, self.n_samples)
#             #print(logits)
#             return torch.multinomial(logits.exp(), self.n_samples)


class CategoricalSampler(Sampler):
    def forward(self, logits, is_end):
        nodes = torch.multinomial(logits.exp(), self.n_samples)
        return nodes


