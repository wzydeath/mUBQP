import torch
import torch.nn as nn
# from torchsummary import summary

from layers import MultiHeadAttention
from data import generate_data
import math
import numpy as np


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super().__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d}.get(normalization, None)
        self.normalizer = normalizer_class(embed_dim, affine=True)

    # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
    # 	self.init_parameters()

    # def init_parameters(self):
    # 	for name, param in self.named_parameters():
    # 		stdv = 1. / math.sqrt(param.size(-1))
    # 		param.data.uniform_(-stdv, stdv)

    def forward(self, x):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            # (batch, num_features)
            # https://discuss.pytorch.org/t/batch-normalization-of-linear-layers/20989
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())

        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return x


class ResidualBlock_BN(nn.Module):
    def __init__(self, MHA, BN, **kwargs):
        super().__init__(**kwargs)
        self.MHA = MHA
        self.BN = BN

    def forward(self, x, mask=None):
        if mask is None:
            return self.BN(x + self.MHA(x))
        return self.BN(x + self.MHA(x, mask))


class SelfAttention(nn.Module):
    def __init__(self, MHA, **kwargs):
        super().__init__(**kwargs)
        self.MHA = MHA

    def forward(self, x, mask=None):
        return self.MHA([x, x, x], mask=mask)


# class EncoderLayer(nn.Module):
#     # nn.Sequential):
#     def __init__(self, n_heads=8, FF_hidden=512, embed_dim=128, **kwargs):
#         super().__init__(**kwargs)
#         self.n_heads = n_heads
#         self.FF_hidden = FF_hidden
#         self.BN1 = Normalization(embed_dim, normalization='batch')
#         self.BN2 = Normalization(embed_dim, normalization='batch')

#         self.MHA_sublayer = ResidualBlock_BN(
#             SelfAttention(
#                 MultiHeadAttention(n_heads=self.n_heads, embed_dim=embed_dim, need_W=True)
#             ),
#             self.BN1
#         )

#         self.FF_sublayer = ResidualBlock_BN(
#             nn.Sequential(
#                 nn.Linear(embed_dim, embed_dim, bias=True)
#                 # nn.ReLU(),
#                 # nn.Linear(FF_hidden, embed_dim, bias=True)
#             ),
#             self.BN2
#         )

#     def forward(self, x, mask=None):
#         """	arg x: (batch, n_nodes, embed_dim)
# 			return: (batch, n_nodes, embed_dim)
# 		"""
#         return self.FF_sublayer(self.MHA_sublayer(x, mask=mask))


class EncoderLayer(nn.Module):
    def __init__(self, n_heads=8, FF_hidden=512, embed_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.FF_hidden = FF_hidden
        self.BN1 = Normalization(embed_dim, normalization='batch')
        self.MHA = MultiHeadAttention(n_heads=n_heads, embed_dim=embed_dim, need_W=True)
        self.ReLU = nn.ReLU()
        self.FF_sublayer = nn.Linear(embed_dim, embed_dim)

    def forward(self, e, mask=None):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        node_num = e.size(1)
        tmp_e = torch.zeros_like(e).to(device)
        for i in range(node_num):
            q_e = e[:, :, i, :]
            k_e = e[:,i,:,:].clone()
            # k_e[index_expanded] = 0
            h_e = self.MHA((q_e, k_e, k_e), mask)
            tmp_e[:, :, i, :] = h_e.clone()
        output = tmp_e
        h = self.FF_sublayer(output)
        out = self.BN1(self.ReLU(h) + e)
        return out  # , v_similarity

def clones(module, N):
    """Produce N identical layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None):
    """
    Compute 'Scaled Dot Product Attention'
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # scores = scores.masked_fill(mask == 0, -1e9)
        scores = scores.masked_fill(mask == 0, -math.inf)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), scores

class AttentionEncoder(nn.Module):
    def __init__(self, hidden_dim):
        """Take in model size and number of heads."""
        super().__init__()
        self.sm = nn.Softmax(dim=-2)
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value, mask=None):
        """

        :param query:[batch_size x node_num x node_num x hidden_dim]
        :param key: [batch_size x node_num x node_num x node_num x hidden_dim]
        :param value: [batch_size x node_num x node_num x node_num x hidden_dim]
        :param mask:
        :return:
        """
        query = self.W1(query)
        key = self.W2(key)
        value = self.W3(value)

        d = key.size(-1)
        query = query.unsqueeze(3)
        temp = query.permute(0, 1, 2, 4, 3)

        # abcd=torch.matmul(key,temp)
        s = torch.matmul(key, temp) / math.sqrt(d)  # [batch_size x node_num x k]
        s = self.sm(s)
        att = torch.matmul(value.permute(0, 1, 2, 4, 3), s).squeeze(-1)
        return att


class GCNLayer(nn.Module):
    """Convnet layer with gating and residual connection.
    """

    def __init__(self, hidden_dim):
        super(GCNLayer, self).__init__()
        self.attn = AttentionEncoder(hidden_dim)
        # self.attn=MultiHeadAttention(8,hidden_dim,need_W=True)
        # edge GCN layers

        self.W_edge = nn.Linear(hidden_dim, hidden_dim)
        self.V_edge_in = nn.Linear(hidden_dim, hidden_dim)
        self.V_edge = nn.Linear(2 * hidden_dim, hidden_dim)
        self.Relu = nn.ReLU()
        self.ln1_edge = nn.LayerNorm(hidden_dim)
        self.ln2_edge = nn.LayerNorm(hidden_dim)

        self.bn1_edge = nn.BatchNorm1d(hidden_dim)
        self.bn2_edge = nn.BatchNorm1d(hidden_dim)

        self.hidden_dim = hidden_dim

    def forward(self, e, k_neighbor):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)

        Returns:
            h: Convolved node features (batch_size, num_nodes, hidden_dim)
        """
        # node embedding
        batch_size = e.size(0)
        node_num = e.size(1)
        edge_hidden_dim = e.size(-1)

        # get k nearest neighbors
        index = k_neighbor.unsqueeze(3).repeat(1, 1, 1, edge_hidden_dim)
        e_out = torch.gather(e, 2, index)

        # edge embedding
        e_out = e_out.unsqueeze(1).repeat(1, node_num, 1, 1, 1)
        att_out = self.attn(e, e_out, e_out)

        h_nb_edge = self.W_edge(att_out)
        h_nb_edge = e + self.Relu(h_nb_edge)
        h_nb_edge = self.bn1_edge(h_nb_edge.permute(0, 3, 1, 2).view(batch_size, edge_hidden_dim, -1)).view(batch_size,
                                                                                                            edge_hidden_dim,
                                                                                                            node_num,
                                                                                                            -1).permute(
            0, 2, 3,
            1)
        h_nb_edge = h_nb_edge + self.Relu(self.V_edge(torch.cat([self.V_edge_in(e), h_nb_edge], dim=-1)))
        h_edge = self.bn2_edge(h_nb_edge.permute(0, 3, 1, 2).view(batch_size, edge_hidden_dim, -1)).view(batch_size,
                                                                                                         edge_hidden_dim,
                                                                                                         node_num,
                                                                                                         -1).permute(0,
                                                                                                                     2,
                                                                                                                     3,
                                                                                                                     1)

        return h_edge



class GraphAttentionEncoder(nn.Module):
    def __init__(self, embed_dim=128, n_heads=8, n_layers=3, FF_hidden=512, n_nodes=10):
        super().__init__()
        # self.LSTM = nn.LSTM(input_size=1, hidden_size=embed_dim, batch_first=True)
        # 1D 卷积加池化
        # self.conv1 = nn.Conv1d(1, embed_dim, kernel_size=1)
        # self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.init_x = nn.Linear(4, embed_dim, bias=False)
        self.encoder_layers = nn.ModuleList([EncoderLayer(n_heads, FF_hidden, embed_dim) for _ in range(3)])
        # self.encoder_layers = nn.ModuleList([GCNLayer(embed_dim) for _ in range(3)])
        self.k = 51 

    def forward(self, x, rf, mask=None):
        x_stack = torch.stack(x, dim=-1)
        rf = rf.unsqueeze(1).unsqueeze(1).repeat(1, x[0].size(-1), x[0].size(-1), 1)
        x_init = torch.cat([x_stack,rf], dim=-1)
        dis = x_init[:, :, :, 0]
        a, b = torch.sort(dis,  dim=2)
        k_neighbor = b[:, :, 1:self.k + 1]
        x_embed = self.init_x(x_init)
        for layer in self.encoder_layers:
            # x_embed = layer(x_embed,  k_neighbor)
            x_embed = layer(x_embed)
        # h_x = torch.cat([x_embed,x_embed.permute(0,2,1,3)], dim=2)
        node_embed = torch.mean(x_embed, dim=2)
        return node_embed, torch.mean(node_embed, dim=1)
        return x_embed, node_embed

        # # 使用LSTM进行编码
        # x_list = []
        # # for i in range(batch):
        # #     _, (h_x, _) = self.LSTM(x.unsqueeze(-1)[i])
        # #     h_x = h_x.permute(1, 0, 2).squeeze(1)
        # #     x_list.append(h_x)
        # for i in range(x[0].size(0)):
        #     h_x = self.conv1(x[0][i].unsqueeze(1))
        #     h_x = self.global_avg_pool(h_x).squeeze(-1)
        #     x_list.append(h_x)
        # x_embed = torch.stack(x_list, dim=0)
        # for layer in self.encoder_layers:
        #     x_embed = layer(x_embed, mask)
        # return x_embed, torch.mean(x_embed, dim=1)


if __name__ == '__main__':
    batch = 5
    n_nodes = 5
    obj_num = 2
    encoder = GraphAttentionEncoder(n_layers=3)
    device = 'cpu'  # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    encoder = GraphAttentionEncoder()
    x_embed, node_embed = encoder(data, point)
    print(x_embed.shape, node_embed.shape)
    # RNN = nn.RNN(input_size=1,hidden_size=20,batch_first=True)
    # x_list = []
    # for i in range(batch):
    #     _, h_x = RNN(data.unsqueeze(-1)[i])
    #     h_x = h_x.permute(1,0,2).squeeze(1)
    #     x_list.append(h_x)
    # x_list = torch.stack(x_list, dim=0)
    # print(x_list.shape)
    # index = torch.zeros((batch, 1), dtype=torch.long)
    # RNN = nn.RNN(input_size=2, hidden_size=20, batch_first=True)
    # node_feat = []
    # for i in range(n_nodes):
    #     index[:, :] = i
    #     x_list = []
    #     for k in range(obj_num):
    #         x = torch.gather(input=data[k], dim=1, index=index[:, :, None].repeat(1, 1, 2))
    #         x_list.append(x.squeeze(1))
    #     x_list = torch.stack(x_list, dim=1)
    #     #print(x_list.shape)
    #     _, h_x = RNN(x_list)
    #     node_feat.append(h_x.permute(1,0,2).squeeze(1))
    # node_feat = torch.stack(node_feat, dim=1)
    # print(node_feat.shape)

# print('output[0].shape:', output[0].size())
# print('output[1].shape', output[1].size())

# summary(encoder, [(2), (20,2), (20)])
# cnt = 0
# for i, k in encoder.state_dict().items():
# 	print(i, k.size(), torch.numel(k))   # torch.numel(k) 统计k中元素的个数
# 	cnt += torch.numel(k)
# print(cnt)

# output[0].mean().backward()
# print(encoder.init_W_depot.weight.grad)
