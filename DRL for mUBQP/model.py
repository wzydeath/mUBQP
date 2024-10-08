import torch
import torch.nn as nn

from data import generate_data
from encoder import GraphAttentionEncoder
from decoder import Decoder
from Buffer import PrioritizedExpReplayBuffer
from decoder_utils import Env, TopKSampler, CategoricalSampler
from Buffer import ExpReplayBuffer
import time

class AttentionModel(nn.Module):
	"""
	Dueling DDQN
	"""
	def __init__(self, embed_dim = 128, n_encode_layers = 3, n_heads = 8, tanh_clipping = 10., FF_hidden = 512):
		super().__init__()
		
		self.Encoder = GraphAttentionEncoder(embed_dim, n_heads, n_encode_layers, FF_hidden)
		self.Decoder = Decoder(embed_dim, n_heads, tanh_clipping)

	def forward(self, x, rf, decode_type = 'sampling'):
		encoder_output = self.Encoder(x, rf)
		decoder_output = self.Decoder(x, encoder_output, rf,decode_type)
		return decoder_output

	def get_obj_costs(self, x, pi):
		sum = []
		for i in range(len(x)):
			d = torch.gather(input=x[i], dim=1, index=pi[:, :, None].repeat(1, 1, 2))
			dist = (torch.sum((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2), dim=1)
					+ (d[:, 0] - d[:, -1]).norm(p=2, dim=1))
			sum.append(dist.mean().item())
		return sum

	def evaluate(self, x, point, decode_type = 'greedy'):
		"""
		对于数据x:[batch, 2]，使用策略Pi生成batch个解
		:param buffer: buffer
		:param x: graphs: [batch, 2]
		:param encoder_output: node embeddings and graph embedding
		:param decode_type: greedy
		:return: the costs and solutions of x
		"""
		with torch.no_grad():
			costs = []
			for i in range(point.size(0)):
				eval_point = point[i].unsqueeze(0).repeat(x[0].size(0), 1)
				output = self.Encoder(x, eval_point)
				cost, ll, pi = self.Decoder(x, output, eval_point, decode_type)
				obj_cost = self.get_obj_costs(x, pi)
				#print(obj_cost)
				costs.append(obj_cost)
				#costs.append(cost.mean())

		return costs

		
if __name__ == '__main__':
	
	device = 'cpu'
	model = AttentionModel()
	model.train()
	data = generate_data(n_samples = 10, n_customer = 14, seed = 1, device='cpu')
	NBUFFER = 100000
	buffer = ExpReplayBuffer(NBUFFER)
	model.expert_generate_sample(buffer, data)

	cost, pi = model.eval(data)
	print(cost, pi)

