import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantize(nn.Module):
	def __init__(self, dim, n_embed, commitment=1.0, decay=0.8, eps=1e-5):
		super().__init__()

		self.dim = dim
		self.n_embed = n_embed
		self.decay = decay
		self.eps = eps
		self.commitment = commitment

		embed = torch.randn(dim, n_embed)
		self.register_buffer('embed', embed)
		self.register_buffer('cluster_size', torch.zeros(n_embed))
		self.register_buffer('embed_avg', embed.clone())

	def forward(self, x, y=None):
		x = x.permute(0, 2, 3, 1).contiguous()
		input_shape = x.shape
		flatten = x.reshape(-1, self.dim)
		dist = (
		    flatten.pow(2).sum(1, keepdim=True)
		    - 2 * flatten @ self.embed
		    + self.embed.pow(2).sum(0, keepdim=True)
		)
		_, embed_ind = (-dist).max(1)
		embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
		embed_ind = embed_ind.view(*x.shape[:-1])
		quantize = self.embed_code(embed_ind).view(input_shape)

		if self.training:
			self.cluster_size.data.mul_(self.decay).add_(
			    1 - self.decay, embed_onehot.sum(0)
			)
			embed_sum = flatten.transpose(0, 1) @ embed_onehot
			self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
			n = self.cluster_size.sum()
			cluster_size = (
			    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
			)
			embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
			self.embed.data.copy_(embed_normalized)

		diff = self.commitment*torch.mean(torch.mean((quantize.detach() - x).pow(2), dim=(1,2)),
		                                  dim=(1,), keepdim=True)
		quantize = x + (quantize - x).detach()
		avg_probs = torch.mean(embed_onehot, 0)
		perplexity = torch.exp(- torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

		return quantize.permute(0, 3, 1, 2).contiguous(), diff, perplexity

	def embed_code(self, embed_id):
		return F.embedding(embed_id, self.embed.transpose(0, 1))


# class VectorQuantizerEMA(nn.Module):
# 	def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
# 		super(VectorQuantizerEMA, self).__init__()

# 		self._embedding_dim = embedding_dim
# 		self._num_embeddings = num_embeddings

# 		self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
# 		self._embedding.weight.data.normal_()
# 		self._commitment_cost = commitment_cost

# 		self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
# 		self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
# 		self._ema_w.data.normal_()

# 		self._decay = decay
# 		self._epsilon = epsilon


# 	def forward(self, inputs):
# 		# convert inputs from BCHW -> BHWC
# 		inputs = inputs.permute(0, 2, 3, 1).contiguous()
# 		input_shape = inputs.shape

# 		# Flatten input
# 		flat_input = inputs.view(-1, self._embedding_dim)

# 		# Calculate distances
# 		distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
# 		             + torch.sum(self._embedding.weight ** 2, dim=1)
# 		             - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

# 		# Encoding
# 		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
# 		encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
# 		encodings.scatter_(1, encoding_indices, 1)

# 		# Quantize and unflatten
# 		quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

# 		# Use EMA to update the embedding vectors
# 		if self.training:
# 			self._ema_cluster_size = self._ema_cluster_size * self._decay + \
# 			                         (1 - self._decay) * torch.sum(encodings, 0)

# 			# Laplace smoothing of the cluster size
# 			n = torch.sum(self._ema_cluster_size.data)
# 			self._ema_cluster_size = (
# 					(self._ema_cluster_size + self._epsilon)
# 					/ (n + self._num_embeddings * self._epsilon) * n)

# 			dw = torch.matmul(encodings.t(), flat_input)
# 			self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

# 			self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

# 		# Loss
# 		e_latent_loss = F.mse_loss(quantized.detach(), inputs)
# 		loss = self._commitment_cost * e_latent_loss

# 		# Straight Through Estimator
# 		quantized = inputs + (quantized - inputs).detach()
# 		avg_probs = torch.mean(encodings, dim=0)
# 		perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

# 		# convert quantized from BHWC -> BCHW
# 		return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
