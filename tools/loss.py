import torch
import torch.nn as nn
from .metric import *


class CrossEntropyLabelSmooth(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.

	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

	def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.use_gpu = use_gpu
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, inputs, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).long().data.cpu(), 1)
		if self.use_gpu: targets = targets.to(torch.device('cuda'))
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (- targets * log_probs).mean(0).sum()
		return loss


class RankingLoss:

	def __init__(self):
		pass

	def _label2similarity(sekf, label1, label2):
		'''
		compute similarity matrix of label1 and label2
		:param label1: torch.Tensor, [m]
		:param label2: torch.Tensor, [n]
		:return: torch.Tensor, [m, n], {0, 1}
		'''
		m, n = len(label1), len(label2)
		l1 = label1.view(m, 1).expand([m, n])
		l2 = label2.view(n, 1).expand([n, m]).t()
		similarity = l1 == l2
		return similarity

	def _batch_hard(self, mat_distance, mat_similarity, more_similar):

		if more_similar is 'smaller':
			sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1,descending=True)
			hard_p = sorted_mat_distance[:, 0]
			sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
			hard_n = sorted_mat_distance[:, 0]
			return hard_p, hard_n

		elif more_similar is 'larger':
			sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1, descending=False)
			hard_p = sorted_mat_distance[:, 0]
			sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1, descending=True)
			hard_n = sorted_mat_distance[:, 0]
			return hard_p, hard_n


class TripletLoss(RankingLoss):
	'''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''

	def __init__(self, margin, metric):
		'''
		:param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
		:param bh: batch hard
		:param metric: l2 distance or cosine distance
		'''
		self.margin = margin
		self.margin_loss = nn.MarginRankingLoss(margin=margin)
		self.metric = metric

	def __call__(self, emb1, emb2, emb3, label1, label2, label3):
		'''

		:param emb1: torch.Tensor, [m, dim]
		:param emb2: torch.Tensor, [n, dim]
		:param label1: torch.Tensor, [m]
		:param label2: torch.Tensor, [b]
		:return:
		'''

		if self.metric == 'cosine':
			mat_dist = cosine_dist(emb1, emb2)
			mat_sim = self._label2similarity(label1, label2)
			hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

			mat_dist = cosine_dist(emb1, emb3)
			mat_sim = self._label2similarity(label1, label3)
			_, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

			margin_label = -torch.ones_like(hard_p)

		elif self.metric == 'euclidean':
			mat_dist = euclidean_dist(emb1, emb2)
			mat_sim = self._label2similarity(label1, label2)
			hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

			mat_dist = euclidean_dist(emb1, emb3)
			mat_sim = self._label2similarity(label1, label3)
			_, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

			margin_label = torch.ones_like(hard_p)

		return self.margin_loss(hard_n, hard_p, margin_label)


class PM_Triplet(RankingLoss):
	'''
	！！最终版！！
	余弦值转化为弧度制

	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''

	def __init__(self, margin, beta=1):
		'''
		:param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
		:param bh: batch hard
		:param metric: l2 distance or cosine distance
		'''
		self.margin = margin
		self.margin_loss = nn.MarginRankingLoss(margin=margin)
		self.beta = beta

	def __call__(self, emb1, emb2, emb3, label1, label2, label3):
		'''

		:param emb1: torch.Tensor, [m, dim]
		:param emb2: torch.Tensor, [n, dim]
		:param label1: torch.Tensor, [m]
		:param label2: torch.Tensor, [b]
		:return:
		'''

		"""cosine"""
		cos_dist = cosine_dist(emb1, emb2)
		cos_dist = torch.acos(torch.clamp(cos_dist, -1 + 1e-7, 1 - 1e-7))
		mat_sim = self._label2similarity(label1, label2)
		hard_p, _ = self._batch_hard(cos_dist, mat_sim.float(), more_similar='larger')  # 相似度最小的正样本对(难样本)

		cos_dist = cosine_dist(emb1, emb3)
		cos_dist = torch.acos(torch.clamp(cos_dist, -1 + 1e-7, 1 - 1e-7))
		mat_sim = self._label2similarity(label1, label3)
		_, hard_n = self._batch_hard(cos_dist, mat_sim.float(), more_similar='larger')

		margin_label = -torch.ones_like(hard_p)
		cos_loss = self.margin_loss(hard_n, hard_p, margin_label)
		"""euclidean"""
		euc_dist = euclidean_dist(emb1, emb2)
		mat_sim = self._label2similarity(label1, label2)
		hard_p, _ = self._batch_hard(euc_dist, mat_sim.float(), more_similar='smaller')

		euc_dist = euclidean_dist(emb1, emb3)
		mat_sim = self._label2similarity(label1, label3)
		_, hard_n = self._batch_hard(euc_dist, mat_sim.float(), more_similar='smaller')

		margin_label = torch.ones_like(hard_p)
		euc_loss = self.margin_loss(hard_n, hard_p, margin_label)
		loss = 1.0 * euc_loss + self.beta * cos_loss  #########################################################################
		return loss


class nRankingLoss_1:

	def __init__(self):
		pass

	def _label2similarity(sekf, label1, label2):
		'''
		compute similarity matrix of label1 and label2
		:param label1: torch.Tensor, [m]
		:param label2: torch.Tensor, [n]
		:return: torch.Tensor, [m, n], {0, 1}
		'''
		m, n = len(label1), len(label2)
		l1 = label1.view(m, 1).expand([m, n])
		l2 = label2.view(n, 1).expand([n, m]).t()
		similarity = l1 == l2
		return similarity

	def _batch_hard(self, mat_distance, mat_similarity, more_similar):

		# for Euclidean
		if more_similar is 'smaller':
			sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
			hard_p = sorted_mat_distance[:, 0]
			mask = sorted_mat_distance > 0.05
			post_dist = torch.masked_select(sorted_mat_distance, mask)
			post_dist, _ = torch.sort(post_dist)
			post_dist = torch.mean(post_dist)

			sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
			hard_n = sorted_mat_distance[:, 0]
			return hard_p, hard_n, post_dist

		# for Cosine
		elif more_similar is 'larger':
			sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1, descending=False)
			hard_p = sorted_mat_distance[:, 0]
			sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1, descending=True)
			hard_n = sorted_mat_distance[:, 0]
			return hard_p, hard_n


class CPM_Triplet(nRankingLoss_1):

	def __init__(self, margin, beta=1.0, gamma=1.0):
		self.margin = margin
		self.margin_loss = nn.MarginRankingLoss(margin=margin)
		self.beta = beta
		self.gamma = gamma

	def hard_sigmoid_post(self, x, c=12):
		"""
		用于单个数
		post_dist: c=12
		post_neighbor: c=3
		"""
		if x < 0:
			return 0
		elif x > 2*c:
			return 1
		else:
			return x / (2 * c) + 0.002

	def __call__(self, emb1, emb2, emb3, label1, label2, label3):
		'''

		:param emb1: torch.Tensor, [m, dim]
		:param emb2: torch.Tensor, [n, dim]
		:param label1: torch.Tensor, [m]
		:param label2: torch.Tensor, [b]
		:return:
		'''

		"""cosine"""
		cos_dist = cosine_dist(emb1, emb2)
		cos_dist = torch.acos(torch.clamp(cos_dist, -1 + 1e-7, 1 - 1e-7))
		mat_sim = self._label2similarity(label1, label2)
		hard_p, _ = self._batch_hard(cos_dist, mat_sim.float(), more_similar='larger')  # 相似度最小的正样本对(难样本)

		cos_dist = cosine_dist(emb1, emb3)
		cos_dist = torch.acos(torch.clamp(cos_dist, -1 + 1e-7, 1 - 1e-7))
		mat_sim = self._label2similarity(label1, label3)
		_, hard_n = self._batch_hard(cos_dist, mat_sim.float(), more_similar='larger')

		margin_label = -torch.ones_like(hard_p)
		cos_loss = self.margin_loss(hard_n, hard_p, margin_label)

		"""euclidean"""
		euc_dist = euclidean_dist(emb1, emb2)
		mat_sim = self._label2similarity(label1, label2)
		hard_p, _, post_dist = self._batch_hard(euc_dist, mat_sim.float(), more_similar='smaller')  

		euc_dist = euclidean_dist(emb1, emb3)
		mat_sim = self._label2similarity(label1, label3)
		_, hard_n, _ = self._batch_hard(euc_dist, mat_sim.float(), more_similar='smaller')  

		margin_label = torch.ones_like(hard_p)
		euc_loss = self.margin_loss(hard_n, hard_p, margin_label)
		euc_pos_loss = self.hard_sigmoid_post(post_dist, c=12)  

		loss = 1.0 * euc_loss + 1.0 * cos_loss + self.beta * euc_pos_loss  
		return loss


