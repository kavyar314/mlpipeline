import torch
import torch.nn as nn
from torch.nn import Module


def loss_selector(loss_name, arguments=None):
	if loss_name == "square":
		return square_loss()
	if loss_name == "torch_ce":
		return nn.CrossEntropyLoss()
	else:
		return None # raise error better here




class square_loss(Module):
	def __init__(self):
		super(square_loss, self).__init__()

	def forward(self, ypred, y):
		return torch.norm(ypred-y)
