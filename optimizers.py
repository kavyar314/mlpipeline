import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt


def optimizer_selector(optimizer_name, optimizer_args):
	if optimizer_name == "torch_sgd":
		return lambda modelparams: optim.SGD(modelparams,lr=optimizer_args["lr"], momentum=optimizer_args["momentum"])
	else:
		return None # raise error here?


class LoggedGradientDescent(optim.Optimizer, OptimizerTermination, Logs):
	def __init__(self, parameters, lr=0.001, termination=("train loss", 1e-4), log_names=[]):
		'''
		termination: tuple of name of termination condition and 
		'''
		super().__init__()
		## TODO: rest of init
		self.termination_function = self.termination_selector(termination[0])
		self.terminal_value = termination[1]
		self.logs = [self.logs_selector(log_name) for log_name in log_names]
		self.terminated = False

	def step(self):
		pass

	def termination_selector(self, termination_name):
		'''
		returns a function that takes in the terminal value of the parameter we are checking and returns a boolean about whether or not to terminate.
		'''
		if termination_name == "train loss":
			return self.loss_converge(self.train_losses) ## ahhh, idk if this will auto update as self.train_losses gets updated. CHECK -- test example in notebook seems to work?
		if termination_name == "validation loss":
			return self.loss_converge(self.validation_losses)
		if termination_name == "averaged train loss":
			return self.averaged_losses_converge(self.train_losses)
		if termination_name == "averaged validation loss":
			return self.averaged_losses_converge(self.validation_losses)
		if termination_name == "fixed epochs":
			return self.epochs_over
		if termination_name == "small last update":
			return self.last_update_norm_small


class OptimizerTermination(OptimizerInstanceFields):
	def __init__(self):
		super().__init__()

	def loss_converge(self, losses):
		'''
		can be used for train losses, validation losses, or even running averaged losses
		'''
		def appropriate_loss_converge(epsilon):
			if losses[-1] < epsilon:
				return True
			else:
				return False
		return appropriate_loss_converge

	def averaged_losses_converge(self, losses, smoothing=3):
		smoothed_losses = [np.mean(losses[i:i+3]) for i in range(len(losses)-2)]
		return self.loss_converge(smoothed_losses)

	def epochs_over(self, limit_epochs):
		if self.epochs_transpired > limit_epochs:
			return True
		else:
			return False

	def last_update_norm_small(self, epsilon):
		if self.last_update_norm < epsilon:
			return True
		else:
			return False


class Logs(OptimizerInstanceFields):
	def __init__(self, plot):
		'''
		plot: bool regarding whether or not to plot the desired logged quantity
		'''
		super().__init__()
		self.plot = plot

	def log_item(self, itemfn):
		pass


class OptimizerInstanceFields():
	def __init__(self):
		self.train_losses = []
		self.validation_losses = []
		self.epochs_transpired = 0
		self.last_update_norm = np.inf

