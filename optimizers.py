import torch.optim as optim
import torch
import numpy as np

import matplotlib.pyplot as plt


def optimizer_selector(optimizer_name, optimizer_args):
	if optimizer_name == "torch_sgd":
		return lambda modelparams: optim.SGD(modelparams,lr=optimizer_args["lr"], momentum=optimizer_args["momentum"])
	else:
		return None # raise error here?


class OptimizerInstanceFields():
	def __init__(self):
		self.train_losses = [np.inf]
		self.validation_losses = [np.inf]
		self.epochs_transpired = 0
		self.last_update_norm = np.inf


class OptimizerTermination(OptimizerInstanceFields):
	def __init__(self):
		super(OptimizerTermination, self).__init__()

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
	def __init__(self):
		'''
		plot: bool regarding whether or not to plot the desired logged quantity
		'''
		super(Logs, self).__init__()
		# self.plot = plot

	def log_item(self, itemfn):
		pass

	def save_log_state(self, path_to_save):
		pass




class LoggedGradientDescent(optim.Optimizer, OptimizerTermination, Logs):
	def __init__(self, parameters, train_loss_fn, val_loss_fn=None, lr=0.001, termination=("train loss", 1e-4), log_names=[]):
		'''
		termination: tuple of name of termination condition and 
		train_loss_fn: closure that already incorporates train data
		val_loss_fn: closure that already incorporates val data
		'''
		# need kwargs to handle `plot` variable for Logs
		optim.Optimizer.__init__(self, params=parameters, defaults={"lr": lr})
		OptimizerTermination.__init__(self)
		Logs.__init__(self)
		## TODO: rest of init
		self.termination_function = self.termination_selector(termination[0])
		self.terminal_value = termination[1]
		self.logs = [self.logs_selector(log_name) for log_name in log_names] # TODO: create a Log class that updates itself according to its instantiating function when .update() is called
		self.terminated = False ## is it better to have an instance field for this or a function that calls the termination_function when called?
		self.train_loss_fn = train_loss_fn
		self.val_loss_fn = val_loss_fn

		## TODO: check that: if logs includes val loss, then val_loss_fn is not None.

	@torch.no_grad()
	def step(self, closure=None):

		if closure is not None:
			with torch.enable_grad():
				loss = [closure()]
		else:
			loss = [self.train_loss_fn, self.val_loss_fn]


		# does not step if termination condition has been met
		if self.termination_function(self.terminal_value):
			self.terminated = True
			return loss

		for g in self.param_groups:
			params_with_grad = []
			d_params = []
			has_sparse_gradient = False
			# import pdb; pdb.set_trace()
			for param in g['params']:
				if param.grad is not None:
					params_with_grad.append(param)
					d_params.append(param.grad)
					# if param.grad_is_sparse:
						# has_sparse_gradient = True

			gd(params_with_grad, d_params, lr=g["lr"], has_sparse_gradient=False)

		losses = [l() for l in loss if l is not None]

		self.train_losses.append(losses[0])

		return losses

	def has_terminated(self):
		return self.termination_function()

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


def gd(params, d_params, lr, has_sparse_gradient):
	# first, don't implement foreach version
	# print("inside gd!",	 lr, params)
	for i, p in enumerate(params):
		# print(lr[i])
		if type(lr) == list:
			p.add_(d_params[i], alpha=-lr[i]) 
		else:
			p.add_(d_params[i], alpha=-lr)



