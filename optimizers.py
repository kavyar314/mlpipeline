import torch.optim as optim
import torch
import numpy as np

import matplotlib.pyplot as plt

# def get_dict_attr(optimizer_args):
# 	all_args = ["lr", "momentum", "train loss closure", "validation loss closure", "termination", "logs"]
# 	lr, momentum, train_loss_fn, val_loss_fn, lr, termination, log_names = None, None, None, None, None, None, None
# 	argument_vars = [lr, momentum, train_loss_fn, val_loss_fn, lr, termination, log_names]
# 	set_args = []
# 	for i, arg in enumerate(all_args):
# 		if arg in optimizer_args:
# 			set_args.append()
# 	# I don't think this is going to work...

def optimizer_selector(optimizer_name, optimizer_args):
	'''
	returns optimzer to use based on the name (string) and any arguments required for construction

	Arguments:
		optimizer_name: string describing which optimizer to use
		optimizer_args: dictionary of required arguments for that optimizer (learning rate, momentum, etc)
	'''
	if optimizer_name == "torch_sgd":
		return lambda modelparams: optim.SGD(modelparams,lr=optimizer_args["lr"], momentum=optimizer_args["momentum"])
	elif optimizer_name == "logged_gd":
		return lambda modelparams: LoggedGradientDescent(modelparams, optimizer_args['train loss closure'], 
													optimizer_args['validation loss closure'], optimizer_args["path for custom"])
	else:
		return None # raise error here?


class OptimizerInstanceFields():
	'''
	collects instance fields necessary for optimizer

	Attributes:
		train_losses: list of train losses over epochs
		validation_losses: list of validation losses over epochs
		epochs_transpired: number of epochs completed
		last_update_norm: the norm of the most recent update to the weights
	'''
	def __init__(self):
		self.train_losses = [np.inf]
		self.validation_losses = [np.inf]
		self.epochs_transpired = 0
		self.last_update_norm = np.inf


class OptimizerTermination(OptimizerInstanceFields):
	'''
	class to specify termination condition

	Attributes:
		inherits from OptimizerInstanceFields

	Methods:
		loss_converge: checks if the loss of the most recent set of weights is small
		averaged_losses_converge: checks if the smoothed losses converge
		epochs_over: check if the desired number of epochs have finished
		last_update_norm_small: checks if the most recent update to the weights has small norm
	'''
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
		smoothed_losses = [np.mean(losses[i:i+smoothing]) for i in range(len(losses)-smoothing+1)]
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
	'''
	implements Gradient Descent as a pytorch optimizer that also has custom termination options and logs

	Attributes:
		inherits from OptimizerInstanceFields along with the additional ones below
		termination_function: chooses the appropriate termination function
		terminal_value: at what value of the desired parameter to terminate
		logs: which logs to include
		terminated: whether or not optimization is complete
		train_loss_fn: function to compute train loss
		val_loss_fn: function to compute validation loss
	'''
	def __init__(self, parameters, train_loss_fn, val_loss_fn, path_for_custom=None, lr=0.001, termination=("train loss", 1e-4), log_names=[]):
		'''
		termination: tuple of name of termination condition and 
		train_loss_fn: closure that already incorporates train data
		val_loss_fn: closure that already incorporates val data
		'''
		# need kwargs to handle `plot` variable for Logs
		optim.Optimizer.__init__(self, params=parameters, defaults={"lr": lr})
		OptimizerTermination.__init__(self)
		Logs.__init__(self)
		## train loss and validation loss functions need to be passed in, since they can only be instantiated after the data is loaded
		self.train_loss_fn = train_loss_fn
		self.val_loss_fn = val_loss_fn
		self.terminated = False
		if path_for_custom is not None:
			spec = importlib.util.spec_from_file_location(path_for_custom, os.path.join("./", path_for_custom))
			attributes = importlib.util.module_from_spec(spec)
			spec.loader.exec_module(attributes)
			self.termination_function = self.termination_selector(attributes.termination[0])
			self.terminal_value = attributes.termination[1]
			self.logs = [self.logs_selector(log_name) for log_name in attributes.log_names] # TODO: create a Log class that updates itself according to its instantiating function when .update() is called
		else:
			self.termination_function = self.termination_selector(termination[0])
			self.terminal_value = termination[1]
			self.logs = [self.logs_selector(log_name) for log_name in log_names] # TODO: create a Log class that updates itself according to its instantiating function when .update() is called

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



