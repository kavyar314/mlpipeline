import torch.optim as optim


def optimizer_selector(optimizer_name, optimizer_args):
	if optimizer_name == "torch_sgd":
		return lambda modelparams: optim.SGD(modelparams,lr=optimizer_args["lr"], momentum=optimizer_args["momentum"])
	else:
		return None # raise error here?


