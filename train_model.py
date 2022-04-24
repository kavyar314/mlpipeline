import argparse, os, importlib, datetime
import numpy as np

import dataload
import losses
import utils
import models
import optimizers

import torch

PATH_TO_CONFIG = './'
VERBOSE = True
n_samples = 100


def train_loop(dataloader, model, loss, optimizer, n_epochs, print_freq=1):
	'''
	training loop function trains model on data using loss and optimizer as specified for the desired number of epochs

	Arguments
	dataloader: type pytorch DataLoader containing the data on which to be trained
	model: type pytorch Module containing architecture and parameters
	loss: pytorch Module with loss function implemented
	optimizer: pytorch Optimizer capable of optimizer.step() to update model parameters
	n_epochs: number of epochs for which to train
	print_freq: how often to log the loss.
	'''
	for i in range(n_epochs):
		print("training: epoch %d" % i)
		epoch_loss = 0

		for (X,y) in dataloader:
			preds = model(X)
			loss_param = loss(preds, y)
			
			loss_param.backward()
			optimizer.step()

			loss_val = loss_param.item() * X.shape[0]
			epoch_loss += loss_val
		avg_epoch_loss = epoch_loss / len(dataloader.dataset)
		print(f"training loss at epoch {i}: {avg_epoch_loss}")

def custom_train_loop(dataloader, model, loss, optimizer, n_epochs, print_freq=1):
	# runs with a while loop -- add a "terminate" flag in optimizer and check it each time
	while optimizer.has_terminated() is not True:
		print("training: epoch %d" % i)
		epoch_loss = 0
		# take actions
		for (X,y) in dataloader:
			preds = model(X)
			loss_param = loss(preds, y)
			
			loss_param.backward()
			optimizer.step()

			loss_val = loss_param.item() * X.shape[0]
			epoch_loss += loss_val
		avg_epoch_loss = epoch_loss / len(dataloader.dataset)
		print(f"training loss at epoch {i}: {avg_epoch_loss}")

def eval_model(dataloader, model, loss):
	correct = 0
	total_test_loss = 0
	total = 0
	for (X, y) in dataloader:
		preds = model(X)
		test_loss = loss(preds, y).item()
		total_test_loss += test_loss*X.shape[0]
		correct += (preds.argmax(1) == y).sum().item()
		total += X.shape[0]
	score = correct/total
	avg_test_loss = total_test_loss/total
	print(f"score: {score},\t average test loss: {avg_test_loss}")
	return score, avg_test_loss

def save_model(model, model_name, dataset, model_args, path):
	## might make sense to save the model just using pytorch presets and then have a separate condition for the VJ with Boosting
	if model_name == "VJ":
		print("not implemented yet...")
	else:
		timestamp = str(datetime.datetime.now())
		save_name = f"{model_name}_{dataset}_{timestamp}.npy"
		torch.save(model, os.path.join(path, save_name))

def load_model(path):
	model_dict = np.load(path)

	model = models.model_selector(model_dict["model constructor"], model_dict["model arguments"])
	model.set_parameters(model_dict["parameters"])

	return model

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data")
	parser.add_argument("--preproc")
	parser.add_argument("--model", required=True)
	parser.add_argument("--loss", required=True)
	parser.add_argument("--optimizer", required=True)
	parser.add_argument("--config", required=True)

	args = parser.parse_args()

	print(args.data, args.preproc, args.model, args.optimizer, args.config)

	if VERBOSE:
		print("loading config....")

	spec = importlib.util.spec_from_file_location(args.config, os.path.join(PATH_TO_CONFIG, args.config))
	configpy = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(configpy)
	batch_size = configpy.batch_size
	loss_args = configpy.loss_args
	model_args = configpy.model_args
	optimizer_args = configpy.optimizer_args
	n_epochs = configpy.n_epochs
	save_path = configpy.save_path

	if not os.path.isdir(save_path):
		os.makedirs(save_path)

	try:
		preproc = configpy.preproc
	except:
		preproc = None

	if args.preproc == "VJ" and args.optimizer == "Boosting":
		## handle this case separately
		# load data into files
		# compute the integral images
		# run Boosting with V-J weak classifiers
		# score on test set
		# save model
		print("unimplemented")

	else:

		if VERBOSE:
			print("handling non-VJ case....")
			print("making train/test split\n\n")

		# make train / test split
		train_files, test_files = utils.train_test_split(args.data, n_samples)

		## todo: implement class that extends Dataset so can use in DataLoader
		train_dataset = dataload.load_data_from_path(args.data, preproc, train_files)
		test_dataset = dataload.load_data_from_path(args.data, preproc, test_files)


		trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

		loss = losses.loss_selector(args.loss, loss_args)
		model = models.model_selector(args.model, model_args)
		optimizer = optimizers.optimizer_selector(args.optimizer, optimizer_args)(model.parameters()) 

		if VERBOSE:
			print("instantiated dataloader, loss, model, optimizer...")
			print("beginning training...\n\n")

		if "torch" in args.optimizer:
			# signature of that which is returned by optimizer_selection is takes in model parameters and has the same interface as optim.SGD
			train_loop(trainloader, model, loss, optimizer, n_epochs)
		else:
			custom_train_loop(trainloader, model, loss, optimizer, n_epochs)

		if VERBOSE:
			print("training complete...")

		score, test_loss = eval_model(testloader, model, loss)

		dataset_name = args.data.split('/')[-1]

		save_model(model, args.model, dataset_name, model_args, save_path)




