import argparse, os, importlib, datetime
import numpy as np

import dataload
import losses
import utils
import models
import optimizers
import violajones

import torch

PATH_TO_CONFIG = './'
VERBOSE = True
n_samples = 100


def train_loop(dataloader, model, loss, optimizer, n_epochs, print_freq=1):
	'''
	training loop function trains model for a specified number of epochs on data 
	using loss and optimizer as specified

	Arguments:
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

def custom_train_loop(dataloader, model, loss, optimizer, print_freq=1):
	'''
	custom train loop trains a model with a custom termination condition in terms of the 
	optimization. Optimizer termination condition can be set through "optimizer args" in the selector function

	Arguments:
		dataloader: type pytorch DataLoader containing the data on which to be trained
		model: type pytorch Module containing architecture and parameters
		loss: pytorch Module with loss function implemented
		optimizer: pytorch Optimizer capable of optimizer.step() to update model parameters
		print_freq: how often to log the loss.
	'''
	# runs with a while loop -- add a "terminate" flag in optimizer and check it each time
	while optimizer.terminated is false:
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
	'''
	evaluates model on a test dataset and returns the score (0-1 loss) and the average test loss for the given loss function

	Arguments:
		dataloader: type pytorch DataLoader containing the data on which to be tested
		model: type pytorch Module containing architecture and parameters
		loss:  
	'''
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

def save_model(model, model_name, dataset, path):
	'''
	saves model to path with name that includes the name of the model, the dataset, and the timestamp. 

	##TODO: should this include optimizer, either in name or in arguments?

	Arguments:
		model: model to be saved. if it is not a torch model, it needs a function called "get_save_attributes()" that returns the dictionary to be saved
		model_name (str): name of the model
		dataset (str): name of the dataset that the model was trained on
		path: location at which to save the model 
	'''
	## might make sense to save the model just using pytorch presets and then have a separate condition for the VJ with Boosting
	timestamp = str(datetime.datetime.now())
	save_name = f"{model_name}_{dataset}_{timestamp}.npy"
	if model_name == "VJ":
		# print("not implemented yet...")
		save_dict = model.get_save_attributes()
		np.save(save_name, save_dict)
	else:
		torch.save(model, os.path.join(path, save_name))

def load_model(path):
	'''
	loads model from path if non-torch. requires model to hav
	## todo: I think this is actually useless b/c the Torch ones need something different, and the Boosting one can be directly instantiated from path
	'''
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

	try:
		img_dim = configpy.img_dim
	except:
		img_dim = None

	try:
		n_learners = configpy.n_learners
	except:
		n_learners = None

	if args.preproc == "VJ" and args.optimizer == "Boosting":
		## handle this case separately
		# load data into files
		# compute the integral images
		# run Boosting with V-J weak classifiers
		# score on test set
		# save model
		if not (os.path.isdir("./vj-data/faces") and os.path.isdir("./vj-data/not_faces")):
			print("data not available")
			return
		if img_dim is None or n_learners is None:
			print("use config that allows for V-J")
			return
		train_files, test_files = utils.train_test_split(args.data, n_sampes, include_classes=True)
		X_train, y_train = utils.load_specified_files_from_path(args.data, train_files, img_dim=img_dim)
		X_test, y_test = utils.load_specified_files_from_path(args.data, test_files, img_dim=img_dim)

		phi_X = violajones.compute_integral_img(X_train)

		boosted_clf_vj = AdaBoost(violajones.V_J_weak, n_learners=n_learners)
		boosted_clf_vj.fit(phi_X, y_train)

		test_phi_X = compute_integral_img(X_test)

		yvj_test_predict = boosted_clf_vj.predict(test_phi_X)
		# eval_model() can I use this here somehow?
		test_accuracy = np.mean(y_test == yvj_test_predict)

		save_model(model, args.model, dataset_name, save_path)

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
			custom_train_loop(trainloader, model, loss, optimizer)

		if VERBOSE:
			print("training complete...")

		score, test_loss = eval_model(testloader, model, loss)

		dataset_name = args.data.split('/')[-1]

		save_model(model, args.model, dataset_name, save_path)




