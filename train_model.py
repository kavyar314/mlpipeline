import argparse, os, importlib

import dataload
import losses
import utils
import models

import torch

PATH_TO_CONFIG = './'
VERBOSE = True


def train_loop(dataloader, model, loss, optimizer, n_epochs, print_freq=1):
	for i in n_epochs:
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


def save_model(model, model_name, model_args, path):
	save_dict = {"model constructor": model_name,
				 "model arguments": model_args,
				 "parameters": model.get_parameters()}
	timestamp = str(datetime.datetime.now())
	save_name = f"{model_name}_{timestamp}.npy"
	np.save(os.path.join(path, save_name), save_dict)

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

	spec = importlib.util.spec_from_file_location(args.config, PATH_TO_CONFIG)
	configpy = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(configpy)
	batch_size = configpy.batch_size
	loss_args = configpy.loss_args
	model_args = configpy.model_args
	optimizer_args = configpy.optimizer_args
	n_epochs = configpy.n_epochs
	save_path = configpy.save_path

	if not os.isdir(save_path):
		os.mkdirs(save_path)

	try:
		preproc = configpy.preproc
	except:
		preproc = None

	if args.preproc == "VJ" and args.optimizer == "Boosting":
		## handle this case separately
		print("unimplemented")

	else:

		if VERBOSE:
			print("handling non-VJ case....")
			print("making train/test split")

		# make train / test split
		train_files, test_files = utils.train_test_split(args.data)

		## todo: implement class that extends Dataset so can use in DataLoader
		train_dataset = dataload.load_data_from_path(args.data, preproc, train_files)
		test_dataset = dataload.load_data_from_path(args.data, preproc, test_files)


		trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

		loss = losses.loss_selector(args.loss, loss_args)

		model = models.model_selector(args.model, model_args)

		optimizer = optimizers.optimizer_selection(args.optimizer, optimizer_args)(model.parameters()) 

		if VERBOSE:
			print("instantiated dataloader, loss, model, optimizer...")
			print("beginning training...")
		# signature of that which is returned by optimizer_selection is takes in model parameters and has the same interface as optim.SGD

		train_loop(trainloader, model, loss, optimizer, n_epochs)

		if VERBOSE:
			print("training complete...")

		score = eval_model(testloader, model)

		save_model(model, save_path)




