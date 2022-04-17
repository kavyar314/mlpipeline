
import os, random

EXCLUDE = ['.DS_Store']

def train_test_split(path, n_samples=None, split=0.8):
	'''
	splits all the files found in the dataset into train and test splits

	Arguments
	path: path leading to the top-level directory consisting of: one subfolder per class, each containing all the images of that class
	n_samples: None if all images are to be used for either train or test; number caps the total number in the train and test splits
	split: proportion in [0, 1] to give to the training set. Defaults to 0.8
	'''
	classes = [c for c in os.listdir(path) if c not in EXCLUDE]
	files = []
	for c in classes:
		files += os.listdir(os.path.join(path, c))

	files = [f for f in files if f not in EXCLUDE]

	random.shuffle(files)

	upper_limit = n_samples if n_samples is not None else len(files)

	return files[:int(split*upper_limit)], files[int(split*upper_limit)+1:upper_limit]