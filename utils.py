
import os, random
import numpy as np

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

def load_raw_data(pos_path="./data/faces_selected/", neg_path="./data/not_faces_selected/", cap_class=np.inf):
	## TODO: make the signature less specific to the two classes etc
    paths = [pos_path, neg_path]
    pos_img_path_list = os.listdir(pos_path)
    neg_img_path_list = os.listdir(neg_path)
    np.random.shuffle(pos_img_path_list)
    np.random.shuffle(neg_img_path_list)
    positives = []
    negatives = []
    img_types = [positives, negatives]
    for i, img_list in enumerate([pos_img_path_list, neg_img_path_list]):
        for j, img_fname in tqdm(enumerate(img_list)):
            if img_fname == ".DS_Store":
                continue
            if j > cap_class:
                break
            img = mpimg.imread(os.path.join(paths[i], img_fname))
            img_types[i].append(img)
    y = np.hstack((np.ones(len(positives)), -np.ones(len(negatives))))
    X_raw = positives + negatives
    idxs = np.random.permutation(range(y.shape[0]))
    return [X_raw[i] for i in idxs], y[idxs] ## returns list of raw images and their labels

def greyscale_and_resize(images, img_dim=64):
    output_images = []
    for img in tqdm(images):
        if len(img.shape)>2:
            grey_scale_img = np.mean(img, axis=2)
        else:
            grey_scale_img = img
        same_size_img = transform.resize(grey_scale_img, (img_dim, img_dim))
        output_images.append(same_size_img)
    return np.array(output_images)
