
import os, random
import numpy as np
from skimage.io import imread
from skimage import transform

EXCLUDE = ['.DS_Store']

def train_test_split(path, n_samples=None, split=0.8, include_classes=False):
	'''
	splits all the files found in the dataset into train and test splits

	Arguments:
		path: path leading to the top-level directory consisting of: one subfolder per class, each containing all the images of that class
		n_samples: None if all images are to be used for either train or test; number caps the total number in the train and test splits
		split: proportion in [0, 1] to give to the training set. Defaults to 0.8
	'''
	classes = [c for c in os.listdir(path) if c not in EXCLUDE]
	files = []
	for c in classes:
		files += [(f, c) for f in os.listdir(os.path.join(path, c))]

	files = [f for f in files if f not in EXCLUDE]

	random.shuffle(files)

	upper_limit = n_samples if n_samples is not None else len(files)
	if not include_classes:
		return [x[0] for x in files[:int(split*upper_limit)]], [x[0] for x in files[int(split*upper_limit)+1:upper_limit]]
	else:
		return files[:int(split*upper_limit)], files[int(split*upper_limit)+1:upper_limit]

def load_specified_files_from_path(data_path, specs, img_dim=19):
	'''
	loads the paths specified in "specs" to images, makes the images greyscale, and resizes them to "img_dim"

	Arguments:
		data_path: the path where the files live, with one subfolder for each class, inside which all the image files live
		specs: list of (fname, class) tuples to load from data_path / class / fname
		img_dim: dimension to which to resize these images
	'''
	X = []
	for fname, cname in specs:
		x = imread(os.path.join(os.path.join(data_path, cname), fname))
		X.append(grey_scale_resize([x], img_dim))
	classes = [s[1] for s in specs]
	class_numbers = dict(zip(list(set(classes)), range(len(set(classes)))))
	return np.array(X), [class_numbers[c] for c in classes]



def load_raw_data(pos_path="./data/faces_selected/", neg_path="./data/not_faces_selected/", cap_class=np.inf):
	'''
	loads positive and ngative data samples from path

	Arguments:
		pos_path: path to positive samples
		neg_path: path to negative samples
		cap_lcass: the number of samples per class to cap at. defaults to infinite
	'''
    paths = [pos_path, neg_path]
    classes = os.lisd
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
	'''
	takes in array of images and returns an array of the same images greyscaled and resized

	Arguments:
		images: array of images to preprocess
		img_dim: the dimension to which to resize
	'''
    output_images = []
    for img in tqdm(images):
        if len(img.shape)>2:
            grey_scale_img = np.mean(img, axis=2)
        else:
            grey_scale_img = img
        same_size_img = transform.resize(grey_scale_img, (img_dim, img_dim))
        output_images.append(same_size_img)
    return np.array(output_images)
