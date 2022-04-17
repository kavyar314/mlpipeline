import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode


EXCLUDE = [".DS_Store"]

def load_data_from_path(data_path, preproc, selector):
	return DatasetFromPath(data_path, preproc, selector)

class DatasetFromPath(Dataset):
	'''
	class to construct a Pytorch Dataset when passed a path that contains folders for 
	each class. e.g., if path = './' the folder structure may look like:
	./
	|__ class1
	    |__ img1.jpg
	    |__ img2.jpg
	    |__ ...
	|__ class2
	    |__ img1.jpg
	    |__ img2.jpg
	    |__ ...
	|__ class3
	    |__ img1.jpg
	    |__ img2.jpg
	    |__ ...
	|__ ...
	'''
	def __init__(self, path, transform, selector):
		'''
		selector: list of names of files that belong to the desired split
		'''
		self.classes = [c for c in os.listdir(path) if c not in EXCLUDE]
		self.class_dict = {}
		self.ys = []
		self.xs = []
		for i, c in enumerate(self.classes):
			xs_c = [fname for fname in os.listdir(os.path.join(path, c)) if fname in selector]

			self.xs += xs_c
			self.ys += [i]*len(xs_c)
			self.class_dict[i] = c
		self.transform = transform
		self.imgs_path = path
		

	def __len__(self):
		return len(self.ys)


	def __getitem__(self, idx):
		img_path = os.path.join(os.path.join(self.imgs_path, self.class_dict[self.ys[idx]]), self.xs[idx])
		img_label = self.ys[idx]
		img = read_image(img_path, ImageReadMode.RGB)
		if self.transform:
			img = self.transform(img)

		return img, img_label