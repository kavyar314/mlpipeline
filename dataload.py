import os
from torchvision.io import read_image
from torch.utils.data import Dataset



# transform = transforms.Compose(
#     						[transforms.ToTensor(),
#      						transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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
		self.classes = os.listdir(path)
		self.class_dict = {}
		self.ys = []
		self.xs = []
		for i, c in enumerate(self.classes):
			xs_c = os.listdir(os.path.join(path, c))
			self.xs += [fname for fname in xs_c if fname in selector]
			self.ys += [i]*len(self.xs)
			self.class_dict[i] = c

		self.transform = transform
		self.imgs_path = path
		

	def __len__(self):
		return len(self.ys)


	def __getitem__(self, idx):
		img_path = os.path.join(os.path.join(self.imgs_path, self.class_dict[self.ys[idx]]), self.xs[idx])
		img_label = self.ys[idx]
		img = read_image(img_path)
		if self.transform:
			img = self.transform(img)

		return img, img_label