
import os, random

def train_test_split(path, split=0.8):
	classes = os.listdir(path)
	files = []
	for c in classes:
		files += os.listdir(os.path.join(path, c))

	random.shuffle(files)

	return files[:int(split*len(files))], files[int(split*len(files))+1:]