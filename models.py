import torchvision.models as models


def model_selector(model_name, model_args):
	if model_name == 'vgg16':
		return models.vgg16()
	else:
		return None # raise error is better here