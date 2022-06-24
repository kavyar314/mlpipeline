import torchvision.transforms as transforms
import torch


batch_size = 4
loss_args = None
model_args = None
optimizer_args = {"lr": 0.001, "momentum":0.9, "path for custom": "./optimizerconfig.py"}
n_epochs = 1
save_path = "./test"


preproc = transforms.Compose(
    						[transforms.Resize((224, 224)),
    						transforms.ConvertImageDtype(torch.float32),
     						transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

img_dim = 19

n_learners = 10