import torchvision.models as models
import torchvision.transforms as transforms
from loader import ImagetNet, Val
from Normalize import Normalize
from torchvision.datasets import ImageFolder
import warnings
from universal_pert import universal_perturbation
warnings.filterwarnings("ignore")
import numpy as np
from torch.utils.data import DataLoader
import torch


epsilon = 10.0 / 255.0
training_data_path = ''
testing_data_path = ''
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# std = [1.0, 1.0, 1.0]
net = torch.nn.Sequential(Normalize(mean, std), models.inception_v3(pretrained=True).eval()).cuda()

transform = transforms.Compose([
    transforms.Resize((330, 330)),
    transforms.CenterCrop(299),
    transforms.ToTensor()])


print('loader data')
X = ImagetNet('../data/ILSVRC2012_train/pick_image/', 1000, 10, transforms = transform)

# X = torch.utils.data.DataLoader(
#     ImageFolder(training_data_path, transforms.Compose([
#         transforms.Resize((330, 330)),
#         transforms.CenterCrop(299),
#         transforms.ToTensor(),
#         ])),
#         batch_size = 1, shuffle=True,
#         pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    ImageFolder(testing_data_path, transforms = transform,
        batch_size = 50, shuffle=False,
        num_workers = 8, pin_memory=True)


print('Computation')
v = universal_perturbation(X, val_loader, net, epsilon)








