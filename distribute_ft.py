
import torch
from models import mnist, cifar10, resnet, queries
from torchsummary import summary
from loaders import get_cifar10_loaders, get_cifar100_loaders, get_svhn_loaders, get_mnist_loaders

CIFAR_QUERY_SIZE = (3, 32, 32)
response_scale = 10

# Load the existing checkpoint
checkpoint = torch.load('./experiments/cifar10_res34_margin_100_/checkpoints/checkpoint_query_best.pt') # C:/Users/Someone/margin-based-watermarking/experiments/cifar10_res34_margin_100_/checkpoints/checkpoint_query_best.pt

# Preparation for loading
query_size = CIFAR_QUERY_SIZE
model_archive = cifar10.models
train_loader, valid_loader, test_loader = get_cifar10_loaders()

# Load the model  structure from checkpoint
water_model = model_archive[checkpoint['model']['type']](num_classes=response_scale)
# Load the model weights
water_model.load_state_dict(checkpoint['model']['state_dict'])

# Load the optimizer from checkpoint
opt = torch.optim.SGD(water_model.parameters(), lr=0.1)
opt.load_state_dict(checkpoint['optimizer'])

