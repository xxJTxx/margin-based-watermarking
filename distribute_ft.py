import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import mnist, cifar10, resnet, queries
from torchsummary import summary
from loaders import get_cifar10_loaders, get_cifar100_loaders, get_svhn_loaders, get_mnist_loaders,get_cifar10_loaders_sub

# Custom Loss function that take into two models water_model and train_model, and return the mean squared error between output of certain layers
class custom_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, water_relu, train_relu):
        return torch.mean(torch.pow((water_relu - train_relu), 2))

# Hyperparameters setting 
dataset = 'cifar10'
subset_rate = 0.1
epoch = 10


CIFAR_QUERY_SIZE = (3, 32, 32) # input size
response_scale = 10 # number of classes

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
# Load the query from checkpoint
query = checkpoint['query_model']['state_dict']['query']
# Load the response from checkpoint
response = checkpoint['query_model']['state_dict']['response']
# Load the original response from checkpoint
original_response = checkpoint['query_model']['state_dict']['original_response']

# Create a same model for training with deep copy
train_model = copy.deepcopy(water_model)
train_model.to("cuda:0")
water_model.to("cuda:0")

# Record the result of neurons from every bn1 and shortcut layer
water_relu = []
train_relu = []

# Define the hook function
w_hooks = [] # list of hook handles, to be removed when you are done
t_hooks = []
def water_hook(module, input, output):
    water_relu.append(input)   
def train_hook(module, input, output):
    train_relu.append(input)


# Hook the function onto conv1 and conv2 of layer1~layer4 of both models.
for idx in range(len(water_model[1].layer1)):
    for name, module in water_model[1].layer1[idx].named_children():
        if name in ['conv1','conv2']:
            w_hooks.append(getattr(water_model[1].layer1[idx], name).register_forward_hook(water_hook))
""" for idx in range(len(water_model[1].layer2)):
    for name, module in water_model[1].layer2[idx].named_children():
        if name in ['conv1','conv2']:
            w_hooks.append(getattr(water_model[1].layer2[idx], name).register_forward_hook(water_hook))
for idx in range(len(water_model[1].layer3)):
    for name, module in water_model[1].layer3[idx].named_children():
        if name in ['conv1','conv2']:
            w_hooks.append(getattr(water_model[1].layer3[idx], name).register_forward_hook(water_hook))
for idx in range(len(water_model[1].layer4)):
    for name, module in water_model[1].layer4[idx].named_children():
        if name in ['conv1','conv2']:
            w_hooks.append(getattr(water_model[1].layer4[idx], name).register_forward_hook(water_hook)) """

for idx in range(len(train_model[1].layer1)):
    for name, module in train_model[1].layer1[idx].named_children():
        if name in ['conv1','conv2']:
            t_hooks.append(getattr(train_model[1].layer1[idx], name).register_forward_hook(train_hook))
""" for idx in range(len(train_model[1].layer2)):
    for name, module in train_model[1].layer2[idx].named_children():
        if name in ['conv1','conv2']:
            t_hooks.append(getattr(train_model[1].layer2[idx], name).register_forward_hook(train_hook))
for idx in range(len(train_model[1].layer3)):
    for name, module in train_model[1].layer3[idx].named_children():
        if name in ['conv1','conv2']:
            t_hooks.append(getattr(train_model[1].layer3[idx], name).register_forward_hook(train_hook))
for idx in range(len(train_model[1].layer4)):
    for name, module in train_model[1].layer4[idx].named_children():
        if name in ['conv1','conv2']:
            t_hooks.append(getattr(train_model[1].layer4[idx], name).register_forward_hook(train_hook))
 """ 

""" # Create testing sample
testing_sample = torch.randn(1, 3, 32, 32).to("cuda:0")
test_label = torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]).to("cuda:0") """


# Generate data loader based on dataset
if dataset == 'cifar10':
    train_loader, valid_loader, test_loader = get_cifar10_loaders_sub(0.1)
elif dataset == 'cifar100':
    train_loader, valid_loader, test_loader = get_cifar100_loaders()
elif dataset == 'svhn':
    train_loader, valid_loader, test_loader = get_svhn_loaders()


best_val_nat_acc = 0
best_val_query_acc = 0

# Remove hooks for model after used and reset the handle lists
for handle in w_hooks:
    handle.remove()
for handle in t_hooks:
    handle.remove()
w_hooks=[]
t_hooks=[]

# Sum up the loss of conv1 and conv2 of layer1~layer4 of both models
new_loss_func = custom_loss()
new_loss = 0
for idx in range(len(water_relu)):
    new_loss += new_loss_func(water_relu[idx][0], train_relu[idx][0])