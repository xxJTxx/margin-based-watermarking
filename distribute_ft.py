
import torch
from models import mnist, cifar10, resnet, queries

# Load the existing checkpoint
checkpoint = torch.load('C:/Users/Someone/margin-based-watermarking/experiments/cifar10_res34_margin_100_/checkpoints/checkpoint_query_best.pt')

# Initialize your model with the checkpoint weights
model = YourModel.load_from_checkpoint(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Initialize your optimizer with the checkpoint weights
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


