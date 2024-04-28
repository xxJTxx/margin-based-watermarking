import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import mnist, cifar10, resnet, queries
#from torchsummary import summary
from loaders import get_cifar10_loaders, get_cifar100_loaders, get_svhn_loaders, get_mnist_loaders,get_cifar10_loaders_sub
from similarity_check import activated_neuron_similarity

if __name__ == "__main__":
    ########################### Hyperparameters setting ###########################
    dataset = 'cifar10'
    subset_rate = 0.1 # 0~1
    epoch = 1
    main_loss_ratio = 1 # >=0
    new_loss_ratio = 0# >=0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    layer_input = ['conv2'] # List of layers that input will be used as relu loss input
    layer_output = ['conv1'] # List of layers that output will be used as custom loss input
    # Non-fixed loss ratio
    main_loss_scheduler = lambda t: np.interp([t],\
            [0  ,   5,     5,    8, 8, 40],\
            [0.8, 0.92, 0.94, 0.98, 1,  1])[0]
    ratio_type = 'fix' #'fix' or 'scheduler'
    main_loss_type = 'KDhard' # 'CE', 'KD', 'KDhard'
    opt_type = 'SGD' # 'Adam' or 'SGD'
    using_checkpoint = 0 #0:false, 1:true
    ###############################################################################

    # Set loss ratio type
    if ratio_type == 'fix':
        the_main_task_r = main_loss_ratio
    elif ratio_type == 'scheduler':
        the_main_task_r = main_loss_scheduler
    else:
        print("Choose the predifined type of ratio.")
        breakpoint()
    
    # Load the existing checkpoint
    checkpoint = torch.load('./experiments/cifar10_res34_margin_100_queryindices/checkpoints/checkpoint_query_best.pt') # C:/Users/Someone/margin-based-watermarking/experiments/cifar10_res34_margin_100_/checkpoints/checkpoint_query_best.pt
    # Preparation for loading
    CIFAR_QUERY_SIZE = (3, 32, 32) # input size
    response_scale = 10 # number of classes
    #query_size = CIFAR_QUERY_SIZE
    model_archive = cifar10.models

    # Load the model  structure from checkpoint
    train_model = model_archive[checkpoint['model']['type']](num_classes=response_scale)
    
    # Load the model weights
    if using_checkpoint:
        print("**********************************Start with checkpoint**********************************")
        train_model.load_state_dict(checkpoint['model']['state_dict'])
    else:
        print("Start with initialed weight")
    
    # Load the optimizer from checkpoint
    if opt_type == 'SGD':
        opt = torch.optim.SGD(train_model.parameters(), lr=0.1)
        opt.load_state_dict(checkpoint['optimizer'])
    elif opt_type == 'Adam':
        opt = torch.optim.Adam(train_model.parameters())
    else:
        print("Choose the predifined type of optimizer.")
        breakpoint()
    
    query_size = CIFAR_QUERY_SIZE
    
    query = queries.queries[checkpoint['query_model']['type']](query_size=(100, *query_size),
                                response_size=(100,), query_scale=255, response_scale=10)
    query.load_state_dict(checkpoint['query_model']['state_dict'], strict=False)
    
    
    # Load the query from checkpoint and the index of them
    queryset = checkpoint['query_model']['state_dict']['query']
    queryset.to(device)
    queryset_indices = checkpoint['query_model']['index']
    
    # Load the response from checkpoint
    response = checkpoint['query_model']['state_dict']['response']
    
    # Load the original response from checkpoint
    original_response = checkpoint['query_model']['state_dict']['original_response']

    query1, response1 = query(discretize=False, num_sample=25)
    for _ in range(5): # 看起來PGA的次數是寫死的?
        query1 = query.detach()
        #mask to limit the PGA only on the trigger part
        mask = np.zeros(query1.shape)
        mask[:, :, 0:4, 0:4] = 1
        query1.requires_grad_(True)
        query_preds = train_model(query)
        query_loss = F.cross_entropy(query_preds, response1)
        query_loss.backward()
        query1 = query1 + mask*query1.grad.sign() * (1/255) # PGA，步長也設定常數1/255...?
        query1 = query.project(query1)
        train_model.zero_grad()