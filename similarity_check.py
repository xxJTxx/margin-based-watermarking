import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import mnist, cifar10, resnet, queries
#from torchsummary import summary
from loaders import get_cifar10_loaders, get_cifar100_loaders, get_svhn_loaders, get_mnist_loaders,get_cifar10_loaders_sub


# Custom Loss function that take into two models water_model and train_model, and return the mean squared error between output of certain layers
def relu_neu_similarity(clean_relu, trigger_relu):
        if len(clean_relu) != len(trigger_relu): 
            print('Relu length not matched')
            breakpoint() 
        else:
            for idx in range(len(clean_relu)):
                if clean_relu[idx].shape != trigger_relu[idx].shape:
                    print(f'Clean Relu {clean_relu[idx].shape} and Trigger Relu {trigger_relu[idx].shape} not matched')
                    breakpoint()
            
            # Create a list to store the absolute differences
            absolute_diff_list = []

            # Iterate over the tensors in both lists and calculate the absolute difference
            for tensor1, tensor2 in zip(clean_relu, trigger_relu):
                absolute_diff = torch.abs(tensor1 - tensor2)
                absolute_diff_list.append(absolute_diff)
            
            # Concatenate the tensors into a single tensor
            concatenated_tensor = torch.cat([tensor.flatten() for tensor in absolute_diff_list])

            # Calculate the mean of the concatenated tensor
            mean_value = torch.mean(concatenated_tensor.float())
            
            return 1-mean_value

# Neuron count
def relu_neu_count(water_relu, sample_num):
        w_denominator=water_relu.detach()       
        w_n = torch.where(water_relu !=0, water_relu/w_denominator, water_relu)
        
        return w_n/sample_num


# Similarity Check
def activated_neuron_similarity(dataset, subset_rate, water_model, device, query, excluded_index=None, layer_input=None):
            
    # Generate subset data loader based on dataset
    if dataset == 'cifar10':
        train_loader, val_loader, test_loader = get_cifar10_loaders_sub(subset_rate, excluded_index)
    elif dataset == 'cifar100':
        train_loader, val_loader, test_loader = get_cifar100_loaders()
    elif dataset == 'svhn':
        train_loader, val_loader, test_loader = get_svhn_loaders()
    
    # Record activated neurons of two kinds of data
    clean_activate_neuron_1 = []
    clean_activate_neuron_2 = []
    clean_activate_neuron_3 = []
    clean_activate_neuron_4 = []
    trigger_activate_neuron_1 = []
    trigger_activate_neuron_2 = []
    trigger_activate_neuron_3 = []
    trigger_activate_neuron_4 = []

    if layer_input is not None:    

        water_relu1 = []
        w_hooks1 = [] # list of hook handles, to be removed when you are done
        def water_hook1(module, input, output):
            if hook_flag:
                nonlocal water_relu1
                water_relu1.append(input)   
        
        water_relu2 = []
        w_hooks2 = [] # list of hook handles, to be removed when you are done
        def water_hook2(module, input, output):
            if hook_flag:
                nonlocal water_relu2
                water_relu2.append(input) 
                
        water_relu3 = []
        w_hooks3 = [] # list of hook handles, to be removed when you are done
        def water_hook3(module, input, output):
            if hook_flag:
                nonlocal water_relu3
                water_relu3.append(input) 
        
        water_relu4 = []
        w_hooks4 = [] # list of hook handles, to be removed when you are done
        def water_hook4(module, input, output):
            if hook_flag:
                nonlocal water_relu4
                water_relu4.append(input) 
        
        
        # Hook the function onto conv1 and conv2 of layer1~layer4 of both models.
        for idx in range(len(water_model[1].layer1)):
            for name, module in water_model[1].layer1[idx].named_children():
                if name in layer_input:
                    w_hooks1.append(getattr(water_model[1].layer1[idx], name).register_forward_hook(water_hook1))
        for idx in range(len(water_model[1].layer2)):
            for name, module in water_model[1].layer2[idx].named_children():
                if name in layer_input:
                    w_hooks2.append(getattr(water_model[1].layer2[idx], name).register_forward_hook(water_hook2))
        for idx in range(len(water_model[1].layer3)):
            for name, module in water_model[1].layer3[idx].named_children():
                if name in layer_input:
                    w_hooks3.append(getattr(water_model[1].layer3[idx], name).register_forward_hook(water_hook3))
        for idx in range(len(water_model[1].layer4)):
            for name, module in water_model[1].layer4[idx].named_children():
                if name in layer_input:
                    w_hooks4.append(getattr(water_model[1].layer4[idx], name).register_forward_hook(water_hook4))

        print(f"{len(w_hooks1)} and {len(w_hooks2)} and {len(w_hooks3)} and {len(w_hooks4)} layers of input are being recorded on water model.")
                    

    water_model.eval()
    
    debugger = []
    
    for batch_idx, batch in enumerate(test_loader):
        
        hook_flag = True
        # Reset the lists
        water_relu1 = []
        water_relu2 = []
        water_relu3 = []
        water_relu4 = []            
        
        images = batch[0]
        labels = batch[1].long()         
        images, labels = images.to(device), labels.to(device)
    
        with torch.no_grad():
            _ = water_model(images) 

        #breakpoint()
        # Add relu neuron into relu list
        for idx in range(len(water_relu1)):
            for sample_idx in range(len(water_relu1[idx][0])):
                if sample_idx == 0 and batch_idx == 0: # first smaple in the first batch should use append to create a item in list
                    clean_activate_neuron_1.append(relu_neu_count(water_relu1[idx][0][sample_idx],10000))
                    debugger.append(1)
                else: # otherwise will be summed up
                    #breakpoint()
                    clean_activate_neuron_1[idx] += relu_neu_count(water_relu1[idx][0][sample_idx],10000)
                    debugger[idx] += 1
                    
        for idx in range(len(water_relu2)):
            for sample_idx in range(len(water_relu2[idx][0])):
                if sample_idx == 0 and batch_idx == 0: # first smaple in the first batch should use append to create a item in list
                    clean_activate_neuron_2.append(relu_neu_count(water_relu2[idx][0][sample_idx],10000))
                else: # otherwise will be summed up
                    #breakpoint()
                    clean_activate_neuron_2[idx] += relu_neu_count(water_relu2[idx][0][sample_idx],10000)
                    
        for idx in range(len(water_relu3)):
            for sample_idx in range(len(water_relu3[idx][0])):
                if sample_idx == 0 and batch_idx == 0: # first smaple in the first batch should use append to create a item in list
                    clean_activate_neuron_3.append(relu_neu_count(water_relu3[idx][0][sample_idx],10000))
                else: # otherwise will be summed up
                    #breakpoint()
                    clean_activate_neuron_3[idx] += relu_neu_count(water_relu3[idx][0][sample_idx],10000)
                
        for idx in range(len(water_relu4)):
            for sample_idx in range(len(water_relu4[idx][0])):
                if sample_idx == 0 and batch_idx == 0: # first smaple in the first batch should use append to create a item in list
                    clean_activate_neuron_4.append(relu_neu_count(water_relu4[idx][0][sample_idx],10000))
                else: # otherwise will be summed up
                    #breakpoint()
                    clean_activate_neuron_4[idx] += relu_neu_count(water_relu4[idx][0][sample_idx],10000)
    

    #print(f"Debugger clean: {debugger}")
    debugger = []
    
    hook_flag = True
    # Reset the lists
    water_relu1 = []
    water_relu2 = []
    water_relu3 = []
    water_relu4 = []    
    images = query.to(device)
    with torch.no_grad():
        _ = water_model(images)    
    # Add relu neuron into relu list
    for idx in range(len(water_relu1)):
        for sample_idx in range(len(water_relu1[idx][0])):
            if sample_idx == 0 : # first smaple should use append to create a item in list
                trigger_activate_neuron_1.append(relu_neu_count(water_relu1[idx][0][sample_idx],len(query)))
                debugger.append(1)
            else: # otherwise will be summed up
                #torbreakpoint()
                trigger_activate_neuron_1[idx] += relu_neu_count(water_relu1[idx][0][sample_idx],len(query))
                debugger[idx] += 1
    
    for idx in range(len(water_relu2)):
        for sample_idx in range(len(water_relu2[idx][0])):
            if sample_idx == 0 : # first smaple in the first batch should use append to create a item in list
                trigger_activate_neuron_2.append(relu_neu_count(water_relu2[idx][0][sample_idx],len(query)))
            else: # otherwise will be summed up
                #breakpoint()
                trigger_activate_neuron_2[idx] += relu_neu_count(water_relu2[idx][0][sample_idx],len(query))                
    
    for idx in range(len(water_relu3)):
        for sample_idx in range(len(water_relu3[idx][0])):
            if sample_idx == 0 : # first smaple in the first batch should use append to create a item in list
                trigger_activate_neuron_3.append(relu_neu_count(water_relu3[idx][0][sample_idx],len(query)))
            else: # otherwise will be summed up
                #breakpoint()
                trigger_activate_neuron_3[idx] += relu_neu_count(water_relu3[idx][0][sample_idx],len(query))
                
    for idx in range(len(water_relu4)):
        for sample_idx in range(len(water_relu4[idx][0])):
            if sample_idx == 0 : # first smaple in the first batch should use append to create a item in list
                trigger_activate_neuron_4.append(relu_neu_count(water_relu4[idx][0][sample_idx],len(query)))
            else: # otherwise will be summed up
                #breakpoint()
                trigger_activate_neuron_4[idx] += relu_neu_count(water_relu4[idx][0][sample_idx],len(query))        
    
    #print(f"Debugger trigger: {debugger}")

    print(f"Layer1 similarity: {relu_neu_similarity(clean_activate_neuron_1, trigger_activate_neuron_1)}")
    print(f"Layer2 similarity: {relu_neu_similarity(clean_activate_neuron_2, trigger_activate_neuron_2)}")
    print(f"Layer3 similarity: {relu_neu_similarity(clean_activate_neuron_3, trigger_activate_neuron_3)}")
    print(f"Layer4 similarity: {relu_neu_similarity(clean_activate_neuron_4, trigger_activate_neuron_4)}")
    
    clean_activate_total = clean_activate_neuron_1 + clean_activate_neuron_2 + clean_activate_neuron_3 + clean_activate_neuron_4
    trigger_activate_total = trigger_activate_neuron_1 + trigger_activate_neuron_2 + trigger_activate_neuron_3 + trigger_activate_neuron_4
    print(f"Total layer similarity: {relu_neu_similarity(clean_activate_total, trigger_activate_total)}")
        
        
    
        
    # Remove hooks for model after used and reset the handle lists
    for handle in w_hooks1:
        handle.remove()
    for handle in w_hooks2:
        handle.remove()
    for handle in w_hooks3:
        handle.remove()
    for handle in w_hooks4:
        handle.remove()
    w_hooks1=[]
    w_hooks2=[]
    w_hooks3=[]
    w_hooks4=[]
           

def model_on_testset(test_model, test_loader, device):
    test_model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch[0]
            labels = batch[1].long()
            images, labels = images.to(device), labels.to(device)
            outputs = test_model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_loss = test_loss / len(test_loader.dataset)
        test_accuracy = correct / total
        print(f"Test Acc: {test_accuracy:.4f}")
    return test_accuracy

def model_on_queryset(test_model, query, response, device):
    test_model.eval()
    images, labels = query.to(device), response.to(device)
    
    with torch.no_grad():
        query_preds = test_model(images)
        query_acc = (query_preds.topk(1, dim=1).indices == labels.unsqueeze(1)).all(1).float().mean()
        print(f"Query Acc: {query_acc:.4f}")
    return query_acc


if __name__ == "__main__":
    ########################### Hyperparameters setting ###########################
    dataset = 'cifar10'
    subset_rate = 0.1 # 0~1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    layer_input = ['conv1','conv2'] # List of layers that input will be used as relu loss input
    layer_output = ['conv1'] # List of layers that output will be used as custom loss input
    ###############################################################################


    # Load the existing checkpoint
    checkpoint = torch.load('./experiments/cifar10_res18_margin_100_/checkpoints/checkpoint_query_best.pt') # C:/Users/Someone/margin-based-watermarking/experiments/cifar10_res34_margin_100_/checkpoints/checkpoint_query_best.pt
    # Preparation for loading
    CIFAR_QUERY_SIZE = (3, 32, 32) # input size
    response_scale = 10 # number of classes
    #query_size = CIFAR_QUERY_SIZE
    model_archive = cifar10.models

    # Load the model  structure from checkpoint
    train_model = model_archive[checkpoint['model']['type']](num_classes=response_scale)
    # Load the model weights
    train_model.load_state_dict(checkpoint['model']['state_dict'])
    # Load the optimizer from checkpoint
    opt = torch.optim.SGD(train_model.parameters(), lr=0.001)
    opt.load_state_dict(checkpoint['optimizer'])
    # Load the query from checkpoint and the index of them
    query = checkpoint['query_model']['state_dict']['query']
    query.to("cpu")
    query_indices = checkpoint['query_model']['index']
    # Load the response from checkpoint
    response = checkpoint['query_model']['state_dict']['response']
    # Load the original response from checkpoint
    original_response = checkpoint['query_model']['state_dict']['original_response']
    
    train_model.to(device)
    
    print(f"Nat acc:{checkpoint['val_nat_acc']}, Query acc:{checkpoint['val_query_acc']}")
    
    # Start training
    activated_neuron_similarity(dataset, subset_rate, train_model, device, query, query_indices, layer_input)
    
    breakpoint()
    
