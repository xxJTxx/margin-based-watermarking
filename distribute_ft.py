import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import mnist, cifar10, resnet, queries
#from torchsummary import summary
from loaders import get_cifar10_loaders, get_cifar100_loaders, get_svhn_loaders, get_mnist_loaders,get_cifar10_loaders_sub
from similarity_check import activated_neuron_similarity

# Custom Loss function that take into two models water_model and train_model, and return the mean squared error between output of certain layers
def relu_neu_loss(water_relu, train_relu):
        w_denominator=water_relu.detach()
        t_denominator=train_relu.detach()       
        w_n = torch.where(water_relu !=0, water_relu/w_denominator, water_relu)
        t_n = torch.where(train_relu !=0, train_relu/t_denominator, train_relu)
        return torch.mean(torch.pow((w_n - t_n), 2))

# Knowledge Distillation Loss    
def loss_fn_kd(outputs, labels, teacher_outputs, alpha, temperature):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = temperature
    labels = torch.tensor(labels, dtype=torch.long)
    KD_loss = nn.KLDivLoss().cuda(device)(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels).cuda(device) * (1. - alpha)

    return KD_loss

# Custom Loss function
def custom_loss1(water_model, train_model, trigger=1e-3):
        # Denominator take abs value to maintain the sign
        w_denominator=torch.abs(water_model.detach())
        t_denominator=torch.abs(train_model.detach()) 
        
        # Creating a new tensor where negative/positive values are changed to -1/1, wheras 0 becomes 0+trigger
        water_one = torch.FloatTensor(water_model.size()).type_as(water_model)
        mask_w = water_model!=0
        water_one[mask_w] = water_model[mask_w]/w_denominator[mask_w]
        mask_w = water_model==0
        water_one[mask_w] = water_model[mask_w] + trigger

        # Creating a new tensor where negative/positive values are changed to -1/1, wheras 0 remains 0
        mask = train_model==0
        train_one = torch.FloatTensor(train_model.size()).type_as(train_model)
        train_one[mask] = train_model[mask]
        mask = train_model!=0
        train_one[mask] = train_model[mask]/t_denominator[mask]
        
        # To give the right direction of gradient when the value from fix tensor is 0
        water_one[mask_w] = (torch.sign(train_one[mask_w])-1)*water_one[mask_w]

       
        ''' for idx in range(len(train_one)):
          print(f"w_o:{water_one[idx]}~~~t_o:{train_one[idx].item(),}") '''
        return torch.mean(torch.pow((1 + water_one*train_one), 2))


# Define the training loop
def start_train_kd_loss1(dataset, subset_rate, train_model, water_model, optimizer, device, query, response, original_response, num_epochs=10, new_loss_r=0, default_loss_r=1, excluded_index=None, layer_output=None, layer_input=None):
    
    for epoch in range(num_epochs):
        
        if epoch == 0:
            
            # Generate subset data loader based on dataset
            if dataset == 'cifar10':
                train_loader, val_loader, test_loader = get_cifar10_loaders_sub(subset_rate, excluded_index)
            elif dataset == 'cifar100':
                train_loader, val_loader, test_loader = get_cifar100_loaders()
            elif dataset == 'svhn':
                train_loader, val_loader, test_loader = get_svhn_loaders()
            
            # Record accuaracy after every epoch
            water_test_acc = []
            train_test_acc = []
            water_query_acc = []
            train_query_acc = []
            neuron_loss_after_epoch = [] 
            task_loss_after_epoch = [] 
            # TO CHECK RELU RESULT
            relu_neuron_loss_after_epoch =[]
            
            if layer_output is not None:
                # Record the input of every hooked layer
                water_relu = []
                train_relu = []            
                # Define the hook function
                w_hooks = [] # list of hook handles, to be removed when you are done
                t_hooks = []
                hook_flag = False
                def water_hook(module, input, output):
                    if hook_flag:
                        nonlocal water_relu
                        water_relu.append(output)   
                def train_hook(module, input, output):
                    if hook_flag:
                        nonlocal train_relu
                        train_relu.append(output)

                # Hook the function onto conv1 and conv2 of layer1~layer4 of both models.
                """ for idx in range(len(water_model[1].layer1)):
                    for name, module in water_model[1].layer1[idx].named_children():
                        if name in layer_output:
                            w_hooks.append(getattr(water_model[1].layer1[idx], name).register_forward_hook(water_hook)) """
                """ for idx in range(len(water_model[1].layer2)):
                    for name, module in water_model[1].layer2[idx].named_children():
                        if name in layer_output:
                            w_hooks.append(getattr(water_model[1].layer2[idx], name).register_forward_hook(water_hook)) """
                for idx in range(len(water_model[1].layer3)):
                    for name, module in water_model[1].layer3[idx].named_children():
                        if name in layer_output:
                            w_hooks.append(getattr(water_model[1].layer3[idx], name).register_forward_hook(water_hook))
                for idx in range(len(water_model[1].layer4)):
                    for name, module in water_model[1].layer4[idx].named_children():
                        if name in layer_output:
                            w_hooks.append(getattr(water_model[1].layer4[idx], name).register_forward_hook(water_hook))
            
                """ for idx in range(len(train_model[1].layer1)):
                    for name, module in train_model[1].layer1[idx].named_children():
                        if name in layer_output:
                            t_hooks.append(getattr(train_model[1].layer1[idx], name).register_forward_hook(train_hook)) """
                """ for idx in range(len(train_model[1].layer2)):
                    for name, module in train_model[1].layer2[idx].named_children():
                        if name in layer_output:
                            t_hooks.append(getattr(train_model[1].layer2[idx], name).register_forward_hook(train_hook)) """
                for idx in range(len(train_model[1].layer3)):
                    for name, module in train_model[1].layer3[idx].named_children():
                        if name in layer_output:
                            t_hooks.append(getattr(train_model[1].layer3[idx], name).register_forward_hook(train_hook))
                for idx in range(len(train_model[1].layer4)):
                    for name, module in train_model[1].layer4[idx].named_children():
                        if name in layer_output:
                            t_hooks.append(getattr(train_model[1].layer4[idx], name).register_forward_hook(train_hook))
                
                print(f"{len(w_hooks)} and {len(t_hooks)} layers of output are being recorded on water/train model.")

            if layer_input is not None:    
                # TO CHECK RELU RESULT
                water_relu1 = []
                train_relu1 = []        
                # TO CHECK RELU RESULT
                w_hooks1 = [] # list of hook handles, to be removed when you are done
                t_hooks1 = []
                def water_hook1(module, input, output):
                    if hook_flag:
                        nonlocal water_relu1
                        water_relu1.append(input)   
                def train_hook1(module, input, output):
                    if hook_flag:
                        nonlocal train_relu1
                        train_relu1.append(input)
                # Hook the function onto conv1 and conv2 of layer1~layer4 of both models.
                """ for idx in range(len(water_model[1].layer1)):
                    for name, module in water_model[1].layer1[idx].named_children():
                        if name in layer_input:
                            w_hooks1.append(getattr(water_model[1].layer1[idx], name).register_forward_hook(water_hook1)) """
                """ for idx in range(len(water_model[1].layer2)):
                    for name, module in water_model[1].layer2[idx].named_children():
                        if name in layer_input:
                            w_hooks1.append(getattr(water_model[1].layer2[idx], name).register_forward_hook(water_hook1)) """
                for idx in range(len(water_model[1].layer3)):
                    for name, module in water_model[1].layer3[idx].named_children():
                        if name in layer_input:
                            w_hooks1.append(getattr(water_model[1].layer3[idx], name).register_forward_hook(water_hook1))
                for idx in range(len(water_model[1].layer4)):
                    for name, module in water_model[1].layer4[idx].named_children():
                        if name in layer_input:
                            w_hooks1.append(getattr(water_model[1].layer4[idx], name).register_forward_hook(water_hook1))
            
                """ for idx in range(len(train_model[1].layer1)):
                    for name, module in train_model[1].layer1[idx].named_children():
                        if name in layer_input:
                            t_hooks1.append(getattr(train_model[1].layer1[idx], name).register_forward_hook(train_hook1)) """
                """ for idx in range(len(train_model[1].layer2)):
                    for name, module in train_model[1].layer2[idx].named_children():
                        if name in layer_input:
                            t_hooks1.append(getattr(train_model[1].layer2[idx], name).register_forward_hook(train_hook1)) """
                for idx in range(len(train_model[1].layer3)):
                    for name, module in train_model[1].layer3[idx].named_children():
                        if name in layer_input:
                            t_hooks1.append(getattr(train_model[1].layer3[idx], name).register_forward_hook(train_hook1))
                for idx in range(len(train_model[1].layer4)):
                    for name, module in train_model[1].layer4[idx].named_children():
                        if name in layer_input:
                            t_hooks1.append(getattr(train_model[1].layer4[idx], name).register_forward_hook(train_hook1))
                
                print(f"{len(w_hooks1)} and {len(t_hooks1)} layers of input are being recorded on water/train model.")
                     
        if epoch == 0 :    
            print("Train model main/query acc eval...")
            main_acc = model_on_testset(train_model, test_loader, device)
            query_acc = round(model_on_queryset(train_model, query, response, device).item(),2)
            recover_acc = round(model_on_queryset(train_model, query, original_response, device).item(),2)
            train_test_acc.append([epoch,main_acc,query_acc,recover_acc])
        
            
        print(f"===============================Now in epoch {epoch+1}...===============================")

        # To track performance of the two losses
        ave_neu_loss_per_epoch = 0.0
        ave_task_loss_per_epoch = 0.0
        # TO CHECK RELU RESULT
        ave_relu_neu_loss_per_epoch = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            
            hook_flag = True
            # Reset the lists
            water_relu = []
            train_relu = []
            water_relu1 = []
            train_relu1 = []
            
            optimizer.zero_grad()
            images = batch[0]
            labels = batch[1].long()         
            images, labels = images.to(device), labels.to(device)
            
            train_model.train()
            # If not set to eval(), the model will still changing even if not being opt.step(), due to the change of BN layer during the forward process
            water_model.eval() 
            
            outputs = train_model(images)
            with torch.no_grad():
                outputs_water = water_model(images) 
            
            # For checking hook functions work correctly
            if layer_input and layer_output:
                if not water_relu and not train_relu:
                    print("No value stored in water_relu and train_relu...")
                    breakpoint()
                elif len(water_relu) != len(train_relu):
                    print("The length of water_relu and train_relu are not equal...")
                    breakpoint()  
            
             # Reset new_loss
            new_loss = 0.0
            # Sum up the loss of conv1 and conv2 of layer1~layer4 of both models
            for idx in range(len(water_relu)):
                new_loss += custom_loss1(water_relu[idx][0].detach(), train_relu[idx][0]) / len(water_relu)

            """ if batch_idx % 5 == 0:
                print(f"{batch_idx+1} batch neuron loss: {new_loss.item()}")     """
            
            # TO CHECK RELU RESULT
            relu_new_loss = 0.0
            # Sum up the loss of conv1 and conv2 of layer1~layer4 of both models
            for idx in range(len(water_relu1)):
                relu_new_loss += relu_neu_loss(water_relu1[idx][0].detach(), train_relu1[idx][0].detach()) / len(water_relu1)
 
            """ if batch_idx % 5 == 0:
                print(f"{batch_idx+1} batch relu neuron loss: {relu_new_loss.item()}")   """
            
            kd_loss = loss_fn_kd(outputs, labels, outputs_water, 0.9, 20)
            #kd_loss = F.cross_entropy(outputs, labels)
             
            """ if batch_idx % 5 == 0:
                print(f"{batch_idx+1} batch kd loss: {kd_loss.item()}") """      
            
            # Combine both losses
            # if default_loss_r is passed in with fixed value
            if not callable(default_loss_r):
                loss = (default_loss_r)*kd_loss + (new_loss_r)*(new_loss)
                """ if batch_idx % 5 == 0:
                    print(f"{batch_idx+1} batch {default_loss_r}:{new_loss_r} combined loss: {loss.item()}") """
            # if default_loss_r is passed in with non_fixed ratio (new_loss_r will be replaced with the number calculated below and ignore what was passing into this function)
            else:
                loss = 10*(10*(default_loss_r(epoch))*kd_loss + (1-default_loss_r(epoch))*(new_loss))
                """ if batch_idx % 5 == 0:
                    print(f"{batch_idx+1} batch non_fixed {default_loss_r(epoch)}:{1-default_loss_r(epoch)} combined loss: {loss.item()}") """
            
            ave_neu_loss_per_epoch += new_loss.item()
            ave_task_loss_per_epoch += kd_loss.item()
            # TO CHECK RELU RESULT
            ave_relu_neu_loss_per_epoch += relu_new_loss.item()
            
            loss.backward()
            optimizer.step()

        if callable(default_loss_r):
            print(f"Non-fixed loss ratio: {default_loss_r(epoch)}:{1-default_loss_r(epoch)}")
        else:
            print(f"Fixed ratio: {default_loss_r}:{new_loss_r}")
        
        if epoch % 2 == 0:    
            ave_task_loss_per_epoch /= len(train_loader)
            ave_neu_loss_per_epoch /= len(train_loader)
            neuron_loss_after_epoch.append(round(ave_neu_loss_per_epoch,4))
            task_loss_after_epoch.append(round(ave_task_loss_per_epoch,4))
        
            # TO CHECK RELU RESULT
            ave_relu_neu_loss_per_epoch /= len(train_loader)
            relu_neuron_loss_after_epoch.append(round(ave_relu_neu_loss_per_epoch,4))
        
        
        # Validate the model
        train_model.eval()
        hook_flag = False # Turn off the hook for validation
        val_loss = 0.0
        correct = 0
        total = 0
        print("Validation Process...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch[0]
                labels = batch[1].long()
                images, labels = images.to(device), labels.to(device)
                outputs = train_model(images)
                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_epoch_loss = val_loss / len(val_loader.dataset)
            val_accuracy = correct / total
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        if epoch % 2 == 1:   
            #Testing train model
            """ print("Train Model Test Process...")
            train_test_acc.append(model_on_testset(train_model, test_loader, device))
            # Query train models
            print("Train Model Query Process...")
            train_query_acc.append(round(model_on_queryset(train_model, query, response, device).item(),4)) """
            print("Train model main/query acc eval...")
            main_acc = model_on_testset(train_model, test_loader, device)
            query_acc = round(model_on_queryset(train_model, query, response, device).item(),2)
            recover_acc = round(model_on_queryset(train_model, query, original_response, device).item(),2)
            train_test_acc.append([epoch+1,main_acc,query_acc,recover_acc])
        
        if epoch == 0 or epoch == num_epochs-1:    
            """ print("Water Model Test Process...")
            water_test_acc.append(model_on_testset(water_model, test_loader, device))
            print("Water Model Query Process...")
            water_query_acc.append(round(model_on_queryset(water_model, query, response, device).item(),4)) """
            print("Water model main/query acc eval...")
            main_acc = model_on_testset(water_model, test_loader, device)
            query_acc = round(model_on_queryset(water_model, query, response, device).item(),2)
            water_test_acc.append(f"{main_acc}/{query_acc}")
        
    # Remove hooks for model after used and reset the handle lists
    for handle in w_hooks:
        handle.remove()
    for handle in t_hooks:
        handle.remove()
    w_hooks=[]
    t_hooks=[]
    
    # TO CHECK RELU RESULT
    for handle in w_hooks1:
        handle.remove()
    for handle in t_hooks1:
        handle.remove()
    w_hooks1=[]
    t_hooks1=[]
    
    print('===============================Finished Training===============================')
    print('===============================Finished Training===============================')
    print('===============================Finished Training===============================')
    print(f"Neuron loss after every epoch: {neuron_loss_after_epoch}")
    print(f"K.D. loss after every epoch: {task_loss_after_epoch}")
    
    # TO CHECK RELU RESULT
    print(f"Relu Neuron loss after every epoch: {relu_neuron_loss_after_epoch}")    
    
    return train_test_acc, train_query_acc, water_test_acc, water_query_acc       

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
    epoch = 50
    default_loss_ratio = 1 # >=0
    new_loss_ratio = 0# >=0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    layer_input = ['conv2'] # List of layers that input will be used as relu loss input
    layer_output = ['conv1'] # List of layers that output will be used as custom loss input
    # Non-fixed loss ratio
    default_loss_scheduler = lambda t: np.interp([t],\
            [0, 3, 3, 50],\
            [0.5, 0.5, 1, 1])[0]
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

    # Create a same model for training with deep copy
    #water_model = copy.deepcopy(train_model)
    water_model = model_archive[checkpoint['model']['type']](num_classes=response_scale)
    # Load the model weights
    water_model.load_state_dict(checkpoint['model']['state_dict'])
    
    train_model.to(device)
    water_model.to(device)

    
    
    # Start training
    train_test_acc, train_query_acc, water_test_acc, water_query_acc = start_train_kd_loss1(dataset, subset_rate, train_model, water_model, opt, device, query, response, original_response, epoch, new_loss_ratio, default_loss_scheduler, query_indices, layer_output, layer_input)
    
    # Print the results
    print(f"===============================Training on {subset_rate} of {dataset} with old/new loss ratio {default_loss_ratio}/{new_loss_ratio} for {epoch} epochs.===============================")
    print("Train model Test Acc:", train_test_acc)
    #print("Train model Query Acc:", train_query_acc)
    print(f"Water model Test/Query Acc:{water_test_acc}")
    
    print("Neuron Similarity of Watered model:")
    activated_neuron_similarity(dataset, subset_rate, water_model, device, query, query_indices, ['conv1','conv2'])
    print("Neuron Similarity of Trained model:")
    activated_neuron_similarity(dataset, subset_rate, train_model, device, query, query_indices, ['conv1','conv2'])
    
    breakpoint()
    
