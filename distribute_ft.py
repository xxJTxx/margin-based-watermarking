import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import mnist, cifar10, resnet, queries
from torchsummary import summary
from loaders import get_cifar10_loaders, get_cifar100_loaders, get_svhn_loaders, get_mnist_loaders,get_cifar10_loaders_sub

import pdb

# Custom Loss function that take into two models water_model and train_model, and return the mean squared error between output of certain layers
class custom_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, water_relu, train_relu):
        return torch.mean(torch.pow((water_relu - train_relu), 2))
        


# Define the training loop
def start_train(train_model, water_model, train_loader, val_loader, optimizer, device, num_epochs=10, new_loss_r=0):
    
    # To save the best model on val set
    #best_val_nat_acc = 0
    #best_val_query_acc = 0
    
    for epoch in range(num_epochs):
        
        if epoch == 0:
            # Record the input of every hooked layer
            water_relu = []
            train_relu = []
            
            # Define the hook function
            w_hooks = [] # list of hook handles, to be removed when you are done
            t_hooks = []
            hook_flag = True
            def water_hook(module, input, output):
                if not hook_flag:
                    return
                nonlocal water_relu
                water_relu.append(input)   
            def train_hook(module, input, output):
                if not hook_flag:
                    return
                nonlocal train_relu
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
                        t_hooks.append(getattr(train_model[1].layer4[idx], name).register_forward_hook(train_hook)) """
        
        print(f"Now in epoch {epoch+1}...")
        
        new_loss_func = custom_loss()

        for batch_idx, batch in enumerate(train_loader):
            train_model.train()
            water_model.train() # Not sure if it's ok to use .eval() and abandon with .no_grad() below?
            
            print(f"Process {batch_idx+1} batch...")
            optimizer.zero_grad()
            images = batch[0]
            labels = batch[1].long()

            # Record the input of every hooked layer
            water_relu = []
            train_relu = []
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = train_model(images)
            with torch.no_grad():
                water_model(images) 
            ce_loss = (1-new_loss_r)*F.cross_entropy(outputs, labels)
            print("Current batch ce loss: ", ce_loss.item())
            print("Current batch ce loss grad: ", ce_loss.grad_fn)
                
            if not water_relu and not train_relu:
                print("No value stored in fix_relu and train_relu...")
            else:
                print(f"{len(water_relu)} layers are store in the water_lists...")
                print(f"{len(train_relu)} layers are store in the train_lists...")    
            
            # Reset new_loss
            new_loss = 0.0
            # Sum up the loss of conv1 and conv2 of layer1~layer4 of both models
            for idx in range(len(water_relu)):
                new_loss += (new_loss_r)*new_loss_func(water_relu[idx][0], train_relu[idx][0])
            # Just for debug
            print("current batch neuron loss: ", new_loss.item())
            print("Current batch neuron loos: ", new_loss.grad_fn)    
            
            # Combine both losses
            loss = ce_loss - new_loss
            print("Current batch combined loss: ", loss.item())
            print("Current batch combined loos: ", loss.grad_fn) 
            #pdb.set_trace()
            loss.backward()
            optimizer.step()
            
            #del water_relu, train_relu
            
            #neuron_loss += new_loss.item() * images.size(0)
            #running_loss += ce_loss.mean().item() * images.size(0)
            #combine_loss += ((1-new_loss_r)*ce_loss - (new_loss_r)*new_loss) * inputs.size(0)
        
        print("===============================1 epoch of training ends===============================")

        # Validate the model
        train_model.eval()
        hook_flag = False # Turn off the hook for validation
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                print("Validation Process...")
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
    
    # Remove hooks for model after used and reset the handle lists
    for handle in w_hooks:
        handle.remove()
    for handle in t_hooks:
        handle.remove()
    w_hooks=[]
    t_hooks=[]
    
    print('Finished Training') 
    

if __name__ == "__main__":
    # Hyperparameters setting 
    dataset = 'cifar10'
    subset_rate = 0.1 # 0~1
    epoch = 1
    new_loss_ratio = 0.5 # 0~1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the existing checkpoint
    checkpoint = torch.load('./experiments/cifar10_res34_margin_100_/checkpoints/checkpoint_query_best.pt') # C:/Users/Someone/margin-based-watermarking/experiments/cifar10_res34_margin_100_/checkpoints/checkpoint_query_best.pt
    # Preparation for loading
    CIFAR_QUERY_SIZE = (3, 32, 32) # input size
    response_scale = 10 # number of classes
    query_size = CIFAR_QUERY_SIZE
    model_archive = cifar10.models



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
    train_model.to(device)
    water_model.to(device)

    """ # Record the input of every hooked layer
    water_relu = []
    train_relu = []
    
    # Define the hook function
    w_hooks = [] # list of hook handles, to be removed when you are done
    t_hooks = []
    def water_hook(module, input, output):
        water_relu.append(input)   
    def train_hook(module, input, output):
        train_relu.append(input) """


    """ # Hook the function onto conv1 and conv2 of layer1~layer4 of both models.
    for idx in range(len(water_model[1].layer1)):
        for name, module in water_model[1].layer1[idx].named_children():
            if name in ['conv1','conv2']:
                w_hooks.append(getattr(water_model[1].layer1[idx], name).register_forward_hook(water_hook))
    for idx in range(len(water_model[1].layer2)):
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
                w_hooks.append(getattr(water_model[1].layer4[idx], name).register_forward_hook(water_hook))

    for idx in range(len(train_model[1].layer1)):
        for name, module in train_model[1].layer1[idx].named_children():
            if name in ['conv1','conv2']:
                t_hooks.append(getattr(train_model[1].layer1[idx], name).register_forward_hook(train_hook))
    for idx in range(len(train_model[1].layer2)):
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
                t_hooks.append(getattr(train_model[1].layer4[idx], name).register_forward_hook(train_hook)) """
 

    """ # Create testing sample
    testing_sample = torch.randn(1, 3, 32, 32).to("cuda:0")
    test_label = torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]).to("cuda:0") """


    # Generate subset data loader based on dataset
    if dataset == 'cifar10':
        train_loader, valid_loader, test_loader = get_cifar10_loaders_sub(subset_rate)
    elif dataset == 'cifar100':
        train_loader, valid_loader, test_loader = get_cifar100_loaders()
    elif dataset == 'svhn':
        train_loader, valid_loader, test_loader = get_svhn_loaders()



    # Start training
    start_train(train_model, water_model, train_loader, valid_loader, opt, device, epoch, new_loss_ratio)


    """ # Start training
    # To save the best model on val set
    best_val_nat_acc = 0
    #best_val_query_acc = 0
        
    for epoch in range(epoch):
            
        print(f"Now in epoch {epoch+1}...")
            
        train_model.train()
        water_model.train() # Not sure if it's ok to use .eval() and abandon with .no_grad() below?
        print("Models in training mode...")
            
        running_loss = 0.0
        neuron_loss = 0.0
        conbine_loss = 0.0
        new_loss = 0.0
        print("Loss initialized...")
        new_loss_func = custom_loss()
        print("New loss function created...")

        for inputs, labels in train_loader:
            print("Process one batch...")

            # Record the input of every hooked layer
            water_relu = []
            train_relu = []

            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            outputs = train_model(inputs)
            with torch.no_grad():
                water_model(inputs)
            ce_loss = F.cross_entropy(outputs, labels)
                
            # Sum up the loss of conv1 and conv2 of layer1~layer4 of both models
            for idx in range(len(water_relu)):
                new_loss += new_loss_func(water_relu[idx][0], train_relu[idx][0])
                
            # Combine both losses
            ((1-new_loss_ratio)*ce_loss - (new_loss_ratio)*new_loss).backward()
            opt.step()
                
            neuron_loss += new_loss.item() * inputs.size(0)
            running_loss += ce_loss.item() * inputs.size(0)
            #combine_loss += ((1-new_loss_r)*ce_loss - (new_loss_r)*new_loss) * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_n_loss = neuron_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epoch}, Training Loss: {epoch_loss:.4f}")
        print(f"Epoch {epoch+1}/{epoch}, Neuron Loss: {epoch_n_loss:.4f}")

        # Validate the model
        train_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = train_model(inputs)
                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_epoch_loss = val_loss / len(valid_loader.dataset)
            val_accuracy = correct / total
            print(f"Epoch {epoch+1}/{epoch}, Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    print('Finished Training') """


    """ # Remove hooks for model after used and reset the handle lists
    for handle in w_hooks:
        handle.remove()
    for handle in t_hooks:
        handle.remove()
    w_hooks=[]
    t_hooks=[] """

