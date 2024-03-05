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
        w_denominator=water_relu.detach()
        t_denominator=train_relu.detach()       
        w_n = torch.where(water_relu !=0, water_relu/w_denominator, water_relu)
        t_n = torch.where(train_relu !=0, train_relu/t_denominator, train_relu)
        return torch.mean(torch.pow((w_n - t_n), 2))

# Define the training loop
def start_train(dataset, subset_rate, train_model, water_model, optimizer, device, query, response,num_epochs=10, new_loss_r=0):
    
    for epoch in range(num_epochs):
        
        if epoch == 0:
            # Check if query data got changed
            query_detach = copy.deepcopy(query)
            response_detach = copy.deepcopy(response)
            
            # Generate subset data loader based on dataset
            if dataset == 'cifar10':
                train_loader, val_loader, test_loader = get_cifar10_loaders_sub(subset_rate)
            elif dataset == 'cifar100':
                train_loader, val_loader, test_loader = get_cifar100_loaders()
            elif dataset == 'svhn':
                train_loader, val_loader, test_loader = get_svhn_loaders()
            
            # Record accuaracy after every epoch
            water_test_acc = []
            train_test_acc = []
            water_query_acc = []
            train_query_acc = []
            neuron_loss_after_epoch = [] # To reduce variable, first item stores the acc of water model on testing before training 
            crosse_loss_after_epoch = [] # To reduce variable, first item stores the acc of water model on query before training
            
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
                    water_relu.append(input)   
            def train_hook(module, input, output):
                if hook_flag:
                    nonlocal train_relu
                    train_relu.append(input)

            # Hook the function onto conv1 and conv2 of layer1~layer4 of both models.
            for idx in range(len(water_model[1].layer1)):
                for name, module in water_model[1].layer1[idx].named_children():
                    if name in ['conv1','conv2']:
                        w_hooks.append(getattr(water_model[1].layer1[idx], name).register_forward_hook(water_hook))
            """ 
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
            """

            for idx in range(len(train_model[1].layer1)):
                for name, module in train_model[1].layer1[idx].named_children():
                    if name in ['conv1','conv2']:
                        t_hooks.append(getattr(train_model[1].layer1[idx], name).register_forward_hook(train_hook))
            """ 
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
                        t_hooks.append(getattr(train_model[1].layer4[idx], name).register_forward_hook(train_hook)) 
            """
            new_loss_func = custom_loss()
            
            print("Watered model Test Acc before training:")
            neuron_loss_after_epoch.append(model_on_testset(water_model, test_loader, device))
            print("Watered model Query Acc before training:")
            crosse_loss_after_epoch.append(round(model_on_queryset(water_model, query, response, device).item(), 4))         
            print("Query data the same:", torch.equal(query, query_detach))
            print("Response data the same:", torch.equal(response, response_detach))
            
        print(f"Now in epoch {epoch+1}...")

        for batch_idx, batch in enumerate(train_loader):
            
            hook_flag = True
            # Record the input of every hooked layer
            water_relu = []
            train_relu = []
            
            optimizer.zero_grad()
            images = batch[0]
            labels = batch[1].long()
                        
            images, labels = images.to(device), labels.to(device)
            
            train_model.eval()
            water_model.eval() # If not set to eval(), the model will still changing even if not being opt.step()
            
            outputs1 = train_model(images)
            with torch.no_grad():
                water_model(images) 
            
            if not water_relu and not train_relu:
                print("No value stored in fix_relu and train_relu...")
                breakpoint()
            #else:
                #print(f"{len(water_relu)} layers are store in the water_lists...")
                #print(f"{len(train_relu)} layers are store in the train_lists...")    
            
             # Reset new_loss
            new_loss = 0.0
            # Sum up the loss of conv1 and conv2 of layer1~layer4 of both models
            for idx in range(len(water_relu)):
                new_loss += new_loss_func(water_relu[idx][0].detach(), train_relu[idx][0]) / len(water_relu)
            # Just for debug
            if batch_idx % 5 == 0:
                print(f"{batch_idx+1} batch neuron loss: {new_loss.item()}")    
            
            train_model.train()
            outputs = train_model(images)
            
            ce_loss = F.cross_entropy(outputs, labels)
            if batch_idx % 5 == 0:
                print(f"{batch_idx+1} batch ce loss: {ce_loss.item()}")      
            
            # Combine both losses
            loss = ce_loss - (new_loss_r)*new_loss
            if batch_idx % 5 == 0:
                print(f"{batch_idx+1} batch combined loss: {loss.item()}")
 
            
            train_model.train()
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 5 == 0:
                print(f"vvvvvvvvv Round {batch_idx} Test Block 3: vvvvvvvvv")
                print((round(model_on_queryset(water_model, query, response, device).item(), 4)))
                water_relu = []
                train_relu = []
                print("Query data the same:", torch.equal(query, query_detach))
                print("Response data the same:", torch.equal(response, response_detach))
        
        neuron_loss_after_epoch.append(round(new_loss.item(),4))
        crosse_loss_after_epoch.append(round(ce_loss.item(),4))
        
        print("===============================1 epoch of training ends===============================")
        
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
          
        #Testing both model
        print("Train Model Test Process...")
        train_test_acc.append(model_on_testset(train_model, test_loader, device))
        print("Water Model Test Process...")
        water_test_acc.append(model_on_testset(water_model, test_loader, device))

        # Query both models
        print("Train Model Query Process...")
        train_query_acc.append(round(model_on_queryset(train_model, query, response, device).item(),4))
        print("Water Model Query Process...")
        water_query_acc.append(round(model_on_queryset(water_model, query, response, device).item(),4))
        
    # Remove hooks for model after used and reset the handle lists
    for handle in w_hooks:
        handle.remove()
    for handle in t_hooks:
        handle.remove()
    w_hooks=[]
    t_hooks=[]
    
    print('===============================Finished Training===============================')
    print(f"Neuron loss after every epoch: {neuron_loss_after_epoch}")
    print(f"CrossE loss after every epoch: {crosse_loss_after_epoch}")    
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
    epoch = 5
    new_loss_ratio = 1 # >=0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    ###############################################################################


    # Load the existing checkpoint
    checkpoint = torch.load('./experiments/cifar10_res34_margin_100_/checkpoints/checkpoint_query_best.pt') # C:/Users/Someone/margin-based-watermarking/experiments/cifar10_res34_margin_100_/checkpoints/checkpoint_query_best.pt
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
    opt = torch.optim.SGD(train_model.parameters(), lr=0.1)
    opt.load_state_dict(checkpoint['optimizer'])
    # Load the query from checkpoint
    query = checkpoint['query_model']['state_dict']['query']
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
    train_test_acc, train_query_acc, water_test_acc, water_query_acc = start_train(dataset, subset_rate, train_model, water_model, opt, device, query, response, epoch, new_loss_ratio)
    print(f"Training on {subset_rate} of {dataset} with new loss ratio {new_loss_ratio} for {epoch} epochs.")
    print("Train model Test Acc:", train_test_acc)
    print("Train model Query Acc:", train_query_acc)
    print("Water model Test Acc:", water_test_acc)
    print("Water model Query Acc:", water_query_acc)
    
    breakpoint()
    

    
    

