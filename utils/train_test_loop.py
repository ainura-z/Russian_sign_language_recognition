import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix


def train(model, optimizer, train_loader, device, epoch, logger, 
          criterion, log_every=10):
    """
    Function for training the LSTM model

    Inputs:
        model - model for training
        optimizer - optimizer for model
        train_loader - loader for training data
        device - device (cuda or cpu)
        epoch - specific epoch
        logger - logger (wandb or tensorboard)
        criterion - loss function

    Returns:
        None
    """
    model.train()
    
    train_loss = 0.0
    total = 0
    correct = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)        

        loss = criterion(outputs, labels)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        optimizer.step()

        # Accumulate training loss and accuracy
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
            
    train_loss /= len(train_loader)
    train_accuracy = 100 * correct / total
    
    # log metrics to wandb
    if logger:
        logger.log({f"Train Accuracy": train_accuracy, 
                    f"Train loss": train_loss},
                 step=epoch, commit=False)
    
    if epoch % log_every == 0:
        print('Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.2f}%'.format(epoch, train_loss, train_accuracy))

def test(model, test_loader, device, epoch, logger, criterion, log_every=10, wandb_prefix=""):
    """
    Function for testing the LSTM model

    Inputs:
        model - model for testing
        test_loader - loader for training data
        device - device (cuda or cpu)
        epoch - specific epoch
        logger - logger (wandb or tensorboard)
        criterion - loss function

    Returns:
        None
    """
   
    model.eval()
    
    test_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Accumulate testing loss and accuracy
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        # Calculate testing accuracy
        test_accuracy = 100*correct / total
        test_loss /= len(test_loader)
        
        # log metrics to wandb
        if logger:
            logger.log({f"Test Accuracy": test_accuracy, 
                        f"Test loss": test_loss}, step=epoch,
                      commit=False)
        
        # Print testing loss and accuracy
        if epoch % log_every == 0:
            print('Epoch: {}, Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(epoch, test_loss, test_accuracy))
    return test_accuracy, model

def cross_validation(num_epochs, k_folds, models, optimizers, 
                     train_dataloaders, test_dataloaders, criterion, device, logger):
    # Run the training loop for defined number of epochs
    for epoch in tqdm(range(0, num_epochs)):
        results = {}
        for fold in range(k_folds):
            models[fold].train()
        
            # Set current loss value
            current_loss = 0.0
            correct, total = 0, 0
            
            all_predicted = []
            all_true = []

            # Iterate over the DataLoader for training data
            for i, data in enumerate(train_dataloaders[fold], 0):
                # Get inputs
                inputs, targets = data[0].to(device).float(), data[1].type(torch.LongTensor).to(device)

                # Zero the gradients
                optimizers[fold].zero_grad()

                # Perform forward pass
                outputs = models[fold](inputs)

                # Compute loss
                loss = criterion(outputs, targets)

                # Perform backward pass
                loss.backward()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                all_predicted.extend(predicted.cpu().detach().numpy().tolist())
                all_true.extend(targets.cpu().detach().numpy().tolist())

                # Perform optimization
                optimizers[fold].step()

                # Print statistics
                current_loss += loss.item()
            
            conf_matrix = confusion_matrix(y_pred=all_predicted, y_true=all_true)
            cls_cnt = conf_matrix.sum(axis=1)  # all labels
            cls_hit = np.diag(conf_matrix)  # true positives
            metrics = [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)]
            for i in range(len(metrics)):
                results[f'{fold}_{i}_train_cls_acc'] = metrics[i] 
            
            results[f'{fold}_train_loss'] = current_loss / len(train_dataloaders[fold])
            results[f'{fold}_train_acc'] = 100.0 * correct / total

            models[fold].eval()
            # Evaluationfor this fold
            correct, total = 0, 0
            current_loss = 0.0
            all_predicted = []
            all_true = []

            with torch.no_grad():
                # Iterate over the test data and generate predictions
                for i, data in enumerate(test_dataloaders[fold], 0):
                    # Get inputs
                    inputs, targets = data[0].to(device).float(), data[1].type(torch.LongTensor).to(device)

                    # Generate outputs
                    outputs = models[fold](inputs)

                    # Set total and correct
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    loss = criterion(outputs, targets)
                    current_loss += loss.item()
                    all_predicted.extend(predicted.cpu().detach().numpy().tolist())
                    all_true.extend(targets.cpu().detach().numpy().tolist())

            conf_matrix = confusion_matrix(y_pred=all_predicted, y_true=all_true)
            cls_cnt = conf_matrix.sum(axis=1)  # all labels
            cls_hit = np.diag(conf_matrix)  # true positives
            metrics = [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)]
            for i in range(len(metrics)):
                results[f'{fold}_{i}_test_cls_acc'] = metrics[i] 
            
            results[f'{fold}_test_loss'] = current_loss / len(test_dataloaders[fold])
            results[f'{fold}_test_acc'] = 100.0 * correct / total
        
        log_dict = {'train_loss':np.mean([results[key] for key in results.keys() if 'train_loss' in key]),
                'train_acc':np.mean([results[key] for key in results.keys() if 'train_acc' in key]),
                'test_loss':np.mean([results[key] for key in results.keys() if 'test_loss' in key]),
                'test_acc':np.mean([results[key] for key in results.keys() if 'test_acc' in key]),
                }
        for i in range(len(metrics)):
            log_dict[f'{i}_test_cls_acc'] = np.mean([results[key] for key in results.keys() if f'{i}_test_cls_acc' in key])
            log_dict[f'{i}_train_cls_acc'] = np.mean([results[key] for key in results.keys() if f'{i}_train_cls_acc' in key])
        
        logger.log(log_dict, step=epoch)
    
    