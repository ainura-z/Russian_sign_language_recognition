import torch


def train(model, optimizer, train_loader, device, epoch, logger, criterion=nn.functional.nll_loss):
    ''' Train the LSTM model '''
    
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
        optimizer.step()

        # Accumulate training loss and accuracy
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
            
    train_loss /= len(train_dataloader)
    train_accuracy = 100 * correct / total
    
    # log metrics to wandb
    logger.log({"Train Accuracy": train_accuracy, "Train loss": train_loss})
    
    if epoch % 100 == 0:
        print('Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.2f}%'.format(epoch, train_loss, train_accuracy))


def test(model, test_loader, device, epoch, logger, criterion=nn.functional.nll_loss):
    ''' Testing'''
    
    model.eval()
    
    test_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):

            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Forward
            outputs = model(inputs)
            print(outputs[0][0])
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
        logger.log({"Test Accuracy": test_accuracy, "Test loss": test_loss})
        
        # Print testing loss and accuracy
        if epoch % 100 == 0:
            print('Epoch: {}, Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(epoch, test_loss, test_accuracy))
