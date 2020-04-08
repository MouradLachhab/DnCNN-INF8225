import torch.nn as nn
import torch.optim as optim

def train(model, X_train, y_train, optimizer, criterion):
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    loss_training = 0.0

    # Training with SGD ?
    for i in range(0,1): # Placeholder loop  
        # output = model(INPUT)
        # loss = criterion(...)
        loss.backward()
        optimizer.step()
        loss_training += loss.item() # not sure if we can use item for SGD

    loss_training /= len(X_train) # averaging the loss for this epoch

    return model, loss_training

def validation(model, X_validation, y_validation, criterion):
    model.eval()

    accuracy = 0
    loss_validation = 0.0

    for i in range(0,1): # Placeholder loop  
        # output = model(INPUT)
        # loss = criterion(...)
        loss_validation += loss.item()
        # accuracy +=

    loss_validation /= len(X_validation)
    accuracy /= len(X_validation)

    return accuracy, loss_validation


def train_model(model, epochs=50, lr=1e-1): # Need to check how to implement exponential decay
    optimizer = optim.Adam()
    criterion = nn.MSELoss()

    best_accuracy = 0.0
    best_model = None

    for epoch in range(1, epochs + 1):
        model, loss_training = train(model, X_train, y_train, optimizer, criterion)
        accuracy, loss_validation = validation(model, X_validation, y_validation, criterion)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    return best_model, best_accuracy
