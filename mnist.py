from nn.module import Sequential, Linear, BatchNorm1d, ReLU
from nn.loss import CrossEntropyLoss
from optim.optimizer import SGD
from tensor import Tensor

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from utils import load_mnist
    

def _train_one_epoch(X_train, Y_train, model, optimizer, criterion, batch_size):
    model.train()
    batch_losses = []
    num_correct = 0

    perm = np.random.permutation(X_train.shape[0])
    X_train = X_train[perm, :]
    Y_train = Y_train[perm]

    num_batches = X_train.shape[0] // batch_size

    for i in range(num_batches):
        Xb = X_train[i * batch_size : (i+1) * batch_size, :]
        Yb = Y_train[i * batch_size : (i+1) * batch_size]

        Xb = Tensor(Xb)
        Yb = Tensor(Yb)

        optimizer.zero_grad() 
        Y_pred = model(Xb) 

        loss = criterion(Y_pred, Yb) 
        loss.backward() 

        optimizer.step()

        predicted_labels = Y_pred.data.argmax(1)
        num_correct += (Yb.data == predicted_labels).sum()
        batch_losses.append(float(loss.data))
    
    return np.mean(batch_losses), num_correct / Y_train.shape[0]


def _valid_one_epoch(X_valid, Y_valid, model, criterion, batch_size=64):
    model.eval()
    batch_losses = [] 
    num_correct = 0

    perm = np.random.permutation(X_valid.shape[0])
    X_valid = X_valid[perm, :]
    Y_valid = Y_valid[perm]

    num_batches = X_valid.shape[0] // batch_size

    for i in range(num_batches):
        Xb = X_valid[i * batch_size : (i+1) * batch_size, :]
        Yb = Y_valid[i * batch_size : (i+1) * batch_size]

        Xb = Tensor(Xb)
        Yb = Tensor(Yb)

        Y_pred = model(Xb)
        loss = criterion(Y_pred, Yb)

        predicted_labels = Y_pred.data.argmax(1)
        num_correct += (Yb.data == predicted_labels).sum()
        batch_losses.append(float(loss.data))
    
    return np.mean(batch_losses), num_correct / Y_valid.shape[0] 


def training_loop(X_train, Y_train, X_valid, Y_valid, 
                  model, optimizer, criterion, num_epochs=30, batch_size=64):
    trn_losses, val_losses = [], []
    trn_accuracies, val_accuracies = [], []

    for epoch in tqdm(range(num_epochs), total=num_epochs):
        # Training
        loss, acc = _train_one_epoch(X_train, Y_train, model, optimizer, criterion, batch_size) 
        trn_losses.append(loss)
        trn_accuracies.append(acc)

        # Validation
        val_loss, val_acc = _valid_one_epoch(X_valid, Y_valid, model, criterion, batch_size)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    return trn_losses, val_losses, trn_accuracies, val_accuracies       


if __name__ == "__main__":
    # Loading data
    X_train, Y_train, X_valid, Y_valid = load_mnist()

    X_train = X_train[16:].reshape(-1, 784).copy()
    Y_train = Y_train[8:]
    X_valid = X_valid[16:].reshape(-1, 784).copy()
    Y_valid = Y_valid[8:]

    # Normalizing data
    X_train = np.divide(X_train, 255.0)
    X_valid = np.divide(X_valid, 255.0)
    
    BobNet = Sequential(
        Linear(784, 20),
        BatchNorm1d(20),
        ReLU(),
        Linear(20, 10)
    )

    optimizer = SGD(BobNet.parameters(), lr=0.1)
    criterion = CrossEntropyLoss()

    trn_losses, val_losses, trn_accuracies, val_accuracies = training_loop(
        X_train=X_train, 
        Y_train=Y_train, 
        X_valid=X_valid, 
        Y_valid=Y_valid, 
        model=BobNet, 
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=30,
        batch_size=64
    )

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    axes[0].plot(trn_losses, 'b-', label="Train")
    axes[0].plot(val_losses, 'r-', label="Valid")
    axes[0].set_title("Loss", fontweight="bold", fontsize=13)

    axes[1].plot(trn_accuracies, 'b-', label="Train")
    axes[1].plot(val_accuracies, 'r-', label="Valid")
    axes[1].set_title("Accuracy", fontweight="bold", fontsize=13)

    fig.suptitle("Training report", fontweight="bold", fontsize=20)

    plt.legend()
    plt.show()

    



