import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from utils.utils import load_mnist


class ConvBobNet(nn.Module):
    def __init__(self):
        super(ConvBobNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(3, 3)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1014, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return nn.LogSoftmax(dim=1)(x)

class BobNet(torch.nn.Module):
  def __init__(self):
    super(BobNet, self).__init__()
    self.l1 = nn.Linear(784, 128)
    self.l2 = nn.Linear(128, 10)
    self.sm = nn.LogSoftmax(dim=1)

  def forward(self, x):
    x = F.relu(self.l1(x))
    x = self.l2(x)
    x = self.sm(x)
    return x

# Loading data
X_train, Y_train, X_test, Y_test = load_mnist()

#X_train = X_train[16:].reshape(-1, 784).copy()
X_train = X_train[16:].reshape(-1, 1, 28, 28).copy()
Y_train = Y_train[8:]
#X_test = X_test[16:].reshape(-1, 784).copy()
X_test = X_test[16:].reshape(-1, 1, 28, 28).copy()
Y_test = Y_test[8:]
    
X_train = np.divide(X_train, 255.0)
X_test = np.divide(X_test, 255.0)

# Model, loss & optim
model = ConvBobNet()
loss_function = nn.NLLLoss(reduction='none')
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0)

# Training loop
BS = 128
losses, accuracies = [], []
STEPS = 2000

for i in tqdm(range(STEPS), total=STEPS):
  samp = np.random.randint(0, X_train.shape[0], size=(BS))
  X = torch.tensor(X_train[samp]).float()
  Y = torch.tensor(Y_train[samp]).long()

  model.zero_grad()

  out = model(X)

  cat = torch.argmax(out, dim=1)
  accuracy = (cat == Y).float().mean()

  loss = loss_function(out, Y)
  loss = loss.mean()

  loss.backward()

  optim.step()

  loss, accuracy = loss.item(), accuracy.item()
  losses.append(loss)
  accuracies.append(accuracy)

Y_test_preds = torch.argmax(model(torch.tensor(X_test).float()), dim=1).numpy()
print((Y_test == Y_test_preds).mean())