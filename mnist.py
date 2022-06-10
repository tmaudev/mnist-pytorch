import os
import numpy as np
from random import randrange
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv2d_relu_stack = nn.Sequential(
            nn.Conv2d(1, 10, 3),
            nn.ReLU(),
            nn.Conv2d(10, 10, 3),
            nn.ReLU(),
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(10),
        )

    def forward(self, x):
        x = self.conv2d_relu_stack(x)
        logits = self.fc_layer(x)
        return logits

def train_loop(device, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(device, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n   Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def visualize_results(dataloader, num_examples):
    with torch.no_grad():
        X, y = next(iter(test_dataloader))
        assert(num_examples <= X.size()[0])
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        pred_labels = pred.argmax(dim=1)

        fig = plt.figure(figsize=(10, 7))
        fig.canvas.manager.set_window_title("Test Examples")
        rows = 2
        columns = int(num_examples / rows)
        for idx in range(num_examples):
            fig.add_subplot(rows, columns, idx + 1)
            title = "Guess: %d, Actual: %d" % (pred_labels[idx], y[idx])
            plt.title(title)
            plt.axis('off')
            plt.imshow(X.to("cpu")[idx].squeeze(), cmap='gray')
        plt.show()

if __name__ == '__main__':
    # Run on GPU (on M1 Mac) if available
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    model = Network().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(device, train_dataloader, model, loss_fn, optimizer)
        test_loop(device, test_dataloader, model, loss_fn)

    visualize_results(test_dataloader, 8)
