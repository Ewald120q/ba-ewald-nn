import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from CarlaDatasets import CarlaDataset
from datetime import datetime
import time
import matplotlib.pyplot as plt

#Code from PyTorch Documentation: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html and https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

def string_to_txt_file(string_data, file_path):
    try:
        with open(file_path, 'w') as file:
            file.write(string_data)
        print("String wurde als Datei gespeichert.")
    except Exception as e:
        print(f"Fehler: {e}")

def train(train_set, test_set, model, name, load = False, num_epochs= 20, start_index = 0, batch_size= 16):

    def check_accuracy(loader, model, word):
        num_correct = 0
        num_samples = 0
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                #x = (x * 2.0) - 1
                x = x.to(device=device)
                y = y.to(device=device)


                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

            print(f'Got {num_correct} / {num_samples} with accuracy {100 * float(num_correct) / float(num_samples)}')
            string_to_txt_file(
                f'Got {num_correct} / {num_samples} with accuracy {100 * float(num_correct) / float(num_samples)}',
                f'{name}_epoch{epoch}_{word}.txt')

        model.train()

    def save_checkpoint(state, filename=name):
        print("Saving Checkpoint")
        torch.save(state, filename)

    def load_checkpoint(checkpoint):
        print("Loading Checkpoint")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    in_channel = 3
    num_classes = 2
    learning_rate = 0.0001
    batch_size = batch_size
    num_epochs = num_epochs

    # Load Data
    train_set = train_set
    test_set = test_set

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


    # Model
    model = model
    model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load:
        load_checkpoint(torch.load(f"{name}_epoch{start_index-1}.pth.tar"))


    #Train Network

    for epoch in range(start_index, num_epochs):
        losses = []
        i = 0
        for batch_dix, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            data = data.float()
            data = (data * 2.0) - 1.0
            print(data)

            data = data.to(device=device)
            targets = targets.to(device=device)


            scores = model(data)
            loss = criterion(scores, targets)

            losses.append(loss.item())


            loss.backward()



            optimizer.step()


            print(f'{i}/{len(train_loader)}:: Cost at epoch {epoch} is {sum(losses)/len(losses)}')
            i+=1

        checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
        save_checkpoint(checkpoint, f"{name}_epoch{epoch}.pth.tar")
        daytime = datetime.now().strftime('%H:%M:%S')
        print("Checking accuracy on Training Set")
        check_accuracy(train_loader, model, "train")
        print("Checking accuracy on Test Set")
        check_accuracy(test_loader, model, "test")
        print(f'daytime: {daytime}')











