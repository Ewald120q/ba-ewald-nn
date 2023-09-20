import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from CarlaDatasets import CarlaDataset, NPDataset
import model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
learning_rate = 0.0001
batch_size = 16


# Load Data
cleartrain_set = NPDataset(csv_file= 'Datasets/labels_ClearTrain.npy', root_dir = 'Datasets/ClearTrain', transform = transforms.ToTensor())
cleartest_set = CarlaDataset(csv_file= 'Datasets/labels_ClearTest.csv', root_dir = 'Datasets/ClearTest', transform = transforms.ToTensor())

cloudytrain_set = CarlaDataset(csv_file= 'Datasets/labels_CloudyTrain.csv', root_dir = 'Datasets/CloudyTrain', transform = transforms.ToTensor())
cloudytest_set = CarlaDataset(csv_file= 'Datasets/labels_CloudyTest.csv', root_dir = 'Datasets/CloudyTest', transform = transforms.ToTensor())

raintrain_set = CarlaDataset(csv_file= 'Datasets/labels_RainTrain.csv', root_dir = 'Datasets/RainTrain', transform = transforms.ToTensor())
raintest_set = CarlaDataset(csv_file= 'Datasets/labels_RainTest.csv', root_dir = 'Datasets/RainTest', transform = transforms.ToTensor())

alltest_set = CarlaDataset(csv_file= 'Datasets/labels_ALLTest.csv', root_dir = 'Datasets/ALLTest', transform = transforms.ToTensor())
alltrain_set = CarlaDataset(csv_file= 'Datasets/labels_ALLTrain.csv', root_dir = 'Datasets/ALLTrain', transform = transforms.ToTensor())


cleartrain_loader = DataLoader(dataset=cleartrain_set, batch_size=batch_size, shuffle=True)
cleartest_loader = DataLoader(dataset=cleartest_set, batch_size=batch_size, shuffle=True)

cloudytrain_loader = DataLoader(dataset=cloudytrain_set, batch_size=batch_size, shuffle=True)
cloudytest_loader = DataLoader(dataset=cloudytest_set, batch_size=batch_size, shuffle=True)

raintrain_loader = DataLoader(dataset=raintrain_set, batch_size=batch_size, shuffle=True)
raintest_loader = DataLoader(dataset=raintest_set, batch_size=batch_size, shuffle=True)

alltrain_loader = DataLoader(dataset=alltrain_set, batch_size=16, shuffle=True)
alltest_loader = DataLoader(dataset=alltest_set, batch_size=16, shuffle=True)



vgg16 = torchvision.models.vgg16(num_classes = 2)
vgg16 = nn.DataParallel(vgg16)
vgg16.to(device)
inception4 = model.inception_v4(num_classes = 2)
inception4 = nn.DataParallel(inception4)
inception4.to(device)
resnet101 = torchvision.models.resnet101(num_classes = 2)
resnet101 = nn.DataParallel(resnet101)
resnet101.to(device)



def string_to_txt_file(string_data, file_path):
    try:
        with open(file_path, 'w') as file:  # 'w' öffnet die Datei im Schreibmodus
            file.write(string_data)        # Schreibe den String in die Datei
        #print("Datei wurde erfolgreich erstellt und der String wurde gespeichert.")
    except Exception as e:
        print(f"Fehler beim Schreiben der Datei: {e}")



def save_checkpoint(state, filename):
    print("Saving Checkpoint")
    torch.save(state, filename)




def check_accuracy(loader, name, word, model, checkpoint):

    model = model
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def load_checkpoint(checkpoint):
        print("Loading Checkpoint")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"device: {device}")


    load_checkpoint(torch.load(checkpoint))




    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

            #print(f'Got {num_correct} / {num_samples} with accuracy {100*float(num_correct)/float(num_samples)}')
            string_to_txt_file(f'Got {num_correct} / {num_samples} with accuracy {100*float(num_correct)/float(num_samples)}', f'E:\\Carla\\{name}_{word}.txt' )
        print(f'{name}: Got {num_correct} / {num_samples} with accuracy {100 * float(num_correct) / float(num_samples)}')
        model.train()

if __name__ == "__main__":

    #Inception4

    check_accuracy(cloudytest_loader,"inception4_new_rain_cloudy", "test", model=inception4, checkpoint=f'inception4_new_rain_rain_epoch16.pth.tar')
    torch.cuda.empty_cache()

    check_accuracy(alltest_loader, "inception4_new_rain_all", "test", model=inception4, checkpoint=f'inception4_new_rain_rain_epoch16.pth.tar')
    torch.cuda.empty_cache()

    check_accuracy(cleartest_loader, "inception4_new_rain_clear", "test", model=inception4, checkpoint=f'inception4_new_rain_rain_epoch16.pth.tar')
    torch.cuda.empty_cache()

    #ResNet101

    check_accuracy(cloudytest_loader, "resnet101_new_rain_cloudy", "test", model=resnet101,
                   checkpoint=f'resnet101_new_rain_rain_epoch17.pth.tar')
    torch.cuda.empty_cache()

    check_accuracy(alltest_loader, "resnet101_new_rain_all", "test", model=resnet101,
                   checkpoint=f'resnet101_new_rain_rain_epoch17.pth.tar')
    torch.cuda.empty_cache()

    check_accuracy(cleartest_loader, "resnet101_new_rain_clear", "test", model=resnet101,
                   checkpoint=f'resnet101_new_rain_rain_epoch17.pth.tar')
    torch.cuda.empty_cache()

    #VGG16

    #check_accuracy(raintest_loader, "vgg16_new_cloudy_rain", "test", model=vgg16,
     #              checkpoint=f'vgg16_new_cloudy_cloudy_epoch4.pth.tar')
    #torch.cuda.empty_cache()

    #check_accuracy(alltest_loader, "vgg16_new_cloudy_all", "test", model=vgg16,
      #             checkpoint=f'vgg16_new_cloudy_cloudy_epoch4.pth.tar')
    #torch.cuda.empty_cache()

    #check_accuracy(cleartest_loader, "vgg16_new_cloudy_clear", "test", model=vgg16,
       #            checkpoint=f'vgg16_new_cloudy_cloudy_epoch4.pth.tar')
    #torch.cuda.empty_cache()




    #check_accuracy(alltest_loader, "vgg16_new_clear_all", "test", model=vgg16, checkpoint=f'vgg16_new_clear_clear_epoch15.pth.tar')
