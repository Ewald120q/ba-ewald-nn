import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
from CarlaDatasets import CarlaDataset, NPDataset
import VGG16
#import test
import model



cleartrain_set = NPDataset(csv_file= 'Datasets/labels_ClearTrain.npy', root_dir = 'Datasets/ClearTrain', transform = transforms.ToTensor())
cleartest_set = CarlaDataset(csv_file= 'Datasets/labels_ClearTest.csv', root_dir = 'Datasets/ClearTest', transform = transforms.ToTensor())

cloudytrain_set = CarlaDataset(csv_file= 'Datasets/labels_CloudyTrain.csv', root_dir = 'Datasets/CloudyTrain', transform = transforms.ToTensor())
cloudytest_set = CarlaDataset(csv_file= 'Datasets/labels_CloudyTest.csv', root_dir = 'Datasets/CloudyTest', transform = transforms.ToTensor())

raintrain_set = CarlaDataset(csv_file= 'Datasets/labels_RainTrain.csv', root_dir = 'Datasets/RainTrain', transform = transforms.ToTensor())
raintest_set = CarlaDataset(csv_file= 'Datasets/labels_RainTest.csv', root_dir = 'Datasets/RainTest', transform = transforms.ToTensor())

raintrain_set = CarlaDataset(csv_file= '~git/Models/labels_RainTrain.csv', root_dir = '~git/Models/RainTrain', transform = transforms.ToTensor())
raintest_set = CarlaDataset(csv_file= '~git/Models/labels_RainTest.csv', root_dir = '~git/Models/RainTest', transform = transforms.ToTensor())

alltest_set = CarlaDataset(csv_file= 'Datasets/labels_ALLTest.csv', root_dir = 'Datasets/ALLTest', transform = transforms.ToTensor())
alltrain_set = CarlaDataset(csv_file= 'Datasets/labels_ALLTrain.csv', root_dir = 'Datasets/ALLTrain', transform = transforms.ToTensor())


cleartrain_loader = DataLoader(dataset=cleartrain_set, batch_size=batch_size, shuffle=True)
cleartest_loader = DataLoader(dataset=cleartest_set, batch_size=batch_size, shuffle=True)

cloudytrain_loader = DataLoader(dataset=cloudytrain_set, batch_size=batch_size, shuffle=True)
cloudytest_loader = DataLoader(dataset=cloudytest_set, batch_size=batch_size, shuffle=True)

raintrain_loader = DataLoader(dataset=raintrain_set, batch_size=batch_size, shuffle=True)
raintest_loader = DataLoader(dataset=raintest_set, batch_size=batch_size, shuffle=True)

alltrain_loader = DataLoader(dataset=alltrain_set, batch_size=32, shuffle=True)
alltest_loader = DataLoader(dataset=alltest_set, batch_size=32, shuffle=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = torchvision.models.vgg16(num_classes=2)
inception4 = model.inception_v4(num_classes=2)
inception4 = nn.DataParallel(inception4)
inception4.to(device)
vgg16 = nn.DataParallel(vgg16)
vgg16.to(device)
resnet101 = torchvision.models.resnet101(num_classes=2)
resnet101 = nn.DataParallel(resnet101)
resnet101.to(device)

if __name__ == '__main__':

    VGG16.train(raintrain_set, raintest_set, vgg16, "vgg16_new_rain_rain", False, start_index=0, num_epochs=30, batch_size=16)
    torch.cuda.empty_cache()
    VGG16.train(cleartrain_set, cleartest_set, vgg16, "vgg16_new_clear_clear", False, start_index=0, num_epochs=30, batch_size=16)
    torch.cuda.empty_cache()
    VGG16.train(cloudytrain_set, cloudytest_set, vgg16, "vgg16_new_cloudy_cloudy", False, start_index=0, num_epochs=30, batch_size=16)
    torch.cuda.empty_cache()
    VGG16.train(alltrain_set, alltest_set, vgg16, "vgg16_new_all_all", False, start_index=0, num_epochs=30, batch_size=20)
    torch.cuda.empty_cache()

    VGG16.train(raintrain_set, raintest_set, resnet101, "resnet101_new_rain_rain", False, start_index=0, num_epochs=30, batch_size=16)
    torch.cuda.empty_cache()
    VGG16.train(cleartrain_set, cleartest_set, resnet101, "resnet101_new_clear_clear", False, start_index=0, num_epochs=30, batch_size=16)
    torch.cuda.empty_cache()
    VGG16.train(cloudytrain_set, cloudytest_set, resnet101, "resnet101_new_cloudy_cloudy", False, start_index=0, num_epochs=30, batch_size=16)
    torch.cuda.empty_cache()
    VGG16.train(alltrain_set, alltest_set, resnet101, "resnet101_new_all_all", False, start_index=0, num_epochs=30, batch_size=20)
    torch.cuda.empty_cache()

    VGG16.train(raintrain_set, raintest_set, inception4, "inception4_new_rain_rain", False, start_index=0, num_epochs=30, batch_size=16)
    torch.cuda.empty_cache()
    VGG16.train(cleartrain_set, cleartest_set, inception4, "inception4_new_clear_clear", False, start_index=0, num_epochs=30, batch_size=16)
    torch.cuda.empty_cache()
    VGG16.train(cloudytrain_set, cloudytest_set, inception4, "inception4_new_cloudy_cloudy", False, start_index=0, num_epochs=30, batch_size=16)
    torch.cuda.empty_cache()
    VGG16.train(alltrain_set, alltest_set, inception4, "inception4_new_all_all", False, start_index=0, num_epochs=30, batch_size=20)
    torch.cuda.empty_cache()






