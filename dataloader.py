from torchvision import datasets , transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

def create_dataset():

    train_data = MNIST(
        root = "data_mnist",
        download = True,
        train = True,
        transform = transforms.ToTensor()
    )

    test_data = MNIST(
        root = "data_mnist",
        download = True,
        train = False,
        transform = transforms.ToTensor()
    )

    return train_data , test_data 

def create_dataloader(train_data, test_data):
    train_dataloader = DataLoader(
        train_data,
        batch_size = 32,
        shuffle = True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size = 32,
        shuffle = False
    )

    return train_dataloader , test_dataloader
