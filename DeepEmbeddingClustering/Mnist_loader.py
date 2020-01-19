import torchvision
import torch.utils.data as Data

def mnist_data_loader():
    train_data = torchvision.datasets.MNIST(root='./mnist',
                                            train=True,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=False)
    test_data = torchvision.datasets.MNIST(root='./mnist',
                                            train=False,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=False)
    train_loader = Data.DataLoader(dataset=train_data,
                                    batch_size=128,
                                    shuffle=True,
                                    num_workers=2)
    test_loader = Data.DataLoader(dataset=test_data,
                                    batch_size=128,
                                    shuffle=True,
                                    num_workers=2)
    return train_loader, test_loader
