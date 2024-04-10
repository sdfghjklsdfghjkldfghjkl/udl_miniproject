import numpy as np
import torch
import torchvision
from torchvision import transforms
import random


def SplitMnistDataloader(class_distribution, batch_size=256, not_mnist=False, seed=0):
    """
    class_distribution: list[list]
    """
    rsz = 28
    transform = transforms.Compose([
        transforms.Resize((rsz, rsz)),
        transforms.ToTensor(),
    ])
    if not_mnist:
        root = "./data/notMNIST"
        dataset = torchvision.datasets.MNIST
        trainset = dataset(root=root, train=True, download=False, transform=transform)
        testset = dataset(root=root, train=False, download=False, transform=transform)
    else:
        root = "./data"
        dataset = torchvision.datasets.MNIST
        trainset = dataset(root=root, train=True, download=True, transform=transform)
        testset = dataset(root=root, train=False, download=True, transform=transform)

    dataloaders = []

    for classes in class_distribution:
        train_idx = _get_class_idx(trainset, classes)
        train_idx = torch.where(train_idx)[0]
        sub_train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        sub_train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, sampler=sub_train_sampler)

        test_idx = _get_class_idx(testset, classes)
        test_idx = torch.where(test_idx)[0]
        sub_test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
        sub_test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, sampler=sub_test_sampler)
        
        dataloaders.append((sub_train_loader, sub_test_loader))

    return dataloaders

def SplitCifar10Dataloader(class_distribution, batch_size=256, grayscale=True, data_subset=1):
    """
    class_distribution: list[list]
    """
    rsz = 32
    if grayscale:
        transform = transforms.Compose([
            transforms.Resize((rsz, rsz)),
            transforms.ToTensor(),
            transforms.Grayscale()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((rsz, rsz)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    root = "./data/cifar10"
    dataset = torchvision.datasets.CIFAR10
    trainset = dataset(root=root, train=True, download=True, transform=transform)
    testset = dataset(root=root, train=False, download=True, transform=transform)

    dataloaders = []

    for classes in class_distribution:
        train_idx = _get_class_idx(trainset, classes)
        train_idx = torch.where(train_idx)[0]

        # Select a subset of indices
        subset_size_train = int(len(train_idx) * data_subset)
        train_idx_subset = torch.randperm(len(train_idx))[:subset_size_train]
        
        sub_train_sampler = torch.utils.data.SubsetRandomSampler(train_idx[train_idx_subset])
        sub_train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, sampler=sub_train_sampler)
        
                # Test subset
        test_idx = _get_class_idx(testset, classes)
        test_idx = torch.where(test_idx)[0]
        # Select a subset of indices
        subset_size_test = int(len(test_idx) * data_subset)
        test_idx_subset = torch.randperm(len(test_idx))[:subset_size_test]
        
        sub_test_sampler = torch.utils.data.SubsetRandomSampler(test_idx[test_idx_subset])
        sub_test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, sampler=sub_test_sampler)
        
        dataloaders.append((sub_train_loader, sub_test_loader))

    return dataloaders

def mini_SplitMnistDataloader(class_distribution, batch_size=256, subset_percentage=0.1, not_mnist=False):
    """
    class_distribution: list[list]
    """
    rsz = 28
    transform = transforms.Compose([
        transforms.Resize((rsz, rsz)),
        transforms.ToTensor(),
    ])

    root = "./data/notMNIST" if not_mnist else "./data"
    dataset = torchvision.datasets.MNIST
    trainset = dataset(root=root, train=True, download=True, transform=transform)
    testset = dataset(root=root, train=False, download=True, transform=transform)

    dataloaders = []

    for classes in class_distribution:
        # Train subset
        train_idx = _get_class_idx(trainset, classes)
        train_idx = torch.where(train_idx)[0]
        # Select a subset of indices
        subset_size_train = int(len(train_idx) * subset_percentage)
        train_idx_subset = torch.randperm(len(train_idx))[:subset_size_train]
        
        sub_train_sampler = torch.utils.data.SubsetRandomSampler(train_idx[train_idx_subset])
        sub_train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, sampler=sub_train_sampler)

        # Test subset
        test_idx = _get_class_idx(testset, classes)
        test_idx = torch.where(test_idx)[0]
        # Select a subset of indices
        subset_size_test = int(len(test_idx) * subset_percentage)
        test_idx_subset = torch.randperm(len(test_idx))[:subset_size_test]
        
        sub_test_sampler = torch.utils.data.SubsetRandomSampler(test_idx[test_idx_subset])
        sub_test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, sampler=sub_test_sampler)
        
        dataloaders.append((sub_train_loader, sub_test_loader))

    return dataloaders


def _get_class_idx(dataset, target_classes):
    """
    dataset: torchvision.datasets.MNIST
    target_classes: list
    """
    dataset.targets = torch.tensor(dataset.targets)
    idx = torch.zeros_like(dataset.targets, dtype=torch.bool)
    for target in target_classes:
        idx = idx | (dataset.targets==target)
    
    return idx
