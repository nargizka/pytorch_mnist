from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

def get_transform() -> transforms.Compose:
    """
    Training transform with optional data augmentation.
    - Resize(32), ToTensor, Normalize
    """
    transforms_list = [
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ]

    return transforms.Compose(transforms_list)

def get_train_test_loaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """
    Returns DataLoader for training and test sets.
    """
    transform_func = get_transform()

    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform_func)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform_func)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
