from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .load_data import get_train_test_loaders
from .model import VanillaMNISTDNN

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """
    Evaluates the model.

    Returns:
        avg_loss, avg_accuracy
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def train_and_test_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> Dict[str, list]:
    """
    Trains the model and prints the loss and accuracy.
    """

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )

    history = {
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
    }

    LOG_INTERVAL = 100
    for epoch in range(0, epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0
        epoch_samples = 0
        epoch_loss = 0.0
        epoch_correct = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # clear any gradients from previous iteration
            optimizer.zero_grad()
            # perform forward pass and get model predictions
            outputs = model(inputs)
            # compute the loss using cross entropy loss
            loss = criterion(outputs, targets)
            # perform backpropatation to compute gradients
            loss.backward()
            # update the step size
            optimizer.step()

            # 
            loss_value = loss.item()
            batch_size = targets.size(0)
            # epoch totals
            epoch_loss += loss_value * batch_size
            epoch_samples += batch_size
            _, preds = outputs.max(dim=1)
            correct_in_batch = preds.eq(targets).sum().item()
            epoch_correct += correct_in_batch

            # interval totals
            running_loss += loss_value * batch_size
            running_correct += correct_in_batch
            running_samples += batch_size

            # prints acc and loss every LOG_INTERVAL batches
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                interval_loss = running_loss / running_samples
                interval_acc = running_correct / running_samples

                print(f"Epoch {epoch+1} | Step {batch_idx+1}/{len(train_loader)} | "
                    f"Loss: {interval_loss:.4f} | "
                    f"Acc: {interval_acc*100:.2f}% ")

                running_loss = 0.0
                running_correct = 0
                running_samples = 0

        train_loss = epoch_loss / epoch_samples
        epoch_accuracy = epoch_correct / epoch_samples

        test_loss, test_acc = evaluate(
            model, test_loader, device, criterion
        )

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(epoch_accuracy)
        history["test_accuracy"].append(test_acc)
        print(60*"=")
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {100*epoch_accuracy:.2f}% "
            f"| Test Loss: {test_loss:.4f}, Test Acc: {100*test_acc:.2f}%"
        )
        print(60*"=")

    return model, history

def main():
    # Parameters
    batch_size = 128
    epochs = 10
    lr = 0.01
    weight_decay = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_train_test_loaders(batch_size=batch_size)

    # model
    model = VanillaMNISTDNN(num_classes=10)
  
    # trains and tests model, prints accuracy of the model for each epoch
    model, history = train_and_test_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )
    return model, history


if __name__ == "__main__":
    model, history = main()
