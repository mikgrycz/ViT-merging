import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
import torchvision
import torchvision.transforms as transforms
import wandb
from torch.utils.data import DataLoader
from torchvision import datasets


def train(model, dataloader, optimizer, criterion, scaler=None):
    model.train()
    running_loss = 0.0
    correct_predictions, total_samples = 0, 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Move data to device
        images, labels = images.to(device, dtype=torch.float32), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scale loss if scaler is provided
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular backward pass
            loss.backward()
            optimizer.step()

        # Update running loss
        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(
                f"Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}, Accuracy: {100 * correct_predictions / total_samples:.2f}")

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = 100 * correct_predictions / total_samples
    print(f"Epoch Loss: {epoch_loss:.4f}, Epoch Accuracy: {epoch_accuracy:.4f}")
    return epoch_loss, epoch_accuracy


# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    running_loss = 0.0
    correct_predictions, total_samples = 0, 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Move data to device
            images, labels = images.to(device, dtype=torch.float32), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Evaluation Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}, Accuracy: {100 * correct_predictions / total_samples:.2f}")

    eval_loss = running_loss / len(dataloader.dataset)
    accuracy = 100 * correct_predictions / total_samples
    print(f"Test Loss: {eval_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return eval_loss, accuracy

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available()else "cpu")
    print(f"Using device: {device}")

    config = {
        "device": device,
        "model_name": "vit_base_patch32_224",
        "dataset": "CIFAR10",
        "transforms": {
            "resize": (224, 224),
            "to tensor": None,
            "normalize": {
                "mean": (0.4914, 0.4822, 0.4465),
                "std": (0.2023, 0.1994, 0.2010)
            }
        },
        "batch_size": 64,
        "num_workers": 10,
        "img_size": 32,
        "n_classes": 10,
        "lr": 1e-6,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0,
        "loss_func": nn.CrossEntropyLoss(),
        "num_epochs": 5,
        "checkpoint_every_num_epochs": 1,
    }

    transform_train = transforms.Compose([
        transforms.RandomCrop(config["img_size"], padding=4),
        transforms.Resize(config["transforms"]["resize"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config["transforms"]["normalize"]["mean"], config["transforms"]["normalize"]["std"]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(config["transforms"]["resize"]),
        transforms.ToTensor(),
        transforms.Normalize(config["transforms"]["normalize"]["mean"], config["transforms"]["normalize"]["std"]),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,
                             num_workers=config["num_workers"])

    model = timm.create_model(model_name=config["model_name"], pretrained=True)
    model.head = nn.Linear(in_features=model.head.in_features, out_features=config["n_classes"])

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], betas=config["betas"], eps=config["eps"],
                           weight_decay=config["weight_decay"])
    criterion = config["loss_func"]

    num_epochs = config["num_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    checkpoint_every_num_epochs = config["checkpoint_every_num_epochs"]
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion)
        test_loss, test_accuracy = evaluate(model, test_loader)
        scheduler.step(epoch - 1)
        print(
            f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f} Test Accuracy: {test_accuracy:.2f}%")

    torch.save(model, "checkpoints/ViT-B-32/Cifar10/finetuned.pt")

    model.eval()

    img, label = test_dataset[0]
    img = img.unsqueeze(0).to(device)

    # Model inference
    model.eval()
    with torch.no_grad():
        logits = model(img)
        probs = F.softmax(logits, dim=-1)
        top_prob, top_class = probs.topk(1, dim=-1)

    print(f"Predicted class: {top_class.item()}, Probability: {top_prob.item():.4f}")
