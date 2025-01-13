import os
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from collections import defaultdict
import random

class SingleCategorySampler(Sampler):
    """Sampler to ensure each batch contains images from a single category."""
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.category_to_indices = defaultdict(list)
        
        # Organize dataset indices by category
        for idx, (_, label) in enumerate(dataset):
            self.category_to_indices[label].append(idx)
        
        # Pre-shuffle indices for each category
        for label in self.category_to_indices:
            random.shuffle(self.category_to_indices[label])
        
        self.categories = list(self.category_to_indices.keys())

    def __iter__(self):
        indices = []
        for category in self.categories:
            category_indices = self.category_to_indices[category]
            # Divide into batches
            for i in range(0, len(category_indices), self.batch_size):
                batch = category_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # Only include full batches
                    indices.extend(batch)
        
        # Shuffle batches
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return sum(len(indices) // self.batch_size for indices in self.category_to_indices.values())


# Function to create dataloaders
def create_dataloader(data_dir, batch_size, num_workers=0, shuffle=False):
    transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(root=data_dir, transform=transform)
    
    if shuffle:
        sampler = SingleCategorySampler(dataset, batch_size=batch_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return dataloader, dataset.classes


# Example usage
if __name__ == "__main__":
    train_dir = "/path/to/dataset/train"
    test_dir = "/path/to/dataset/test"
    batch_size = 32
    num_workers = 4

    train_loader, train_classes = create_dataloader(train_dir, batch_size, num_workers, shuffle=True)
    test_loader, test_classes = create_dataloader(test_dir, batch_size, num_workers, shuffle=False)

    print(f"Train classes: {train_classes}")
    print(f"Test classes: {test_classes}")

    # Example iteration
    for images, labels in train_loader:
        print(f"Batch labels: {labels}")
        break
