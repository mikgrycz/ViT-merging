import os
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from collections import defaultdict
import random
# TODO: multiple classes are put into batch 
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
        # Create batches for each category
        all_batches = []
        for category in self.categories:
            category_indices = self.category_to_indices[category]
            # Divide into full batches
            for i in range(0, len(category_indices), self.batch_size):
                batch = category_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    all_batches.append(batch)

        # Shuffle all batches
        random.shuffle(all_batches)
        # Flatten the list of batches into a single list of indices
        return iter([idx for batch in all_batches for idx in batch])

    def __len__(self):
        # Total number of full batches across all categories
        return sum(len(indices) // self.batch_size for indices in self.category_to_indices.values())



# Function to create dataloaders
def create_dataloader(data_dir, batch_size, num_workers=5):
    """
    Create a DataLoader for the given data directory.
    
    Args:
        data_dir (str): Path to the directory containing training or testing data.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
    
    Returns:
        DataLoader, List[str]: DataLoader and list of class names.
    """
    transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(root=data_dir, transform=transform)
    sampler = SingleCategorySampler(dataset, batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return dataloader, dataset.classes


if __name__ == "__main__":
    # Paths to train and test directories
    base_dir = "/path/to/stansord_cars"
    train_dir = os.path.join(base_dir, "cars_train")
    test_dir = os.path.join(base_dir, "cars_test")

    batch_size = 32
    num_workers = 4

    # Create DataLoaders
    train_loader, train_classes = create_dataloader(train_dir, batch_size, num_workers)
    test_loader, test_classes = create_dataloader(test_dir, batch_size, num_workers)

    print(f"Train classes: {train_classes}")
    print(f"Test classes: {test_classes}")

    # Validate batches
    for images, labels in train_loader:
        print(f"Batch labels: {labels.unique()}")
