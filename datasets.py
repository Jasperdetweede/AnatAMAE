import torch
import os
from torchvision import datasets, transforms
from checkDataset import CheckDataset
import random


def getbasicdataset(batch_size):
    if not os.path.exists('./data'):
        os.mkdir('./data')

    print("Loading dataset, requires download the first time.")
    
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            './data', 
            train=True, 
            download=True, 
            transform=transforms.ToTensor()
        ), 
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            './data', 
            train=False, 
            download=True, 
            transform=transforms.ToTensor()
        ), 
        batch_size=batch_size, 
        shuffle=True
    )

    print("Finished loading dataset")
    return train_loader, test_loader

def take_subset(dataset, subset_percentage, seed=0):
    n = len(dataset)
    k = int(n * subset_percentage)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()[:k]
    return torch.utils.data.Subset(dataset, indices)

def getdataset(batch_size, augment=False, frac_of_training_data=1.00):
    assert frac_of_training_data >= .0 and frac_of_training_data <= 1.0 
 
    print("\nLoading dataset")

    dataset_name = "check"
    print("Using dataset: " + dataset_name.upper())

    output = {}
    total_samples = 0
    for set in ["train", "test", "val"]:
        dataset = CheckDataset(
            processed_root="data/" + dataset_name,
            patients_path="data/" + dataset_name + "/patients",
            targets_json="data/"+ dataset_name + "/targets.json",
            split_txt="data/" + dataset_name + "/train_test_val_split.txt",
            desired_split=set,
            transform=transforms.Normalize(mean=[0.5], std=[0.5]), # normalize image intensity
            augment=augment) 
        
        if frac_of_training_data < 1.00 and set == "train": 
            dataset = take_subset(dataset, frac_of_training_data)
            print(f"Took only {frac_of_training_data*100} percent of training data for training the classifier. New training set has {len(dataset)} samples.") 
        
        print(f"{set}set contains {len(dataset)} samples.")
        total_samples = total_samples + len(dataset)
        
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=10,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2)
        
        output[set] = data_loader
        print("Finished loading " + set + " set")
    
    print(f"Full dataset contains {total_samples} samples.\n")
    return output["train"], output["test"], output["val"]  

def mask_batch(batch, mask_params):
    """
    Apply random masking to each image in a batch.

    Args:
        batch: dict with keys "image", "target", optionally "points"
        mask_params: dict with keys 'height', 'width', 'num_masked'
        patch_size: int, patch size to mask

    Returns:
        A new dict like `batch` but with masked "image" values
    """
    
    images = batch["image"].clone()  # Clone to avoid modifying original
    B, C, H, W = images.shape

    height = mask_params['height']
    width = mask_params['width']
    num_masked = mask_params['num_masked']
    patch_size = mask_params['patch_size']

    assert height % patch_size == 0 and width % patch_size == 0, "Patch size must divide image dimensions"

    n_cols = width // patch_size
    n_rows = height // patch_size
    n_patches = n_rows * n_cols

    for i in range(B):
        mask_id = random.sample(range(n_patches), num_masked)
        mask_positions = [(i // n_rows, i % n_cols) for i in mask_id] 

        for row, col in mask_positions:
            images[i, :, 
                   row * patch_size : (row + 1) * patch_size,
                   col * patch_size : (col + 1) * patch_size] = -1

    # Return a new batch dict
    return {
        "image": images,
        "target": batch["target"],
        "points": batch["points"]
    }