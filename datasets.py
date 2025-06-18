import torch
import os
from torchvision import datasets, transforms
from checkDataset import CheckDataset

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
 
def getdataset(batch_size, augment_training_set=False, frac_of_training_data=1.00, num_workers=10, prefetch_factor=2):
    """
        Args: 
        - batch_size: batch size
        - finetune: means augment, but masking will not work correctly. 
        - frac_of_training_data: fraction of data returned from the training data. 
        
        THE VAL SET IS ADDED TO THE TEST SET 

        Returns: 
        - train_loader, test_loader
    """

    assert frac_of_training_data >= .0 and frac_of_training_data <= 1.0 
 
    print("\nLoading dataset")

    dataset_name = "check"
    print("Using dataset: " + dataset_name.upper())

    output = {}
    total_samples = 0
    for set in ["train", "test"]:

        do_augmentations = (augment_training_set and set == "train" )
        do_shuffle = (set == "train")

        dataset = CheckDataset(
            processed_root="data/" + dataset_name,
            patients_path="data/" + dataset_name + "/patients",
            targets_json="data/"+ dataset_name + "/targets.json",
            split_txt="data/" + dataset_name + "/train_test_val_split.txt",
            desired_split=set,
            transform=transforms.Normalize(mean=[0.5], std=[0.5]), # normalize image intensity
            augment=do_augmentations) 
        
        if frac_of_training_data < 1.00 and set == "train": 
            dataset = take_subset(dataset, frac_of_training_data)
            print(f"Took only {frac_of_training_data*100} percent of training data for training the classifier. New training set has {len(dataset)} samples.") 
        
        print(f"{set}set contains {len(dataset)} samples.")
        total_samples = total_samples + len(dataset)
        
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=do_shuffle, #dont shuffle test sets 
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor)
        
        output[set] = data_loader
        print("Finished loading " + set + " set")
    
    print(f"Full dataset contains {total_samples} samples.\n")
    return output["train"], output["test"] 