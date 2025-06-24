# Anatomy-Aware Masked Autoencoder
## Introduction
This repository contains the code used for the bachelor thesis project called "Anatomy-Aware Masked Autoencoders for Hip Osteoarthritis Classification in X-ray Images". It contains the full preprocessing, pre-training and fine-tune pipeline. Furthermore, for transparency, the results of all experiments can be found in the experiments folder. This contains the parameters for the trained models, as well as the log files. More information on the background of the project can be found in the aforementioned thesis. 

## How to use
### Getting started 
The requirements can be downloaded from the requirements.txt file using "pip install -r requirements.txt". Additionally, a requirements_slim.txt file was added, which should contain all essential packages, but was not tested. 

### Preprocessing
The preprocessing pipeline was created to automatically read the files from the CHECK dataset for osteoarthritis. For scientific purposes, this dataset can be downloaded on request (https://lifesciences.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/dans-zc8-g4cw). 

In order, the steps applied during preprocessing are: 
- Photometric interpretation: assure all images use MONOCHROMATIC1.
- Resample the data to all be in 1mm/pixel
- Crop the image centred on the femoral head, by using the accompanied BoneFinder pointfiles. Specifically, the mean position of the points surrounding the femoral head defined this centre. 
- Downsample the images to 256x256 px to save computational power.
- The left hip is mirrored to have all images appear the same.

The preprocessing pipeline can be instantiated by running the preproces.py file. Most parameters can be changed at the bottom of the file. The image below shows an overview of the preprocessing steps, as well as the expected input and output file structure.  

![alt text](figures/Preprocessing_with_file_structure.png)

### Pre-training
The pre-training can be initiated from the command line, using: "python pretrain.py", where the following flags can be used: 
- --save_every: every n epochs, a .pth file will be created for the model parameters.
- --epoch: the total number of epochs the model will run for.
- --batch_size: the batch size used by the model.
- --mask_rate: masking ratio for the masking strategy. 
- --show_per_epoch: every n epochs, the reconstruction examples will be shown.
- --patch_size: patch size for the masking strategy.
- --seed: the used seed for all randomizers. 
- --lr: learning rate of the model.
- --output_folder: relative folder path in which all results will be stored.
- --mask_roi: string "true" or "false", to determine if the ROI should be masked.
- --mask_non_roi: string "true" or "false", to determine if the background should be masked.

The image below shows an overview of the pre-training model.

![alt text](figures/pretrain_pipeline.png)

### Fine-tuning
The fine-tuning is done or a certain number of epochs. After, the model with the best AUC is calculated and every saved model, except the best one, and the two around it, is deleted from the output path. The pre-training can be initiated from the command line, using: "python finetune.py", where the following flags can be used: 

- --epoch: number of epochs the model will fine-tune for.
- --batch_size: batch size used by the model.
- --lr: the learning rate of the model.
- --model_path_in_folder: pre-trained model path in within the folder path.
- --folder_path: the folder where the pre-trained model is stored, and the output will be stored.
- --seed: the used seed for all randomizers. Currently, specifying a specific seed uses this seed for every repeat.
- --repeats: how many times the finetuning is run with a different seed.
- --add_to_run_num: adds this number to the repeats, useful when you already ran x models before and do not want to overwrite them. 

The image below shows an overview of the fine-tune model.

![alt text](figures/fine-tune_pipeline.png)

### Dataset, model and masking
All logic behind instantiating the dataset is kept inside the checkDataset file. A wrapper for instantiating a DataLoader for this dataset can be found inside the datasets.py file. The model.py file contains the full model for the CNN-based autoencoder.

All masking is handled by the mask_batch function inside the masking.py file. 

### Others 
The repository also contains a file called sample_dataset, which will count the items inside the dataset, and show a parameterized number of samples from the dataset.

Lastly, the utils.py file is used for plotting and saving figures, and setting the seed and logger.

## Credits: 
- The codebase architecture was adapted from https://github.com/JJLi0427/CNN_Masked_Autoencoder
- Special thanks to Gijs van Tulder and Jesse Krijthe for both the guidance and providing the CHECK dataset and example notebooks on which the preprocessing was based.
- LLMs were used during coding to assist in the creation of very niche functionality, and to quickly generate method descriptions.
