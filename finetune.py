import torch
import argparse
import numpy as np
from torch import nn, optim
from datasets import getdataset
from model import Autoencoder, Classifier
from utils import loss_figure, set_logger, set_seed


def get_args():
    # define some parameters
    parser = argparse.ArgumentParser(description='Finetune')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    # parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--model_path', type=str, default='ckpt/pretrain/100epoch.pth', help='Model path')
    parser.add_argument('--save_every', type=int, default=5, help='Save every n epoch')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    return args

def train(
    model: torch.nn.Module, 
    optimizer: torch.optim, 
    criterion: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader,
    device
):
    model.train()
    batch_losses = []

    for batch in data_loader:
        images = batch["image"].to(device)         # [B,1,H,W]
        labels = batch["target"].to(device)

        labels =labels.float()     

        # Forward pass and reconstruction loss
        outputs = model(images)      
        loss = criterion(outputs, labels)

        # Backprop + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

    # Return average loss for this epoch
    return float(np.mean(batch_losses))

def test(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
):
    model.eval()
    batch_losses = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)      # [B,1,H,W] or [B,3,H,W]
            labels = batch["target"].to(device)     # bool or int

            labels = labels.float()                 # To be compatible with BCE
   
            outputs = model(images)                 # [B] for BCE
            loss = criterion(outputs, labels)
            batch_losses.append(loss.item())

    return float(np.mean(batch_losses))

def finetune():
    args = get_args()
    set_seed(args.seed)
    logging = set_logger('finetune')
    logging.info(f'Start finetuning:')
    logging.info(f'Arguments: {args}')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')

    # Data loader
    train_loader, test_loader, val_loader = getdataset(args.batch_size, False, frac_of_training_data=0.3)

    # Initialize model
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(args.model_path)["model_state"])
    model = Classifier(autoencoder.encoder, autoencoder.latent_dim).to(device)

    # freeze the encoder
    # for param in autoencoder.encoder.parameters():
    #     param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': args.lr * 0.2},
        {'params': model.classifier.parameters(), 'lr': args.lr}
    ])

    # Start finetuning
    train_loss_list = []
    test_loss_list = []
    
    print("Start finetuning")

    for epoch in range(args.epoch):
        train_loss = train(
            model, 
            optimizer, 
            criterion, 
            train_loader, 
            device
        )
        test_loss = test(
            model, 
            criterion, 
            test_loader, 
            device
        )
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        
        logging.info(f'Epoch: {epoch + 1:3d} | train loss: {train_loss:.6f} | test loss: {test_loss:.6f}')
        if (epoch+1) % args.save_every == 0 or (epoch+1) == args.epoch:
            checkpoint = {
                'epoch':        epoch,
                'model_state':  model.state_dict(),
                'optim_state':  optimizer.state_dict(),
            }
            torch.save(checkpoint, f'./ckpt/finetune/{epoch+1}epoch.pth')
            
        loss_figure(
            train_loss_list, 
            test_loss_list, 
            epoch+1, 
            'finetune'
        )

if __name__ == "__main__":
    finetune()