import glob
import os
from sklearn.metrics import roc_auc_score
import torch
import argparse
import numpy as np
from torch import nn, optim
from datasets import getdataset
from model import Autoencoder, Classifier
from utils import auc_figure, loss_figure, set_logger, set_seed


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
    parser.add_argument('--patience', type=int, default=10, help='Patience for auc improvement')
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

def compute_auc(model: nn.Module, data_loader, device) -> float:
    """Compute ROC AUC on data_loader in one pass."""
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["target"].float().to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_probs)
    # handle edge: if only one class present, roc_auc_score errors -> return 0.5
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return 0.5

def finetune():
    args = get_args()
    set_seed(args.seed)
    logging = set_logger('finetune')
    logging.info(f'Start finetuning:')
    logging.info(f'Arguments: {args}')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')

    # Data loader
    train_loader, test_loader = getdataset(args.batch_size, augment_training_set=True, frac_of_training_data=0.3, num_workers=6, prefetch_factor=3)
    # Initialize model
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(args.model_path)["model_state"])
    model = Classifier(autoencoder.encoder, autoencoder.latent_dim).to(device)

    # freeze the encoder (Not included in experiments)
    #for param in autoencoder.encoder.parameters():
    #    param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': args.lr * 0.2},
        {'params': model.classifier.parameters(), 'lr': args.lr}
    ])

    # Start finetuning
    train_loss_list = []
    test_loss_list = []
    val_auc_list = []

    best_auc = 0
    epochs_no_improve = 0
    patience = args.patience
    
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

        # Compute AUC on test set
        val_auc = compute_auc(model, test_loader, device)
        val_auc_list.append(val_auc)
        
        avg_auc_of_window = np.average(val_auc_list[max(0, epoch-10): epoch])

        logging.info(f'Epoch: {epoch + 1:3d} | train loss: {train_loss:.6f} | test loss: {test_loss:.6f} | auc: {val_auc:.6f} | avg of last 10 auc: {avg_auc_of_window:.6f}')

        # Check for improvement
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            print("new best AUC found")
            checkpoint = {
                'epoch':        epoch,
                'model_state':  model.state_dict(),
                'optim_state':  optimizer.state_dict(),
                'val_auc':      best_auc
            }
            torch.save(checkpoint, f'./ckpt/finetune/best_auc.pth')
        else:
            epochs_no_improve += 1

        try:
            if (epoch+1) % args.save_every == 0 or (epoch+1) == args.epoch:
                checkpoint = {
                    'epoch':        epoch,
                    'model_state':  model.state_dict(),
                    'optim_state':  optimizer.state_dict(),
                }
                torch.save(checkpoint, f'./ckpt/finetune/{epoch+1}epoch.pth')
        except:
            logging.info(f'Failed to write. Probably could not open ./ckpt/finetune/{epoch+1}epoch.pth. Skipped this save.') # To catch flaky problem where the file cannot be opened
            
        loss_figure(
            train_loss_list, 
            test_loss_list, 
            epoch+1, 
            'finetune'
        )

        auc_figure(val_auc_list, epoch+1)

        # # Early stopping
        # if epochs_no_improve >= patience:
        #     logging.info(f'Early stopping triggered after {epoch} epochs (no AUC improvement in {patience} epochs)')
        #     break

if __name__ == "__main__":
    finetune()