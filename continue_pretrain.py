import os
import torch
import numpy as np
import torch
import random
import argparse
from torch import nn, optim
from model import Autoencoder
from datasets import getdataset, mask_batch
from utils import visualize_pretrain, loss_figure, set_logger, set_seed

###
#
#
#
#
#
#   This only partly works, as the model is not yet configured to also store the optimizer dict. DO NOT USE until this is implemented, or only for testing, not for real experiments. 
#
#
#
#
#
###


def get_args():
    # define some parameters
    parser = argparse.ArgumentParser(description='Pretrain')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--show_per_epoch', type=int, default=10, help='Show per epoch')
    parser.add_argument('--show_img_count', type=int, default=2, help='Show image count')
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size')
    parser.add_argument('--mask_rate', type=float, default=0.75, help='Mask rate')
    parser.add_argument('--save_every', type=int, default=10, help='Save every n epoch')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--model_path', type=str, required=True, help='Model path')
    parser.add_argument('--last_epoch', type=int, required=True, help='Last epoch')
    args = parser.parse_args()
    return args


def train(
    model: torch.nn.Module, 
    optimizer: torch.optim, 
    criterion: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader,
    mask_params: dict,
    device
):
    model.train()
    batch_losses = []

    for batch in data_loader:
        # Original images (targets for reconstruction)
        orig_imgs = batch["image"].to(device)         # [B,1,H,W]

        # Apply masking on the fly
        masked_batch = mask_batch(batch, mask_params)
        masked_imgs  = masked_batch["image"].to(device)

        # Forward pass and reconstruction loss
        reconstruction = model(masked_imgs)            
        loss = criterion(reconstruction, orig_imgs)

        # Backprop + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

    # Return average loss for this epoch
    return float(np.mean(batch_losses))

def test(
    epoch: int,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    mask_params: dict,
    device,
    show_per_epoch: int,
    show_img_count: int,
    last_epoch: int
):
    """
    Evaluate the autoencoder on masked inputs.
    Returns:
      - compare: list of grid tensors [3,1,H,W] for visualization
      - avgloss: float average reconstruction loss
    """
    model.eval()
    losses = []
    compare = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            orig = batch["image"].to(device)  # [B,1,H,W]
            # mask_batch requires patch_size
            mb = mask_batch(batch, mask_params)
            masked = mb["image"].to(device)   # [B,1,H,W]

            # forward
            reconstruction = model(masked)        # [B,1,H,W]
            loss = criterion(reconstruction, orig)
            losses.append(loss.item())

            # gather some images for visualization
            if batch_idx == 0 and (epoch == last_epoch or (epoch+1) % show_per_epoch == 0):
                # number to show is min(show_img_count, B)
                B = orig.size(0)
                n = min(show_img_count, B)
                for i in range(n):
                    # stack original / masked / recon along batch dim
                    trio = torch.cat([
                        orig[i:i+1],
                        masked[i:i+1],
                        reconstruction[i:i+1]
                    ], dim=0)  # [3,1,H,W]
                    compare.append(trio)

    avgloss = float(np.mean(losses))
    return compare, avgloss


def get_mask_params(batch_img, args):
    _, height, width = batch_img.shape
    num_patches = height // args.patch_size * width // args.patch_size
    patch_size = args.patch_size
    num_masked = int(num_patches * args.mask_rate)
    mask_params = {
        'height': height,
        'width': width,
        'num_patches': num_patches,
        'num_masked': num_masked, 
        'patch_size': patch_size
    }
    # Generate the mask parameters
    return mask_params

def pretrain():
    args = get_args()
    set_seed(args.seed)
    logging = set_logger('pretrain')
    logging.info(f'Start training:')
    logging.info(f'Arguments: {args}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')
    
    # Data loader
    train_loader, test_loader, val_loader = getdataset(args.batch_size)

    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(args.model_path))
    model = autoencoder.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    mask_params = get_mask_params(train_loader.dataset[0]["image"], args)
    comparison = []
    train_loss_list = []
    test_loss_list = []

    print("Starting pre-training")

    for epoch in range(args.last_epoch, args.last_epoch + args.epoch):
        train_loss = train(
            model, 
            optimizer, 
            criterion, 
            train_loader,
            mask_params, 
            device
        )
        train_loss_list.append(train_loss)

        compare, test_loss = test(
            epoch, 
            model, 
            criterion, 
            test_loader,
            mask_params,
            device, 
            args.show_per_epoch, 
            args.show_img_count, 
            args.last_epoch
        )
        test_loss_list.append(test_loss)
        comparison.extend(compare)
        
        logging.info(f'Epoch: {epoch + 1:3d} | train loss: {train_loss:.6f} | test loss: {test_loss:.6f}')
        if not os.path.exists("ckpt/continued_pretraining"):
            os.makedirs("ckpt/continued_pretraining")
        if (epoch+1) % args.save_every == 0 or (epoch+1) == args.epoch:
            torch.save(model.state_dict(), f'./ckpt/continued_pretraining/{epoch+1}epoch.pth')

        # Show progress every epoch and after pre-training
        visualize_pretrain(
            comparison, 
            args.show_per_epoch, 
            args.show_img_count, 
            add_prefix="continued"
        )

        loss_figure(
            train_loss_list, 
            test_loss_list, 
            epoch+1-args.last_epoch, 
            'continued_pretrain'
        )
    
if __name__ == "__main__":
    pretrain()