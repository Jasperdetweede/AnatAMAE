import torch
import numpy as np
import torch
import random
import argparse
from torch import nn, optim
from model import Autoencoder
from datasets import getdataset
from masking import mask_batch
from utils import visualize_pretrain, loss_figure, set_logger, set_seed


def get_args():
    # define some parameters
    parser = argparse.ArgumentParser(description='Pretrain')
    parser.add_argument('--epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--show_per_epoch', type=int, default=2, help='Show per epoch')
    parser.add_argument('--show_img_count', type=int, default=2, help='Show image count')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--mask_rate', type=float, default=0.75, help='Mask rate')
    parser.add_argument('--save_every', type=int, default=5, help='Save every n epoch')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--mask_roi', type=str, required=True, help='Mask ROI area')
    parser.add_argument('--mask_non_roi', type=str, required=True, help='Mask non ROI area')
    parser.add_argument('--output_folder', type=str, default='ckpt', help='Folder name for output, for running multiple experiments consecutively. Standard is \'ckpt\'')
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
        masked_batch = mask_batch(batch, mask_params.get("mask_roi"), mask_params.get("mask_non_roi"), mask_params)
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
        
        # Pick a random batch to pick some examples from
        visualize_batch_idx = random.randrange(len(data_loader))

        for batch_idx, batch in enumerate(data_loader):
            orig = batch["image"].to(device)  # [B,1,H,W]
            
            masked_batch = mask_batch(batch, mask_params.get("mask_roi"), mask_params.get("mask_non_roi"), mask_params)
            masked = masked_batch["image"].to(device)   # [B,1,H,W]

            # forward
            reconstruction = model(masked)        # [B,1,H,W]
            loss = criterion(reconstruction, orig)
            losses.append(loss.item())

            # gather some images for visualization
            if batch_idx == visualize_batch_idx and (epoch == 0 or (epoch+1) % show_per_epoch == 0):
                # number to show is min(show_img_count, B)
                B = orig.size(0)
                n = min(show_img_count, B)

                idxs = random.sample(range(B), n) # Pick random indices to show from the random batch
                for i in idxs:
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

    if args.mask_roi.lower() == "true": mask_roi = True
    elif args.mask_roi.lower() == "false": mask_roi = False
    else: raise Exception("mask_roi should be true or false")

    if args.mask_non_roi.lower() == "true": mask_non_roi = True
    elif args.mask_non_roi.lower() == "false": mask_non_roi = False
    else: raise Exception("mask_non_roi should be true or false")

    mask_params = {
        'height': height,
        'width': width,
        'num_patches': num_patches,
        'num_masked': num_masked, 
        'patch_size': patch_size,
        'mask_rate': args.mask_rate,
        'mask_roi': mask_roi,
        'mask_non_roi': mask_non_roi
    }

    print(f"PATCH ROI IS: {args.mask_roi} and PATCH NON-ROI IS: {args.mask_non_roi}")
    # Generate the mask parameters
    return mask_params


def pretrain():
    args = get_args()
    set_seed(args.seed)
    logging = set_logger('pretrain', args.output_folder)
    logging.info(f'Start training:')
    logging.info(f'Arguments: {args}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')
    
    # Data loader
    train_loader, test_loader = getdataset(args.batch_size)
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    mask_params = get_mask_params(train_loader.dataset[0]["image"], args)
    comparison = []
    train_loss_list = []
    test_loss_list = []

    print("Starting pre-training")

    for epoch in range(args.epoch):
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
        )
        test_loss_list.append(test_loss)
        comparison.extend(compare)
        
        logging.info(f'Epoch: {epoch + 1:3d} | train loss: {train_loss:.6f} | test loss: {test_loss:.6f}')
        if (epoch+1) % args.save_every == 0 or (epoch+1) == args.epoch:
            checkpoint = {
                'epoch':        epoch,
                'model_state':  model.state_dict(),
                'optim_state':  optimizer.state_dict(),
            }
            torch.save(checkpoint, f'./{args.output_folder}/pretrain/{epoch+1}epoch.pth')

        # Show progress every epoch and after pre-training
        visualize_pretrain(
            comparison, 
            args.show_per_epoch, 
            args.show_img_count, 
            output_folder_name=args.output_folder
        )

        loss_figure(
            train_loss_list, 
            test_loss_list, 
            epoch+1, 
            'pretrain',
            output_folder_name=args.output_folder
        )
    
if __name__ == "__main__":
    pretrain()