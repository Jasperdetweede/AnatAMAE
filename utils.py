import torch
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import logging
import random
import numpy as np


def visualize_pretrain(
    comparison,
    show_per_epoch, 
    show_img_count, 
    output_folder_name = None
):
    if output_folder_name is not None:
        output_base = os.path.join(output_folder_name, 'figure')
    else: 
        output_base = './figure'

    os.makedirs(output_base, exist_ok=True)
    all_comparisons = torch.cat(comparison, dim=0)
    name = 'show_per' + str(show_per_epoch) + 'epoch' + '.png'
    path = os.path.join(output_base, name)
    save_image(all_comparisons.cpu(), path, nrow=show_img_count*3, normalize=True, value_range=(-1,1))


def loss_figure(
    train_loss_list, 
    test_loss_list, 
    epoch,
    mode,
    output_folder_name=None
):
    if output_folder_name is not None:
        output_base = os.path.join(output_folder_name, 'figure')
    else: 
        output_base = './figure'
    os.makedirs(output_base, exist_ok=True)
    if mode == 'pretrain':
        title = 'Image L1Loss'
        figure_path = os.path.join(output_base, 'pretrain_loss.png')
    elif mode == 'finetune':
        title = 'BCEWithLogitsLoss'
        figure_path = os.path.join(output_base, 'finetune_loss.png')
    else:
        raise ValueError('Invalid mode')
    
    plt.plot(range(epoch), train_loss_list, label='train loss')
    plt.plot(range(epoch), test_loss_list, label='test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(figure_path)
    plt.close()
    # Save the loss curve


def auc_figure(
    auc_list,   # List[float] of length == epoch
    epoch: int, # current epoch count
):
    os.makedirs('./figure', exist_ok=True)

    # Plot
    plt.plot(range(1, epoch + 1), auc_list, marker='o', label='val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.ylim(0.0, 1.0)
    plt.title("AUC values over epochs")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # Save & close
    plt.tight_layout()
    plt.savefig('./figure/auc_per_epoch.png')
    plt.close()

def set_logger(mode, output_folder_name='ckpt'):
    os.makedirs(f'./{output_folder_name}', exist_ok=True)

    if mode == 'pretrain':
        os.makedirs(f'./{output_folder_name}/pretrain', exist_ok=True)
        filename = f'./{output_folder_name}/pretrain/pretrain.log'
    elif mode == 'finetune':
        os.makedirs(f'./{output_folder_name}/finetune', exist_ok=True)
        filename = f'./{output_folder_name}/finetune/finetune.log'
    else:
        raise ValueError('Invalid mode')

    logging.basicConfig(
        filename=filename, 
        level=logging.INFO, 
        format='%(asctime)s %(levelname)s: %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    
    return logging


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
