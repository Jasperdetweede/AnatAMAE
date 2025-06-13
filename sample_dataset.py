import matplotlib.pyplot as plt
import torch
from datasets import getdataset, mask_batch


def sample_and_plot_from_loader(
    dataloader: torch.utils.data.DataLoader, 
    mask_params: dict,
    num_samples: int = 1,
    masked: bool = False
):
    """
    Loads CheckDataset, samples `num_samples` images from a DataLoader, and
    plots them with their numbered in-frame points.
    """

    iterator = iter(dataloader)

    for c in range(num_samples):
        batch = next(iterator)

        if masked: 
            batch = mask_batch(batch, mask_params)

        images  = batch["image"]  # [B, 1, H, W]
        points  = batch["points"] # list of [N_i, 2]
        targets = batch["target"]

        for i in range(len(images)):
            img = images[i].squeeze(0).numpy()  # [H, W]
            pts = points[i].numpy()             # [N, 2]

            plt.figure(figsize=(5, 5))
            plt.imshow(img, cmap="gray")
            plt.title(f"Sample {c+1} — Kellgren: {targets[i]}")
            plt.axis("off")

            for j, (x, y) in enumerate(pts):
                plt.scatter(x, y, c="red", s=15)
                plt.text(x+2, y, str(j), fontsize=6, color="yellow")

            # Ignore masks while calculating intensity (This should not be done in the model)
            nonzero = img[img > -1]
            vmin = nonzero.min() if nonzero.size > 0 else 0
            vmax = nonzero.max() if nonzero.size > 0 else 1
            plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)

            plt.show(block=False)


if __name__ == "__main__":

    print_masked = False
    train_loader, test_loader, val_loader = getdataset(1)
    patch_size = 16
    mask_rate = 0.5
    num_samples = 2
    
    _, height, width = train_loader.dataset[0]["image"].shape
    num_patches = height // patch_size * width // patch_size
    num_masked = int(num_patches * mask_rate)

    mask_params = {
        'height': height,
        'width': width,
        'num_patches': num_patches,
        'num_masked': num_masked,
        'patch_size': patch_size 
    }

    sample_and_plot_from_loader(train_loader, mask_params, num_samples, print_masked)
    input("Press key to close windows and quit program")