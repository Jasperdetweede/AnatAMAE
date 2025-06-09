from matplotlib import transforms
import matplotlib.pyplot as plt
import torch
from checkDataset import CheckDataset
from datasets import getDataset


def sample_and_plot_from_loader(
    batch_size: int = 4,
    num_samples: int = 4,
):
    """
    Loads CheckDataset, samples `num_samples` images from a DataLoader, and
    plots them with their numbered in-frame points.
    """
    loader = getDataset(batch_size)

    iterator = iter(loader)

    for c in range(num_samples):
        sample = next(iterator)

        images  = sample["image"]  # [B, 1, H, W]
        points  = sample["points"] # list of [N_i, 2]
        targets = sample["target"]

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

            plt.show(block=False)
            
if __name__ == "__main__":
    sample_and_plot_from_loader(
        batch_size=1,
        num_samples=4
    )
    input("Press key to close windows and quit program")
