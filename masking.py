import random
import torch
import torch.nn.functional as F

def mask_batch(batch, mask_ROI: bool, mask_non_ROI: bool, mask_params):

    if not mask_ROI and not mask_non_ROI:
        return batch
    elif not mask_ROI and mask_non_ROI:     
        return mask_batch_non_ROI(batch, mask_params)
    elif mask_ROI and not mask_non_ROI:
        return mask_batch_ROI(batch, mask_params)
    else:
        return mask_batch_full(batch, mask_params)
    

def mask_batch_non_ROI(batch, mask_params):
    """
    Masks only patches where NO pixel overlaps with the ROI.
    Masked patches are set to black (-1).
    """

    images    = batch["image"]  # [B,C,H,W]
    roi_mask  = batch["roi_mask"].to(images.device)  # [B,H,W]

    patch_size = mask_params.get("patch_size")
    rate       = mask_params.get("mask_rate")
    
    if rate <= 0:
        return batch

    B, C, H, W = images.shape
    ph, pw = patch_size, patch_size
    n_ph, n_pw = H // ph, W // pw  # patches vertically & horizontally

    # Divide ROI mask into patches and check if any pixel in patch overlaps ROI
    roi_mask_unfolded = roi_mask.unfold(1, ph, ph).unfold(2, pw, pw)  # [B, n_ph, n_pw, ph, pw]
    patch_roi_presence = roi_mask_unfolded.any(dim=(-1, -2))  # [B, n_ph, n_pw], True if overlaps with ROI

    # Invert patch_roi_presence to pick all pacthes with zero overlap with the ROi
    eligible_mask = ~patch_roi_presence 
    eligible_mask_flat = eligible_mask.view(B, -1)  # [B, N_patches]

    # Randomly select patches to mask from eligible ones
    num_to_mask = (eligible_mask_flat.sum(dim=1).float() * rate).long()  # [B]
    masks = torch.zeros_like(eligible_mask_flat, dtype=torch.bool)  # [B, N_patches]
    for b in range(B):
        eligible_indices = torch.where(eligible_mask_flat[b])[0]
        if len(eligible_indices) == 0 or num_to_mask[b] == 0:
            continue
        chosen = eligible_indices[torch.randperm(len(eligible_indices), device=images.device)[:num_to_mask[b]]]
        masks[b, chosen] = True

    # Expand patch mask to full resolution [B, H, W]
    masks_reshaped = masks.view(B, n_ph, n_pw)  # [B, n_ph, n_pw]
    masks_upsampled = masks_reshaped.repeat_interleave(ph, dim=1).repeat_interleave(pw, dim=2)  # [B,H,W]
    masks_upsampled = masks_upsampled[:, :H, :W]  # crop in case of rounding

    # put mask over image
    masks_upsampled = masks_upsampled.unsqueeze(1).expand(-1, C, -1, -1)  # [B,C,H,W]
    black = torch.full_like(images, -1.0)
    masked_images = torch.where(masks_upsampled, black, images)

    return {
        "image": masked_images,
        "target": batch["target"],
        "points": batch["points"]
    }


def mask_batch_ROI(batch, mask_params):
    """
    Masks a specified fraction of pixels inside the ROI, leaving the rest untouched.
    """

    images = batch["image"]                     # [B,C,H,W]
    roi    = batch["roi_mask"].unsqueeze(1)     # [B,1,H,W]

    patch_size = mask_params.get("patch_size")
    rate       = mask_params.get("mask_rate")

    if rate <= 0:
        return batch

    B, C, H, W = images.shape
    device = images.device
    out = images.clone()

    # Unfold ROI into patches: [B, patch_area, num_patches]
    roi_patches = F.unfold(roi.float(), kernel_size=patch_size, stride=patch_size)
    roi_overlap = (roi_patches.sum(dim=1) > 0)  # [B, num_patches], True if any pixel in ROI

    for b in range(B):
        maskable_idxs = torch.nonzero(roi_overlap[b], as_tuple=False).squeeze(1)
        num_to_mask = int(len(maskable_idxs) * rate)
        if num_to_mask == 0:
            continue

        rand_perm = torch.randperm(len(maskable_idxs), device=device)
        selected = maskable_idxs[rand_perm[:num_to_mask]]

        # Convert patch idx to 2D coordinates
        grid_W = W // patch_size
        y = (selected // grid_W) * patch_size
        x = (selected %  grid_W) * patch_size

        for yi, xi in zip(y, x):
            out[b, :, yi:yi+patch_size, xi:xi+patch_size] = -1.0

    return {
        "image": out,
        "target": batch["target"],
        "points": batch["points"]
    }

    
def mask_batch_full(batch, mask_params):
    """
    Apply random masking to each image in a batch.

    Args:
        batch: dict with keys "image", "target" and "points"
        mask_params: dict with keys 'height', 'width', 'num_masked'
        patch_size: int, patch size to mask

    Returns:
        A new dict like batch but with masked image values
    """

    if mask_params["num_masked"] == 0: 
        return batch

    
    images = batch["image"].clone()
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