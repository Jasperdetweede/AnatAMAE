import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from skimage.draw import polygon
from scipy.ndimage import binary_dilation

class CheckDataset(Dataset):
    def __init__(self, processed_root: str, patients_path: str, targets_json: str, split_txt: str, desired_split="train", transform=None, augment=False):
        """
        Args:
            processed_root: root folder containing patient subdirs
            targets_json:   path to your targets.json file
            transform:      optional torchvision transforms on the image tensor
        """
        self.processed_root = processed_root
        self.patients_path = patients_path
        self.desired_split = desired_split

        self.transform = transform
        self.augment = augment
        

        # Load split
        split_map = {}
        with open(split_txt, 'r') as f:
            for line in f:
                CHECK_patient_id, split = line.strip().split(',')
                patient_id = CHECK_patient_id.replace("CHECK-", "")
                split_map[patient_id] = split

        # Load targets
        with open(targets_json, 'r') as f:
            self.targets = json.load(f)

        # Build samples list
        self.samples = []  # each entry: dict with paths + meta
        for patient_id in os.listdir(patients_path):
            
            split = split_map.get(patient_id)
    
            if split == "val": 
                split = "test"

            if split != self.desired_split:
                continue

            patient_dir = os.path.join(patients_path, patient_id)
            meta_dir    = os.path.join(patient_dir, "metadata")
            if not os.path.isdir(patient_dir) or not os.path.isdir(meta_dir):
                continue

            # find all *_right.npy and *_left.npy
            for fname in os.listdir(patient_dir):
                if not fname.endswith("_right.npy") and not fname.endswith("_left.npy"):
                    continue

                side = 'right' if fname.endswith('_right.npy') else 'left'
                stem = fname[:-4]  # strip .npy
                npy_path  = os.path.join(patient_dir, fname)
                json_path = os.path.join(meta_dir, stem + ".json")

                # parse subject, visit from stem: CHECK-0003088/T08/left -> 0003088_T08_APO_left
                parts = stem.split('_')
                if len(parts) < 3:
                    continue
                subject_id, visit = parts[0], parts[1]
                key = f"CHECK-{subject_id}/{visit}/{side}"

                # lookup target
                entry = self.targets.get(key)
                if entry is None:
                    continue
                kl = entry.get("kellgren")
                if not isinstance(kl, (int, float)):
                    continue

                # Define target
                label = kl > 1

                # ensure files exist
                if not (os.path.isfile(npy_path) and os.path.isfile(json_path)):
                    continue

                self.samples.append({
                    'image_path': npy_path,
                    'meta_path':  json_path,
                    'target': label,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx=0):
        sample = self.samples[idx]

        # Load image
        arr = np.load(sample['image_path'])            # (H,W) float32
        arr = (arr - arr.min()) / (arr.max() - arr.min()) # normalize range of DICOM to [0,1]
        img = torch.from_numpy(arr).unsqueeze(0)       # [1,H,W]

        # Load metadata and filter in-frame points
        with open(sample['meta_path'], 'r') as f:
            meta = json.load(f)

        pts = meta['adjusted_points']   # dict: str(idx) -> [x,y]

        # build an (N,2) array of only in-frame points, sorted by idx
        coords = []
        for idx_str, (x, y) in pts.items():
            if pts.get(idx_str, False):
                coords.append([x, y])
        pts_arr = torch.tensor(coords, dtype=torch.float32)  # [N,2]

        # Optional augmentation
        if self.augment:
            img = augment_data(img)

        # Optional transform on image
        if self.transform is not None:
            img = self.transform(img)
        
        # Create roi_mask
        _, H, W = img.shape
        pts_list = []
        for idx_str, (x, y) in pts.items():
            pts_list.append([x, y])

        roi_indices = [18,19,20,21,22,23,24,25,26,75,74,73,72,71,69,18]
        roi_pts = [tuple(pts_list[i]) for i in roi_indices]
        roi_mask_np = make_roi_mask(H, W, roi_pts, dilation_radius=5)
        roi_mask    = torch.from_numpy(roi_mask_np).to(torch.bool)  # BoolTensor [H,W]

        return {
            'image':  img,          # FloatTensor [1,H,W]
            'points': pts_arr,      # FloatTensor [N,2]
            'roi_mask': roi_mask,
            'target': sample['target'],  # boolean
        }

def augment_data(data):
    finetune_transforms = T.Compose([
        # 1) Geometric
        T.RandomRotation(10, fill=0),
        T.RandomAffine(0, translate=(0.05,0.05), scale=(0.95,1.05), shear=5, fill=0),

        # 2) Intensity
        T.RandomApply([T.RandomAdjustSharpness(2)], p=0.3),
        T.RandomApply([T.RandomAutocontrast()], p=0.3),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1,2.0))], p=0.2),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),

        # 3) Noise & occlusion
        T.Lambda(lambda img: img + torch.randn_like(img) * 0.01),
        T.RandomErasing(p=0.3, scale=(0.02,0.1), ratio=(0.3,3.3), value=0),
    ])
    return finetune_transforms(data)

def make_roi_mask(height, width, roi_pts, dilation_radius=2):
    """
    roi_pts: list of (x,y) in pixel coords, closed polygon.
    dilation_radius: how many pixels to grow the mask.
    """
    # Unzip and swap to (row, col)
    xs, ys = zip(*roi_pts)
    rr, cc = polygon(ys, xs, shape=(height, width))
    mask = np.zeros((height, width), dtype=bool)
    mask[rr, cc] = True

    if dilation_radius > 0:
        mask = binary_dilation(mask, iterations=dilation_radius)

    return mask