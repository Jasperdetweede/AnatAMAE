import os
import json
import numpy as np
import skimage
import torch
from torch.utils.data import Dataset

class CheckDataset(Dataset):
    def __init__(self, processed_root: str, patients_path: str, targets_json: str, transform=None, augment=False):
        """
        Args:
            processed_root: root folder containing patient subdirs
            targets_json:   path to your targets.json file
            transform:      optional torchvision transforms on the image tensor
        """
        self.processed_root = processed_root
        self.patients_path = patients_path
        self.transform = transform
        self.augment = augment

        # Load targets
        with open(targets_json, 'r') as f:
            self.targets = json.load(f)

        # Build samples list
        self.samples = []  # each entry: dict with paths + meta
        for patient_id in os.listdir(patients_path):
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
        img = torch.from_numpy(arr).unsqueeze(0)       # [1,H,W]

        # Load metadata and filter in-frame points
        with open(sample['meta_path'], 'r') as f:
            meta = json.load(f)

        pts = meta['adjusted_points']   # dict: str(idx) -> [x,y]
        # in_frame = meta['in_frame']     # dict: str(idx) -> bool

        # build an (N,2) array of only in-frame points, sorted by idx
        coords = []
        for idx_str, (x, y) in pts.items():
            if pts.get(idx_str, False):
                coords.append([x, y])
        pts_arr = torch.tensor(coords, dtype=torch.float32)  # [N,2]

        # 3. Optional transform on image
        if self.transform is not None:
            img = self.transform(img)

        return {
            'image':  img,          # FloatTensor [1,H,W]
            'points': pts_arr,      # FloatTensor [N,2]
            'target': sample['target'],  # dict with jsn_*, osteo_*, kellgren
        }

    def _augment_rotate(self, img: torch.FloatTensor, pts: torch.FloatTensor):
        """
        Random rotation between -10° and +10° around image center.
        Rotates both img (on CPU) and updates pts accordingly.
        """
        angle = float(torch.randint(-10, 11, (1,)).item())  # degrees
        # Convert to skimage’s expectation: img HWC, normalize to [0,1]
        arr = img.squeeze(0).numpy()
        center = (np.array(arr.shape[::-1]) - 1) / 2  # (x_center, y_center)

        # Rotate image
        rot_img = skimage.transform.rotate(
            arr,
            angle=angle,
            center=center,
            mode='constant',
            cval=arr.min(),
            preserve_range=True
        ).astype(np.float32)
        rot_img = torch.from_numpy(rot_img).unsqueeze(0)

        # Rotate points
        theta = np.deg2rad(angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
        pts_np = pts.numpy()  # [N,2]
        # shift to origin, rotate, shift back
        shifted = pts_np - center[None, :]
        rotated = shifted.dot(R.T) + center[None, :]

        return rot_img, torch.from_numpy(rotated).float()