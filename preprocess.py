import matplotlib.pyplot as plt
import pydicom
import numpy as np
import skimage
import os
import json
import shutil

############################
# Resampling 
############################

def resample(img: np.ndarray, ds: pydicom.Dataset, target_spacing: float):
    """
    Load a DICOM file and resample the image array to a constant mm/pixel spacing.

    Args:
        img: Image pixels in ndarray.
        ds: DICOM file
        target_spacing: Desired pixel spacing in mm/pixel.

    Returns:
        img_resampled: 2D NumPy array of the resampled image.
    """
    
    # Extract pixel spacing
    spacing = getattr(ds, 'PixelSpacing', None) or getattr(ds, 'ImagerPixelSpacing', None)
    orig_spacing = [float(spacing[0]), float(spacing[1])]  # [row_spacing, col_spacing]

    # Load and apply rescale slope/intercept
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    img = img * slope + intercept

    # Compute scale factor
    scale_factor = orig_spacing[0] / target_spacing

    # Resample
    img_resampled = skimage.transform.rescale(
        img,
        scale=scale_factor,
        preserve_range=True,
        anti_aliasing=True
    ).astype(np.float32)

    return img_resampled


############################
# Photometric interpretation 
############################

def standardise_intensity(ds: pydicom.Dataset, img_pixels: np.ndarray) -> np.ndarray:
    """
    Apply photometric interpretation to a DICOM image.

    Args:
        ds: pydicom Dataset
        img_pixels: 2D NumPy array of pixel intensities

    Returns:
        Adjusted image as a NumPy array.
    """
    
    photometric = ds.get('PhotometricInterpretation', 'MONOCHROME2')
    
    if photometric == 'MONOCHROME1':
        img_pixels = np.max(img_pixels) - img_pixels
    elif photometric != 'MONOCHROME2':
        raise ValueError(f'PhotometricInterpretation "{photometric}" not supported')

    # Check VOILUTFunction
    voilut = ds.get('VOILUTFunction', 'LINEAR')
    if voilut != 'LINEAR':
        raise ValueError(f'Only VOILUTFunction=LINEAR supported, got: {voilut}')
    
    return img_pixels

############################
# Cropping 
############################

def load_bonefinder_pts(pts_path: str) -> np.ndarray:
    """Load a BoneFinder .pts file as an (N,2) array of (x_mm, y_mm)."""
    with open(pts_path, 'r') as f:
        # skip to opening brace
        for line in f:
            if line.strip() == '{':
                break
        pts = []
        for line in f:
            if line.strip() == '}':
                break
            x_mm, y_mm = map(float, line.split())
            pts.append([x_mm, y_mm])
    return np.array(pts, dtype=float)

def crop_hips(
    img: np.ndarray,
    pts_path: str,
    crop_size: int,
    output_size: int,
    target_spacing: float,
) -> dict:
    """
    1) Crops left/right hip centered on femoral-head subcurve.
    2) Downsamples to `output_size`×`output_size`.
    3) Returns per-side (image, metadata) with global BoneFinder point indices.

    metadata:
      - crop_origin: {x, y} in original image (px)
      - pixel_spacing: mm per pixel
      - downsample_factor
      - adjusted_points: dict of global point_idx -> [x, y] in crop-space
      - in_frame: dict of global point_idx -> bool
      - subcurves: dict of subcurve name -> list of [x, y]
    """
    # constants
    SIDES = { 'right': 0, 'left': 80 }
    SUB_CURVES = {'femoral head': [18, 19, 20, 21, 22, 23, 24, 25, 26, 27]}

    # load
    pts_mm = load_bonefinder_pts(pts_path)   # (160, 2)
    pts_px = pts_mm / target_spacing

    results = {}

    for side, offset in SIDES.items():
        # find crop center from femoral head
        fh_idxs = SUB_CURVES['femoral head']
        fh_pts = pts_px[offset + np.array(fh_idxs)]  # (10, 2)
        xc, yc = fh_pts.mean(axis=0)

        # crop image 
        h, w = img.shape
        half = crop_size // 2
        x0 = int(np.clip(xc - half, 0, w - crop_size))
        y0 = int(np.clip(yc - half, 0, h - crop_size))
        crop = img[y0:y0+crop_size, x0:x0+crop_size]

        # downsample 
        factor = output_size / crop_size
        crop_ds = skimage.transform.rescale(
            crop,
            scale=factor,
            preserve_range=True,
            anti_aliasing=True
        ).astype(np.float32)

        # transform all 80 points on this side
        adjusted_points = {}
        in_frame = {}

        for i in range(80):
            global_idx = offset + i
            x_px, y_px = pts_px[global_idx]
            x_adj = (x_px - x0) * factor
            y_adj = (y_px - y0) * factor
            adjusted_points[str(global_idx)] = [float(x_adj), float(y_adj)]
            in_frame[str(global_idx)] = (0 <= x_adj < output_size and 0 <= y_adj < output_size)

        # build subcurve data in crop-space
        subcurves = {}
        for name, idxs in SUB_CURVES.items():
            subcurves[name] = [
                adjusted_points[str(offset + idx)] for idx in idxs
            ]

        meta = {
            "crop_origin": {"x": x0, "y": y0},
            "pixel_spacing": float(target_spacing),
            "downsample_factor": factor,
            "adjusted_points": adjusted_points,  # global idx -> [x,y]
            "in_frame": in_frame
        }

        results[side] = (crop_ds, meta)

    return results


############################
# Mirroring
############################

def mirror_image_and_points(img: np.ndarray, meta: dict):
    """
    Horizontally flip img_crop and remap all landmarks in meta
    so that x_new = (W - 1) - x_old.
    """
    H, W = img.shape
    flipped_img = img[:, ::-1]

    flipped_points = {}
    for idx, (x, y) in meta['adjusted_points'].items():
        flipped_points[idx] = [(W - 1) - x, y]

    # Flip subcurves
    flipped_subcurves = {
        name: [[(W - 1) - x, y] for (x, y) in pts_list]
        for name, pts_list in meta.get('subcurves', {}).items()
    }

    new_meta = {
        **meta,
        'adjusted_points': flipped_points,
        'subcurves': flipped_subcurves
    }
    return flipped_img, new_meta


############################
# Util 
############################

def _json_converter(o):
    """
    JSON serializer for objects not serializable by default json cod
    """
    if isinstance(o, np.generic):
        return o.item() 
    raise TypeError(f"Type {type(o)} not serializable")

def write_data(root_dir: str, original_fname: str, crops: dict):
    """
    Store cropped NumPy arrays and metadata in a clean folder structure,
    serializing NumPy scalars correctly.
    """
    patient_id = original_fname.split('_', 1)[0]
    patient_dir = os.path.join(root_dir, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    meta_dir = os.path.join(patient_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    for side, (img_crop, meta) in crops.items():
        base = f"{original_fname}_{side}"
        npy_path  = os.path.join(patient_dir, f"{base}.npy")
        json_path = os.path.join(meta_dir, f"{base}.json")

        # Save the crop
        np.save(npy_path, img_crop)

        # Save the metadata, with our converter for NumPy types
        with open(json_path, "w") as jf:
            json.dump(meta, jf, indent=2, default=_json_converter)

    print(f"Saved crops and metadata for patient {patient_id} under {patient_dir}")

def add_meta_file(dataset_path, output_size, crop_size, target_spacing):
    # define the curves: right first, then left
    SIDES = { 'right': 0, 'left': 80 }
    CURVES = {
        'proximal femur':     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        'greater trochanter': [6, 35, 36, 37, 38, 39],
        'posterior wall':     [40, 41, 42, 43, 44],
        'ischium and pubis':  [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
        'foramen':            [60, 61, 62, 63, 64, 65, 66],
        'acetabular roof':    [67, 68, 69, 70, 71, 72, 73, 74],
        'teardrop':           [75, 76, 77, 78, 79],
    }
    SUB_CURVES = {
        'femoral head':       [18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        'sourcil':            [70, 71, 72, 73, 74],
    }

    meta = {
        "image_size": output_size,
        "cropped_size": crop_size,
        "resample_spacing": target_spacing,
        "pointmap": {
            "sides": SIDES,
            "curves": CURVES,
            "sub_curves": SUB_CURVES
        }
    }

    json_path = os.path.join(dataset_path, "metadata.json")

    # Save to JSON
    with open(json_path, "w") as jf: 
        json.dump(meta, jf, indent=2, default=_json_converter)

def find_valid_images(images_root, pointfiles_root, targets_path):
    """
    Finds image files in subfolders of `images_root` and checks if a corresponding
    .dcm.pts file exists in the matching subfolder under `pointfiles_root`.
    
    Returns a list of (relative_subdir, filename_wo_ext) tuples for valid image/pointfile pairs.
    """
    valid_images = []

    # Load metadata
    if not os.path.exists(targets_path):
        print(f"Missing targets.json file at {targets_path}")

    with open(targets_path, "r") as jf:
        targets = json.load(jf)

    for subdir_name in os.listdir(images_root):
        image_subdir = os.path.join(images_root, subdir_name)
        pointfile_subdir = os.path.join(pointfiles_root, subdir_name)

        if not os.path.isdir(image_subdir) or not os.path.isdir(pointfile_subdir):
            continue  # skip if not a matching subfolder in both

        for file in os.listdir(image_subdir):
            name, ext = os.path.splitext(file)

            # Check if the file is a dcm file
            if ext.lower() != ".dcm":
                continue

            # Check if there is a valid target in targets.json
            parts = name.split("_")
            if len(parts) < 2:
                raise ValueError(f"Unexpected filename format: {file}")   
            
            # Check left side target
            keyleft = f"CHECK-{parts[0]}/{parts[1]}/left"
            if targets.get(keyleft) is None: continue
            if not isinstance(targets.get(keyleft).get("kellgren"), (int, float)): continue

            # Check right side target
            keyright = f"CHECK-{parts[0]}/{parts[1]}/right"
            if targets.get(keyright) is None: continue
            if not isinstance(targets.get(keyright).get("kellgren"), (int, float)): continue

            # Check if there is a corresponding pts file
            pointfile_path = os.path.join(pointfile_subdir, name + ".dcm.pts")
            if os.path.isfile(pointfile_path):
                valid_images.append((subdir_name, name)) 
                
    return valid_images


############################
# Main function
############################

#  main preprocessing function
def preprocess(images_path, pointfiles_path, dataset_out_path, targets_path, split_path, output_size, crop_size, target_spacing=0.1) -> dict:

    if not os.path.exists(dataset_out_path):
        os.makedirs(dataset_out_path)
        
    # Add metadata file and define patients folder
    add_meta_file(dataset_out_path, output_size, crop_size, target_spacing)
    shutil.copy(split_path, dataset_out_path)
    shutil.copy(targets_path, dataset_out_path)
    output_path = os.path.join(dataset_out_path, "patients")

    # List all items in the images directory
    images = find_valid_images(images_path, pointfiles_path, targets_path)
    
    # Preprocess for all images
    for subdir, image in images: 
        try:
            image_path = os.path.join(images_path, subdir, image + ".dcm")
            pointfile_path = os.path.join(pointfiles_path, subdir, image + ".dcm.pts")

            # read image
            dcm = pydicom.dcmread(image_path)
            img = dcm.pixel_array.astype(np.float32)

            # correct intensity, resample, crop, flip left hip
            correct_intens_img = standardise_intensity(dcm, img)
            resampled_img = resample(correct_intens_img, dcm, target_spacing)
            cropped = crop_hips(resampled_img, pointfile_path, crop_size, output_size, target_spacing)
            cropped['left'] = mirror_image_and_points(cropped['left'][0], cropped['left'][1])

            # store data in the correct folders 
            write_data(output_path, image, cropped)
        except: 
            print("Error occured, skipping image")

    
if __name__ == "__main__":
    images_path = "check_raw/images"
    pointfiles_path = "check_raw/pointfiles"
    dataset_out_path = "data/check"
    targets_path = "check_raw/targets.json"
    split_path = "check_raw/train_test_val_split.txt"
    output_size = 256
    crop_size = 1024

    print("Creating datatset in: " + dataset_out_path.upper())
    crops = preprocess(images_path, pointfiles_path, dataset_out_path, targets_path, split_path, output_size, crop_size)