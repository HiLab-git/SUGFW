import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
import pickle
from tqdm import tqdm
from scipy.ndimage import zoom


# 定义原始数据和目标路径
RAW_DATA_DIR = "../data/Promise12/raw"

Foreslices_IMAGE_DIR = "../data/Promise12/Foreslices/image"
Foreslices_LABEL_DIR = "../data/Promise12/Foreslices/label"


for dir_path in [
    Foreslices_IMAGE_DIR,
    Foreslices_LABEL_DIR,
]:
    os.makedirs(dir_path, exist_ok=True)

global_idx = 0
alls = ("training_data", "livechallenge_test_data", "test_data")
a = []
for s in alls:
    base_dir = os.path.join(RAW_DATA_DIR, s)
    files = [
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.endswith("mhd") and "seg" not in f
    ]
    files.sort()
    for file in tqdm(files):
        image = sitk.GetArrayFromImage(sitk.ReadImage(file))
        label = sitk.GetArrayFromImage(
            sitk.ReadImage(file.replace(".mhd", "_segmentation.mhd"))
        )

        """ Intensity Truncation """
        percentile_99 = np.percentile(image, 99)
        image = np.clip(image, a_min=None, a_max=percentile_99)

        """ Normalization """
        image = (image - image.mean() + 1e-6) / (image.std() + 1e-6)

        """ Scale to Dx256x256 """
        image = zoom(image, (1.0, 256 / image.shape[1], 256 / image.shape[2]), order=3)
        label = zoom(label, (1.0, 256 / label.shape[1], 256 / label.shape[2]), order=0)



        fore = np.any(label, axis=(1, 2))
        nonzero_indices = np.where(fore)[0]
        z_min, z_max = nonzero_indices[0], nonzero_indices[-1]
        image = np.transpose(image, (1, 2, 0))
        label = np.transpose(label, (1, 2, 0))

        for i in range(image.shape[2]):
            slice_id = i

            image_slice = image[:, :, i : i + 1]
            label_slice = label[:, :, i : i + 1]

            """ generate fore slices """
            if i >= z_min - 2 and i <= z_max + 2:
                image_path = os.path.join(
                    Foreslices_IMAGE_DIR,
                    f"Case{global_idx:03d}_Slice{slice_id:03d}.pkl",
                )
                label_path = os.path.join(
                    Foreslices_LABEL_DIR,
                    f"Case{global_idx:03d}_Slice{slice_id:03d}.pkl",
                )
                with open(image_path, "wb") as f:
                    pickle.dump(image_slice, f)
                with open(label_path, "wb") as f:
                    pickle.dump(label_slice, f)

            
        global_idx += 1
