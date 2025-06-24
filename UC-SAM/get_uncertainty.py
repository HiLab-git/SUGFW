import argparse
import pandas as pd
import os
import pickle
import sys
import json

from tqdm import tqdm
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from augmentation import *
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from numpy import ndarray
from typing import *
import random


def get_aug_image(image: ndarray) -> ndarray:
    aug_dict = {}
    image = (image - image.min()) / (image.max() - image.min() + 1e-6)
    image, gamma = gamma_correction(image)
    aug_dict["gamma"] = float(gamma)
    image, axis = random_flip(image)
    aug_dict["flip"] = float(axis)
    image, angle = random_rotate(image)
    aug_dict["rotate"] = float(angle)
    image, bits = posterization(image)
    aug_dict["posterization"] = float(bits)
    image, factor = contrast_adjustment(image)
    aug_dict["contrast"] = float(factor)
    image, factor = sharpness_enhancement(image)
    aug_dict["sharpness"] = float(factor)
    image, factor = brightness_modification(image)
    aug_dict["brightness"] = float(factor)

    return image, aug_dict


def get_mask(image: ndarray, mask_generator) -> List[ndarray]:
    width, height = image.shape[:2]
    with torch.no_grad():
        masks = mask_generator.generate(image)
    solid_masks = np.zeros((image.shape[0], image.shape[1])).astype(np.uint8)
    for mask in masks:
        if (
            mask["predicted_iou"] > 0.95
            and mask["stability_score"] > 0.95
            and mask["area"] < width * height / 4
        ):
            solid_masks |= mask["segmentation"]
    # print('-' * 100)
    solid_masks[solid_masks > 0] = 1
    return solid_masks


def read_image(image_path: str) -> ndarray:
    if image_path.endswith(".nii.gz") or image_path.endswith(".nii"):
        image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image)
    elif image_path.endswith(".png") or image_path.endswith(".jpg"):
        image = Image.open(image_path)
        image = np.array(image)
    elif image_path.endswith(".pkl"):
        image = pickle.load(open(image_path, "rb"))

    return image


def get_uncertainty(args):
    model_type, checkpoint, csv_path, k_times_aug = (
        args.model_type,
        args.checkpoint,
        args.csv_path,
        args.k_times_aug,
    )
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam = sam.cuda()
    mask_generator = SamAutomaticMaskGenerator(sam)

    files = pd.read_csv(csv_path)["image_pth"].tolist()
    masks = []

    log_data = {}
    dir_name, base_name = os.path.dirname(csv_path), os.path.basename(csv_path)
    log_file_path = dir_name.replace("splits", "un_log")
    output_feature_dir = dir_name.replace("splits", "feature")
    output_uncertainty_dir = dir_name.replace("splits", "uncertainty")
    os.makedirs(log_file_path, exist_ok=True)
    os.makedirs(output_feature_dir, exist_ok=True)
    os.makedirs(output_uncertainty_dir, exist_ok=True)
    log_file_path = os.path.join(
        log_file_path, base_name.replace(".csv", f"un{k_times_aug}log.json")
    )
    for file in tqdm(files):
        image_path = file
        image = read_image(image_path)

        masks = []
        aug_dicts = []  # 用于存储每次增强的参数

        image = image.repeat(3, axis=-1)
        for _ in tqdm(range(k_times_aug)):
            aug_image, aug_dict = get_aug_image(image)
            mask = get_mask(aug_image, mask_generator)
            if "rotate" in aug_dict:
                mask, _ = random_rotate(mask, 360 - int(aug_dict["rotate"]))
            if "flip" in aug_dict:
                mask, _ = random_flip(mask, int(aug_dict["flip"]))

            masks.append(mask)
            aug_dicts.append(aug_dict)

        uncertainty = np.zeros((image.shape[0], image.shape[1]))
        for mask in masks:
            uncertainty += mask

        uncertainty /= k_times_aug
        uncertainty = uncertainty * (1 - uncertainty) + 0.5  # 转换为不确定性

        # 保存不确定性文件
        suffix = os.path.splitext(image_path)[1]
        uncertainty_path = image_path.replace(
            suffix, f"_uncertainty_{k_times_aug}.pkl"
        ).replace("image", "uncertainty")
        pickle.dump(uncertainty, open(uncertainty_path, "wb"))

        if not os.path.exists(image_path.replace(suffix, "_feature.pkl")):
            with torch.no_grad():
                image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).cuda()
                image = (image - image.min()) / (image.max() - image.min()) * 255
                image = sam.preprocess(image).float()
                feature = sam.image_encoder(image)
                pickle.dump(
                    feature.cpu().numpy(),
                    open(
                        image_path.replace(suffix, "_feature.pkl").replace(
                            "image", "feature"
                        ),
                        "wb",
                    ),
                )

        # 生成日志信息
        log_message = {
            "uncertainty_mean": float(uncertainty.mean()),
            "uncertainty_max": float(uncertainty.max()),
            "uncertainty_min": float(uncertainty.min()),
            "aug_dicts": aug_dicts,  # 保存增强参数
        }

        # 将日志信息添加到 JSON 数据中
        log_data[file] = log_message
        tqdm.write(
            f"Processed {file}, uncertainty_mean: {uncertainty.mean()}, uncertainty_std:{uncertainty.std()}, uncertainty_max: {uncertainty.max()}"
        )

    with open(log_file_path, "w") as log_file:
        json.dump(log_data, log_file, indent=4)

    print(f"Log data has been saved to {log_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../checkpoint/sam_vit_b_01ec64.pth",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="../data/Promise12/Foreslices/splits/train.csv",
    )

    parser.add_argument("--k_times_aug", type=int, default=10)
    args = parser.parse_args()
    get_uncertainty(args=args)
