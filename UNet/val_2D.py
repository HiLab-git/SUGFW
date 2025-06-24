from ctypes.wintypes import HDC
import logging
import sys
import SimpleITK as sitk
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import pandas as pd
import pickle
import os
import argparse
from tqdm import tqdm
from networks.net_factory import net_factory

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, net, patch_size, volume_id, save_path=None):
    image = np.transpose(image, (2, 0, 1))  
    label = np.transpose(label, (2, 0, 1))  
    prediction = np.zeros_like(label)
    for ind in tqdm(range(image.shape[0]), desc=f"Inferencing volume {volume_id}"):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred

    pred = prediction.astype(np.float32)
    Dice, HD95 = calculate_metric_percase(pred, label)
    logging.info(f"Case{volume_id:03d} - Dice: {Dice:.4f}, HD95: {HD95:.4f}")
       
    if save_path is not None:
        sitk.WriteImage(sitk.GetImageFromArray(pred), save_path)

    return Dice, HD95


def val_binary_2d(csv_path, net, patch_size, output_path=None):
    if output_path is not None:
        stage = os.path.basename(csv_path).split('.')[0]
        output_path = os.path.join(output_path, stage) 
        os.makedirs(output_path, exist_ok=True)
    net.eval()
    df = pd.read_csv(csv_path)

    # 提取volume和slice编号
    df['volume_id'] = df['image_pth'].apply(lambda x: int(os.path.basename(x)[4:7]))
    df['slice_id'] = df['image_pth'].apply(lambda x: int(os.path.basename(x)[13:16]))

    # 按volume_id分组并按slice_id排序
    grouped = df.sort_values('slice_id').groupby('volume_id')
    # 加载pkl文件
    def load_pkl(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data[:, :, 0] if data.ndim == 3 and data.shape[-1] == 1 else data

    all_dice = []
    all_hd95 = []
    for volume_id, group in grouped:
        volume_slices = []
        label_slices = []
        for _, row in group.iterrows():
            volume_slices.append(load_pkl(row['image_pth']))
            label_slices.append(load_pkl(row['mask_pth']))

        
        volume = np.stack(volume_slices, axis=-1)  
        label = np.stack(label_slices, axis=-1)  
        if output_path is not None:
            save_path = os.path.join(output_path, f"Case{volume_id:03d}_pred.nii.gz")
        else:
            save_path = None
        Dice, HD95 = test_single_volume(volume, label, net, patch_size, volume_id, save_path=save_path)
        all_dice.append(Dice)
        all_hd95.append(HD95)
    all_dice = np.array(all_dice)
    all_hd95 = np.array(all_hd95)

    net.train()
    return all_dice.mean(), all_dice.std(), all_hd95.mean(), all_hd95.std()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Validation script for 2D UNet")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing image paths")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256], help="Patch size for inference")
    parser.add_argument("--class_num", type=int, default=2, help="Number of classes for segmentation")

    args = parser.parse_args()
    
        # Load the model
    net = net_factory(net_type="unet", in_chns=1, class_num=args.class_num)
    net.load_state_dict(torch.load(args.model_path))
    net.eval()
    
    output_path = os.path.dirname(args.model_path)
    # Run validation
    logging.basicConfig(
        filename=output_path + "/testing_log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    dice_mean, dice_std, hd95_mean, hd95_std = val_binary_2d(args.csv_path, net, args.patch_size, output_path)
    logging.info(f"Dice Mean: {dice_mean:.4f}, Dice Std: {dice_std:.4f}")
    logging.info(f"HD95 Mean: {hd95_mean:.4f}, HD95 Std: {hd95_std:.4f}")