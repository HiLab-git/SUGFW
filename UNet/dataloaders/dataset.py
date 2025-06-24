import pickle
import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd


class base_dataset(Dataset):
    """
    base_dir: the location of data.
    split: the part used for training, validation or testing.
    transform: augmentation to images.
    """

    def __init__(self, csv_path, transform=None):
        self.image_paths = pd.read_csv(csv_path)["image_pth"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = image_path.replace("image", "label")
        with open(image_path, "rb") as f:
            image = pickle.load(f)
        with open(label_path, "rb") as f:
            label = pickle.load(f)
        
        sample = {"image": image.squeeze(2), "label": label.squeeze(2)}
        if self.transform is not None:
            sample = self.transform(sample) 
        

        return sample
        


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if self.with_sdf:
            sdf = sample["sdf"]

        # pad the sample if necessary
        if (
            label.shape[0] <= self.output_size[0]
            or label.shape[1] <= self.output_size[1]
        ):
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            image = np.pad(
                image, [(pw, pw), (ph, ph)], mode="constant", constant_values=0
            )
            label = np.pad(
                label, [(pw, pw), (ph, ph)], mode="constant", constant_values=0
            )
            if self.with_sdf:
                sdf = np.pad(
                    sdf, [(pw, pw), (ph, ph)], mode="constant", constant_values=0
                )

        (w, h) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])

        label = label[w1 : w1 + self.output_size[0], h1 : h1 + self.output_size[1]]
        image = image[w1 : w1 + self.output_size[0], h1 : h1 + self.output_size[1]]
        if self.with_sdf:
            sdf = sdf[w1 : w1 + self.output_size[0], h1 : h1 + self.output_size[1]]
            return {"image": image, "label": label, "sdf": sdf}
        else:
            return {"image": image, "label": label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {"image": image, "label": label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample["image"]
        image = np.expand_dims(image, axis=0).astype(np.float32)
        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(sample["label"]).float(),
        }
