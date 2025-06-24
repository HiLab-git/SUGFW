from PIL import Image
import os
import pickle
import numpy as np
import pandas as pd
from skimage.measure import block_reduce
from sklearn.cluster import KMeans
from tqdm import tqdm
import argparse

""" 
    image: H, W, 1
    feature: 1, 256, 64, 64
    uncertinaty: H, W
"""


class Sample_Selector(object):
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.files = pd.read_csv(csv_path)["image_pth"].tolist()
        self.suffix = os.path.splitext(self.files[0])[1]

        # select_num = int(len(self.files) * select_ratio)
        self.features, self.name_list = self._load_features()

    @staticmethod
    def average_pooling(
        uncertainty: np.ndarray, target_size: tuple = (64, 64)
    ) -> np.ndarray:
        """
        Perform average pooling to resize the uncertainty map to the target size.

        Args:
            uncertainty (np.ndarray): Input uncertainty map of shape (H, W).
            target_size (tuple): Target size (height, width), default is (64, 64).

        Returns:
            np.ndarray: Resized uncertainty map of shape (64, 64).
        """
        input_height, input_width = uncertainty.shape
        target_height, target_width = target_size

        # Calculate pooling block size
        block_size = (input_height // target_height, input_width // target_width)

        # Perform average pooling using block_reduce
        pooled_uncertainty = block_reduce(
            uncertainty, block_size=block_size, func=np.mean
        )

        return pooled_uncertainty

    @staticmethod
    def weighted_feature_fusion(
        feature: np.ndarray, pooled_uncertainty: np.ndarray
    ) -> np.ndarray:
        """
        Perform weighted fusion of feature maps using pooled uncertainty.

        Args:
            feature (np.ndarray): Input feature map of shape (1, 256, 64, 64).
            pooled_uncertainty (np.ndarray): Pooled uncertainty map of shape (64, 64).

        Returns:
            np.ndarray: Weighted feature of shape (1, 256).
        """
        # Normalize pooled_uncertainty to sum to 1
        weights = pooled_uncertainty / np.sum(pooled_uncertainty)

        # Reshape weights to match feature dimensions
        # print(feature.shape, weights.shape)
        weights = weights.reshape(1, 1, weights.shape[0], weights.shape[1])

        # Perform weighted sum along spatial dimensions
        weighted_feature = np.sum(feature * weights, axis=(2, 3))  # Shape: (1, 256)

        return weighted_feature
    def my_select(self, select_ratio):

        select_num = int(select_ratio * len(self.files))
        results = []  # To store weighted_feature and avg_uncertainty for each file

        for image_file in tqdm(self.files):
            feature_file = image_file.replace(".pkl", "_feature.pkl").replace(
                "image", "feature"
            )
            uncertianty_file = image_file.replace(
                ".pkl", "_uncertainty_10.pkl"
            ).replace("image", "uncertainty")
            feature, uncertainty = (
                pickle.load(open(feature_file, "rb")),
                pickle.load(open(uncertianty_file, "rb")),
            )

            pooled_uncertainty = self.average_pooling(
                uncertainty, target_size=(feature.shape[-1], feature.shape[-1])
            )
            avg_uncertainty = uncertainty.mean()

            weighted_feature = self.weighted_feature_fusion(feature, pooled_uncertainty)

            results.append(
                {
                    "image_pth": image_file,
                    "weighted_feature": weighted_feature,
                    "avg_uncertainty": avg_uncertainty,
                }
            )

        # Prepare data for clustering
        weighted_features = np.array(
            [result["weighted_feature"].flatten() for result in results]
        )  # Shape: (num_files, 256)

        # Perform KMeans clustering
        n_clusters = select_num
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=2025)
        clusters = kmeans.fit_predict(weighted_features)
        # Add cluster labels to results
        for i, result in enumerate(results):
            result["cluster"] = int(clusters[i])

        # Select one sample from each cluster
        selected_samples = []  # To store selected samples
        for cluster_id in range(n_clusters):
            cluster_samples = [
                result for result in results if result["cluster"] == cluster_id
            ]

            if not selected_samples:
                # For the first cluster, select the sample with avg_uncertainty closest to the median
                uncertainties = np.array(
                    [sample["avg_uncertainty"] for sample in cluster_samples]
                )
                median_uncertainty = np.median(uncertainties)
                closest_sample = cluster_samples[
                    np.argmin(np.abs(uncertainties - median_uncertainty))
                ]
            else:
                # For subsequent clusters, select the sample with avg_uncertainty farthest from already selected samples
                selected_uncertainties = np.array(
                    [sample["avg_uncertainty"] for sample in selected_samples]
                )
                distances = np.array(
                    [
                        np.min(
                            np.abs(selected_uncertainties - sample["avg_uncertainty"])
                        )
                        for sample in cluster_samples
                    ]
                )
                closest_sample = cluster_samples[np.argmax(distances)]

            selected_samples.append(
                {
                    "image_pth": closest_sample["image_pth"],
                    "mask_pth": closest_sample["image_pth"].replace("image", "label"),
                    "avg_uncertainty": closest_sample["avg_uncertainty"],
                }
            )

        # Save selected samples to a CSV file
        output_csv_file = os.path.join(
            os.path.dirname(self.csv_path),
            f"my_select_{n_clusters}.csv",
        )
        selected_samples_df = pd.DataFrame(selected_samples)
        selected_samples_df.to_csv(output_csv_file, index=False)

        print(
            f"Successfully saved selected samples to {output_csv_file} with my method!"
        )

    def my_select_wo_featureweighting(self, select_ratio):
        select_num = int(select_ratio * len(self.files))
        results = []  # To store weighted_feature and avg_uncertainty for each file

        for image_file in tqdm(self.files):
            feature_file = image_file.replace(".pkl", "_feature.pkl").replace(
                "image", "feature"
            )
            uncertianty_file = image_file.replace(
                ".pkl", "_uncertainty_10.pkl"
            ).replace("image", "uncertainty")
            feature, uncertainty = (
                pickle.load(open(feature_file, "rb")),
                pickle.load(open(uncertianty_file, "rb")),
            )

            weighted_feature = feature.mean(axis=(2, 3))

            avg_uncertainty = uncertainty.mean()
            results.append(
                {
                    "image_pth": image_file,
                    "weighted_feature": weighted_feature,
                    "avg_uncertainty": avg_uncertainty,
                }
            )

        # Prepare data for clustering
        weighted_features = np.array(
            [result["weighted_feature"].flatten() for result in results]
        )  # Shape: (num_files, 256)

        # Perform KMeans clustering
        n_clusters = select_num
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=2025)
        clusters = kmeans.fit_predict(weighted_features)
        # Add cluster labels to results
        for i, result in enumerate(results):
            result["cluster"] = int(clusters[i])

        # Select one sample from each cluster
        selected_samples = []  # To store selected samples
        for cluster_id in range(n_clusters):
            cluster_samples = [
                result for result in results if result["cluster"] == cluster_id
            ]

            if not selected_samples:
                # For the first cluster, select the sample with avg_uncertainty closest to the median
                uncertainties = np.array(
                    [sample["avg_uncertainty"] for sample in cluster_samples]
                )
                median_uncertainty = np.median(uncertainties)
                closest_sample = cluster_samples[
                    np.argmin(np.abs(uncertainties - median_uncertainty))
                ]
            else:
                # For subsequent clusters, select the sample with avg_uncertainty farthest from already selected samples
                selected_uncertainties = np.array(
                    [sample["avg_uncertainty"] for sample in selected_samples]
                )
                distances = np.array(
                    [
                        np.min(
                            np.abs(selected_uncertainties - sample["avg_uncertainty"])
                        )
                        for sample in cluster_samples
                    ]
                )
                closest_sample = cluster_samples[np.argmax(distances)]

            selected_samples.append(
                {
                    "image_pth": closest_sample["image_pth"],
                    "mask_pth": closest_sample["image_pth"].replace("image", "label"),
                    "avg_uncertainty": closest_sample["avg_uncertainty"],
                }
            )

        # Save selected samples to a CSV file
        output_csv_file = os.path.join(
            os.path.dirname(self.csv_path),
            f"my_select_{n_clusters}_wo_featureweighting.csv",
        )
        selected_samples_df = pd.DataFrame(selected_samples)
        selected_samples_df.to_csv(output_csv_file, index=False)

        print(
            f"Successfully saved selected samples to {output_csv_file} with my method!"
        )

    def my_select_wo_GS(self, select_ratio):

        select_num = int(select_ratio * len(self.files))
        results = []  # To store weighted_feature and avg_uncertainty for each file

        for image_file in tqdm(self.files):
            feature_file = image_file.replace(".pkl", "_feature.pkl").replace(
                "image", "feature"
            )
            uncertianty_file = image_file.replace(
                ".pkl", "_uncertainty_10.pkl"
            ).replace("image", "uncertainty")
            feature, uncertainty = (
                pickle.load(open(feature_file, "rb")),
                pickle.load(open(uncertianty_file, "rb")),
            )

            pooled_uncertainty = self.average_pooling(
                uncertainty, target_size=(feature.shape[-1], feature.shape[-1])
            )
            avg_uncertainty = uncertainty.mean()

            weighted_feature = self.weighted_feature_fusion(feature, pooled_uncertainty)

            results.append(
                {
                    "image_pth": image_file,
                    "weighted_feature": weighted_feature,
                    "avg_uncertainty": avg_uncertainty,
                }
            )

        # Prepare data for clustering
        weighted_features = np.array(
            [result["weighted_feature"].flatten() for result in results]
        )  # Shape: (num_files, 256)

        # Perform KMeans clustering
        n_clusters = select_num
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=2025)
        clusters = kmeans.fit_predict(weighted_features)
        # Add cluster labels to results
        for i, result in enumerate(results):
            result["cluster"] = int(clusters[i])

        # Select one sample from each cluster
        selected_samples = []  # To store selected samples
        for cluster_id in range(n_clusters):
            cluster_samples = [
                result for result in results if result["cluster"] == cluster_id
            ]

            choosed_sample = np.random.choice(cluster_samples)

            selected_samples.append(
                {
                    "image_pth": choosed_sample["image_pth"],
                    "mask_pth": choosed_sample["image_pth"].replace("image", "label"),
                    "avg_uncertainty": choosed_sample["avg_uncertainty"],
                }
            )

        # Save selected samples to a CSV file
        output_csv_file = os.path.join(
            os.path.dirname(self.csv_path),
            f"my_select_{n_clusters}_wo_GS.csv",
        )
        selected_samples_df = pd.DataFrame(selected_samples)
        selected_samples_df.to_csv(output_csv_file, index=False)

        print(
            f"Successfully saved selected samples to {output_csv_file} with my method!"
        )

    def _load_features(self):
        """
        Load features and corresponding names.
        """
        # Implement feature loading logic here
        features = []
        name_list = []
        for image_file in tqdm(self.files):
            feature_file = image_file.replace(".pkl", "_feature.pkl").replace(
                "image", "feature"
            )
            with open(feature_file, "rb") as f:
                feature = pickle.load(f)

            feature = feature.mean(axis=(2, 3))  # Shape: (1, 256)
            feature = feature.flatten()  # Shape: (256,)

            features.append(feature)
            name_list.append(image_file)
        features = np.array(features)  # Shape: (num_files, 256)
        print("Loaded features finish!")
        return features, name_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample Selector")
    parser.add_argument(
        "--base_csv_path",
        type=str,
        default="../data/Promise12/Foreslices/splits/train.csv",
        help="Path to the CSV file containing training image paths.",
    )
    parser.add_argument(
        "--select_ratios",
        type=float,
        nargs='+',
        help="Ratio of samples to select from the dataset.",
    )
    args = parser.parse_args()
    selector = Sample_Selector(args.base_csv_path)
    for select_ratio in args.select_ratios:
        selector.my_select(select_ratio=select_ratio)
        selector.my_select_wo_featureweighting(select_ratio=select_ratio)
        selector.my_select_wo_GS(select_ratio=select_ratio)
            