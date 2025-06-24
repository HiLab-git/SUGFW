import numpy as np
import os.path as osp
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def select_valuable_samples_FPS(feats, name_list, num_samples):
    """
    Select valuable samples using KMeans clustering and Euclidean distance.

    Parameters:
    - feats: ndarray of shape (n, p), feature vectors.
    - name_list: ndarray of shape (n,), corresponding sample names.
    - num_samples: int, the number of valuable samples to select.

    Returns:
    - select_samples_list: list of selected sample names.
    """
    num_clusters = num_samples // 2  # Number of clusters for KMeans

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(feats)

    # Get the labels of each point
    labels = kmeans.labels_

    select_samples_list = []

    # Handle odd number of samples
    extra_sample_needed = num_samples % 2 == 1

    for i in range(num_clusters):
        # Get the indices of samples in the current cluster
        cluster_indices = np.where(labels == i)[0]

        # Get the feature vectors of the samples in the cluster
        cluster_feats = feats[cluster_indices]

        # Compute pairwise Euclidean distances within the cluster
        distances = cdist(cluster_feats, cluster_feats, metric='euclidean')

        # Find the indices of the two samples with the maximum distance
        max_dist_indices = np.unravel_index(np.argmax(distances), distances.shape)

        # Get the original indices of these two samples
        sample1_index = cluster_indices[max_dist_indices[0]]
        sample2_index = cluster_indices[max_dist_indices[1]]

        # Add the corresponding sample names to the list
        select_samples_list.append(name_list[sample1_index])
        select_samples_list.append(name_list[sample2_index])

    # If an extra sample is needed (odd number of samples), select one sample from the last cluster
    if extra_sample_needed:
        last_cluster_indices = np.where(labels == num_clusters - 1)[0]
        # Select only one sample, which could be the first one in the last cluster
        select_samples_list.append(name_list[last_cluster_indices[0]])

    return select_samples_list[:num_samples]  # Return exactly num_samples

def generate_FPS_plan(organ, feats_file, num_samples):
    np.random.seed(0)
    data_dict = dict(np.load(feats_file, allow_pickle=True))
    feats = np.array(data_dict['feats'])
    name_list = np.array(data_dict['name_list'])

    select_samples_list = select_valuable_samples_FPS(feats=feats, name_list=name_list, num_samples=num_samples)
    paths = [osp.join(f'./{organ}/data/{pid}.npz') for pid in select_samples_list]

    print(f'Select:\n{paths}')
    save_path = f'./{organ}/plans/FPS_{num_samples}.npz'
    np.savez(save_path, paths=paths)

def compute_typicality(cluster_feats):
    """
    Compute typicality for each sample in the cluster.

    Parameters:
    - cluster_feats: ndarray of shape (n_c, p), feature vectors in the cluster.

    Returns:
    - typicalities: ndarray of shape (n_c,), typicality of each sample.
    """
    # Compute pairwise Euclidean distances within the cluster
    distances = cdist(cluster_feats, cluster_feats, metric='euclidean')

    # Compute average distance to all other points for each sample
    avg_distances = np.mean(distances, axis=1)

    # Compute typicality as the inverse of the average distance
    typicalities = 1 / avg_distances

    return typicalities

def select_valuable_samples_TypiClust(feats, name_list, num_samples):
    """
    Select valuable samples using KMeans clustering and typicality.

    Parameters:
    - feats: ndarray of shape (n, p), feature vectors.
    - name_list: ndarray of shape (n,), corresponding sample names.
    - num_samples: int, the number of valuable samples to select.

    Returns:
    - select_samples_list: list of selected sample names.
    """
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_samples, random_state=0)
    kmeans.fit(feats)

    # Get the labels of each point
    labels = kmeans.labels_

    select_samples_list = []

    for i in range(num_samples):
        # Get the indices of samples in the current cluster
        cluster_indices = np.where(labels == i)[0]

        # Get the feature vectors of the samples in the cluster
        cluster_feats = feats[cluster_indices]

        # Compute typicality for each sample in the cluster
        typicalities = compute_typicality(cluster_feats)

        # Get the index of the sample with the highest typicality
        highest_typicality_index = cluster_indices[np.argmax(typicalities)]

        # Add the corresponding sample name to the list
        select_samples_list.append(name_list[highest_typicality_index])

    return select_samples_list

def generate_TypiClust_plan(organ, feats_file, num_samples):
    np.random.seed(1001)
    data_dict = dict(np.load(feats_file, allow_pickle=True))
    feats = np.array(data_dict['feats'])
    name_list = np.array(data_dict['name_list'])

    select_samples_list = select_valuable_samples_TypiClust(feats=feats, name_list=name_list, num_samples=num_samples)
    paths = [osp.join(f'./{organ}/data/{pid}.npz') for pid in select_samples_list]

    print(f'Select:\n{paths}')
    save_path = f'./{organ}/plans/TypiClust_{num_samples}.npz'
    np.savez(save_path, paths=paths)

def compute_information_density(cluster_feats):
    """
    Compute information density for each sample in the cluster.

    Parameters:
    - cluster_feats: ndarray of shape (n_c, p), feature vectors in the cluster.

    Returns:
    - densities: ndarray of shape (n_c,), information density of each sample.
    """
    # Compute cosine similarity matrix for the cluster
    similarity_matrix = cosine_similarity(cluster_feats)

    # Compute information density for each sample
    densities = similarity_matrix.mean(axis=1)

    return densities

def select_valuable_samples_CALR(feats, name_list, num_samples):
    """
    Select valuable samples using BIRCH clustering and information density.

    Parameters:
    - feats: ndarray of shape (n, p), feature vectors.
    - name_list: ndarray of shape (n,), corresponding sample names.
    - num_samples: int, the number of valuable samples to select.

    Returns:
    - select_samples_list: list of selected sample names.
    """
    # Perform BIRCH clustering
    birch = Birch(n_clusters=num_samples)
    birch.fit(feats)

    # Get the labels of each point
    labels = birch.labels_

    select_samples_list = []

    for i in range(num_samples):
        # Get the indices of samples in the current cluster
        cluster_indices = np.where(labels == i)[0]

        # Get the feature vectors of the samples in the cluster
        cluster_feats = feats[cluster_indices]
        if cluster_feats.shape[0] == 0: 
            continue

        # Compute information density for each sample in the cluster
        densities = compute_information_density(cluster_feats)

        # Get the index of the sample with the highest information density
        highest_density_index = cluster_indices[np.argmax(densities)]

        # Add the corresponding sample name to the list
        select_samples_list.append(name_list[highest_density_index])

    return select_samples_list

def generate_CALR_plan(organ, feats_file, num_samples):
    np.random.seed(0)
    data_dict = dict(np.load(feats_file, allow_pickle=True))
    feats = np.array(data_dict['feats'])
    name_list = np.array(data_dict['name_list'])

    select_samples_list = select_valuable_samples_CALR(feats=feats, name_list=name_list, num_samples=num_samples)
    paths = [osp.join(f'./{organ}/data/{pid}.npz') for pid in select_samples_list]

    print(f'Select:\n{paths}')
    save_path = f'./{organ}/plans/CALR_{num_samples}.npz'
    np.savez(save_path, paths=paths)

def select_valuable_samples_ALPS(feats, name_list, num_samples):
    """
    Select valuable samples using KMeans clustering.

    Parameters:
    - feats: ndarray of shape (n, p), feature vectors.
    - name_list: ndarray of shape (n,), corresponding sample names.
    - num_samples: int, the number of valuable samples to select.

    Returns:
    - select_samples_list: list of selected sample names.
    """
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_samples, random_state=0)
    kmeans.fit(feats)

    # Get the cluster centers
    centers = kmeans.cluster_centers_

    # Get the labels of each point
    labels = kmeans.labels_

    select_samples_list = []

    for i in range(num_samples):
        # Get the indices of samples in the current cluster
        cluster_indices = np.where(labels == i)[0]

        # Compute the distance from each sample in the cluster to the cluster center
        distances = np.linalg.norm(feats[cluster_indices] - centers[i], axis=1)

        # Get the index of the sample closest to the cluster center
        closest_index = cluster_indices[np.argmin(distances)]

        # Add the corresponding sample name to the list
        select_samples_list.append(name_list[closest_index])

    return select_samples_list

def generate_ALPS_plan(organ, feats_file, num_samples):
    np.random.seed(0)
    data_dict = dict(np.load(feats_file, allow_pickle=True))
    feats = np.array(data_dict['feats'])
    name_list = np.array(data_dict['name_list'])

    select_samples_list = select_valuable_samples_ALPS(feats=feats, name_list=name_list, num_samples=num_samples)
    paths = [osp.join(f'./{organ}/data/{pid}.npz') for pid in select_samples_list]

    print(f'Select:\n{paths}')
    save_path = f'./{organ}/plans/ALPS_{num_samples}.npz'
    np.savez(save_path, paths=paths)

def read_tsv(file_path):
    scores = []
    with open(file_path, 'r') as file:
        for line in file:
            scores.append(float(line.strip()))
    return np.array(scores)


def calculate_typicality(cluster_points):
    distances = cosine_distances(cluster_points)
    typicality = 1 / (np.mean(distances, axis=1) + 1e-10)
    return typicality

    
def select_valuable_samples_ProbCover(feats, name_list, num_samples, delta=None, alpha=0.95):
    
    n_samples = feats.shape[0]
    
    if delta is None:
        num_classes = num_samples  
        delta = estimate_delta(feats, num_classes, alpha)
        print(f'Estimated delta: {delta}')
    
    dist_matrix = distance_matrix(feats, feats)
    
    adjacency_matrix = (dist_matrix <= delta).astype(int)
    
    selected_indices = []
    
    for _ in range(num_samples):
        out_degrees = adjacency_matrix.sum(axis=1)
       
        max_out_degree_index = np.argmax(out_degrees)
        selected_indices.append(max_out_degree_index)
        
        covered_indices = np.where(adjacency_matrix[max_out_degree_index] > 0)[0]
        adjacency_matrix[:, covered_indices] = 0
        adjacency_matrix[covered_indices, :] = 0  

    # Convert selected indices to name list
    selected_names = [name_list[idx] for idx in selected_indices]
    return selected_names

def estimate_delta(embedding, num_classes, alpha=0.95):
    
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=num_classes)
    cluster_labels = kmeans.fit_predict(embedding)
    

    dist_matrix = distance_matrix(embedding, embedding)
    
    delta_values = np.linspace(0, np.max(dist_matrix), num=100)
    best_delta = 0
    for delta in delta_values:
        pure_balls = 0
        total_balls = 0
        
        for i in range(len(embedding)):
            neighbors = np.where(dist_matrix[i] <= delta)[0]
            if len(neighbors) > 0:
                if np.all(cluster_labels[neighbors] == cluster_labels[i]):
                    pure_balls += 1
                total_balls += 1
        
        purity = pure_balls / total_balls
        if purity >= alpha:
            best_delta = delta
        else:
            break
    
    return best_delta
def generate_Probcover_plan(organ, feats_file, num_samples):
    np.random.seed(0)
    data_dict = dict(np.load(feats_file, allow_pickle=True))
    feats = np.array(data_dict['feats'])
    name_list = np.array(data_dict['name_list'])
    
    select_samples_list = select_valuable_samples_ProbCover(feats=feats, name_list=name_list, num_samples=num_samples)
    paths = [osp.join(f'./{organ}/data/{pid}.npz') for pid in select_samples_list]  # Modified this line
    print(f'Select:\n{paths}')
    save_path = f'./{organ}/plans/Probcover_{num_samples}.npz'
    np.savez(save_path, paths=paths)

# generate_ALPS_plan(organ='Spleen', feats_file='./Spleen/feats/Ours.npz', num_samples=4)
# generate_CALR_plan(organ='Lung', feats_file='./Lung/feats/Ours.npz', num_samples=5)
# generate_TypiClust_plan(organ='Heart', feats_file='./Heart/feats/Ours.npz', num_samples=3)
# generate_TypiClust_plan(organ='Heart', feats_file='./Heart/feats/Ours.npz', num_samples=5)
# generate_TypiClust_plan(organ='Heart', feats_file='./Heart/feats/Ours.npz', num_samples=3)
# generate_TypiClust_plan(organ='Heart', feats_file='./Heart/feats/Ours.npz', num_samples=3)
# generate_TypiClust_plan(organ='Heart', feats_file='./Heart/feats/Ours.npz', num_samples=3)
# generate_FPS_plan(organ='Spleen', feats_file='./Spleen/feats/Ours.npz', num_samples=3)
# generate_FPS_plan(organ='Spleen', feats_file='./Spleen/feats/Ours.npz', num_samples=5)
# generate_Probcover_plan(organ='Spleen',feats_file='./Spleen/feats/Ours.npz', num_samples=4)