## "SUGFW: A SAM-based Uncertainty-guided Feature Weighting Framework for Cold Start Active Learning."

### 0. Accepted by MICCAI 2025

### 1. Environment
```sh
conda create -n sgfw python=3.10
conda activate sgfw
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
cd SUGFW
pip install -e . 
```

### 2. Download and Preprocessing
#### 2.1 Download sam checkpoint 
```sh
cd ./checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
#### 2.2 Download data
Click [here](https://zenodo.org/records/8026660) to download **Promise12** and [here](https://www.dropbox.com/scl/fi/nero2nlaocdcdfhzwu5h0/2018_UTAH_MICCAI.zip?rlkey=vkkfrkc2l6x1e61jqyutb35qn&e=1&dl=0) to download **UTAH**
##### Place the raw data as follows for preprocessing:
```sh
SUGFW
├── data
    ├── Promise12
    │   └── raw
    │       ├── LICENSE.TXT
    │       ├── livechallenge_test_data
    │       ├── test_data
    │       └── training_data
    ├── tree.txt
    └── UTAH
        └── raw
            ├── preprocess_data.py
            ├── Testing Set
            ├── Training Set
            └── Unet.py
```
#### 2.3 Preprocessing
* Each image is **z-score normalized** and outliers greater than the **99.5th** percentile are removed. All images are resized to **256x256** for Promise12 while **512x512** for UTAH.
* For both datasets, we followed the official splits provided by the challenge.


### 3. How to use this code

#### 3.1 Run preprocessing
```sh
cd preprocessing
python Promise12_pre.py
python split_Promise12.py

python UTAH_pre.py
python split_UTAH.py
```

#### 3.2 Get uncertainty
```sh
cd UC-SAM
bash ./command/getun_Promise12.sh
bash ./command/getun_UTAH.sh
```

#### 3.3 Sampling
```sh
cd UC-SAM
bash ./command/select_Promise12.sh
bash ./command/select_UTAH.sh
```
#### 3.4 Train
##### For different selection methods, just change the "csv_path" of selected samples
```sh
cd UNet
bash ./command/train_Promise12.sh
bash ./command/train_UTAH.sh
```
#### 3.5 Test
```sh
cd UNet
bash ./command/inference.sh
```

### 4. Citation