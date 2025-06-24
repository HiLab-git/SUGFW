import os
import pandas as pd

# 假设图片和掩码文件夹路径如下
def get_case_num(filename):
    # 例如: Case000_Slice010.pkl
    return int(filename.split('_')[0][4:])

def make_records(file_list):
    records = []
    for img_file in file_list:
        img_path = os.path.join(IMAGE_DIR, img_file)
        mask_path = os.path.join(LABEL_DIR, img_file)
        records.append({'image_pth': img_path, 'mask_pth': mask_path})
    return records


IMAGE_DIR = f'../data/Promise12/Foreslices/image'
LABEL_DIR = f'../data/Promise12/Foreslices/label'
SPLITS_DIR = f'../data/Promise12/Foreslices/splits'

os.makedirs(SPLITS_DIR, exist_ok=True)

# 获取所有图片文件名
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.pkl')])


# 按casenum分组
train_files = []
valid_files = []
test_files = []

for f in image_files:
    casenum = get_case_num(f)
    if casenum < 50:
        train_files.append(f)
    elif 50 <= casenum < 70:
        valid_files.append(f)
    elif 70 <= casenum < 100:
        test_files.append(f)



# 生成csv
pd.DataFrame(make_records(train_files)).to_csv(os.path.join(SPLITS_DIR, 'train.csv'), index=False)
pd.DataFrame(make_records(valid_files)).to_csv(os.path.join(SPLITS_DIR, 'valid.csv'), index=False)
pd.DataFrame(make_records(test_files)).to_csv(os.path.join(SPLITS_DIR, 'test.csv'), index=False)