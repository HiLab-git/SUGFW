CUDA_VISIBLE_DEVICES=0 python train.py \
    --csv_path ../data/UTAH/Foreslices/splits/train.csv \
    --output_path ../outputs/UTAH/train \
    --patch_size 480 480 \
    --batch_size 16 \
    --base_lr 0.01 \
    --max_epoch 400 \
    --class_num 2 \
    --deterministic 1 \
    --seed 2025
