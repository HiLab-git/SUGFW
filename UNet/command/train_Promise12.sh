 CUDA_VISIBLE_DEVICES=0 python train.py \
    --csv_path ../data/Promise12/Foreslices/splits/train.csv \
    --output_path ../outputs/Promise12/train \
    --patch_size 224 224 \
    --batch_size 16 \
    --base_lr 0.01 \
    --max_epoch 400 \
    --class_num 2 \
    --deterministic 1 \
    --seed 2025
