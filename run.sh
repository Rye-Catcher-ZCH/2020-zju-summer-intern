set -e  # 有非零值就推出
set -x  # debug

python -u run.py \
    --task="train" \
    --model="MLP" \
    --data_path="datasets/hog_feature_85000.h5" \
    --save_path="saved/LR/model_1000_100_0-2.npy" \
    --roc_path="SVM_test_c.png" \
    --max_iter=5 \
    --batch_size=200 \
    --learning_rate=0.2 \
    --lamda=0 \
    --option="one-layer"
