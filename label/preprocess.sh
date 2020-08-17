set -e  # 有非零值就推出
set -x  # debug

python -u preprocess.py \
    --task="hog_extract" \
    --video_path="/Users/maitianshouwangzhe/Desktop/zju-2020-summer-intern/label/video/basketball-video-03.avi" \
    --data_store_path="/Users/maitianshouwangzhe/Desktop/zju-2020-summer-intern/label/data/data3.h5" \
    --annotation_path="/Users/maitianshouwangzhe/Desktop/zju-2020-summer-intern/label//annotation.txt" \
    --anno_data_store_path="/Users/maitianshouwangzhe/Desktop/zju-2020-summer-intern/label/data/data3_anno.h5" \
    --hog_feature_store_path="/Users/maitianshouwangzhe/Desktop/zju-2020-summer-intern/label/data/data3_hog.h5" \
    --crop_size=100 \
    --feature_type="two_frame_hog" \
    --hog_cell_size=8 \
    --hog_block_size=2 \
    --hog_stride=8 \
    --hog_bins=9 \
