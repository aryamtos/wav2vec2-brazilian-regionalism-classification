DATASET_TRAIN="dataset/train/"
DATASET_VAL="dataset/validation/"
MAX_SAMPLES=7998
MODEL_NAME=

python3 main.py train --dataset_train=$DATASET_TRAIN --dataset_val=$DATASET_VAL --max_samples=$MAX_SAMPLES --model_name=$MODEL_NAME 