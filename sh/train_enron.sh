export PYTHONPATH=.
export HF_DATASETS_CACHE="/mnt/16tb/minyoung/data/huggingface"

CUDA_VISIBLE_DEVICES=3 \
python3 run_email_extractor.py \
    --do_train \
    --dataset_name enron \
    --model_name_or_path gpt2-medium \
    --output_dir /mnt/16tb/minyoung/checkpoints/email_content_extraction/enron_preprocessed \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 16 \
    --save_strategy epoch \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --logging_steps 200 \
    --seed 1234 \
    --remove_unused_columns False \
    --report_to tensorboard 