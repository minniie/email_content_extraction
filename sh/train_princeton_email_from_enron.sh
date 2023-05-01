export PYTHONPATH=.
export HF_DATASETS_CACHE="/mnt/16tb/minyoung/data/huggingface"

CUDA_VISIBLE_DEVICES=5 \
python3 run_email_extractor.py \
    --do_train \
    --dataset_name princeton_email \
    --load_finetuned_model True \
    --model_name_or_path /mnt/16tb/minyoung/checkpoints/email_content_extraction/enron_preprocessed/checkpoint-22353 \
    --output_dir /mnt/16tb/minyoung/checkpoints/email_content_extraction/princeton_email_enron \
    --num_train_epochs 20 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 16 \
    --save_strategy epoch \
    --evaluation_strategy steps \
    --eval_steps 20 \
    --logging_steps 20 \
    --seed 1234 \
    --remove_unused_columns False \
    --report_to tensorboard