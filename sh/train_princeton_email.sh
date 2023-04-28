export PYTHONPATH=.
export HF_DATASETS_CACHE="/mnt/16tb/minyoung/data/huggingface"

CUDA_VISIBLE_DEVICES=3 \
python3 run_email_extractor.py \
    --do_train \
    --dataset_name princeton_email \
    --dataset_format squad \
    --load_finetuned_model True \
    --model_name_or_path /mnt/16tb/minyoung/checkpoints/email_content_extraction/squad/checkpoint-10950 \
    --output_dir /mnt/16tb/minyoung/checkpoints/email_content_extraction/princeton_email \
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