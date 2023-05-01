export PYTHONPATH=.
export HF_DATASETS_CACHE="/mnt/16tb/minyoung/data/huggingface"

CUDA_VISIBLE_DEVICES=4 \
python3 run_email_extractor.py \
    --do_eval \
    --dataset_name princeton_email \
    --load_finetuned_model True \
    --model_name_or_path /mnt/16tb/minyoung/checkpoints/email_content_extraction/princeton_email_squad/checkpoint-654 \
    --output_dir /mnt/16tb/minyoung/checkpoints/email_content_extraction/princeton_email_squad/checkpoint-654 \
    --per_device_eval_batch_size 16 \
    --seed 1234 \
    --remove_unused_columns False \
    --report_to tensorboard