export PYTHONPATH=.
export HF_DATASETS_CACHE="/mnt/16tb/minyoung/data/huggingface"

CUDA_VISIBLE_DEVICES=3 \
python3 run_email_extractor.py \
    --do_train \
    --dataset_name emailsum \
    --model_name_or_path gpt2-medium \
    --output_dir /mnt/16tb/minyoung/checkpoints/email_content_extraction/dummy \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-4 \
    --save_strategy epoch \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --logging_steps 100 \
    --seed 1234 \
    --report_to tensorboard