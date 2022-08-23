export TASK_NAME=glue
export DATASET_NAME=kaggle_tweets
export CUDA_VISIBLE_DEVICES=1

bs=8
lr=2e-5
dropout=0.1
psl=20
epoch=30

python3 run.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --corruption_file None \
  --do_train \
  --do_eval \
  --do_predict \
  --max_grad_norm 10.0 \
  --max_seq_length 256 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --load_best_model_at_end True \
  --save_strategy epoch \
  --evaluation_strategy epoch \

  
  