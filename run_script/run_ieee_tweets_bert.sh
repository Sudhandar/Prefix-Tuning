export TASK_NAME=glue
export DATASET_NAME=ieee_tweets
export CUDA_VISIBLE_DEVICES=0

bs=8
lr=1e-2
dropout=0.1
psl=20
epoch=30

python3 run.py \
  --model_name_or_path bert-large-cased \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-bert/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix