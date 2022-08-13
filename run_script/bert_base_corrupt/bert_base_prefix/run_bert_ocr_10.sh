export TASK_NAME=glue
export DATASET_NAME=financial_phrasebank
export CUDA_VISIBLE_DEVICES=0

bs=8
lr=1e-2
dropout=0.1
psl=20
epoch=30

python3 run.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --corruption_file ./generate_new_datasets/financial_phrasebank/corrupt_data/ocr_replacement/financial_phrasebank_corrupt_10.hf \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 256 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-bert-ocr_replacement-10/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --load_best_model_at_end True \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --prefix \

  