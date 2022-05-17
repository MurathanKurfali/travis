#!/usr/bin/env bash

trained_bert_model=$1

seq_len=400
batch_size=12
out_dir="saved_models"
data_dir="processed_data"
prediction_dir="predictions"

mkdir ${prediction_dir}

python run_ner.py \
  --data_dir "${data_dir}" \
  --model_name_or_path ${trained_bert_model} \
  --output_dir "${trained_bert_model}" \
  --max_seq_length $seq_len \
  --num_train_epochs 3 \
  --per_gpu_train_batch_size $bs \
  --per_gpu_eval_batch_size $bs \
  --do_predict \
