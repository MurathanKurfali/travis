#!/usr/bin/env bash

bert_model=$1
lang=$2
seq_len=$3 # was 250 in the original experiments
batch_size=$4

out_dir="saved_models"
data_dir="processed_data"
prediction_dir="predictions"

mkdir ${prediction_dir}
out_path="${out_dir}/${bert_model}_${lang}"
rm "${data_dir}/cached_*"  # remove previously cached files to be on the safe side

for i in {1..4};   do    # do four different runs
    seed=$(shuf -i1-1000000 -n1)
    python run_ner.py \
      --data_dir "${data_dir}" \
      --lang ${lang} \
      --model_name_or_path ${bert_model} \
      --output_dir "${out_path}_${seed}" \
      --max_seq_length $seq_len \
      --num_train_epochs 3 \
      --per_gpu_train_batch_size $bs \
      --per_gpu_eval_batch_size $bs \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --seed $seed
done