# TRAVIS: (multilingual) Multiword Expression Identification model

This repository contains the code for the TRAVIS model built for the PARSEME Shared Task 2020 on semisupervised identification of verbal multiword expressions. TRAVIS is a fully feature-independent
model, relying only on the contextual embeddings. The model ranked second in the open track of the shared task, see [(kurfali, 2020)](https://www.diva-portal.org/smash/get/diva2:1524071/FULLTEXT01.pdf) for details.



TRAVIS is a tool for performing multi-word expression (MWE) identification via fine-tuning language models.

## Shared Task Data
The shared task data used to train and evaluate TRAVIS can be obtained via:

```bash
    git clone https://gitlab.com/parseme/sharedtask-data.git
```

## Pre-processing
TRAVIS models MWE identification as a token classification task. Therefore, it expects files with the standard "one token \t label" pair per line format where empty lines
specify a new training instance. The .cupt files can be converted to desired format using 

```bash
   python utils/preprocess.py  sharedtask-data_path/1.2 output_path  target_language[optional]
```
Preprocess script will produce the corresponding csv files in the output_path. 

## Training
Here is a sample script to train a model using TRAVIS:

```bash
    python run_ner.py \
      --data_dir path_to_data \
      --lang "tr" \
      --model_name_or_path dbmdz/bert-base-turkish-128k-cased \
      --output_dir path_to_output_dir \
      --max_seq_length 256 \
      --num_train_epochs 3 \
      --per_gpu_train_batch_size 16 \
      --per_gpu_eval_batch_size 32 \
      --do_train \
      --do_eval \
      --overwrite_output_dir
```
BERT fine-tuning procedure is known to be prone to variance; therefore, you may want to consider it to fine-tune it for several times with different seeds (see `train.sh`). 
In the shared task, we submitted the predictions of the model with the best performance on the development set out of four runs.

See `run_experiments.sh` to run all the experiments described in the paper.

## PREDICTION

To run a trained model on your validation file, you can use the following command (predict.sh):
```bash
python run_ner.py \
  --data_dir path_to_data \
  --model_name_or_path path_to_trained_model \
  --output_dir path_to_trained_model \
  --max_seq_length 512 \
  --num_train_epochs 3 \
  --per_gpu_eval_batch_size 32 \
  --do_predict \
  --predict_file path_to_target_file
```

The predictions of the model will be saved to the "predictions" folder. If no predict_file is provided, the prediction will be run on the original test file which is assumed to be in --data_dir.

The predictions on the shared task files can be converted back to the original .cupt format using
```bash
python utils/postprocess.py prediction_file_path original_cupt_file_path output_dir
```

# Publications

If you use this resource, please consider citing

    @inproceedings{kurfali2020travis,
      title={TRAVIS at PARSEME Shared Task 2020: How good is (m) BERT at seeing the unseen?},
      author={Kurfali, Murathan},
      booktitle={International Conference on Computational Linguistics (COLING), Barcelona, Spain (Online), December 13, 2020},
      pages={136--141},
      year={2020}
    }
