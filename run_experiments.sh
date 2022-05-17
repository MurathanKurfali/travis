#!/usr/bin/env bash

### Multilingual-bert
batch_size=8
langs=('de' 'el' 'eu' 'fr' 'ga', 'hi' 'it' 'pl' 'pt' 'ro' 'sv' 'tr' 'zh')
for lang in "${langs[@]}"; do
echo $lang
    ./train.sh "bert-base-multilingual-cased" $lang $seq_len $batch_size
    sleep 3
    #rm -rf saved_models*  ## consider removing saved models to save up from space
done

echo "Running Monolingual Models"
declare -A monolingual_models
monolingual_models=(["DE"]="bert-base-german-cased" ["EL"]="nlpaueb/bert-base-greek-uncased-v1" ["FR"]="camembert-base"
["IT"]="dbmdz/bert-base-italian-cased" ["HI"]="hindi-bert" ["PL"]="dkleczek/bert-base-polish-cased-v1"
["RO"]="dumitrescustefan/bert-base-romanian-cased-v1" ["SV"]="KB/bert-base-swedish-cased"
["TR"]="dbmdz/bert-base-turkish-128k-cased" ["ZH"]="bert-base-chinese")
# missing:  Basque (EU),  Irish (GA), Hebrew (HE), Brazilian Portuguese (PT),

for lang in "${!monolingual_models[@]}";
do
    ./train.sh ${monolingual_models[$lang]} $lang $seq_len $batch_size
done

