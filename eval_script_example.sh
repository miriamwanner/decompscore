#!/bin/bash

export FILENAME=Pythia-12B

export PATH_TO_ENV=/home/mwanner5/miniconda3/envs/factscore/bin
export PATH_TO_HOME=/brtx/602-nvme1/mwanner5
export PYTHONPATH=/home/mwanner5/factscore/FActScore-main
export PATH_TO_DATA=/brtx/602-nvme1/mwanner5/decontext-decomp

${PATH_TO_ENV}/python ${PYTHONPATH}/factscore/factscorer_sentence_level.py \
                --input_path ${PATH_TO_DATA}/data/decontext-then-decomp/${FILENAME}_decontext_then_decomp_formatted_decompose.jsonl \
                --model_name retrieval+llama \
                --openai_key ${PATH_TO_HOME}/factscore/gpt_key.txt \
                --model_dir /brtx/602-nvme1/mwanner5/data/models \
                --data_dir /brtx/602-nvme1/mwanner5/data \
                --cache_dir ${PATH_TO_DATA}/cache/${FILENAME} \
                --demon_file demons.json \
                --use_atomic_facts \
                --verbose
