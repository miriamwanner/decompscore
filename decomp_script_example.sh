#!/bin/bash

export FILENAME=Alpaca-65B

export PATH_TO_ENV=/home/mwanner5/miniconda3/envs/factscore/bin
export PATH_TO_HOME=/brtx/602-nvme1/mwanner5
export PYTHONPATH=${PATH_TO_HOME}/decontext-decomp

${PATH_TO_ENV}/python ${PYTHONPATH}/decomp/decompose_prompt.py \
                --input_path ${PYTHONPATH}/data/decontext-then-decomp/${FILENAME}_decontext_then_decomp.jsonl \
                --openai_key ${PYTHONPATH}/data/gpt_key.txt \
                --data_dir ${PYTHONPATH}/data \
                --cache_dir ${PYTHONPATH}/cache/${FILENAME} \
                --demon_file russelian_demons.json

${PATH_TO_ENV}/python ${PYTHONPATH}/decomp/fs_formatting.py \
                --annotated_dict ${PYTHONPATH}/data/decontext-then-decomp/${FILENAME}_decontext_then_decomp_annotated_decompose.jsonl \
                --factscore_dict ${PYTHONPATH}/data/decontext-then-decomp/${FILENAME}_decontext_then_decomp.jsonl \
                --decomp_type decompose

rm ${PYTHONPATH}/data/decontext-then-decomp/${FILENAME}_decontext_then_decomp_annotated_decompose.jsonl
