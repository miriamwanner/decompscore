import argparse
import string
import json
import numpy as np
import os
import logging

from tqdm import tqdm
from factscore.factscorer import FactScorer
from split_paragraph import split_paragraph



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        type=str,
                        default="data/labeled/InstructGPT.jsonl")
    parser.add_argument('--gamma',
                        type=int,
                        default=10,
                        help="hyperparameter for length penalty")

    parser.add_argument('--openai_key',
                        type=str,
                        default="api.key")
    parser.add_argument('--data_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--cache_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--demon_file',
                        type=str,
                        default="russelian_demons.json")

    parser.add_argument('--cost_estimate',
                        type=str,
                        default="consider_cache",
                        choices=["consider_cache", "ignore_cache"])
    parser.add_argument('--abstain_detection_type',
                        type=str,
                        default=None,
                        choices=["perplexity_ai", "generic", "none"])
    parser.add_argument('--verbose',
                        action="store_true",
                        help="for printing out the progress bar")    
    parser.add_argument('--print_rate_limit_error',
                        action="store_true",
                        help="for printing out rate limit error when using OpenAI keys")

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR if args.print_rate_limit_error else logging.CRITICAL)

    fs = FactScorer(data_dir=args.data_dir,
                    cache_dir=args.cache_dir,
                    openai_key=args.openai_key,
                    cost_estimate=args.cost_estimate,
                    demon_file=args.demon_file,
                    prompt_prefix="Please decompose the following sentence into individual facts: ",
                    abstain_detection_type=args.abstain_detection_type)

    tot = 0
    topics, generations, atomic_facts, sentences, para_breaks = [], [], [], [], []
    logging.critical("Splitting paragraphs")
    with open(args.input_path) as f:
        for line in tqdm(f):
            dp = json.loads(line)
            tot += 1
            topics.append(dp["topic"])
            gen = dp["output"]
            generations.append(gen)
            sent, pb = split_paragraph(gen)
            sentences.append(sent)
            para_breaks.append(pb)

    atomic_facts = fs.get_af(topics=topics,
                       sentences=sentences,
                       generations=generations,
                       para_breaks=para_breaks,
                       gamma=args.gamma,
                       verbose=args.verbose)
    
    # with open(args.input_path.replace(".jsonl", f"_annotated_fs.json"), 'w') as f:
    #     json.dump(atomic_facts, f)
    with open(args.input_path.replace(".jsonl", f"_annotated_decompose.jsonl"), 'w') as f:
        logging.critical("Writing to "+str(args.input_path.replace(".jsonl", f"_annotated_decompose.jsonl")))
        for topic in atomic_facts:
            json_record = json.dumps(topic)
            f.write(json_record+'\n')

