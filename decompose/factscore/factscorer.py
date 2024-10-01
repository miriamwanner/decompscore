import argparse
import string
import json
import numpy as np
import os
import logging

from tqdm import tqdm
from factscore.abstain_detection import is_response_abstained
from factscore.atomic_facts import AtomicFactGenerator
from factscore.openai_lm import OpenAIModel

class FactScorer(object):

    def __init__(self,
                 data_dir=".cache/factscore",
                 cache_dir=".cache/factscore",
                 openai_key="api.key",
                 cost_estimate="consider_cache",
                 demon_file="demons.json",
                 prompt_prefix="Please breakdown the following sentence into independent facts: ",
                 abstain_detection_type=None,
                 conllu=False,
                 batch_size=256):

        self.openai_key = openai_key
        self.abstain_detection_type = abstain_detection_type

        self.data_dir = data_dir
        self.demon_file = demon_file
        self.prompt_prefix = prompt_prefix
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.af_generator = None
        self.cost_estimate = cost_estimate
        self.conllu=conllu

    def print_cost_estimates(self, total_words, task, model):
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        # Number of tokens are roughly 4/3 of the number of words
        total_tokens = total_words * 4.0 / 3

        # https://openai.com/pricing
        # if we use davinci-003, the cost is $0.02 per 1000 tokens
        # if we use gpt-3.5-turbo, the cost is $0.002 per 1000 tokens
        if model == "davinci-003":
            rate = 0.02
        elif model == "gpt-3.5-turbo":
            rate = 0.002
        elif model == "gpt-3.5-turbo-instruct":
            rate = 0.002

        total_cost = total_tokens * rate / 1000

        # print the total words, tokens, and cost along with rate
        logging.critical("Estimated OpenAI API cost for %s ($%.3f per 1000 tokens): $%.2f for %d words and %d tokens" % (task, rate, total_cost, total_words, total_tokens))

    def get_af(self,
                topics,
                sentences,
                generations,
                para_breaks,
                gamma=10,
                conllus=None,
                verbose=False):

        assert type(topics)==type(sentences)==type(generations)==list, "`topics` and `sentences` should be lists."
        assert len(topics)==len(generations)==len(sentences), "`topics` and `sentences` should have the same length"

        # added for output
        out_af_lst = []

        if self.af_generator is None:
            self.af_generator = AtomicFactGenerator(key_path=self.openai_key,
                                                    demon_dir=os.path.join(self.data_dir, "demos"),
                                                    demon_file=self.demon_file,
                                                    prompt_prefix=self.prompt_prefix,
                                                    conllu=self.conllu,
                                                    gpt3_cache_file=os.path.join(self.cache_dir, "TurboInstructGPT.pkl"))

            # estimate the total cost of atomic fact generation
            total_words = 0
            if self.conllu:
                for c, sent, pb in zip(conllus, sentences, para_breaks):
                    total_words += self.af_generator.run(sent, pb, conllus=c, cost_estimate=self.cost_estimate)
            else:
                for sent, pb in zip(sentences, para_breaks):
                    total_words += self.af_generator.run(sent, pb, cost_estimate=self.cost_estimate)

            self.print_cost_estimates(total_words, task="atomic fact generation", model="gpt-3.5-turbo-instruct")

            if verbose:
                topics = tqdm(topics)

            atomic_facts = []
            if conllus == None:
                conllus = [None for t in topics]

            logging.critical("Generating atomic facts")
            for topic, gen, sent, pb, c in tqdm(zip(topics, generations, sentences, para_breaks, conllus)):
                # optionally, first detect if the response is abstained
                response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                if response_abstained:
                    atomic_facts.append(None)
                    continue
                # continue only when the response is not abstained
                if self.conllu:
                    curr_afs, _ = self.af_generator.run(sent, pb, conllus=c)
                else:
                    curr_afs, _ = self.af_generator.run(sent, pb, conllus=c)

                # added for output
                if self.conllu:
                    af_method = "conllu prompt"
                else:
                    af_method = "factscore prompt"
                out_af_dict = {}
                out_af_dict["topic"] = topic
                out_af_dict["paragraph"] = gen
                out_af_dict["decomposition"] = []
                for sent, facts in curr_afs:
                    out_af_dict["decomposition"].append({"sentence": sent, af_method: facts})
                out_af_lst.append(out_af_dict)



                curr_afs = [fact for _, facts in curr_afs for fact in facts]
                if len(curr_afs)==0:
                    atomic_facts.append(None)
                else:
                    atomic_facts.append(curr_afs)
                if len(atomic_facts) % 10 == 0:
                    self.af_generator.save_cache()


            assert len(atomic_facts)==len(topics)
            self.af_generator.save_cache()

        respond_ratio = np.mean([facts is not None for facts in atomic_facts])
        
        return out_af_lst
