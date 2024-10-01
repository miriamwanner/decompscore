import json
import numpy as np
import re
import functools
import string
import spacy
import sys
import nltk
import openai
import logging
from rank_bm25 import BM25Okapi
import os
import time
from nltk.tokenize import sent_tokenize

from factscore_utils.openai_lm import OpenAIModel

nltk.download("punkt")


class AtomicFactGenerator(object):
    def __init__(self, key_path, demon_dir, demon_file, prompt_prefix, conllu, gpt3_cache_file=None):
        self.nlp = spacy.load("en_core_web_sm")
        self.is_bio = True
        self.demon_path = os.path.join(demon_dir, demon_file if self.is_bio else "demons_complex.json")
        self.prompt_prefix = prompt_prefix
        self.conllu = conllu

        self.openai_lm = OpenAIModel("TurboInstructGPT", cache_file=gpt3_cache_file, key_path=key_path)

        # get the demos
        with open(self.demon_path, 'r') as f:
            self.demons = json.load(f)

        tokenized_corpus = [doc.split(" ") for doc in self.demons.keys()]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save_cache(self):
        self.openai_lm.save_cache()

    def run(self, sentences, para_breaks, conllus=None, cost_estimate=None):
        """Convert the generation into a set of atomic facts. Return a total words cost if cost_estimate != None."""
        assert isinstance(sentences, list), "generation must be a string"
        if self.conllu:
            return self.get_atomic_facts_from_conllu(conllus, sentences, para_breaks, cost_estimate=cost_estimate)
        # paragraphs = [para.strip() for para in generation.split("\n") if len(para.strip()) > 0]
        return self.get_atomic_facts_from_paragraph(sentences, para_breaks, cost_estimate=cost_estimate)

    def get_atomic_facts_from_paragraph(self, sentences, para_breaks, cost_estimate=None):

        atoms_or_estimate = self.get_init_atomic_facts_from_sentence(sentences, cost_estimate=cost_estimate)

        if cost_estimate:
            return atoms_or_estimate
        else:
            atoms = atoms_or_estimate

        atomic_facts_pairs = []
        for i, sent in enumerate(sentences):
            if not self.is_bio and ( \
                (i==0 and (sent.startswith("Sure") or sent.startswith("Here are"))) or \
                (i==len(sentences)-1 and (sent.startswith("Please") or sent.startswith("I hope") or sent.startswith("Here are")))):
                atomic_facts_pairs.append((sent, []))
            elif self.is_bio and sent.startswith("This sentence does not contain any facts"):
                atomic_facts_pairs.append((sent, []))
            elif sent.startswith("Sure") or sent.startswith("Please") or (i==0 and sent.startswith("Here are")):
                atomic_facts_pairs.append((sent, []))
            else:
                atomic_facts_pairs.append((sent, atoms[sent]))

        # postprocess_atomic_facts will fix minor issues from InstructGPT
        # it is supposed to handle sentence splitter issue too, but since here
        # we fixed sentence splitter issue already,
        # the new para_breaks should be identical to the original para_breaks
        if self.is_bio:
            atomic_facts_pairs, para_breaks = postprocess_atomic_facts(atomic_facts_pairs, list(para_breaks), self.nlp)

        return atomic_facts_pairs, para_breaks


    def get_atomic_facts_from_conllu(self, conllus, sentences, para_breaks, cost_estimate=None):
        
        conllu_sentences = conllus.split("\n\n")

        atoms_or_estimate = self.get_init_atomic_facts_from_conllu(sentences, conllu_sentences, cost_estimate=cost_estimate)

        if cost_estimate:
            return atoms_or_estimate
        else:
            atoms = atoms_or_estimate

        atomic_facts_pairs = []
        for i, sent in enumerate(sentences):
            if self.is_bio and sent.startswith("This sentence does not contain any facts"):
                atomic_facts_pairs.append((sent, []))
            elif sent.startswith("Sure") or sent.startswith("Please") or (i==0 and sent.startswith("Here are")):
                atomic_facts_pairs.append((sent, []))
            else:
                atomic_facts_pairs.append((sent, atoms[sent]))

        # postprocess_atomic_facts will fix minor issues from InstructGPT
        # it is supposed to handle sentence splitter issue too, but since here
        # we fixed sentence splitter issue already,
        # the new para_breaks should be identical to the original para_breaks
        if self.is_bio:
            atomic_facts_pairs, para_breaks = postprocess_atomic_facts(atomic_facts_pairs, list(para_breaks), self.nlp)

        return atomic_facts_pairs, para_breaks


    def get_init_atomic_facts_from_sentence(self, sentences, cost_estimate=None):
        """Get the initial atomic facts from the sentences. Return a total words cost if cost_estimate != None."""

        is_bio = self.is_bio
        demons = self.demons

        k = 1 if is_bio else 0
        n = 7 if is_bio else 8

        prompts = []
        prompt_to_sent = {}
        atoms = {}
        for sentence in sentences:
            if sentence in atoms:
                continue
            top_machings = best_demos(sentence, self.bm25, list(demons.keys()), k)
            prompt = ""

            for i in range(n):
                # prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(list(demons.keys())[i])
                prompt = prompt + self.prompt_prefix + str(list(demons.keys())[i]) + "\n"
                for fact in demons[list(demons.keys())[i]]:
                    prompt = prompt + "- {}\n".format(fact)
                prompt = prompt + "\n"

            for match in top_machings:
                prompt = prompt + self.prompt_prefix + str(match) + "\n"
                for fact in demons[match]:
                    prompt = prompt + "- {}\n".format(fact)
                prompt = prompt + "\n"
            prompt = prompt + self.prompt_prefix + str(sentence) + "\n"
            prompts.append(prompt)
            prompt_to_sent[prompt] = sentence

        if cost_estimate:
            total_words_estimate = 0
            for prompt in prompts:
                if cost_estimate == "consider_cache" and (prompt.strip() + "_0") in self.openai_lm.cache_dict:
                    continue
                total_words_estimate += len(prompt.split())
            return total_words_estimate
        else:
            for prompt in prompts:
                output, _ = self.openai_lm.generate(prompt)
                atoms[prompt_to_sent[prompt]] = text_to_sentences(output)

            for key, value in demons.items():
                if key not in atoms:
                    atoms[key] = value
            return atoms

    def get_init_atomic_facts_from_conllu(self, sentences, conllus, cost_estimate=None):
        """Get the initial atomic facts from the sentences. Return a total words cost if cost_estimate != None."""

        is_bio = self.is_bio
        demons = self.demons

        k = 1 if is_bio else 0
        n = 7 if is_bio else 8
        if self.conllu:
            n = 1

        prompts = []
        short_prompts = []
        prompt_to_sent = {}
        short_prompt_to_sent = {}
        atoms = {}
        for conllu, sentence in zip(conllus, sentences):
            if sentence in atoms:
                continue
            top_machings = best_demos(sentence, self.bm25, list(demons.keys()), k)
            prompt = ""
            short_prompt = ""

            for i in range(n):
                # prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(list(demons.keys())[i])
                prompt = prompt + self.prompt_prefix + str(list(demons.keys())[i]) + "\n"
                for fact in demons[list(demons.keys())[i]]:
                    prompt = prompt + "- {}\n".format(fact)
                prompt = prompt + "\n"

            for match in top_machings:
                prompt = prompt + self.prompt_prefix + str(match) + "\n"
                short_prompt = short_prompt + self.prompt_prefix + str(match) + "\n"
                for fact in demons[match]:
                    prompt = prompt + "- {}\n".format(fact)
                    short_prompt = short_prompt + "- {}\n".format(fact)
                prompt = prompt + "\n"
                short_prompt = short_prompt + "\n"
            prompt = prompt + self.prompt_prefix + str(conllu) + "\n"
            short_prompt = short_prompt + self.prompt_prefix + str(conllu) + "\n"
            prompts.append(prompt)
            short_prompts.append(short_prompt)
            prompt_to_sent[prompt] = sentence
            short_prompt_to_sent[short_prompt] = sentence

        if cost_estimate:
            total_words_estimate = 0
            for prompt in prompts:
                if cost_estimate == "consider_cache" and (prompt.strip() + "_0") in self.openai_lm.cache_dict:
                    continue
                total_words_estimate += len(prompt.split())
            return total_words_estimate
        else:
            for prompt, short_prompt in zip(prompts, short_prompts):
                try:
                    output, _ = self.openai_lm.generate(prompt)
                    atoms[prompt_to_sent[prompt]] = text_to_sentences(output)
                except Exception:
                    try:
                        logging.critical("Using shorter prompt")
                        output, _ = self.openai_lm.generate(short_prompt)
                        atoms[short_prompt_to_sent[short_prompt]] = text_to_sentences(output)
                    except Exception:
                        logging.critical("Sentence is too long: setting sentence as atomic fact.")
                        atoms[prompt_to_sent[prompt]] = [sentence]
                

            # for key, value in demons.items():
            #     if key not in atoms:
            #         atoms[key] = value
            return atoms



def best_demos(query, bm25, demons_sents, k):
    tokenized_query = query.split(" ")
    top_machings = bm25.get_top_n(tokenized_query, demons_sents, k)
    return top_machings


# transform InstructGPT output into sentences
def text_to_sentences(text):
    sentences = text.split("- ")[1:]
    sentences = [sent.strip()[:-1] if sent.strip()[-1] == '\n' else sent.strip() for sent in sentences]
    if len(sentences) > 0: 
        if sentences[-1][-1] != '.':
            sentences[-1] = sentences[-1] + '.' 
    else:
        sentences = []
    return sentences


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MONTHS = [m.lower() for m in MONTHS]

def is_num(text):
    try:
        text = int(text)
        return True
    except Exception:
        return False

def is_date(text):
    text = normalize_answer(text)
    for token in text.split(" "):
        if (not is_num(token)) and token not in MONTHS:
            return False
    return True

def extract_numeric_values(text):
    pattern = r'\b\d+\b'  # regular expression pattern for integers
    numeric_values = re.findall(pattern, text)  # find all numeric values in the text
    return set([value for value in numeric_values])  # convert the values to float and return as a list


def detect_entities(text, nlp):
    doc = nlp(text)
    entities = set()

    def _add_to_entities(text):
        if "-" in text:
            for _text in text.split("-"):
                entities.add(_text.strip())
        else:
            entities.add(text)


    for ent in doc.ents:
        # spacy often has errors with other types of entities
        if ent.label_ in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:

            if is_date(ent.text):
                _add_to_entities(ent.text)
            else:
                for token in ent.text.split():
                    if is_date(token):
                        _add_to_entities(token)
        
    for new_ent in extract_numeric_values(text):
        if not np.any([new_ent in ent for ent in entities]):
            entities.add(new_ent)

    return entities

def postprocess_atomic_facts(_atomic_facts, para_breaks, nlp):

    verbs = ["born.", " appointed.", " characterized.", " described.", " known.", " member.", " advocate.", "served.", "elected."]
    permitted_verbs = ["founding member."]

    atomic_facts = []
    new_atomic_facts = []
    new_para_breaks = []

    for i, (sent, facts) in enumerate(_atomic_facts):
        sent = sent.strip()
        if len(sent.split())==1 and i not in para_breaks and i > 0:
            assert i not in para_breaks
            atomic_facts[-1][0] += " " + sent
            atomic_facts[-1][1] += facts
        else:
            if i in para_breaks:
                new_para_breaks.append(len(atomic_facts))
            atomic_facts.append([sent, facts])

    for i, (sent, facts) in enumerate(atomic_facts):
        entities = detect_entities(sent, nlp)
        covered_entities = set()
        # print (entities)
        new_facts = []
        for i, fact in enumerate(facts):
            if any([fact.endswith(verb) for verb in verbs]) and not any([fact.endswith(verb) for verb in permitted_verbs]):
                if any([fact[:-1] in other_fact for j, other_fact in enumerate(facts) if j != i]):
                    continue
            sent_entities = detect_entities(fact, nlp)
            covered_entities |= set([e for e in sent_entities if e in entities])
            new_entities = sent_entities - entities
            if len(new_entities) > 0:
                do_pass = False
                for new_ent in new_entities:
                    pre_ent = None
                    for ent in entities:
                        if ent.startswith(new_ent):
                            pre_ent = ent
                            break
                    if pre_ent is None:
                        do_pass = True
                        break
                    fact = fact.replace(new_ent, pre_ent)
                    covered_entities.add(pre_ent)
                if do_pass:
                    continue
            if fact in new_facts:
                continue
            new_facts.append(fact)
        try:
            assert entities==covered_entities
        except Exception:
            new_facts = facts # there is a bug in spacy entity linker, so just go with the previous facts

        new_atomic_facts.append((sent, new_facts))

    return new_atomic_facts, new_para_breaks

def is_integer(s):
    try:
        s = int(s)
        return True
    except Exception:
        return False

def main():
    generator = AtomicFactGenerator("api.key", "demos", gpt3_cache_dir=None)
    atomic_facts, para_breaks = generator.run("Thierry Henry (born 17 August 1977) is a French professional football coach, pundit, and former player. He is considered one of the greatest strikers of all time, and one the greatest players of the Premier League history. He has been named Arsenal F.C's greatest ever player.\n\nHenry made his professional debut with Monaco in 1994 before signing for defending Serie A champions Juventus. However, limited playing time, coupled with disagreements with the club's hierarchy, led to him signing for Premier League club Arsenal for £11 million in 1999.")

    print(atomic_facts)
    print(para_breaks)

if __name__ == "__main__":
    main()
