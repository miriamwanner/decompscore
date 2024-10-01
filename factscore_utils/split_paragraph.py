import json
import argparse
import re
import nltk
import numpy as np

from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from pathlib import Path

nltk.download("punkt")


def detect_initials(text): # copied from factscore
    pattern = r"[A-Z]\. ?[A-Z]\."
    match = re.findall(pattern, text)
    return [m for m in match]

def fix_sentence_splitter(curr_sentences, initials): # copied from factscore
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split(".") if len(t.strip())>0]
            for i, (sent1, sent2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                    # merge sentence i and i+1
                    curr_sentences = curr_sentences[:i] + [curr_sentences[i] + " " + curr_sentences[i+1]] + curr_sentences[i+2:]
                    break
    sentences = []
    combine_with_previous = None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split())<=1 and sent_idx==0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split())<=1:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combined_with_previous = False
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)
    return sentences

def split_paragraph(generation): # copied from factscore
    paragraphs = [para.strip() for para in generation.split("\n") if len(para.strip()) > 0]
    sentences = []
    para_breaks = []
    for para_idx, paragraph in enumerate(paragraphs):
        if para_idx > 0 :
            para_breaks.append(len(sentences))

        initials = detect_initials(paragraph)

        curr_sentences = sent_tokenize(paragraph)
        curr_sentences_2 = sent_tokenize(paragraph)

        curr_sentences = fix_sentence_splitter(curr_sentences, initials)
        curr_sentences_2 = fix_sentence_splitter(curr_sentences_2, initials)

        # checking this, just to ensure the crediability of the sentence splitter fixing algorithm
        assert curr_sentences == curr_sentences_2, (paragraph, curr_sentences, curr_sentences_2)

        sentences += curr_sentences

    sentences = [sent for i, sent in enumerate(sentences) if not ((i==0 and (sent.startswith("Sure") or sent.startswith("Here are"))) or \
                    (i==len(sentences)-1 and (sent.startswith("Please") or sent.startswith("I hope") or sent.startswith("Here are"))))]

    return sentences, para_breaks