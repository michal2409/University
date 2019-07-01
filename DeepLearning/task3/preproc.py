import re
import random
import numpy as np
import torch
from util import *

random.seed(0)

class CorpusPreprocessor():
    def __init__(self, text):
        self.head_prob = 0.5
        self.mask = "mask"
        self.dictionary = sorted(list(set(self.transform_text(text).split())))
    
    def transform_text(self, text):
        text = re.sub('[^a-zA-ZżźćńółęąśŻŹĆĄŚĘŁÓŃ]+', ' ', text)
        text = re.sub(' +', ' ', text)
        text = text.strip()
        text = unicodeToAscii(text)
        return text
        
    def mask_text(self, text):
        words = text.split()
        orig_word_idx = random.randint(0, len(words)-1)
        orig_word = words[orig_word_idx]
        words[orig_word_idx] = self.mask
        mask_sent = " ".join(words)
        
        return ((mask_sent, orig_word, 1) if random.random() < self.head_prob
                else (mask_sent, random.choice(self.dictionary), 0))
        