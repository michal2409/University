import unicodedata
import string
import torch
from torch import FloatTensor

def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(1, len(line), n_letters)
    for li, letter in enumerate(line):
        tensor[0][li][letterToIndex(letter)] = 1
    return tensor

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def sentToTensor(sent):
    words_one_hot = [lineToTensor(word) for word in sent.split()]
    words_len = [len(word) for word in sent.split()]
    tensor = torch.zeros(len(words_len), max(words_len), n_letters)
    for idx, (word, wordLen) in enumerate(zip(words_one_hot, words_len)):
        tensor[idx, :wordLen] = FloatTensor(words_one_hot[idx])
    return tensor

all_letters = string.ascii_letters + " "
n_letters = len(all_letters)