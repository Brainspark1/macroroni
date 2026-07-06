# credit for most of vectorizer code: https://agombert.github.io/AdvancedNLPClasses/chapter3/Session_3_1_Word2Vec_Training/

import re
from collections import Counter

SPECIAL_TOKENS = [
    "<PAD>", # makes every sentence have the same length
    "<UNK>", # unknown word
    "<SOS>", # start of sequence/sentence
    "<EOS>", # end of sequence/sentence
    "<FILLER>", # filler token
    "<NEGATION>" # negation token
]

FILLERS = {
    "uh", "um", "er", "hmm", "umm", "uhh"
}

NEGATIONS = {
    "don't", "never","no"
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z'\s]", "", text)
    text_split = text.split()
    return text_split

def tokenize(sentence):
    tokens = preprocess_text(sentence)

    output_list = []

    for token in tokens:
        if token in FILLERS:
            output_list.append("<FILLER>")
        elif token in NEGATIONS:
            output_list.append("<NEGATION>")
        else:
            output_list.append(token)

    not_duplicated = []
    for token in output_list:
        if not not_duplicated or not_duplicated[-1] != token: # if the current token is not in the list already or has not most recently been added to the list
            not_duplicated.append(token)

    return not_duplicated

# building a vocabulary from training data by assigning each unique token a unique integer id
def build_vocab(sentences):

    # counting times each token appears in training data
    counter = Counter()

    # tokenize every training sentence and update the token counts
    for sentence in sentences:
        counter.update(tokenize(sentence))

    # tokens to integer ids
    vocab = {}

    # add special tokens
    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab) # next id determined by dictionary size/what index value is next

    # add every unique token
    for token in counter:
        if token not in vocab:
            vocab[token] = len(vocab)

    # swaps positions of index and word to map id back to word for decoding down the line
    id_to_word = {index: word for word, index in vocab.items()}

    return vocab, id_to_word


# converting tokenized sentence into sequence of integer ids to put into seq2seq model
def encode(sentence, vocab):

    # tokenize input sentence
    tokens = tokenize(sentence)

    # add start of sentence token
    ids = [vocab["<SOS>"]]

    # converting each token to its integer id, where unknown words mapped to unknown
    for token in tokens:
        ids.append(vocab.get(token, vocab["<UNK>"])) # get id of token, if nothing then id of unknown token

    # add end of sentence token
    ids.append(vocab["<EOS>"])

    return ids


# converting sequence of integer ids back into text to see seq2seq model's output
def decode(ids, id_to_word):

    # list of words to be outputted
    words = []

    # convert each integer id back into its corresponding token
    for index in ids:
        word = id_to_word[index]

        # ignore internal special tokens, removing them from output sentence
        if word in ("<PAD>", "<SOS>", "<EOS>"):
            continue

        # adding decoded word to word list
        words.append(word) 

    # returning list of words by adding them all to same string
    output_string = " ".join(words)
    return output_string
