import re

FILLERS = {
    "uh",
    "um",
    "er",
    "hmm"
}

CORRECTIONS = {
    "wait",
    "actually",
    "instead",
    "rather"
}

NEGATIONS = {
    "don't",
    "never",
    "no"
}

OUTPUT = []

def preprocess_text(text):
    text = text.lower() # lower case
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation
    tokens = text.split() # splitting
    return tokens

def output(text):
    OUTPUT.append(text)    

sentence = input()
tokens = preprocess_text(sentence)

for token in tokens:
    if token in FILLERS:
        output("<FILLER>")
    elif token in CORRECTIONS:
        output("<CORRECTION>")
    elif token in NEGATIONS:
        output("<NEGATION>")
    else:
        output(token)

OUTPUT = [OUTPUT[out_token] for out_token in range(len(OUTPUT)) if out_token == 0 or OUTPUT[out_token] != OUTPUT[out_token - 1]]

print(OUTPUT)
