import re

FILLERS = {
    "uh", "um", "er", "hmm", "umm", "uhh"
}

NEGATIONS = {
    "don't", "never","no"
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z'\s]", "", text)
    return text.split()

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

sentence = input()
result = tokenize(sentence)
print(result)
