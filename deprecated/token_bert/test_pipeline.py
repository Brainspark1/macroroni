from transformers import pipeline
text = input('Type something: ')
classifier = pipeline("ner", model="Saggarwal/token_bert")
print(classifier(text))