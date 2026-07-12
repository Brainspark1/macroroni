import json
 
SRC = "/mnt/user-data/uploads/output_multilabel.json"
OUT = "/home/claude/token_classification_data.json"
 
with open(SRC) as f:
    data = json.load(f)
 
examples = data["rasa_nlu_data"]["common_examples"]
 
# Optional heuristic entity tagging, keyed by words that commonly signal
# a direction or action in this dataset.
DIRECTION_WORDS = {"left", "right", "backwards", "back", "behind"}
ACTION_WORDS = {"jump", "run", "fire", "shoot", "down", "duck", "crouch"}
 
label_list = ["O", "B-DIRECTION", "I-DIRECTION", "B-ACTION", "I-ACTION"]
label2id = {l: i for i, l in enumerate(label_list)}
 
converted = []
for ex in examples:
    words = ex["text"].split()
    tags = []
    for w in words:
        lw = w.lower().strip(".,!?")
        if lw in DIRECTION_WORDS:
            tags.append("B-DIRECTION")
        elif lw in ACTION_WORDS:
            tags.append("B-ACTION")
        else:
            tags.append("O")
    converted.append({
        "tokens": words,
        "ner_tags": [label2id[t] for t in tags],
        "ner_tags_str": tags,  
        # drop before training
        "intent": ex["intent"], 
        # kept for reference
    })
 
with open(OUT, "w") as f:
    json.dump({
        "label_list": label_list,
        "label2id": label2id,
        "id2label": {v: k for k, v in label2id.items()},
        "data": converted,
    }, f, indent=2)
 
print(f"Wrote {len(converted)} examples to {OUT}")
print("\nSample:")
print(json.dumps(converted[0], indent=2))
print(json.dumps(converted[300], indent=2))
