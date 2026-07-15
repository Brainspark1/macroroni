from transformers import DataCollatorForTokenClassification, AutoTokenizer
tokenizer =  AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from deprecated.token_bert.compute_metrics import compute_metrics

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
ds = load_from_disk('..\dataset\mario_token_classification_tokenized')
split_ds = ds.train_test_split(test_size=0.2, seed=42)
label2id = {
    "O": 0,
    "B-DIRECTION": 1,
    "I-DIRECTION": 2,
    "B-ACTION": 3,
    "I-ACTION": 4
}
id2label = {
    0: "O",
    1: "B-DIRECTION",
    2: "I-DIRECTION",
    3: "B-ACTION",
    4: "I-ACTION"
}
model = AutoModelForTokenClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=13, id2label=id2label, label2id=label2id)

training_args = TrainingArguments(
    output_dir="token_bert",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= split_ds["train"],
    eval_dataset= split_ds["test"],
    processing_class=tokenizer, 
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model("./token_classifier")
trainer.push_to_hub()