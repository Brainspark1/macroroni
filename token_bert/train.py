from transformers import DataCollatorForTokenClassification, AutoTokenizer
tokenizer =  AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)