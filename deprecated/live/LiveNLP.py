import whisper
import re
import numpy as np
import pandas as pd
import speech_recognition as sr
import torch
import io
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

whisper_model = whisper.load_model("base")

def preprocess_text(text):
    text = text.lower() # lower case
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation
    tokens = text.split() # splitting
    return tokens

df = pd.read_csv("actual_dataset.csv")

dataset = list(df.itertuples(index=False))

texts = [preprocess_text(x[0]) for x in dataset]
labels = [x[1] for x in dataset]

vectorizer = TfidfVectorizer(analyzer=lambda x: x) # forces vectorizer to read text as already tokenized
X = vectorizer.fit_transform(texts)

# splitting into train, temp
X_train, X_temp, y_train, y_temp = train_test_split(X, labels, test_size=0.3, random_state=42, stratify=labels)

# splitting temp into validation, test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

print("\nDistribution of labels in each set:")
print(f"Training labels:\n{pd.Series(y_train).value_counts(normalize=True)}")
print(f"\nValidation labels:\n{pd.Series(y_val).value_counts(normalize=True)}")
print(f"\nTest labels:\n{pd.Series(y_test).value_counts(normalize=True)}")

"""## Training"""

model = LogisticRegression(C=1, penalty='l2', random_state=42, solver='lbfgs', max_iter=1000) # c = 10 with higher accuracy
model.fit(X_train, y_train)

def predict(text):
    processed_text = preprocess_text(text)
    x = vectorizer.transform([processed_text])

    probs = model.predict_proba(x)[0]
    best_idx = probs.argmax()

    label = model.classes_[best_idx]
    confidence = probs[best_idx]

    return label, confidence

"""## Validation"""

C_values = np.logspace(-4, 2, 7)

train_accuracies = []
val_accuracies = []

print("Hyperparameter Tuning for C value:\n")
for C in C_values:
    model_tuned = LogisticRegression(C=C, random_state=42, solver='lbfgs', max_iter=1000)
    model_tuned.fit(X_train, y_train)

    train_acc = model_tuned.score(X_train, y_train)
    train_accuracies.append(train_acc)

    val_acc = model_tuned.score(X_val, y_val)
    val_accuracies.append(val_acc)

    print(f"C: {C:<7.4f} | Training Accuracy: {train_acc:.4f} | Validation Accuracy: {val_acc:.4f}")

best_C_idx = np.argmax(val_accuracies)
best_C = C_values[best_C_idx]
best_val_accuracy = val_accuracies[best_C_idx]

print(f"\nBest C value found: {best_C:.4f} with Validation Accuracy: {best_val_accuracy:.4f}")

best_model = LogisticRegression(C=best_C, random_state=42, solver='lbfgs', max_iter=1000)
best_model.fit(X_train, y_train)

print("Model retrained with the best C value on the training set (stored as `best_model`).")

"""## Testing"""

test_accuracy = best_model.score(X_test, y_test)
print(f"Test Accuracy with best C ({best_C:.4f}): {test_accuracy:.4f}")

y_pred_test = best_model.predict(X_test)

report_test = classification_report(y_test, y_pred_test, target_names=best_model.classes_)
print("\nClassification Report (Test Data):\n", report_test)

model = best_model
print("\nGlobal 'model' variable updated to the best performing model (`best_model`).")

"""## Live Translation"""

recognizer = sr.Recognizer()
microphone = sr.Microphone(sample_rate=16000)

with microphone as source:
    print("Adjusting for background noise")
    recognizer.adjust_for_ambient_noise(source, duration=7)
    print("Completed adjusting for background noise")

print("Start speaking into your microphone. Press Ctrl+C to stop.\n")

def audio_callback(recognizer, audio):
    try:
        wav_data = io.BytesIO(audio.get_wav_data())

        with open("temp_chunk.wav", "wb") as f:
            f.write(wav_data.read())

        result = whisper_model.transcribe("temp_chunk.wav", fp16=torch.cuda.is_available())
        text = result["text"].strip()

        if text:
            print(f"Transcript: {text}")

            predicted_label, confidence = predict(text)
            print(f"Predicted Intent: {predicted_label}, Confidence: {confidence:.4f}")

        if os.path.exists("temp_chunk.wav"):
            os.remove("temp_chunk.wav")

    except Exception as e:
        print(f"Error during transcription segment: {e}")

stop_listening = recognizer.listen_in_background(
    microphone,
    audio_callback,
    phrase_time_limit=3
)

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopping live transcription. Goodbye!")
    stop_listening(wait_for_stop=False)
