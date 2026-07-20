import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

json_data = """
{
    "red_koopa": { "address": "0x01", "description": "Yellow Turtle with Red shell" },
    "buzzy_beetle": { "address": "0x02", "description": "short yellow guy with blue shell" }
}
"""

# loading data into python dictionary
data = json.loads(json_data)
target_sentence = "A small yellow creature wearing a bright red shell"

target_names = list(data.keys())
descriptions = [data[name]["description"] for name in target_names]

# combining to be vectorized
all_documents = [target_sentence] + descriptions

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_documents)

target_vector = tfidf_matrix[0]
description_vectors = tfidf_matrix[1:]
similarity_scores = cosine_similarity(target_vector, description_vectors)[0]

max_idx = np.argmax(similarity_scores)
max_score = similarity_scores[max_idx]
max_target_name = target_names[max_idx]

print(f"Name: {max_target_name}, Score: {max_score}")
