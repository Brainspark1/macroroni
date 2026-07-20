from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentence = ""
embedding = model.encode(sentence, convert_to_tensor=True)

similarities = util.cos_sim(embedding, )

# get similarity value with similarities.item()
