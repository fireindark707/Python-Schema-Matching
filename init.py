from sentence_transformers import SentenceTransformer
print("schema_matching|Loading sentence transformer, this will take a while...")
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
print("schema_matching|Done loading sentence transformer")