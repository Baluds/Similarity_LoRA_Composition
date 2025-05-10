import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from taskSpecs import TASK_SPECS
import numpy as np

client = chromadb.PersistentClient(path="./chroma_store") 
collection = client.get_or_create_collection(name="task_embeddings",
    metadata={"hnsw:space": "cosine"}
    )

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')



def load_and_sample_data(task_specs, total_samples=2000, seed=42):
    random.seed(seed)
    sampled_data = []
    for spec in task_specs:
        ds = load_dataset(*spec['load_args'])
        take = min(2000, len(ds))
        ds_shuff = ds.shuffle(seed=seed).select(range(take))
        for idx, example in enumerate(ds_shuff):
            text = ds_shuff['text'][idx]
            meta = {"dataset": spec['name']}
            meta["text"] = text  # store the text itself as metadata for reference
            sampled_data.append((text, meta))
    return sampled_data

sampled_examples = load_and_sample_data(TASK_SPECS, total_samples=2000, seed=42)


with open("outputnew.txt", "w") as f:
    for item in sampled_examples:
        f.write(str(item) + "\n")

texts = [text for text, meta in sampled_examples]
metadatas = [meta for text, meta in sampled_examples]

embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

ids = [f"{meta['dataset']}_{i}" for i, meta in enumerate(metadatas)]

batch_size = 4000 

for i in range(0, len(embeddings), batch_size):
    collection.add(
        embeddings=embeddings[i:i+batch_size].tolist(),
        metadatas=metadatas[i:i+batch_size],
        ids=ids[i:i+batch_size]
    )

print(f"Indexed {collection.count()} embeddings in Chroma.")
