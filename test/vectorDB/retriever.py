import chromadb
import math
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np

#todo maake it top 100 and also from the list calculate percentage of similarity for each dataset

def weigh_datasets(query_text, temp=1.0, topK=5):
    """
    Turn Chroma query results into dataset-level weights.
    Uses a softmax over (‑distance/temp) within the top‑k.
    """
    client = chromadb.PersistentClient(path="./chroma_store") 
    collection = client.get_or_create_collection("task_embeddings")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query_text])  
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=100  
    )
    sims_by_ds = defaultdict(float)

    ids         = results["ids"][0]
    metadatas   = results["metadatas"][0]
    distances   = results["distances"][0]

    #  exp(‑d/T) normalized
    exp_sim = [math.exp(-d / temp) for d in distances]

    # add the distances for each dataset
    for sim, meta in zip(exp_sim, metadatas):
        ds_name = meta["dataset"]
        sims_by_ds[ds_name] += sim

    # normalize by dividing by sum
    Z = sum(sims_by_ds.values())
    weights = {ds: val / Z for ds, val in sims_by_ds.items()}
    weights_sorted = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:topK]

    Z_top = sum(w for _, w in weights_sorted)          # re-normalise
    weights_top = {ds: w / Z_top for ds, w in weights_sorted}

    return weights_top

# w = weigh_datasets(temp=0.3,topK=6)
# print(w)

