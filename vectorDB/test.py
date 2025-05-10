import chromadb
import math
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np

#todo maake it top 100 and also from the list calculate percentage of similarity for each dataset

def weigh_datasets(results, temp=1.0):
    """
    Turn Chroma query results into dataset-level weights.
    Uses a softmax over (‑distance/temp) within the top‑k.
    """
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
    weights_sorted = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    return weights_sorted

client = chromadb.PersistentClient(path="./chroma_store") 
collection = client.get_or_create_collection("task_embeddings")
model = SentenceTransformer("all-MiniLM-L6-v2")
query_text ='''I was truly and wonderfully surprised at "O\' Brother, Where Art Thou?" The video store was out of all the movies I was planning on renting, so then I came across this. I came home and as I watched I became engrossed and found myself laughing out loud. The Coen\'s have made a magnificiant film again. But I think the first time you watch this movie, you get to know the characters. The second time, now that you know them, you laugh sooo hard it could hurt you. I strongly would reccomend ANYONE seeing this because if you are not, you are truly missing a film gem for the ages. 10/10'''
query_embedding = model.encode([query_text])  
query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

results = collection.query(
    query_embeddings=query_embedding,
    n_results=100  
)

w = weigh_datasets(results, temp=0.5)
for rank, (ds, weight) in enumerate(w, 1):
    print(f"{rank:>2}. {ds:<15} {weight:.3f}")
print(sum(weight for _, weight in w))

# print(weigh_datasets(results=results,temp=0.3))
# for i in range(len(results["ids"][0])):
#     print(f"Rank {i+1}:")
#     print("ID:      ", results["ids"][0][i])
#     print("Text:    ", results["metadatas"][0][i]["text"])
#     print("Dataset: ", results["metadatas"][0][i]["dataset"])
#     print("Distance:", results["distances"][0][i])
#     print("-" * 40)
