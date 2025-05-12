import chromadb
import math
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np

#todo maake it top 100 and also from the list calculate percentage of similarity for each dataset

def weigh_datasets(query_text, top_p=0.9):
    """
    Turn Chroma query results into dataset-level weights.
    Uses a softmax over (‑distance) within the top-p.
    """
    print("started retrieving")
    client = chromadb.PersistentClient(path="/home/sgovindan_umass_edu/Similarity_LoRA_Composition/test/vectorDB/chroma_store/") 
    collection = client.get_or_create_collection("task_embeddings")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query_text])  
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=100  
    )
    
    metadatas   = results["metadatas"][0]
    distances   = results["distances"][0]

    #  exp(‑d/T) normalized
    exp_sim = [math.exp(-d) for d in distances]

    sims_by_ds = defaultdict(float)

    # add the distances for each dataset
    for sim, meta in zip(exp_sim, metadatas):
        sims_by_ds[meta["dataset"]] += sim

    # normalize by dividing by sum
    Z = sum(sims_by_ds.values())
    weights = {ds: val / Z for ds, val in sims_by_ds.items()}
    weights_sorted = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    cumulative = 0.0
    nucleus = []
    for ds, w in weights_sorted:
        nucleus.append((ds, w))
        cumulative += w
        if cumulative >= top_p:
            break

    Z_nucleus = sum(w for _, w in nucleus)          # re-normalise
    weights_top = {ds: w / Z_nucleus for ds, w in nucleus}
    print(weights_top)

    return weights_top

#For testing
# w = weigh_datasets('''Answer the question based on the following paragraph with True or False.
# Paragraph:
# All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. All of these steps require resources and an infrastructure. The total amount of energy input into the process compared to the energy released by burning the resulting ethanol fuel is known as the energy balance (or ``energy returned on energy invested''). Figures compiled in a 2007 report by National Geographic Magazine point to modest results for corn ethanol produced in the US: one unit of fossil-fuel energy is required to create 1.3 energy units from the resulting ethanol. The energy balance for sugarcane ethanol produced in Brazil is more favorable, with one unit of fossil-fuel energy required to create 8 from the ethanol. Energy balance estimates are not easily produced, thus numerous such reports have been generated that are contradictory. For instance, a separate survey reports that production of ethanol from sugarcane, which requires a tropical climate to grow productively, returns from 8 to 9 units of energy for each unit expended, as compared to corn, which only returns about 1.34 units of fuel energy for each unit of energy expended. A 2006 University of California Berkeley study, after analyzing six separate studies, concluded that producing ethanol from corn uses much less petroleum than producing gasoline.
# Question:
# does ethanol take more energy make that produces
# Answer:''')

