# Task-Aware LoRA Adapter Composition  
_Dynamic retrieval & fusion of task-specific LoRA adapters_

## 1 What is this?
This repo implements a **retrieve-then-compose framework** that stores 22 LoRA adapters in a ChromaDB vector index and **dynamically merges them at inference**. Given an input, we  
1. embed the prompt with **MiniLM-L6-v2**,  
2. retrieve the top-p (p = 0.9) nearest training examples,  
3. derive a task-similarity distribution, and  
4. fuse the corresponding adapters with one of four PEFT strategies: **Linear, Magnitude-Prune, TIES, Concatenation**.

> **Why?** This lets a frozen Llama-2-7B backbone generalise to unseen tasks _without_ extra training, while using ~3 × less GPU memory than multi-task fine-tuning.


## 2  Key Results (Llama-2-7B, 15 NLP Tasks)

| Task         | Best Dynamic Merge | Task specific Fine-tuned Adapter | Δ (points) |
|--------------|-------------------|---------------------------|-----------|
| **PIQA**        | Linear – **70.95** | 46.00 | **+24.95** |
| **HellaSwag**   | Concatenation – **91.32** | 46.00 | **+45.32** |
| **RTE**         | Linear – **77.62** | 52.00 | **+25.62** |
| **MultiRC**     | Magnitude-Prune – **81.35** | 68.00 | **+13.35** |
| **StoryCloze**  | TIES – **70.09** | 72.00 | –1.91 |

*Linear or Magnitude-Prune beats individually fine-tuned adapters on **9 / 15** tasks while reducing GPU memory ~3×.*

---

## 3  Repository Layout

```text
.
├── adapters/           # 22 task-specific LoRA weights
├── chroma_index/       # MiniLM embeddings for retrieval
├── src/
│   ├── retrieve.py     # top-p similarity search & weighting
│   ├── merge.py        # Linear / TIES / Cat / MagPrune fusion
│   └── evaluate.py     # zero-shot evaluation harness
└── demo.py             # CLI demo for composed inference
