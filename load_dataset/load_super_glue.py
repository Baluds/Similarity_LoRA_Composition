import ast
from pathlib import Path
from datasets import load_dataset
import numpy as np

# ["cb", "copa", "multirc", "record", "rte", "wic"]
config_name = "rte"
dataset = load_dataset("aps/super_glue", cache_dir="/project/pi_wenlongzhao_umass_edu/6/sudharshan/data", trust_remote_code=True, name=config_name)
train_dataset = dataset['train']
train_df = train_dataset.to_pandas()
test_dataset = dataset['validation']
test_df = test_dataset.to_pandas()
output_file_path = f"/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/{config_name}/"

dir_path = Path(output_file_path)

# Create it if it doesn't exist
# dir_path.mkdir(parents=True, exist_ok=True)

# #### CB
# train_df["label"] = train_df["label"].map({
#     0: "entailment",
#     1: "contradiction",
#     2: "neutral"
# })
# test_df["label"] = test_df["label"].map({
#     0: "entailment",
#     1: "contradiction",
#     2: "neutral"
# })

#### COPA
# def format_copa_prompt(row):
#     instruction = f"Given the premise and two possible choices, select the most plausible {row['question']}."
#     return (
#         f"{instruction}\n\n"
#         f"Premise: {row['premise']}\n"
#         f"A: {row['choice1']}\n"
#         f"B: {row['choice2']}"
#     )

# train_df['combinedOption'] = train_df.apply(format_copa_prompt, axis=1)
# test_df['combinedOption'] = test_df.apply(format_copa_prompt, axis=1)
# train_df["label"] = train_df["label"].map({0: "A", 1: "B"})
# test_df["label"] = test_df["label"].map({0: "A", 1: "B"})

####MultiRC
# train_df["label"] = train_df["label"].map({0: "No", 1: "Yes"})
# test_df["label"] = test_df["label"].map({0: "No", 1: "Yes"})

####Record
# def format_copa_prompt(row):
#     answers = row['answers']
    
#     # Convert numpy array to list
#     if isinstance(answers, np.ndarray):
#         answers = answers.tolist()
    
#     # Extract the first item if available
#     if isinstance(answers, list) and len(answers) > 0:
#         return answers[0]
#     else:
#         return "UNKNOWN"
# train_df['answer'] = train_df.apply(format_copa_prompt, axis=1)
# test_df['answer'] = test_df.apply(format_copa_prompt, axis=1)

#### RTE
train_df["label"] = train_df["label"].map({
    0: "entailment",
    1: "not entailment"
})
test_df["label"] = test_df["label"].map({
    0: "entailment",
    1: "not entailment"
})

#### WIC
# def format_wic_prompt(row):
#     question = f"Does the word '{row['word']}' have the same meaning in both sentences? Answer with: yes or no."
#     return question

# train_df['question'] = train_df.apply(format_wic_prompt, axis=1)
# test_df['question'] = test_df.apply(format_wic_prompt, axis=1)
# train_df["label"] = train_df["label"].map({0: "no", 1: "yes"})
# test_df["label"] = test_df["label"].map({0: "no", 1: "yes"})

train_df.to_csv(f"{output_file_path}/train.csv", index=False)
test_df.to_csv(f"{output_file_path}/test.csv", index=False)
print(train_df.head())