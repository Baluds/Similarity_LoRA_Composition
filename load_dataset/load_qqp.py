import pandas as pd
from datasets import load_dataset

dataset = load_dataset("nyu-mll/glue", cache_dir="/project/pi_wenlongzhao_umass_edu/6/sudharshan/data", name="qqp")
train_dataset = dataset['train']
train_df = train_dataset.to_pandas()
test_dataset = dataset['validation']
test_df = test_dataset.to_pandas()
# train_df = pd.read_parquet("/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/qqp/train-00000-of-00001.parquet")
# test_df = pd.read_parquet("/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/qqp/validation.parquet")

output_file_path = "/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/qqp/"

def format_options_s1(row):
    return f"Question 1:\n{row['question1']}"
def format_options_s2(row):
    return f"Question 2:\n{row['question2']}"

# Add the new column
train_df['question1'] = train_df.apply(format_options_s1, axis=1)
train_df['question2'] = train_df.apply(format_options_s2, axis=1)
test_df['question1'] = test_df.apply(format_options_s1, axis=1)
test_df['question2'] = test_df.apply(format_options_s2, axis=1)
train_df["label"] = train_df["label"].map({0: "Not a Paraphrase", 1: "Paraphrase"})
test_df["label"] = test_df["label"].map({0: "Not a Paraphrase", 1: "Paraphrase"})
train_df.to_csv(f"{output_file_path}/train.csv", index=False)
test_df.to_csv(f"{output_file_path}/test.csv", index=False)
print(train_df.head())