import random
from datasets import load_dataset

dataset = load_dataset("pkavumba/balanced-copa", cache_dir="/project/pi_wenlongzhao_umass_edu/6/sudharshan/data", trust_remote_code=True)
train_dataset = dataset['train']
train_df = train_dataset.to_pandas()
test_dataset = dataset['test']
test_df = test_dataset.to_pandas()
output_file_path = "/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/copa/"
def format_options(row):
    row["label"] = "A" if row["label"] == 0 else "B"
    return f"Option A: {row['choice1']}\nOption B: {row['choice2']}"

# Add the new column
train_df['options'] = train_df.apply(format_options, axis=1)
test_df['options'] = test_df.apply(format_options, axis=1)
train_df["label"] = train_df["label"].map({0: "A", 1: "B"})
test_df["label"] = test_df["label"].map({0: "A", 1: "B"})
train_df.to_csv(f"{output_file_path}/train.csv", index=False)
test_df.to_csv(f"{output_file_path}/test.csv", index=False)
print(train_df.head())