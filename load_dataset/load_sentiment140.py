import random
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/sentiment140", cache_dir="/project/pi_wenlongzhao_umass_edu/6/sudharshan/data", trust_remote_code=True)
train_dataset = dataset['train']
train_df = train_dataset.to_pandas()
test_dataset = dataset['test']
test_df = test_dataset.to_pandas()
output_file_path = "/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/sentiment140/"
# def format_options(row):
#     return f"Option A: {row['sol1']}\nOption B: {row['sol2']}"

# Add the new column
# train_df['options'] = train_df.apply(format_options, axis=1)
# test_df['options'] = test_df.apply(format_options, axis=1)
train_df["sentiment"] = train_df["sentiment"].map({0: "negative", 4: "positive"})
test_df["sentiment"] = test_df["sentiment"].map({0: "negative", 4: "positive"})
train_df.to_csv(f"{output_file_path}/train.csv", index=False)
test_df.to_csv(f"{output_file_path}/test.csv", index=False)
print(train_df.head())