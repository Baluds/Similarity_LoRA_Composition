from datasets import load_dataset

dataset = load_dataset("Rowan/hellaswag", cache_dir="/project/pi_wenlongzhao_umass_edu/6/sudharshan/data")
train_dataset = dataset['train']
train_df = train_dataset.to_pandas()
test_dataset = dataset['validation']
test_df = test_dataset.to_pandas()
output_file_path = "/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/hellaswag/"

def format_options(row):
    num_char_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    return "\n".join([f"Option {num_char_map[i]}: {option}" for i, option in enumerate(row["endings"])])

# Add the new column
train_df['CombinedOptions'] = train_df.apply(format_options, axis=1)
test_df['CombinedOptions'] = test_df.apply(format_options, axis=1)

train_df["label"] = train_df["label"].map({"0": "A", "1": "B", "2": "C", "3": "D"})
test_df["label"] = test_df["label"].map({"0": "A", "1": "B", "2": "C", "3": "D"})

train_df.to_csv(f"{output_file_path}/train.csv", index=False)
test_df.to_csv(f"{output_file_path}/test.csv", index=False)
print(train_df.head())