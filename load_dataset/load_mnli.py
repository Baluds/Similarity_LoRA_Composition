from datasets import load_dataset

dataset = load_dataset("nyu-mll/glue", cache_dir="/project/pi_wenlongzhao_umass_edu/6/sudharshan/data", name="mnli")
train_dataset = dataset['train']
train_df = train_dataset.to_pandas()
test_matched_dataset = dataset['validation_matched']
test_matched_df = test_matched_dataset.to_pandas()
test_mismatched_dataset = dataset['validation_mismatched']
test_mismatched_df = test_mismatched_dataset.to_pandas()
output_file_path = "/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/mnli/"

train_df["label"] = train_df["label"].map({0: "entailment", 1: "neutral", 2: "contradiction"})
test_matched_df["label"] = test_matched_df["label"].map({0: "entailment", 1: "neutral", 2: "contradiction"})
test_mismatched_df["label"] = test_mismatched_df["label"].map({0: "entailment", 1: "neutral", 2: "contradiction"})

train_df.to_csv(f"{output_file_path}/train.csv", index=False)
test_matched_df.to_csv(f"{output_file_path}/test_matched.csv", index=False)
test_mismatched_df.to_csv(f"{output_file_path}/test_mismatched.csv", index=False)
print(train_df.head())