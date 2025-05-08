from datasets import load_dataset

dataset = load_dataset("nyu-mll/glue", cache_dir="/project/pi_wenlongzhao_umass_edu/6/sudharshan/data", name="mrpc")
train_dataset = dataset['train']
train_df = train_dataset.to_pandas()
test_dataset = dataset['test']
test_df = test_dataset.to_pandas()
output_file_path = "/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/mrpc/"

train_df["label"] = train_df["label"].map({0: "not equivalent", 1: "equivalent"})
test_df["label"] = test_df["label"].map({0: "not equivalent", 1: "equivalent"})

train_df.to_csv(f"{output_file_path}/train.csv", index=False)
test_df.to_csv(f"{output_file_path}/test.csv", index=False)
print(train_df.head())