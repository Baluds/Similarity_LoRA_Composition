from datasets import load_dataset

dataset = load_dataset("google/boolq", cache_dir="/project/pi_wenlongzhao_umass_edu/6/sudharshan/data")
train_dataset = dataset['train']
train_df = train_dataset.to_pandas()
test_dataset = dataset['validation']
test_df = test_dataset.to_pandas()
output_file_path = "/project/pi_wenlongzhao_umass_edu/6/sudharshan/data/boolq/"

train_df["answer"] = train_df["answer"].map({False: "False", True: "True"})
test_df["answer"] = test_df["answer"].map({False: "False", True: "True"})
train_df.to_csv(f"{output_file_path}/train.csv", index=False)
test_df.to_csv(f"{output_file_path}/test.csv", index=False)
print(train_df.head())